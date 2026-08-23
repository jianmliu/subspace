# Proof of Resident Weights (PoRW) 设计文档

**基于 TEE 硬件证明 + 推理搭便车审计的显存驻留共识**

状态：研究草案 v0.4（2026-08）

> English edition: [`proof-of-resident-weights.en.md`](proof-of-resident-weights.en.md)。
> 本中文版为规范版本，两版有出入时以中文为准。

> v0.2 变更：删除「专用扫描」模式——审计完全跟随真实推理（单一模式）；
> 抽签权重改为「唯一覆盖门票 × TEE 计量的服务量乘数（硬件包络封顶）」，
> 即多劳多得；MoE 天然支持（覆盖集 = 实际激活的专家）；
> 覆盖位图使副本交叉核验不依赖推理输入即可重算，验证性比 v0.1 更强。

---

## 0. 一句话概述

把 Subspace 的 Proof-of-Archival-Storage 骨架（PoT 时钟 + 每 slot 挑战 + 容量加权抽签 +
solution range 难度调整）保留，把「稀缺资源」从 SSD 上的唯一编码 plot 换成
**经 TEE 证明驻留在 GPU 显存中的大模型原始权重**；副本唯一性问题不再用密码学密封解决，
而是用 GPU/CPU 机密计算的远程证明（attestation）把「一张物理卡」作为防 Sybil 的计数单位；
审计不引入任何额外的显存读取——承诺只在推理 decode 本来就在做的权重扫描上
顺带累计，抽签权重正比于真实发生的服务量（多劳多得），「跑推理即出块」。

## 1. 背景与动机

### 1.1 Subspace 现状（本仓库）

Subspace 共识由三层配合构成：

1. **资源 = 存储的历史数据量（存量）**。农民把纠删码后的历史 plot 成 sector
   （每 sector ≤1000 piece，`crates/subspace-runtime/src/lib.rs` 的
   `MAX_PIECES_IN_SECTOR`），中签概率正比于存储量。
2. **审计**：每个 slot（1s，`SLOT_DURATION = 1000`）由 PoT 挑战为每个 sector 派生一个
   s-bucket，农民读取几十 KB 并检查 32 字节 chunk 是否落入 `solution_range`
   （`crates/subspace-farmer-components/src/auditing.rs`）。
3. **防「以算代存」**：plot 编码使用 Chia 风格 PoS 表 + KZG（`subspace-proof-of-space`），
   按需重算远贵于读盘；PoT（顺序 AES，`subspace-proof-of-time`）作为不可加速的硬时钟，
   把响应窗口卡死。

关键性质：**每个农民的 plot 字节是唯一的**——`SectorId::new(public_key_hash,
sector_index, history_size)` 用农民公钥播种编码。这是防 Sybil 的前提：没有唯一编码，
一份物理数据可以替任意多个身份应答审计。

### 1.2 为什么要改

AI 推理是带宽/显存受限的工作负载。我们希望共识的稀缺资源与 AI 硬件重合：
**驻留在 GPU 显存中的模型权重**。这带来根本矛盾：

- 共识要求副本唯一（否则 Sybil）→ 需要每农民密封编码；
- 推理要求权重是原始格式 → 密封后的字节不能喂给推理框架；
- 「便宜可逆」的密封也不行：解码便宜 ⇒ 按需编码也便宜 ⇒ 一份原始数据 + 少量算力
  可以现场伪造任意身份的「唯一副本」。

不可能三角：**推理可用（解码快）／副本唯一（按需重编码贵）／不额外占空间**，最多取二。
纯密码学框架下无解。本设计用 TEE 换掉「密码学唯一性」这一角。

### 1.3 设计目标

| # | 目标 | 度量 |
|---|------|------|
| G1 | 同一份显存字节既是共识抵押又是推理权重 | 无双副本，无密封编码 |
| G2 | 防 Sybil：奖励与物理硬件成正比，与身份数无关 | 一卡一注册 |
| G3 | 审计零额外带宽：只计量推理真实发生的读取，无专用扫描模式 | 单一执行模式 |
| G4 | 多劳多得：抽签权重正比于实际服务量，MoE 等部分激活模型天然支持 | 覆盖门票 × 服务量乘数 |
| G5 | 保留 Subspace 的抽签/难度调整/PoT 骨架 | 最小化共识层改动 |
| G6 | 信任依赖显式化、有界、可治理 | 度量白名单上链；TEE 破裂时通胀被硬件包络封顶 |

**明确接受的信任依赖**：NVIDIA（GPU 设备身份与 CC 固件）、Intel TDX / AMD SEV-SNP
（CVM）、经治理白名单的 agent 度量值。这是设计前提，不再重复论证。

## 2. 架构总览

```
                          ┌────────────────────────────────────┐
                          │  链上 (Substrate runtime)           │
                          │  pallet-subspace (改)               │
                          │   ├─ 设备注册表 DeviceRegistry      │
                          │   ├─ 模型注册表 ModelRegistry       │
                          │   ├─ 度量白名单 MeasurementSet      │
                          │   ├─ 抽签验证 (solution range)      │
                          │   └─ 质押/罚没 Bond & Slashing      │
                          └───────────▲────────────────────────┘
                                      │ Solution{sketch, sig, device_id}
        PoT 链 (不变) ──challenge──►  │
                          ┌───────────┴────────────────────────┐
                          │  节点侧 CVM (TDX/SEV-SNP)           │
                          │  PoRW Agent (度量、开源、可复现构建)  │
                          │   ├─ 持有节点密钥 (CVM 内生成)       │
                          │   ├─ 管理权重加载/驻留               │
                          │   ├─ 推理服务 (对接 ai3-inference)   │
                          │   └─ sketch 采集与签名               │
                          └───────────▲────────────────────────┘
                                      │ CC 加密 PCIe / NVLink
                          ┌───────────┴────────────────────────┐
                          │  GPU (H100/H200/B200, CC mode)      │
                          │   HBM: 模型权重 (原始格式, Merkle 化) │
                          │   Kernel: 推理 + 融合 sketch 累加器  │
                          └────────────────────────────────────┘
```

四个信任支柱，缺一不可、互为冗余：

1. **B 支柱（设备 attestation）**：设备唯一身份 + 代码度量 ⇒ 防 Sybil、协议合规性。
2. **C 支柱（sketch）**：挑战随机化的字粒度线性草图 ⇒ 证明字节真实**驻留**且被
   读取，是 TEE 被攻破时的第二道防线（timing + 副本交叉核验仍然成立）。
3. **D 支柱（推理证明）**：每请求 attestation quote ⇒ 证明**输出真实性**
   `O = M_registered(I)` 在 attested TEE 内产生（§4.7）。
4. **PoT（不变）**：不可加速的时钟 ⇒ sketch 的响应期限有客观依据。

## 3. B 支柱：设备证明与注册

### 3.1 硬件基础

- **GPU**：NVIDIA Hopper 起支持机密计算（CC mode）。设备内熔断唯一身份密钥，
  证书链锚定到 NVIDIA 根 CA；attestation report 覆盖 VBIOS/固件/驱动度量与 CC 状态，
  经 SPDM 会话获取。Blackwell 改善了 CC 性能开销并支持受保护 NVLink 的多卡 CC 域。
- **CPU/VM**：GPU CC 要求宿主为机密虚拟机（Intel TDX 或 AMD SEV-SNP）。
  CVM 的 launch measurement 覆盖 PoRW Agent 镜像。
- **限制**：消费卡（RTX 40/50 系）无 CC。B+C 路线是数据中心卡专属；
  消费卡的参与走降级路径（§8.3）。

### 3.2 注册流程（一卡一身份）

```
1. Agent 在 CVM 内生成节点密钥对 (sk, pk)，sk 永不出 CVM。
2. Agent 采集: CVM attestation (含 agent 度量、pk 绑定)
             + GPU attestation report (含 device_id、CC 状态)。
3. 链上 register_device(evidence):
   a. 验证 NVIDIA / Intel / AMD 证书链与报告签名。
   b. 检查 agent 度量 ∈ MeasurementSet (治理维护的白名单)。
   c. 检查 device_id 未注册 (防同卡多注册)。
   d. 要求质押 bond (罚没抵押)。
   e. 写入 DeviceRegistry: device_id → (pk, vram_capacity, epoch)。
4. 周期性 re-attestation (epoch 级)，过期自动移出抽签集合。
```

证书链验证较重，不必全部在线上执行：可采用「乐观验证 + 欺诈证明窗口」或由轻量
验证委员会预验签、链上只验委员会聚合签名。原型阶段直接 native 验证即可。

### 3.3 attestation 提供了什么、没提供什么

| 提供 | 不提供（由 C 支柱补） |
|------|----------------------|
| 这是一张真实的、唯一的物理 GPU | 权重此刻真的在 HBM 里 |
| 运行的 agent 代码是白名单版本 | 权重被真实读取（而非声明） |
| 节点密钥被正确的代码持有 | 响应的时效性 |
| PCIe/显存内容对宿主不可见 | TEE 自身未被攻破 |

由于 agent 代码本身是被度量的、诚实的，「权重放在主存里分页进来」这类作弊在 TEE
未被攻破时**由代码构造排除**——作弊者无法运行改过的 agent（度量对不上）。
C 支柱的 sketch 时限是针对「TEE 被攻破」情形的纵深防御。

## 4. C 支柱：跟随推理的驻留审计与服务量计量

### 4.1 核心观察

LLM decode 每生成一个 token，都要把全部权重从 HBM 完整扫一遍（这正是推理带宽
受限的原因）。共识想强制的行为——「每 slot 高强度读取驻留数据」——**推理已经在做了**。
审计只需在这条已付费的数据流上顺带累计一个承诺。

### 4.2 Sketch 定义（整数域，与推理数值完全解耦）

设权重按 4 KiB tile 划分：`W = {T_0, T_1, ..., T_{n-1}}`，Merkle 根 `R_W` 登记在
ModelRegistry。每个 slot 由 PoT 得到全局挑战 `c_t`（复用现有
`global_challenge` 派生路径）。设本 slot 内推理**实际读取过**的 tile 集合为
`C_t ⊆ {0..n-1}`（覆盖集，dense 模型的忙时 decode 下 ≈ 全集，MoE 下 =
实际被路由到的专家 + 共享层）。定义：

```
r_i      = PRF(c_t, device_id, i)              // 每 tile 的种子
c_{i,j}  = PRF'(r_i, j)                        // 每【字】的系数, 寄存器内生成
sketch_t = Σ_{i ∈ C_t} Σ_j  c_{i,j} ⊙ w_{i,j}  // 模 2^32/2^64 乘加
```

> **系数必须细到字粒度且每 slot 新鲜**（v0.2.1 修正，详见
> `porw-p1-feasibility.md` §3）：若 tile 内共用系数，sketch 坍缩为
> `r_i · L(T_i)`（L 为固定线性泛函），作弊者每 tile 存 4 字节的 L 值
> 即可通过所有未来审计（1024 倍压缩）。PoC 中该攻击有可执行演示，
> 字粒度 PRF 系数（murmur3 fmix32，~6 条整数指令/字）下攻击失败。

要点：

- **只在推理已经发生的读取上累计**。乘加融合进 decode kernel 的
  K 循环（权重 tile 此刻已在寄存器/SMEM 中；注意是 mainloop 而非
  epilogue——GEMM epilogue 只见输出 C tile，见不到权重字节流），
  不发起任何额外的 HBM 读取。没有专用扫描模式，卡在做什么就计量什么。
  实际部署为 **S1+S2 混合**（`porw-p1-feasibility.md` §2）：融合手术只在
  覆盖依赖负载的 MoE Triton kernel 必要；dense/闭源（cuBLAS）路径覆盖
  恒为全量，用每 slot 一遍的独立扫描（~2.4% 带宽税）零语义损失地兜底。
- **对原始字节做整数乘加**，不是浮点运算——完全确定、跨卡跨驱动可复现，
  与推理框架的数值行为无关。
- 系数每 slot 更换 ⇒ `sketch_t` 无法用旧 slot 的结果拼凑，`C_t` 中每个 tile
  必须在**本 slot 内**被真实读过至少一次（Freivalds 式随机线性草图：
  存低精度/低秩近似的作弊者必然算错）。
- `device_id` 混入 PRF ⇒ 不同卡的 sketch 不同，无法转发抄袭。

### 4.3 密码学的硬边界：sketch 只能证明「至少一次」

一个必须直面的事实：系数以 slot 为粒度，同一 tile 在 slot 内被读 1 次与
k 次，对 sketch 的贡献可以由一次读取推算（乘以 k 即可）。因此
**sketch 能证明的物理量上限是「每 slot 每 tile 至少一次真实读取」**，
即唯一覆盖 `C_t`；超出一遍的读取次数、token 数量等「劳动强度」
在密码学上不可由 sketch 证明。

由此，抽签权重拆成两个因子，各自由能负责它的机制背书：

| 因子 | 含义 | 背书机制 |
|------|------|----------|
| **覆盖门票** `|C_t|`（字节计） | 这些权重字节驻留且本 slot 被真实读过 | sketch（密码学，可交叉核验） |
| **服务量乘数** `m_t` | 本 slot 实际完成的服务量（见 §4.4） | TEE 计量 + 硬件包络封顶（§4.6） |

### 4.4 多劳多得：服务量计量

Agent（被度量的可信代码）在 CVM 内如实统计本 slot 的服务量：

```
m_t = (本 slot 内完成的 decode 步数 × 该步实际读取的权重字节) / |C_t| 字节
```

即「以覆盖集为单位，权重被完整扫过多少遍」。性质：

- **dense 模型忙时**：每 token 一遍全权重 ⇒ m_t ≈ slot 内 token 步数
  （batch 内多个请求共享同一遍读取，m_t 计带宽遍数而非 token 数——
  见 §10 关于是否引入 token 加权的讨论）。
- **MoE**：冷专家不在 C_t 里就不计票，热专家读多少算多少。
  无需补扫，多劳多得的语义天然成立。
- **闲置卡**：没有推理就没有票。理性农民会自发跑自生成负载
  （batch=1 的 decode 紧循环，物理上趋同于 v0.1 的 sweep kernel）。
  协议不区分真实/自生成负载，也不需要区分：这构成显卡收益下限，
  真实服务的溢价由推理费用市场承担（§5.1）。协议因此少一个执行模式。

### 4.5 时序与抽签（复用 Subspace 骨架）

```
slot t:  PoT ──► c_t
         │
         ├─ GPU: decode 照常进行, epilogue 顺带累计 sketch_t 与覆盖位图 C_t
         ├─ Agent: tickets = hash_expand(sketch_t, |C_t| × m_t)   // 展开长度∝票数
         │         对每 32B chunk 检查 is_within_solution_range(...)  // 复用现逻辑
         └─ 中签 ⇒ Solution {
               device_id, model_id, slot,
               sketch_t, coverage_bitmap, partials_root, m_t, chunk_index,
               //        ^ per-tile sketch 值的 Merkle 根（kernel 本就产出
               //          per-tile partials）——使核验可按单 tile 抽查（§4.6）
               sig = Sign_sk(...)            // 节点密钥, attestation 背书
            } ──► 出块
```

- **工作量加权**：chunk 流展开长度 ∝ `|C_t| × m_t`（覆盖字节 × 扫描遍数 =
  本 slot 真实流过的权重字节），与 Subspace「每 chunk 一张彩票」语义一致，
  难度调整逻辑（pallet-subspace 的 solution range era 调整）原样复用，
  全网难度自动跟随全网真实推理吞吐。
- **响应期限**：沿用 `BLOCK_AUTHORING_DELAY`，作为 TEE 破裂时的纵深防御
  （PCIe/网络流式作弊来不及，§6.2）。

### 4.6 链上验证

快速路径（每块必做，代价 ~几次签名验证）：

1. `device_id ∈ DeviceRegistry` 且 attestation 未过期；
2. `sig` 对应注册的 `pk`；
3. `model_id ∈ ModelRegistry` 且该设备登记了此模型；
4. **硬件包络检查**：`|C_t| × m_t ≤ 该卡型号登记的 HBM 带宽 × slot 时长`。
   这给 TEE 计量器封了物理顶：即使 agent 被完全攻破，单卡可虚报的票数
   也不超过其真实带宽上限的常数倍——TEE 信任的失效模式是**有界通胀**，
   不是无界伪造；
5. chunk 落入 solution range（同现有 `subspace-verification` 逻辑）。

深度路径（抽查 + 欺诈证明）——分两级，均以 `partials_root`
（per-tile sketch 值的 Merkle 根）为锚，欺诈证明因此是**单 tile 粒度**：
取回一个 tile 的字节 → 重算 `s_tile` → 与 Merkle 路径上的承诺比对，
O(4KiB) + 一条路径，而非重算整个模型：

- **快速抓骗 = VRAM 副本交叉核验**：每 epoch 随机指派持有同一模型的
  其他注册设备，对声明的 `C_t` 抽查若干 tile（权重就在自己 HBM 里，
  抽查近乎免费；全量重算也仅 ~24ms）。**不需要知道被核验者的推理输入**
  （sketch 只依赖系数、tile 字节与覆盖集，不依赖激活值）。不一致 ⇒
  欺诈证明 ⇒ 罚没 bond 并吊销 device_id，举报人分赏金。诚实多数假设
  仅需在「同模型副本集合」内成立——热模型副本多，天然满足，
  故绝大多数票权被本级覆盖。
- **终审 = 存储轨仲裁**（§5.6 风险 2）：孤本模型、或交叉核验双方各执
  一词时，DSN 碎片对 `R_W` 是规范字节。任何 PoAS 农民检索含争议 tile
  的 piece（~MiB 级）重算并提交欺诈证明——最终裁决不依赖任何 VRAM
  副本存在，也裁决核验者本身的作恶。挑战期长度按 DSN 检索延迟定参。

#### 4.6.1 副本交叉核验的 epoch 调度（已实现）

交叉核验可行的根源是一条不对称性：sketch 的 slot seed 是
`derive_slot_seed(global_challenge, device_id)`——**公开**且**逐设备**。
逐设备保证同一模型的两个副本对同一 tile 算出不同 sketch（副本唯一性
所需）；公开则意味着**任何持有该模型真实字节的副本，都能用目标设备的
seed 复算其任意 tile 的承诺值**。所以「只有副本能审计副本，而任一副本
能审计任意同模型对等设备」——调度就是给这条既有欺诈证明路径排班。

**两 epoch 流水线 + 奖励托管**：

```
epoch e      提交期：正常出块，累积 partials_root 承诺
e 边界       定信标 B_e（epoch 边界随机性）→ 全网本地推出派单
epoch e+1    审计窗口：被指派副本复算比对 e 的承诺；不一致 ⇒ TileFraudProof
e+2 结算     e 的区块奖励此时才从托管释放（铸造）
```

关键是**奖励托管**（`EscrowedRewards`）：epoch e 挣的区块奖励**不铸造**，
押到 e+2 结算、审计窗口无欺诈后才 mint 给设备所有者。窗口内被证欺诈 ⇒
托管直接删除（从未进入供给）+ bond 罚给举报人 + 吊销设备。托管未释放
期间禁止注销设备（`EscrowPending`），堵住「带着待审计的报酬跑路」；
bond 本身的退出延迟（pending-exit 状态机）留作后续工作。

**派单是纯函数,几乎零链上状态**（`subspace-proof-of-residency`）：

- `audit_beacon(epoch, entropy)`：epoch 信标。entropy 必须到 epoch 边界
  才可知（生产取 PoT 派生随机性；pallet 目前用边界块 parent hash 占位,
  已注明可被边界块作者在其解集内 grind,上线前须换）——否则说谎者可
  预判被抽 tile、只对那些 tile 备好真值；
- `select_auditors(B, model, target, replicas, k)`：对每个目标取副本集内
  rank hash 最小的 k 个（排除自身),扇出 k≈3；
- `audit_tile_sample(B, model, target, auditor, n_tiles, t)`：每对
  (审计者,目标) 独立抽 t 个不重复 tile,不同审计者的样本互异,合并覆盖
  面更宽。

链上只存信标（`AuditBeaconValue`,epoch 边界随结算写入）；无派单表、
无 ack、审计者不领干净审计的酬劳——调度只为诚实节点**划定看哪里、
把带宽约束住**,强制力全在无许可的欺诈证明路径上（举报有赏）。

**抽样量与成本**：设伪造比例 f,总抽样 N,漏检率 = (1−f)^N。
f=1% 时 N≈690 达 99.9% 检出；每 tile 4 KiB,对 TB/s 级 HBM
复算成本可忽略,每副本每 epoch 平均只审 k≈3 个对等设备、MB 级流量。
真正的约束不是算力而是 **opening 可得性**：审计者需要目标承诺 tile 的
Merkle opening（partials 树按 coverage 顺序建,故欺诈证明携带
`partials_index` 定位叶位置——叶哈希本身绑定 tile_idx,位置说谎只会
验证失败,不可能移花接木）。目标须在审计窗口内按请求提供 opening,
拒供即无法自证承诺,按不可用处置——这是一个小 DA 子问题。

**单副本模型**（`ReplicaCount == 1`）没有对等审计者：回退存储轨仲裁
（下节终审路径),这正是 `min_replicas` 作为服务参数的意义。副本
中途加入/退出时,只审计双方同在副本集期间的承诺,以 `registered_at` /
announce 块界定。

实现映射：派单纯函数与 `partials_index` 修复在
`subspace-proof-of-residency`；托管、信标、罚没、注销门控在
`pallet-porw-registry`（`note_block_reward` / `EscrowedRewards` /
`AuditBeaconValue` / `forfeit_escrow`）；审计者侧
（`audit_duties` / `cross_check`,含产出可直接提交的欺诈证明）在
`porw-agent::audit`；端到端(说谎副本被审计员抓获→bond 罚没+托管追回)
见 `porw-devnet` 的 `cross_audit_catches_a_lying_replica_end_to_end`。

**本栈的边界（诚实重申）**：以上抓的是「驻留与覆盖造假」。`m_t` 的
包络内虚报密码学上不可抓（§4.3 硬边界），其防线仅为 TEE 度量 +
包络封顶 + 统计异常检测。

但该软点的经济敞口远小于表面，且随时间萎缩：

- **`m_t` 只影响区块奖励，不影响服务收入**。推理费用按请求结算、由
  D 支柱 quote（§4.7）逐笔背书、用户终验输出——虚报 `m_t` 多拿不到
  一分钱服务费；刷服务费须真实燃烧手续费（§5.1），是付费不是作弊。
- **作弊收益被包络精确限定为「省电」**。忙碌农民（真实负载）、闲置
  诚实农民（垃圾 decode 打满带宽、付真实电费）、`m_t` 谎报者三者的
  票数同被包络封顶——谎报买不到超额票数，只省下未执行 sweep 的功耗
  （单卡百瓦级），而被抓代价是罚没 bond + 吊销设备，天然负期望。
- **软点权重随收入结构衰减**：`m_t` 只作用于通胀补贴；随网络成熟、
  费用收入（密码学上硬）占比上升（§5.2 补贴→费用过渡），唯一软点
  在总收入中的权重单调下降。
- `m_t` 本身不可被外部重算（它是对真实负载的计数），其防线是
  快速路径的包络封顶 + attestation 度量 + 统计异常检测
  （长期贴着包络顶且无对应推理收入的设备可被治理层调查）。

### 4.7 D 支柱：推理证明与端到端 agent 生命周期

现今 TEE-ML 生态（Phala、Atoma、NVIDIA CC 等）已能对**每个推理请求**签发
attestation quote，把 `(input_hash, output_hash, model_measurement,
cvm_measurement)` 绑定到设备密钥。把它作为第四支柱纳入，`O =
M_registered(I)` 在 attested TEE 内产生这一事实变得**可验证**。

**为什么 C 与 D 都需要、不冗余**（关键澄清）：

| | C 支柱 sketch | D 支柱 推理证明 |
|---|---|---|
| 证明什么 | 权重**驻留**且被读取（容量） | **输出真实性** `O=M(I)`（有用性） |
| 何时发生 | 挑战驱动，**每 slot**，无请求也有 | 请求驱动，**有请求才有** |
| 服务对象 | 共识抽签（稀缺资源度量） | 费用市场（可验证收据） |

空 slot：只有 C（自生成负载 / sweep）→ 收益下限（§4.4、§5.1）。有请求
的 slot：C+D → 费用市场结算 + 费用燃烧加权（§5.1）有了可验证依据。
D 证明不了「此刻持续驻留 X 字节」这个抽签需要的连续容量承诺——所以
不能用 D 取代 C。两者的模型身份复用**同一个 `R_W` root**，互相加固。

**D 绕过了确定性难题**：验证推理最难的是浮点跨节点不可复现（ZKML 昂贵、
TOPLOC 类重算脆弱）。TEE attestation 不需要任何重算或 bit 一致——只证明
「此 attested CVM 运行被度量的代码，对 I 产出了 O」，信任在 TEE 而非复现。
这正是本设计选 TEE 信任根后，proof-of-useful-work 从「难做」变「可落地」
的原因。

**闭环为端到端 agent 生命周期**，每步 attested：

| 生命周期步 | 承载 | 证明 |
|---|---|---|
| 感知（读 memory） | 存储轨（§5.4） | 永存 + 信封加密，明文仅在 attested CVM 内解开 |
| 思考（驻留权重推理） | 显存轨 | 驻留（C:sketch）+ 设备（B:attestation） |
| 行动（产出输出） | 推理证明 | D 支柱：`O=M(I)` 在 attested TEE 内 |
| 记忆（写 memory） | 存储轨 | 永存 + memory root 上链 |
| 身份 / 密钥 | CVM | 节点密钥 CVM 内生成、永不出 TEE |

四支柱共用同一 TEE 信任根，覆盖 agent「感知→思考→行动→记忆」的完整
回路。落地上，D 支柱对接 `ai3-inference`（推理层）、记忆对接
`aigg-memory`、结算与费用燃烧对接 `aigg-facilitator` / `aigg-wallet`——
PoRW 共识 + 四支柱证明构成链原生、可验证、可永生的 agent 生命周期底座。

## 5. 经济与治理

### 5.1 自生成负载（wash trading）：接受它，并给它定价

抽签权重跟随实际推理量后，必须回答：农民给自己发请求刷量怎么办？
本设计的立场是**不在共识层阻止它**，理由：

- 自生成负载也必须真实消耗 HBM 带宽（sketch + 包络保证），
  它买不到超过物理上限的票——这就是「junk 挖矿 = 带宽 PoW」，
  和闲置扫描是同一件事，构成收益下限而非漏洞；
- 区分「真实用户」与「自己」在无许可网络里不可判定；
- 真正的分层应发生在费用侧：**共识奖励**支付给物理资源
  （驻留 + 带宽，junk 与真实同价），**推理费用**支付给有用性
  （只有真实用户才付费）。若希望共识奖励也向真实服务倾斜，
  可引入费用燃烧加权（付费请求燃烧的手续费按比例加成票数——
  自刷需真金白银燃烧，wash trading 有了成本），作为治理可调参数。
  该加权的「付费请求确为真实推理」由 D 支柱（§4.7）的推理证明背书，
  不再是纯信任 agent 自报。

### 5.2 区块奖励作为推理费用的原生货币

出块奖励发给的正是提供推理容量的资源（驻留权重的显卡），所以用同一代币
给 AI agent 的推理费计价，是闭环而非牵强——这也是 Filecoin/Render 等
资源网络的标准形态。飞轮：

```
通胀奖励 → 补贴待机容量（收益下限, §5.1）
agent 付费推理 → 费用部分燃烧 + 部分给服务方
需求↑ → 燃烧↑ → 净通胀↓ → 代币升值 → 显卡收益↑ → 容量↑ → 服务更好/更便宜
```

成立的三个理由：

1. **内生结算**：agent（尤其链上 agent）需要可编程、可流式、可按 token
   微支付的结算方式，原生代币天然满足；与费用燃烧加权（§5.1）无汇率
   摩擦地衔接——同一资产既是奖励、又是费用、又是抗刷量的燃烧标的。
2. **需求驱动安全**：费用燃烧把推理需求直接转化为共识安全预算
   （EIP-1559 式），网络越有用越安全。
3. **供给侧自举**：早期无需求时，通胀奖励维持容量在线（standby 补贴）；
   需求起来后收入结构自然从通胀转向费用（Bitcoin 补贴→手续费的过渡曲线，
   但这里过渡由真实服务收入驱动）。

三个必须直面的设计点：

- **计价 ≠ 结算**。推理有真实法币成本（电、折旧），币价波动会打穿服务方
  利润。报价应锚定算力成本（法币或计算单位），代币仅作结算资产
  （oracle 换算）；把波动资产直接当记账单位是资源网络的经典死因。
- **死亡螺旋**：币价跌 → 显卡退出 → 容量降 → 服务差 → 需求再降。
  缓解：注册 bond 的退出延迟（本设计已有）、长期驻留奖励加成、
  协议金库在低价期回购容量。
- **纯通胀风险**：无需求时代币只有挖矿产出没有消耗场景。对冲：燃烧之外,
  可要求 agent/聚合商质押代币获取服务配额与优先级（work-token 模式），
  给代币一个与用量成正比的锁定需求。

### 5.3 模型准入：质押上架 + 需求跟随权重

> ★ 需求跟随权重已在 `pallet-porw-registry` 落地:每模型 `demand_ema`(费用燃烧的 EMA)+ `floor_weight`;`record_inference_fee` **真实燃烧**调用者货币后累加需求信号(刷量有真金白银成本,§5.1);`settle_epoch` 每 epoch 折算 `有效权重 = clamp(demand_ema/费用单位, floor, 上限)`;`model_reward_weight` 供奖励层读取。EMA 平滑提供 §10.1 的滞后。runtime 已接入,14 项测试。

- **ModelRegistry**：`model_id → (R_W merkle root, size, version, min_replicas,
  reward_weight)`。
- **准入三阶段**：
  1. 初期治理白名单——把有限显存集中到少数模型凑够副本数
     （`min_replicas` 是安全参数：副本交叉核验 §4.6 依赖同模型副本集合）。
  2. 成熟期开放上架：任何人质押 + 付上架费 + 许可证声明即可提案；
     治理保留「移出奖励权重」的下架权（停止补贴 ≠ 抹除数据，见 §5.4）。
  3. **奖励权重跟随需求**：每模型权重 ∝ 其近期付费推理费用燃烧量的
     滑动平均。无人使用 → 权重衰减 → 农民腾显存给热模型；刷量者必须
     真实燃烧费用（= 付费预订容量，市场行为而非攻击，与 §5.1 一致）；
     通胀补贴精确流向被验证的有用性。
- **冷启动补贴**：新模型无需求 → 无权重 → 无人加载的死锁，由上架质押
  的一部分转化为前 N epoch 的引导权重解决（提案者购买初始容量；
  社区持续质押可维持长尾模型存活）。
- 均衡预期为幂律：少数开源旗舰占据多数副本，长尾靠质押维持。
  奖励权重曲线即网络的容量规划器。
- **权重分发**：权重本体作为数据上传进 Subspace DSN（现有 archiver/gateway
  路径原样可用，激励来源见 §5.4 存储轨），新节点从 DSN 拉取，
  对着 `R_W` 验证。

### 5.4 数据永存：双轨共识（v0.3 修正）

**PoRW 显存层不提供、也不应提供任何永存性**——它是被经济锚定的热缓存：
模型奖励权重衰减到零，农民即清出显存。永存必须由 DSN/归档层承诺，
而这暴露了 v0.2 的一个结构缺口：单轨 PoRW 裁掉了原本付钱给存储层的
PoAS 抽签，「权重上传进历史」失去激励来源。修正为**双轨共识**：

| 轨道 | 奖励份额 | 机制 | 购买的东西 |
|------|---------|------|-----------|
| 显存轨 (PoRW) | X% | 本设计（§3–4） | 热模型的推理容量 |
| 存储轨 (PoAS) | Y% | 原 Subspace（唯一编码 plot + s-bucket 审计，SSD） | 全部历史数据的永存 |

X:Y 为治理参数。双轨同时解决两个遗留问题：历史与退役权重住在便宜的
SSD 上（显存太贵、存不下历史的问题消解）；chiapos/KZG/plotting 在
存储轨原样保留——§7 表中「不再需要」仅指显存轨不再需要密封编码。

**永存语义（白皮书措辞级）**：所有上架过的模型权重进入归档历史，
纠删码分片散布于全体 SSD 农民；模型退役出显存层后字节仍在档案层，
需求回归可重新上架。这是**经济-概率性**保证（存储轨通胀持续 +
存储成本下降快于历史增长 ⇒ 副本数维持），不是密码学绝对保证——
链活着数据在，不应对外承诺「绝对永存」。

**永存 vs 下架的张力在分层中化解**：治理只能动奖励权重（停止补贴
某模型的显存副本），动不了归档层——DSN 分片内容无关、单农民持有的
是不可解读的碎片。审查抵抗留在档案层，合规作用在补贴层。退役模型
版本对可复现性研究、模型谱系追溯有长期价值——「模型的国家图书馆」
是存储轨通胀开支的正当性叙事之一。

**叙事分层：记忆归存储轨，思考归显存轨。** agent = 记忆（数据）+
思考（在共享模型上的推理），映射到双轨严丝合缝：

- 存储轨承载 **agent memory 的永存**（对话史、经验、知识库、embedding、
  LoRA 增量；memory root 上链，内容纠删码归档）——agent 的「灵魂」，
  写入一次付费、几乎零边际成本地永续；
- 显存轨承载**算力市场**——agent 的「大脑」，按 token 付费租用。

由此得到中心化平台给不了的性质：**agent 可暂停、可复活**——状态独立
于任何服务商，停用多年后任何显存轨节点拉取 memory + 加载注册模型即可
原地复活。「记忆永存 + 算力按需 = 链上永生的 agent」是压轴叙事，
模型图书馆退居支撑性基础设施。经济上 agent 的代币消耗从单一推理费
扩展为「生存开销」（写记忆 + 做推理），两条轨的费用流各自闭环。
技术边界天然正确：sketch 只对注册权重计票，memory/KV/RAG 读取不计票
（§6.1），两轨审计对象零重叠。**记忆隐私**：memory 密文上链
（持有者密钥信封加密），明文仅在 attestation 通过的 CVM 内解开——
隐私叙事与算力叙事共用同一个 TEE 信任根。

### 5.5 模型生命周期
- **版本更新**：新版本 = 新 `model_id`。旧版本设置 deprecation epoch，
  期内双计奖励，之后移出抽签集合。无 re-plot 成本（没有密封编码），
  切换成本 = 下载 + 载入显存。
- **多模型/分片**：一张卡可登记多个模型（7B+13B），多卡 NVLink CC 域可登记
  张量并行分片的大模型——sketch 按分片累加，聚合签名。

### 5.6 热层驱逐：非热点模型的三个风险与补强

显存层是需求驱动的热缓存（§5.4），非热点模型不保证有 VRAM 副本。
数据不丢（存储轨），但有三个真实风险，各配一个补强机制：

**前置澄清：同一份权重的三种存储形态。** 存储轨的均匀随机抽样
（农民不可挑选碎片）与推理需要完整权重**不冲突**——因为推理热路径
从不直接读存储轨，桥梁是「DSN gather 重组 + 本地完整缓存」：

| 形态 | 内容 | 谁选择 | 激励 | 用途 |
|---|---|---|---|---|
| VRAM 原始权重 | 完整、自选模型 | 农民自由挑（§5.3 需求跟随） | C 支柱抽签 + 推理费 | 推理热路径 |
| 本地 SSD 完整副本 | 完整、原始格式 | 农民自由囤 | 间接：秒级重载 + 复活赏金（下述温层） | 显存载入源 |
| DSN 唯一编码碎片 | 均匀抽样、不可选 | 协议指派 | PoAS 存储轨奖励 | 永存 + 终审（风险 2） |

上架流程：DSN gather（凑纠删码阈值重组完整权重，分钟级）→ 对
`R_W` Merkle 根逐 tile 验证 → 本地 SSD 留完整副本 → 载入显存。
此后一切重载走本地副本（秒级）；gather 仅首次或副本丢失时发生。
本地完整副本不是唯一编码 plot、不参与共识计票——它是纯效用存储，
由重载速度与复活赏金间接激励。

**风险 1：冷启动延迟（服务可用性）。** 被逐出的模型来了请求要「复活」：
本地无副本时需 DSN gather 几十 GB + 载入显存，分钟级。
**补强：分层服务等级 + 预热市场。** 模型层级可查询、延迟可预期：
热（≥min_replicas，即时）/ 温（农民本地 SSD 投机缓存，秒~分钟）/
冷（仅 DSN，分钟~小时）。付费请求可附**复活赏金**：第一个加载该模型并
交付 attested 服务的节点领赏——冷启动变成市场行为，且农民有动机用
便宜的本地 SSD 囤「可能复活」的退役模型抢赏金，温层自发形成。

**风险 2：孤本模型的核验失效（安全级，最重要）。** §4.6 的副本交叉核验
要求同模型副本集合内诚实多数；只剩 1–2 个副本的模型，其 sketch 无人
能核验，TEE 破裂时纵深防御对孤本失效。
**补强：存储轨即仲裁法庭。** sketch 只依赖（挑战系数 × 权重字节 ×
覆盖集），而存储轨持有全量权重字节——**任何存储轨农民都能从 DSN 重算
任意模型任意 slot 的 sketch** 提交欺诈证明，慢（DSN 读取）但在挑战期内
完全可行。由此 sketch 争议的**最终裁决不依赖 VRAM 副本存在**：
交叉核验只是「快速抓骗」，存储轨是终审。`min_replicas` 从安全参数
降级为服务参数——这是双轨设计的第二个结构性协同
（第一个是永存，§5.4）。

**风险 3：需求反馈回路不稳定。** 权重跟随需求是正反馈：需求降→权重降→
弃载→服务差→需求更降（单模型死亡螺旋）；反向亦然（突发需求→无副本→
服务不了→需求消失）。
**补强：保底权重（floor）+ 滞后。** 仍在注册集合的模型获得
`max(需求跟随权重, floor)`，floor 由上架方持续质押支付（驻留租金）——
想让模型留在热层，要么需求付费、要么提案者付费，明码标价；权重调整
加滞后窗口（EMA + 驱逐冷却期），抑制震荡。agent 侧可据层级做 SLA
决策或预付预热。

## 6. 威胁模型

### 6.1 TEE 完好时

| 攻击 | 防御 |
|------|------|
| Sybil（一卡多身份） | device_id 唯一注册 |
| 声称驻留但放主存/SSD | agent 代码被度量，不会配合；何况 sketch 时限 |
| 伪造 sketch | agent 不会签；覆盖位图 + 副本交叉核验兜底 |
| 虚报服务量 m_t | agent 被度量不会虚报；包络封顶兜底 |
| 自生成负载刷量 | 不视为攻击（§5.1）：真实消耗带宽，构成收益下限 |
| 转发他人 sketch | PRF 混入 device_id |
| 存压缩/低秩权重 | sketch 是对原始字节的字粒度随机线性组合，近似必错 |
| 长上下文垃圾请求刷量 | KV cache 不是注册权重，attention 读取不计票 |
| 即时租卡攻击 | 注册需 bond + attestation + epoch 生效延迟（替代 plotting 慢的作用）；租入的卡还需先获得权重副本（DSN 下载受带宽约束） |

### 6.2 TEE 被攻破时（纵深防御）

假设攻击者能伪造 agent 行为但不能伪造设备证书链：

- 虚报 m_t ⇒ 被硬件包络封顶，通胀有界（≤ 该卡带宽上限/实际用量的比值）；
- 伪造 sketch 值 ⇒ 被副本交叉核验抓到，罚没（经济防御）；
- 权重放主存流式计算 sketch ⇒ PCIe 5.0 x16 仅 64GB/s，80GB 权重需 1.25s
  > 响应期限，且 HBM 直读只要 24ms——50 倍时差（物理防御）；
- 权重放远端 ⇒ 网络带宽更不可能（100Gbps ≈ 12.5GB/s）；
- 设备证书链本身被伪造（NVIDIA 根 CA 泄露级别）⇒ 系统性风险，治理层吊销
  受影响度量版本 + 过渡到修复固件（§8.3 降级路径）。

### 6.3 经济安全

攻击成本 = 获得多数「已注册、已过生效延迟的显存字节」。由于必须是真实 CC 硬件
+ bond + 延迟，等价于「买下/控制多数参与共识的 H100 显存」。与 PoS 类似地，
bond 罚没使得已注册算力作恶有直接经济损失；与 Subspace 一样，PoT 保证无法
通过快算未来挑战做 long-range 攻击。

### 6.4 stake 的三种角色：门控保留、有界调节、保真押金

关键原则：**容量（驻留）永远是硬门控，stake 绝不取代它**。但在此前提下
stake 有三个正当角色。区分它们，避免把 PoRW 误做成纯 PoS。

**必须拒绝的：纯 PoS（票数 ∝ stake，无容量门控）**——那样富人不驻留也能
出块，VRAM 证明沦为摆设。这是唯一的红线。

**采用的：容量门控 + 有界 sqrt 质押调节**（本仓库 `PoS` 分支的机制）。
`scale_solution_range` 把中签判定 `solution_distance ≤ scaled_range/2` 中的
range 按 `sqrt(stake)/sqrt(MaxVotingBalance)` 缩放，即：

```
中签概率 ∝ |C_t|(驻留) × m_t(服务) × sqrt(有效质押)/sqrt(上限)
             └──── 容量门控，零则出局 ────┘   └─ 有界经济对齐 ─┘
```

该分支的三个设计恰好守住了红线，值得原样采纳：

- **容量仍是门控**：零容量 = 零 solution candidate = 零权重，stake 再多无用
  ⇒ VRAM 驻留证明不是摆设；
- **`sqrt` 次线性 + `MaxVotingBalance` 硬封顶**：翻倍影响力需 4× 质押，且
  超过上限的巨鲸与刚到顶者等权 ⇒ 强反财阀，stake 买不到无界影响；
- **零总质押 → 退化为纯容量共识**（`voting_stake_weight` 返回 max_weight）
  ⇒ staking 是叠加项而非网络必需，可平滑启用。

**保真押金（fidelity bond）——与上面的调节质押可合一**。PoRW 用 TEE 换掉
密码学唯一性，引入两处可信断言需押金威慑：(1) 服务量 `m_t`（§4.4）agent
自报、包络内可虚报；(2) TEE 被攻破时的 sketch 伪造（§6.2）需可罚没对象。
同一笔质押既作罚没抵押、又经 sqrt 给奖励加成——bond 与调节合一，一石二鸟。

两个协同与一个张力：

- **协同 1**：stake 顺带给代币一个与用量成正比的锁定需求，补上 §5.2
  tokenomics 的「纯通胀风险」对冲。
- **协同 2**：`domain operator` 层的 staking（Subspace 执行层本就有）与此
  正交，可直接叠加。
- **张力**：PoRW 本欲让奖励跟随有用服务（`m_t`、费用燃烧），stake 调节让
  奖励同时（次线性地）跟随资本。`sqrt` + `cap` 正是把资本这一维压住、
  不让其盖过驻留与服务的旋钮——这是治理权衡，`MaxVotingBalance` /
  `MinVotingBalance` / sqrt 曲率是可调参数。

**long-range / nothing-at-stake** 由继承的 PoT 硬时钟接住，与 Subspace 一样，
不依赖 stake 补足。

**实现落地**：直接复用 `PoS` 分支——`pallet-voting-stake` +
`scale_solution_range` + `voting_stake_weight/max_voting_stake_weight`
runtime API。PoRW 侧唯一需替换的是被缩放的「base 容量」语义：从
Subspace 的纯 plot 容量，改为 `|C_t| × m_t`（驻留 × 服务，§4.3–4.4），
stake 缩放层原样套用。

## 7. 与现有代码的映射

| 现有组件 | PoRW 中的命运 |
|----------|--------------|
| `subspace-proof-of-time` | **原样保留**（时钟与挑战源） |
| `pallet-subspace` 难度调整 / solution range | **原样保留**（计量对象变为驻留字节） |
| `subspace-verification::is_within_solution_range` | **复用**（作用于 sketch 展开的 chunk 流） |
| `auditing.rs` 的挑战派生骨架 | **改写**为 sketch 系数派生（PRF(c_t, device_id, i)） |
| `subspace-proof-of-space` (chiapos) / KZG chunk witness / 唯一编码 plotting | **显存轨不再需要**（TEE 取代密码学唯一性）；**存储轨（§5.4 双轨）原样保留**，继续保障历史与退役权重的永存 |
| archiver / DSN / gateway | **保留**，用途转为模型权重的存储分发（+可选的历史归档双轨） |
| `subspace-farmer` | 被 **PoRW Agent** 取代：CVM 内运行，管理 attestation、权重、sketch kernel、推理服务对接 |
| `shared/subspace-proof-of-space-gpu` (CUDA/ROCm) | 作为 GPU kernel 工程的起点参考 |
| `Solution` 结构 (`subspace-core-primitives`) | 改造：`{device_id, model_id, sketch, chunk_index, sig}` 替代 KZG witness 字段 |

新增组件（★ = 已在本分支落地）：

1. ★ `crates/subspace-proof-of-residency`：PoRW 共识原语——sketch 规范的
   规范 Rust 实现（与 Python/Triton 跨语言测试向量逐位一致）、tile Merkle
   承诺（`R_W` 与 `partials_root`）、票数展开、包络检查、
   `verify_tile_fraud_proof`。no_std，9 项测试。
2. ★ `crates/pallet-porw-registry`：设备/模型/度量注册表 + fidelity bond
   （fungible holds）+ 单 tile 欺诈证明罚没（赏金归举报人）+
   `check_solution` 快速路径（注册/生效延迟/度量吊销/模型声明/包络）。
   attestation 经可插拔 `AttestationVerifier` trait（P4 接真实 NVIDIA
   CC/TDX/SNP 验证）。wasm 可编译，7 项 mock runtime 测试。
3. ★ 已落地：`crates/porw-agent`——节点侧 agent(状态机
   Unregistered→Registered→Active、solution 组装+设备签名、pluggable
   `SketchBackend` trait GPU/CPU 可换、CPU 后端用规范 sketch)+
   `crates/porw-devnet`——纯 CPU 端到端集成测试:agent 产签名 solution →
   真实 attestation 证据注册 → 激活 → 链上快速路径接受 + 共识距离校验,
   零 GPU 零 TEE 全绿。剩余:GPU `SketchBackend` 实现、接进活跃节点服务。
4. `porw-sketch-gpu`：S1-over-coverage 定向 sweep（PoC 已有 Triton 版）
   + 可选融合加固模式。
5. ★ 已落地：`crates/porw-attestation`——attestation 证据验证库,真实
   ed25519 签名链(厂商根 → 设备身份证书 → 报告),绑定 device_id +
   节点密钥 + measurement,8 项测试。pallet 接入:`TrustedRoots` 治理
   存储 + `PorwAttestation` 实现,register_device 走真实验证(runtime
   已从 insecure stub 切换)。**剩余硬件专属部分**:NVIDIA NRAS/SPDM 与
   TDX/SNP 的 wire format 解析(替换 `Evidence::decode`)+ 真实 NVIDIA/
   Intel/AMD 根(替换治理测试根)——验证核心格式无关、原样复用。

## 8. 分期路线图

### P0 — 机制验证（无 GPU、无 TEE，纯 DRAM）
- 在现有 farmer 上实现「审计放大」：挑战派生 N 个链式 s-bucket、可调每 slot
  审计比例 f；验证侧配套。目的：验证 solution range 在放大审计下的
  统计行为与难度调整稳定性。**这是所有后续路线的共同底座。**

### P1 — 融合 sketch kernel（有 GPU、无 TEE）
- 先做独立验证工具：整数域 tile 乘加的跨卡确定性测试、覆盖位图重算核验
  （这也是 §4.6 深度路径的核验器，交付物直接复用）。
- 核心：vLLM/SGLang 融合 PoC（自定义 epilogue 或旁路 kernel 蹭 L2/SMEM），
  量化 sketch 累加对推理吞吐的净开销（目标 <2%），并在真实 MoE 负载下
  测量覆盖集统计。**这是本设计风险最高的工程假设，应最早证伪/证实。**

### P2 — attestation 闭环（有 TEE）
- TDX/SNP CVM + H100 CC 环境搭建；agent 原型：密钥生成、双 attestation 采集、
  链下验证器；`pallet-porw-registry` 最小实现（native 验证证书链）。

### P3 — 共识集成
- ★ 已落地：`PorwApi` runtime API（sp-consensus-subspace 声明、
  subspace-runtime 实现——注册快速路径 + 票数计算）；两 pallet 挂入
  subspace-runtime（`PorwRegistry` = index 10，`PorwBond` hold reason，
  wasm 构建通过）；`sc-consensus-subspace::porw` 客户端验证胶水
  （票据展开 → 与 PoT 挑战的双向距离 → 质押缩放 solution range，
  与 farming 路径同语义）。
- ★ 已落地（P3 第一阶段）：PoRW 预摘要载体（`PorwPreDigest`，独立
  `PORW` 引擎 ID，与 farming 预摘要共存）+ header 提取 + 导入侧验证入口
  `verify_porw_block`（提取预摘要 → 从 PoT 导出 slot 挑战 → 全量校验）；
  快速路径升级为 `check_solution_signed`（注册 + 包络 + **设备签名**），
  故未签名/伪造的 solution 无法出块。
- ★ 已落地（P3 授权逻辑）：`claim_porw_slot`（选最优 solution + 构建
  预摘要）、`porw_pre_digest_logs`/`porw_seal_digest`（header 日志）、
  `verify_porw_seal`（设备节点密钥对区块 pre-hash 的封印,防 solution
  被套到别的区块体）。授权侧与导入侧现共用同一套验证与 digest 载体,
  各自单元测试齐备。
- ★ 已落地（`porw-devnet` 产块循环）：`claim_porw_slot` 驱动的真实
  产块→封印→链接→`verify_porw_block` 导入循环,形成一条增长的、每块
  device-key 封印且父子链接的区块链(mock runtime-API client 做真实
  快速路径检查,无 substrate storage)。伪造封印被拒。CPU 全绿。
- 余下(需真实基础设施):接进真正的 `subspace-node` 服务(PoT gadget、
  网络、多节点);GPU `SketchBackend`;副本交叉核验调度;testnet。

### 8.3 降级路径（设计保险）
- **无 CC 的消费卡**：不参与 PoRW 主抽签，可参与 P0 式 DRAM/显存审计放大的
  平行彩票（较低奖励权重），保持网络的长尾去中心化。
- **TEE 信任崩塌**：治理开关将共识权重滑向「副本交叉核验 + 时限」的纯经济/
  物理防御模式（安全性降级但不停机），同时保留回退到经典 PoAS 的终极选项。

## 9. 相关工作（定位差异）

- **Filecoin**：密码学密封 + 双副本换取用数据可检索——我们用 TEE 免掉密封。
- **Bittensor / io.net 类**：无硬件证明或仅软证明，计量依赖委员会主观评分——
  我们的中签与字节驻留有物理绑定。
- **Prime Intellect TOPLOC（2025）**：激活值局部敏感哈希验证「推理结果真实」——
  与本设计正交互补：PoRW 证明「权重驻留与被读取」，TOPLOC 类方案证明
  「输出确由该权重算出」，二者可叠加（sketch 融合层顺带承诺激活值）。
- **NVIDIA CC 生态（Phala、Atoma 等）**：已验证「TEE-GPU 跑推理并出证明」的
  工程可行性；本设计的差异在于把它接到 Nakamoto 式容量抽签共识上，
  而非仅做任务级验证。

## 10. 开放问题

1. **m_t 的计量单位是否引入 token 加权**：当前 m_t 计「带宽遍数」，
   batching 中立（batch 128 与 batch 1 的一遍扫描同票）——物理上干净，
   但不奖励高效批处理的真实吞吐。若改为 token 加权（一遍扫描 × batch 内
   token 数），票数上限从带宽包络变为算力包络，且自刷负载的收益被放大，
   需配合 §5.1 的费用燃烧加权。建议 P0 阶段用两种权重回放真实负载数据对比。
2. 融合 kernel 在投机解码、prefix cache、量化 kernel（权重在 SMEM 反量化）
   等路径下的覆盖位图正确性与开销。
3. sketch 的 PRF 与 hash_expand 选型（BLAKE3 XOF vs ChaCha）与 GPU 上的
   吞吐权衡；覆盖位图的紧凑编码（MoE 下高度结构化，可按专家粒度压缩）。
4. 证书链链上验证的成本与乐观验证的博弈参数（挑战期、保证金）。
5. 多租户推理（一卡多模型动态换入换出）下的驻留计量语义。
6. ~~与推理真实性证明的叠加~~ → 已纳入为 D 支柱（§4.7）。剩余开放子问题：
   D 支柱 quote 的链上验证 / 抽样策略与成本，及其与 B 支柱设备证书链
   验证（本节 4）的合并批处理。

---

*本文档是研究草案，参数（tile 大小、期限、bond 数额、epoch 长度）均为占位，
待 P0/P1 实验校准。*
