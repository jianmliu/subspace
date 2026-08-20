# Proof of Resident Weights (PoRW) 设计文档

**基于 TEE 硬件证明 + 推理搭便车审计的显存驻留共识**

状态：研究草案 v0.1（2026-08）

---

## 0. 一句话概述

把 Subspace 的 Proof-of-Archival-Storage 骨架（PoT 时钟 + 每 slot 挑战 + 容量加权抽签 +
solution range 难度调整）保留，把「稀缺资源」从 SSD 上的唯一编码 plot 换成
**经 TEE 证明驻留在 GPU 显存中的大模型原始权重**；副本唯一性问题不再用密码学密封解决，
而是用 GPU/CPU 机密计算的远程证明（attestation）把「一张物理卡」作为防 Sybil 的计数单位；
审计带宽开销通过把承诺计算融合进推理 decode 本来就在做的全权重扫描来摊销
（「跑推理即出块」）。

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
| G2 | 防 Sybil：奖励与物理显存字节成正比，与身份数无关 | 一卡一注册，容量加权 |
| G3 | 审计带宽开销在推理繁忙时趋近于零 | 承诺计算融合进 decode 的权重扫描 |
| G4 | 保留 Subspace 的抽签/难度调整/PoT 骨架 | 最小化共识层改动 |
| G5 | 信任依赖显式化、可治理、可退出 | 度量白名单上链，TEE 破裂有降级路径 |

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

三个信任支柱，缺一不可、互为冗余：

1. **B 支柱（attestation）**：设备唯一身份 + 代码度量 ⇒ 防 Sybil、协议合规性。
2. **C 支柱（sketch）**：挑战随机化的全权重线性草图 ⇒ 证明字节真实驻留且被读取，
   是 TEE 被攻破时的第二道防线（timing + 副本交叉核验仍然成立）。
3. **PoT（不变）**：不可加速的时钟 ⇒ sketch 的响应期限有客观依据。

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

## 4. C 支柱：推理搭便车的驻留审计（sketch）

### 4.1 核心观察

LLM decode 每生成一个 token，都要把全部权重从 HBM 完整扫一遍（这正是推理带宽
受限的原因）。共识想强制的行为——「每 slot 高强度读取驻留数据」——**推理已经在做了**。
审计只需在这条已付费的数据流上顺带累计一个承诺。

### 4.2 Sketch 定义（整数域，与推理数值完全解耦）

设权重按 4 KiB tile 划分：`W = {T_0, T_1, ..., T_{n-1}}`，Merkle 根 `R_W` 登记在
ModelRegistry。每个 slot 由 PoT 得到全局挑战 `c_t`（复用现有
`global_challenge` 派生路径）。定义：

```
r_i     = PRF(c_t, device_id, i)          // 每 tile 的挑战系数, u64
sketch_t = Σ_i  r_i ⊙ bytes(T_i)          // ⊙: 按 64-bit 字的模 2^64 乘加
```

要点：

- **对原始字节做整数乘加**，不是浮点运算——完全确定、跨卡跨驱动可复现，
  与推理框架的数值行为无关（G3 的可行性关键）。
- 计算 `sketch_t` 必须**每 slot 完整读取全部权重字节一次**（系数每 slot 换，
  无法跨 slot 缓存部分和）。这就是带宽下限证明，等价于 Freivalds 式随机线性草图：
  存了低精度/低秩近似的作弊者会产出错误的 sketch。
- `device_id` 混入 PRF ⇒ 不同卡的 sketch 不同，无法转发抄袭。

### 4.3 两种执行模式

| 模式 | 触发 | 实现 | 带宽开销 |
|------|------|------|----------|
| **搭便车** | 推理繁忙 | sketch 乘加融合进 decode kernel 的 epilogue（权重 tile 已在寄存器/SMEM 中） | ≈0（蹭已有读取），仅少量 ALU |
| **专用扫描** | 卡空闲 | 独立 sweep kernel 按随机链式顺序扫权重 | 打满 HBM 带宽（即「闲置挖矿」） |

两种模式产出的 sketch 值**完全相同**（同一定义、整数域），链上无法也无需区分——
这正是「推理与挖矿同一枚硬币两面」的形式化表达。

### 4.4 时序与抽签（复用 Subspace 骨架）

```
slot t:  PoT ──► c_t
         │
         ├─ GPU: 累计 sketch_t（一次全权重扫描, 80GB@3.35TB/s ≈ 24ms）
         ├─ Agent: chunks = split_32B(hash_expand(sketch_t))
         │         对每个 chunk 检查 is_within_solution_range(...)   // 复用现逻辑
         └─ 中签 ⇒ Solution {
               device_id, model_id, slot,
               sketch_t, chunk_index, 
               sig = Sign_sk(...)            // 节点密钥, attestation 背书
            } ──► 出块
```

- **容量加权**：`solution_range` 判定作用在 `hash_expand(sketch_t)` 展开出的
  chunk 流上，展开长度正比于该设备登记的驻留字节数 ⇒ 期望中签率 ∝ 显存中的
  权重字节数，与 Subspace「每 chunk 一张彩票」语义一致，难度调整逻辑
  （pallet-subspace 的 solution range era 调整）原样复用。
- **响应期限**：沿用 `BLOCK_AUTHORING_DELAY`。sketch 计算仅 ~24ms（H100 满带宽），
  1s slot 有巨大裕量；期限的真正作用是纵深防御（§6.2 的 PCIe/网络流式作弊在
  TEE 破裂时也来不及）。

### 4.5 链上验证

快速路径（每块必做，代价 ~几次签名验证）：

1. `device_id ∈ DeviceRegistry` 且 attestation 未过期；
2. `sig` 对应注册的 `pk`；
3. `model_id ∈ ModelRegistry` 且该设备登记了此模型；
4. chunk 落入 solution range（同现有 `subspace-verification` 逻辑）。

深度路径（抽查 + 欺诈证明，防 TEE 破裂后的伪造 sketch）：

- 每 epoch 随机指派 k 个持有同一模型的其他注册设备重算若干历史 slot 的
  `sketch`（它们有权重，重算只花 24ms 带宽）；不一致 ⇒ 提交欺诈证明 ⇒
  罚没 bond 并吊销 device_id。诚实多数假设仅需在「同模型副本集合」内成立。

## 5. 模型治理与生命周期

- **ModelRegistry**：`model_id → (R_W merkle root, size, version, min_replicas,
  reward_weight)`。哪些模型可用于共识、各自的奖励权重，由治理决定
  （初期白名单，后期可按推理需求市场化加权）。
- **权重分发**：权重本体作为数据上传进 Subspace DSN（现有 archiver/gateway
  路径原样可用）——网络同时承担「模型的永久存储与分发」，新节点从 DSN 拉取，
  对着 `R_W` 验证。这保留了原网络的实用性叙事且与共识解耦。
- **版本更新**：新版本 = 新 `model_id`。旧版本设置 deprecation epoch，
  期内双计奖励，之后移出抽签集合。无 re-plot 成本（没有密封编码），
  切换成本 = 下载 + 载入显存。
- **多模型/分片**：一张卡可登记多个模型（7B+13B），多卡 NVLink CC 域可登记
  张量并行分片的大模型——sketch 按分片累加，聚合签名。

## 6. 威胁模型

### 6.1 TEE 完好时

| 攻击 | 防御 |
|------|------|
| Sybil（一卡多身份） | device_id 唯一注册 |
| 声称驻留但放主存/SSD | agent 代码被度量，不会配合；何况 sketch 时限 |
| 伪造 sketch | agent 不会签；副本交叉核验兜底 |
| 转发他人 sketch | PRF 混入 device_id |
| 存压缩/低秩权重 | sketch 是对原始字节的随机线性组合，近似必错 |
| 即时租卡攻击 | 注册需 bond + attestation + epoch 生效延迟（替代 plotting 慢的作用）；租入的卡还需先获得权重副本（DSN 下载受带宽约束） |

### 6.2 TEE 被攻破时（纵深防御）

假设攻击者能伪造 agent 行为但不能伪造设备证书链：

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

## 7. 与现有代码的映射

| 现有组件 | PoRW 中的命运 |
|----------|--------------|
| `subspace-proof-of-time` | **原样保留**（时钟与挑战源） |
| `pallet-subspace` 难度调整 / solution range | **原样保留**（计量对象变为驻留字节） |
| `subspace-verification::is_within_solution_range` | **复用**（作用于 sketch 展开的 chunk 流） |
| `auditing.rs` 的挑战派生骨架 | **改写**为 sketch 系数派生（PRF(c_t, device_id, i)） |
| `subspace-proof-of-space` (chiapos) / KZG chunk witness / 唯一编码 plotting | **不再需要**（TEE 取代密码学唯一性）——这是巨大的简化 |
| archiver / DSN / gateway | **保留**，用途转为模型权重的存储分发（+可选的历史归档双轨） |
| `subspace-farmer` | 被 **PoRW Agent** 取代：CVM 内运行，管理 attestation、权重、sketch kernel、推理服务对接 |
| `shared/subspace-proof-of-space-gpu` (CUDA/ROCm) | 作为 GPU kernel 工程的起点参考 |
| `Solution` 结构 (`subspace-core-primitives`) | 改造：`{device_id, model_id, sketch, chunk_index, sig}` 替代 KZG witness 字段 |

新增组件：

1. `pallet-porw-registry`：设备/模型/度量注册表 + bond + 罚没 + 欺诈证明。
2. `porw-agent`：CVM 内节点程序（Rust，度量白名单化、可复现构建）。
3. `porw-sketch-gpu`：CUDA sketch kernel（独立 sweep 版 + 推理框架融合版）。
4. attestation 验证库（NVIDIA NRAS / TDX / SNP 报告解析与证书链验证）。

## 8. 分期路线图

### P0 — 机制验证（无 GPU、无 TEE，纯 DRAM）
- 在现有 farmer 上实现「审计放大」：挑战派生 N 个链式 s-bucket、可调每 slot
  审计比例 f；验证侧配套。目的：验证 solution range 在放大审计下的
  统计行为与难度调整稳定性。**这是所有后续路线的共同底座。**

### P1 — sketch kernel（有 GPU、无 TEE）
- 实现 `porw-sketch-gpu` 独立 sweep 版：整数域 tile 乘加、随机链式访问序、
  跨卡确定性测试；测量 H100/4090 上的实际扫描耗时与推理共存时的吞吐影响。
- 实现 vLLM/SGLang 的融合 PoC（自定义 epilogue 或旁路 kernel 蹭 L2），
  量化「搭便车」相对「专用扫描」的净开销。**这是本设计风险最高的工程假设，
  应最早证伪/证实。**

### P2 — attestation 闭环（有 TEE）
- TDX/SNP CVM + H100 CC 环境搭建；agent 原型：密钥生成、双 attestation 采集、
  链下验证器；`pallet-porw-registry` 最小实现（native 验证证书链）。

### P3 — 共识集成
- Solution 结构改造、出块路径接通、副本交叉核验与罚没、模型治理流程；
  testnet。

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

1. 融合 sketch 在真实推理引擎（paged attention、投机解码、MoE 稀疏激活）下的
   覆盖率：MoE 每 token 只读部分专家 ⇒ 搭便车模式对冷专家覆盖不足，
   需混合专用扫描补齐——比例待测。
2. sketch 的 PRF 与 hash_expand 选型（BLAKE3 XOF vs ChaCha）与 GPU 上的
   吞吐权衡。
3. 证书链链上验证的成本与乐观验证的博弈参数（挑战期、保证金）。
4. 多租户推理（一卡多模型动态换入换出）下的驻留计量语义。
5. 奖励曲线：farming 保底 vs 推理溢价的分成如何避免「纯挖矿卡」挤占
   真实服务容量（可能需要服务量证明加权，接 TOPLOC 类机制）。

---

*本文档是研究草案，参数（tile 大小、期限、bond 数额、epoch 长度）均为占位，
待 P0/P1 实验校准。*
