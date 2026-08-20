# PoRW P1 可行性报告：sketch 与 vLLM/SGLang 的融合

状态：v0.1（2026-08）。配套代码：`porw-poc/`（CPU 可运行的验证套件）。

## 1. 结论摘要

1. **融合点真实存在且可改**：vLLM 的 MoE 路径是开源 Triton kernel
   （`vllm/model_executor/layers/fused_moe/fused_moe.py` 的
   `fused_moe_kernel`，权重 tile 在 K 循环内 `tl.load` 进寄存器），
   sketch 乘加可直接插在加载点之后。PoC 已复刻该 kernel 结构并完成融合，
   在 Triton CPU 解释器下验证了语义正确性（10/10 测试通过）。
2. **发现并修复了一个方案级密码学缺陷**（§3）：tile 粒度常数系数可被
   4 字节/tile 摘要完全伪造（1024 倍压缩）。系数必须字粒度、每 slot 新鲜，
   由寄存器内廉价 PRF（murmur3 fmix32，~6 条整数指令）生成。
   这把融合的 ALU 成本从"可忽略"提高到"每 16-bit 字约 8-10 条整数指令"，
   使 GPU 实测成为必要（§5）。
3. **覆盖语义按设计成立**：冷专家不产生覆盖；sketch/覆盖对 batch 组成不变
   （幂等存储，无原子操作）；融合路径与独立扫描路径逐 tile 一致，
   dense 层（S1 独立扫描）与 MoE 层（S2 融合）可共用一个验证器。
4. **dense 未量化路径不可融合**（cuBLAS 闭源，见 §2），采用 S1 独立扫描：
   dense 层覆盖恒为全量，无 MoE 语义问题，每 slot 一遍的额外带宽 ≈
   模型字节/带宽/slot（80GB @ 3.35TB/s ≈ 2.4%），可接受。

## 2. vLLM 权重路径盘点（融合可行性）

decode 时权重字节流经的 kernel，按可改性分类
（基于 vllm-project/vllm 主干，2026-08 浅克隆）：

| 路径 | 实现 | 融合可行性 |
|------|------|-----------|
| MoE 专家 GEMM（默认） | Triton `fused_moe_kernel`（fused_moe.py:299），b tile 于 :537 `tl.load` | **S2 直接融合**（PoC 已验证语义） |
| MoE GPTQ/AWQ | Triton `fused_moe_kernel_gptq_awq`（fused_moe.py:65） | S2 同上（对存储的量化字节做 sketch） |
| MoE 其他后端（DeepGEMM、CUTLASS 变体、flydsl） | csrc / 外部库 | 逐后端处理或对该层退化为 S1 |
| dense 未量化 Linear | `UnquantizedLinearMethod` → `torch.nn.functional.linear` → cuBLAS（闭源） | **不可融合 → S1 独立扫描** |
| dense 量化（Marlin/Machete/w8a8 等） | csrc CUTLASS/CUDA（源码可得） | 可融合但需 mainloop 手术，工程量大；先 S1 |
| attention（FlashAttn/FlashInfer） | 读的是 KV cache，不是注册权重 | **无关**（KV 不计票，天然排除长上下文刷量） |

两个结构性结论：

- **融合手术恰好只在覆盖依赖负载的地方必要**：MoE 专家是唯一"读不读取决于
  路由"的权重；dense 层覆盖恒为全量，独立扫描在协议语义上零损失。
  这把 S2 的工程面收窄到 Triton MoE kernel 一族——vLLM 与 SGLang 的
  MoE kernel 同源（SGLang 的 fused_moe 是 vLLM 内核的近亲分支），
  一次融合两家受益（SGLang 侧待逐版本核实）。
- **GEMM epilogue 不是融合点**：CUTLASS/cuBLAS 的 epilogue 作用于输出
  C tile，看不到权重 B tile 的字节流；对权重做 sketch 必须进 mainloop。
  这就是 cuBLAS 闭源路径判死、CUTLASS 路径工程量大的原因，
  也是 Triton kernel（B tile 就是普通张量变量）成为首选融合点的原因。

## 3. 密码学缺陷与修复：系数必须字粒度、slot 新鲜

设计文档 v0.2 中 sketch 定义为 `Σ r_i ⊙ bytes(T_i)`（tile 粒度系数）。
本次 PoC 把它形式化时发现该构造**不成立**：

- 若 tile 内所有字共用系数 `r_i`，则 `s_i = r_i · L(T_i)`，其中
  `L(T_i) = Σ_j w_j` 是与 slot 无关的固定线性泛函。作弊者只需存每 tile
  4 字节的 `L(T_i)` 即可对**所有未来 slot** 正确应答——1024 倍压缩，
  驻留证明彻底失效。`tests/test_sketch.py::test_per_tile_coeff_scheme_is_broken`
  给出了 100 个 slot 全部伪造成功的可执行演示。
- 一般化：只要字系数是每 slot 随机量的低秩函数（如 `r_i·(2j+1)`、
  `r_i·j + s_i`），泛函族就坍缩到少数固定线性泛函，作弊者存对应少数
  摘要即可通过。**修复要求：字粒度系数对 slot 的依赖必须"满秩"**——
  即每字系数由 PRF 现场生成：

```
r_tile = fmix32(fmix32(slot_seed ^ tile_idx))          // 每 tile 一次
c_j    = fmix32(r_tile + j·GOLDEN32)                    // 每字 ~6 条整数指令
s_tile = Σ_j c_j · w_j   (mod 2^32)
```

- 修复后，4 字节摘要攻击者与"64 个固定泛函 + 最小二乘重构"攻击者
  （256 字节/tile，仍 16 倍压缩）在全部测试 slot 失败
  （`test_per_word_coeff_scheme_resists_compression`）；单比特篡改必检出。
- 模加交换律保证 `s_tile` 对任意分块与求和顺序不变——kernel 的 block
  形状、launch 顺序、幂等存储竞争都不影响结果，这是融合可以随 kernel
  配置自由变化的前提（`test_spec_partition_independence`）。

**代价**：每 16-bit 字约 8–10 条整数指令（PRF 6 条 + 乘加掩码）。
决定了 §5 的 GPU 实测必要性。设计文档的 sketch 定义需按本节更新。

## 4. PoC 交付物与验证结果

`porw-poc/`，无 GPU 环境可完整运行（Triton `TRITON_INTERPRET=1` CPU
解释器；int64+显式掩码算术使解释器/GPU/numpy 三方逐位一致）：

- `moe_gemm_sketch_kernel`：`fused_moe_kernel` 的结构复刻（同款 grouped
  pid 映射、`sorted_token_ids`/`expert_ids` 路由、K 循环），sketch 融合于
  b tile 加载点，`ENABLE_SKETCH` constexpr 开关（基准/融合 A/B 同 kernel）。
- 幂等存储替代原子操作：同专家的每个 token-block 算出相同的 per-tile 值,
  竞争写良性，sketch/覆盖与 batch 组成无关（`test_moe_kernel_batch_invariance`）。
- 10/10 测试通过：规范性质 4 项、攻击演示 2 项、kernel 语义 4 项。

## 5. GPU 实测结果（A100 80GB PCIe，2026-08-20）

环境：torch 2.10.0+cu128，Triton 原生后端，全部正确性测试通过
（CPU 解释器与 GPU 后端位级一致得证）。

| 配置 | base | fused | 开销 | sweep 带宽 |
|------|------|-------|------|-----------|
| E=8 N=1024 K=2048 M=4 topk=2 | 0.171ms | 0.222ms | **30.0%** | 947 GB/s |
| E=8 N=1024 K=2048 M=64 topk=2 | 0.190ms | 0.247ms | **30.2%** | 957 GB/s |
| E=64 N=512 K=2048 M=64 topk=8 | 0.500ms | 0.582ms | **16.3%** | 1119 GB/s |

A100 HBM2e 理论峰值 ~1935 GB/s，sweep 实测 49–58% 利用率；
配置越大开销越低、带宽利用越高。

### 5.1 解读：三个系统性偏差与一个反转性结论

数字解读要先扣掉三个 PoC 自身的偏差：

1. **int64+掩码算术**是为跨端位级一致设计的，GPU 上 64 位整乘是多条
   32 位指令模拟——生产 u32 版指令数约减半以上；
2. **基线用的是 fp32 dot**（同为跨端一致设计），比生产 fp16 tensor core
   路径慢——真实基线更快，百分比开销会**更高**（两个偏差方向相反）；
3. **测试规模太小**：B 仅 32MB（E=8 配置），整个放得进 A100 的 40MB L2，
   负载不是真实 decode 的 HBM 流式形态。E=64 配置更接近真实
   （开销降到 16.3%），趋势支持"规模越大 ALU 越好隐藏"。

**反转性结论：按当前数据，S1（独立扫描）全面优于 S2（融合）。**
算一笔账：sweep 以 ~1.1 TB/s 流过 70GB 权重 = 64ms ≈ slot 的 6.4%
带宽税（u32 优化后预计 <5%）；而融合路径对每次 decode 收 16–30% 的
吞吐税。覆盖语义（MoE 只计实际激活的专家）并不非融合不可：agent 从
路由遥测取 `C_t`，然后**只对 `C_t` 内的 tile 跑 sweep**——绑定由
被度量的 agent 背书，信任强度与 m_t 计量完全一致（agent 本来就是
可信计量器）。融合的真实增量价值只剩「TEE 被攻破时的纵深防御」。

由此修订架构建议：

- **主路线 = S1-over-C_t**：遥测定覆盖集 + 定向 sweep。零推理侵入、
  兼容所有后端（cuBLAS/CUTLASS/FlashInfer 全部无关）、带宽税 ~5% 且
  只在 slot 内一次；工程面从"给每个 kernel 家族做手术"缩小为
  "一个独立 kernel + 路由遥测接口"。
- **S2 融合降级为可选加固**：TEE 信任折扣期或高保证模式启用，
  接受吞吐税换取"覆盖由密码学而非遥测背书"。
- 设计文档 §4.2/§4.3 的表述需随此更新（sketch 语义不变，执行策略变）。

### 5.2 后续优化与复测清单

1. **u32 原生算术 + 32-bit 字粒度**（规范 v0.2：字从 16 位改 32 位，
   PRF 调用减半，安全粒度仍是字级、无损失）→ 预期 sweep >1.5 TB/s、
   融合开销降到 ~8–15%；
2. 真实规模负载：B >1GB（如 E=64 N=4096 K=2048）压出 L2，
   测 HBM 流式形态下的真实开销曲线；
3. 融合版若继续：fp16 tensor core dot + warp 级异步 sketch
  （CUDA 层，Triton 之外）。

## 6. GPU 待验证项（原计划，部分已完成）

`bench_gpu.py` 已就绪，按优先级：

1. **融合开销**：decode 形状负载下 fused vs baseline tokens/s，目标 <2%。
   风险点：int 指令与 tensor core 流水的争用、寄存器压力对 occupancy 的
   影响。生产 kernel 需换原生 u32 算术（较 PoC 的 int64+掩码约省一半指令）
   与 fp16 tensor core dot。
2. **独立扫描带宽**：`sketch_sweep_kernel` 应逼近 HBM 峰值带宽的 80%+，
   否则 S1 的 2.4% 估算失真。
3. **真实引擎集成**：vLLM 补丁版 `fused_moe_kernel`（改动面：K 循环内
   +4 行、kernel 参数 +3 个、host 侧 partials/coverage 分配）跑真实
   MoE 模型，验证 tile 索引与权重内存布局的对应、量化路径的字节语义。
4. 端到端：每 slot 换系数的调度（CUDA graph 兼容性——slot_seed 作为
   kernel 参数传入，不破坏 graph capture；partials buffer 双缓冲）。

## 7. 对设计文档的反馈

1. §4.2 sketch 定义更新为字粒度系数（本报告 §3）——已是硬性要求而非优化。
2. 融合策略确认为 **S1+S2 混合**：MoE Triton kernel 融合（覆盖语义所在），
   dense/闭源路径独立扫描（覆盖恒全量，2.4% 带宽税）。§4.3 的
   "零额外带宽" 表述应修正为 "MoE 零额外、dense ~2.4%"。
3. KV cache 天然不参与 sketch（attention kernel 不触注册权重），
   长上下文刷量不可行——可作为 §6.1 威胁表新增一行。
