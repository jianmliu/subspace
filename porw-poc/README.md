# PoRW P1 PoC — 推理融合 sketch 可行性验证

见设计文档 `docs/proof-of-resident-weights.md` 与可行性报告
`docs/porw-p1-feasibility.md`。

## 内容

- `porw_sketch/spec.py` — sketch 规范 + numpy 参考实现（协议的规范文本）。
- `porw_sketch/kernels.py` — Triton kernel：
  - `moe_gemm_sketch_kernel`：vLLM `fused_moe_kernel` 的结构复刻 +
    在权重 tile 加载点融合 sketch（`ENABLE_SKETCH` constexpr 可开关）；
  - `sketch_sweep_kernel`：独立扫描（S1，用于 dense/cuBLAS 层）。
- `porw_sketch/reference.py` — `moe_align_block_size` 最小复刻与 GEMM 参考。
- `tests/test_sketch.py` — 10 项验证（无 GPU 时自动走 Triton CPU 解释器）。
- `bench_gpu.py` — GPU 开销基准（需真实 GPU，测融合开销 % 与扫描 GB/s）。

## 运行

```bash
pip install numpy torch triton pytest   # CPU 环境即可
python -m pytest tests/ -q              # TRITON_INTERPRET=1 自动启用
```

## 已验证（CPU 解释器，与 GPU 后端语义一致）

1. sketch 规范：确定性、slot 敏感性、任意分块/求和顺序不变性、单比特篡改检出。
2. 融合 kernel：GEMM 结果正确；覆盖到的 tile 的 sketch 与规范参考逐位一致；
   冷专家不产生覆盖；不同 batch 组成下 sketch/覆盖不变（幂等存储语义）；
   融合路径与独立扫描路径逐 tile 一致（dense 与 MoE 可共用验证器）。
3. 攻击演示：tile 级常数系数方案被 4 字节/tile 的摘要 100% 伪造（1024 倍压缩）；
   字级 slot 新鲜系数方案下，同一攻击者及 64 泛函/最小二乘重构攻击者
   在所有测试 slot 全部失败。

## 规范 v2（u32 优化版）

- 字粒度 16-bit → **32-bit**（PRF 调用减半，安全粒度不变）；
- kernel 全部**原生 u32** 环绕算术（不再 int64+掩码模拟）；
  >int31 的常数经 int32 张量传入、kernel 内 bitcast，规避 Triton 字面量
  提升为 int64 的跨端不一致；
- **系数强制为奇数**（`c_j |= 1`）：模 2^32 下奇数乘子是双射，修复
  偶系数时字高位翻转以 ≤50% 概率逃逸检测的规范级弱点（v2 测试新增覆盖）；
- sweep kernel 支持覆盖子集（`tile_ids`）——S1-over-coverage 主路线的
  执行原语；
- 基准新增 1 GB 权重配置（压穿 A100 L2，真实 HBM 流式形态）。

## 待 GPU 复测（bench_gpu.py，同一条 ssh 命令）

- 1 GB 配置下的融合开销与 sweep GB/s（预期 sweep 显著高于 v1 的
  947–1119 GB/s）；
- 融合版剩余差距归因（fp32 dot 基线 vs fp16 tensor core 为后续项）。
