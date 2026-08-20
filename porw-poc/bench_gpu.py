"""GPU benchmark harness for the P1 overhead measurement (run on a real GPU).

Measures, per (E, N, K, M, top_k) config:
  1. fused MoE GEMM baseline (ENABLE_SKETCH=False) tokens/s
  2. fused MoE GEMM + sketch (ENABLE_SKETCH=True) tokens/s  -> overhead %
  3. standalone sweep kernel GB/s vs theoretical HBM bandwidth

Target: fused overhead < 2% on decode-shaped workloads (small M, large E*N*K).
Note: this PoC kernel is correctness-first (fp32 tl.dot, int64 sketch math).
Before trusting absolute numbers, switch the dot to native fp16/bf16 and the
sketch math to u32 — relative overhead is what matters here.
"""

import time

import numpy as np
import torch

from porw_sketch.kernels import run_moe_gemm, run_sketch_sweep
from porw_sketch.spec import TILE_WORDS

CONFIGS = [
    # (E, N, K, M, top_k)  — K must equal TILE_WORDS in the PoC
    (8, 1024, TILE_WORDS, 4, 2),  # decode, tiny batch
    (8, 1024, TILE_WORDS, 64, 2),  # decode, medium batch
    (64, 512, TILE_WORDS, 64, 8),  # DeepSeek-ish expert count
]


def bench(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    assert torch.cuda.is_available(), "run this on a GPU box"
    dev = torch.device("cuda")
    rng = np.random.default_rng(0)
    for E, N, K, M, top_k in CONFIGS:
        a = torch.randn(M, K, dtype=torch.float16, device=dev)
        b = torch.randn(E, N, K, dtype=torch.float16, device=dev)
        topk_ids = torch.from_numpy(
            rng.integers(0, E, size=(M, top_k)).astype(np.int32)
        ).to(dev)

        t_base = bench(lambda: run_moe_gemm(a, b, topk_ids, 1, enable_sketch=False))
        t_fused = bench(lambda: run_moe_gemm(a, b, topk_ids, 1, enable_sketch=True))

        buf = b.contiguous().view(torch.uint8).flatten()
        t_sweep = bench(lambda: run_sketch_sweep(buf, 1))
        gbps = buf.numel() / t_sweep / 1e9

        print(
            f"E={E} N={N} K={K} M={M} topk={top_k}: "
            f"base {t_base * 1e3:.3f}ms  fused {t_fused * 1e3:.3f}ms  "
            f"overhead {(t_fused / t_base - 1) * 100:.2f}%  "
            f"sweep {gbps:.0f} GB/s"
        )


if __name__ == "__main__":
    main()
