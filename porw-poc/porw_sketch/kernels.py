"""Triton kernels for the PoRW P1 feasibility PoC.

Two kernels:

- ``moe_gemm_sketch_kernel`` — a structural replica of vLLM's
  ``fused_moe_kernel`` (vllm/model_executor/layers/fused_moe/fused_moe.py):
  same grouped pid mapping, same ``sorted_token_ids``/``expert_ids`` routing,
  same K-loop with the weight tile ``b`` loaded per iteration — plus the PoRW
  sketch fused at the exact point where ``b`` sits in registers.
  ``ENABLE_SKETCH`` is a constexpr so the no-sketch baseline compiles to the
  plain GEMM for A/B benchmarking on real GPUs.

- ``sketch_sweep_kernel`` — the standalone per-slot sweep (strategy S1),
  used for dense/cuBLAS-handled layers where coverage is trivially full,
  and as an independent cross-check of the fused path.

Portability notes (correctness-first PoC):

- All sketch arithmetic is int64 + explicit ``& 0xFFFFFFFF`` masking, so the
  Triton CPU interpreter (``TRITON_INTERPRET=1``, numpy backend), a real GPU
  backend, and the numpy reference in ``spec.py`` agree bit-exactly.
  A production kernel would use native u32 ops (~2x fewer instructions).
- The PoC fixes K == TILE_WORDS (2048), so one canonical 4 KiB tile == one
  (expert, n) weight row and the per-row accumulator is flushed once after
  the K loop.  General K needs a flush at every 2048-word boundary
  (BLOCK_SIZE_K must divide TILE_WORDS) — mechanical, not fundamental.

Multi-load semantics: every token-block routed to expert ``e`` computes the
same per-row sketch value, and the store is idempotent (same address, same
value), so racing writes across token-blocks are benign and the result is
independent of batch size and launch order — matching the protocol's
"at least once per slot" semantics without atomics.
"""

import os

import torch

if not torch.cuda.is_available():
    # Interpreter mode must be set before importing triton.
    os.environ.setdefault("TRITON_INTERPRET", "1")

import triton
import triton.language as tl

M32 = 0xFFFFFFFF
GOLDEN32 = 0x9E3779B9


@triton.jit
def _fmix32(h):
    """murmur3 finalizer on int64 values, masked to u32 (result < 2^32)."""
    h = h & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = (h ^ (h >> 13)) & 0xFFFFFFFF
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = (h ^ (h >> 16)) & 0xFFFFFFFF
    return h


@triton.jit
def moe_gemm_sketch_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    partials_ptr,
    coverage_ptr,
    N,
    K,
    EM,
    num_valid_tokens,
    slot_seed,
    stride_am,
    stride_ak,
    stride_be,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    top_k: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ENABLE_SKETCH: tl.constexpr,
):
    # ---- pid mapping: verbatim structure of vLLM fused_moe_kernel ----
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(EM, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
        return

    offs_m = tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    offs_token = tl.load(sorted_token_ids_ptr + pid_m * BLOCK_SIZE_M + offs_m).to(
        tl.int64
    )
    token_mask = offs_token < num_valid_tokens

    off_expert = tl.load(expert_ids_ptr + pid_m).to(tl.int64)
    if off_expert == -1:
        return  # PoC: expert-parallel write-zeros path omitted

    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K).to(tl.int64)
    a_ptrs = a_ptr + (offs_token[:, None] // top_k) * stride_am + offs_k[
        None, :
    ] * stride_ak
    b_ptrs = (
        b_ptr
        + off_expert * stride_be
        + offs_k[:, None] * stride_bk
        + offs_bn[None, :] * stride_bn
    )

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # ---- PoRW additions: per-row (== per-tile, since K == TILE_WORDS)
    # sketch accumulator and coefficients ----
    sk = tl.zeros((BLOCK_SIZE_N,), dtype=tl.int64)
    tile_idx = off_expert * N + offs_bn
    r_tile = _fmix32(_fmix32(slot_seed ^ tile_idx))

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_rem = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_rem),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        # (PoC casts to fp32 for interpreter/GPU parity; the GPU benchmark
        # build uses the native fp16 tensor-core path.)
        accumulator += tl.dot(a.to(tl.float32), b.to(tl.float32))

        if ENABLE_SKETCH:
            # The weight tile is in registers right now: sketch it.
            w = b.to(tl.uint16, bitcast=True).to(tl.int64)  # [K_blk, N_blk]
            j = k * BLOCK_SIZE_K + offs_k  # word index within the row/tile
            c = _fmix32(r_tile[None, :] + ((j[:, None] * 0x9E3779B9) & 0xFFFFFFFF))
            sk += tl.sum((c * w) & 0xFFFFFFFF, axis=0)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ENABLE_SKETCH:
        # Idempotent per-tile store (same value from every token-block of
        # this expert) + coverage flag.
        out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = out_n < N
        tl.store(partials_ptr + tile_idx, sk & 0xFFFFFFFF, mask=n_mask)
        tl.store(
            coverage_ptr + tile_idx,
            tl.full((BLOCK_SIZE_N,), 1, tl.int8),
            mask=n_mask,
        )

    c_out = accumulator.to(tl.float16)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c_out, mask=token_mask[:, None] & (offs_cn[None, :] < N))


@triton.jit
def sketch_sweep_kernel(
    buf_ptr,  # int16 view of the weight buffer (little-endian words)
    out_ptr,  # int64, one sketch per tile
    n_tiles,
    slot_seed,
    TILE_WORDS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    if pid >= n_tiles:
        return
    r_tile = _fmix32(_fmix32(slot_seed ^ pid))
    s = tl.zeros((BLOCK,), dtype=tl.int64)
    for start in range(0, TILE_WORDS, BLOCK):
        j = start + tl.arange(0, BLOCK).to(tl.int64)
        w = tl.load(buf_ptr + pid * TILE_WORDS + j).to(tl.int64) & 0xFFFF
        c = _fmix32(r_tile + ((j * 0x9E3779B9) & 0xFFFFFFFF))
        s += (c * w) & 0xFFFFFFFF
    tl.store(out_ptr + pid, tl.sum(s, axis=0) & 0xFFFFFFFF)


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def run_moe_gemm(
    a: torch.Tensor,  # [M, K] fp16
    b: torch.Tensor,  # [E, N, K] fp16, contiguous
    topk_ids: torch.Tensor,  # [M, top_k] int32
    slot_seed: int,
    *,
    enable_sketch: bool = True,
    block_m: int = 16,
    block_n: int = 64,
    block_k: int = 64,
    group_m: int = 1,
):
    """Launch the fused kernel; returns (c [M*top_k, N], partials, coverage)."""
    from .reference import moe_align

    M, K = a.shape
    E, N, Kb = b.shape
    assert K == Kb and b.is_contiguous()
    top_k = topk_ids.shape[1]
    num_valid_tokens = M * top_k

    sorted_token_ids, expert_ids, num_post_padded = moe_align(
        topk_ids.cpu().numpy(), E, block_m
    )
    dev = a.device
    sorted_token_ids = torch.from_numpy(sorted_token_ids).to(dev)
    expert_ids_t = torch.from_numpy(expert_ids).to(dev)
    num_post_padded_t = torch.tensor([num_post_padded], dtype=torch.int32, device=dev)
    EM = sorted_token_ids.numel()

    c = torch.zeros((num_valid_tokens, N), dtype=torch.float16, device=dev)
    n_tiles = E * N  # K == TILE_WORDS: one tile per (expert, n) row
    partials = torch.zeros(n_tiles, dtype=torch.int64, device=dev)
    coverage = torch.zeros(n_tiles, dtype=torch.int8, device=dev)

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)
    moe_gemm_sketch_kernel[grid](
        a,
        b,
        c,
        sorted_token_ids,
        expert_ids_t,
        num_post_padded_t,
        partials,
        coverage,
        N,
        K,
        EM,
        num_valid_tokens,
        slot_seed,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(2),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        top_k=top_k,
        BLOCK_SIZE_M=block_m,
        BLOCK_SIZE_N=block_n,
        BLOCK_SIZE_K=block_k,
        GROUP_SIZE_M=group_m,
        ENABLE_SKETCH=enable_sketch,
    )
    return c, partials, coverage


def run_sketch_sweep(buf_bytes: torch.Tensor, slot_seed: int, block: int = 256):
    """Standalone sweep over a uint8 buffer; returns per-tile sketches."""
    from .spec import TILE_WORDS

    assert buf_bytes.dtype == torch.uint8 and buf_bytes.numel() % (TILE_WORDS * 2) == 0
    words = buf_bytes.view(torch.int16)
    n_tiles = words.numel() // TILE_WORDS
    out = torch.zeros(n_tiles, dtype=torch.int64, device=buf_bytes.device)
    sketch_sweep_kernel[(n_tiles,)](
        words, out, n_tiles, slot_seed, TILE_WORDS=TILE_WORDS, BLOCK=block
    )
    return out
