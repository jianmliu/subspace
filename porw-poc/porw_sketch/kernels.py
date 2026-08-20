"""Triton kernels for the PoRW P1 feasibility PoC — spec v2, native u32.

Two kernels:

- ``sketch_sweep_kernel`` — the standalone per-slot sweep, now the PRIMARY
  strategy (S1-over-coverage per the A100 measurements): the agent derives
  the coverage set from router telemetry and sweeps exactly those tiles.
  Native u32 arithmetic throughout.

- ``moe_gemm_sketch_kernel`` — structural replica of vLLM's
  ``fused_moe_kernel`` with the sketch fused at the weight-tile load point
  (hardening mode / S2).  The sketch reads the weight tile again through a
  32-bit view of the same addresses — these loads hit L1/L2 (the fp16 tile
  was just loaded), so no extra HBM traffic.

Portability notes:

- All sketch arithmetic is native u32 (wrapping) — no int64 emulation.
  Triton promotes >int31 literals to int64 (and the CPU interpreter's numpy
  promotion differs again), so the fmix multipliers, GOLDEN32 and the slot
  seed are passed in a small int32 tensor and bitcast to u32 in-kernel:
  bit-exact on interpreter, GPU backend and the numpy reference.
- The PoC fixes K*2 bytes == TILE_BYTES (K == 2048 fp16 elements == 1024
  u32 words), so one canonical 4 KiB tile == one (expert, n) weight row.
  General K needs a flush at tile boundaries — mechanical, not fundamental.

Multi-load semantics: every token-block routed to expert ``e`` computes the
same per-row sketch value, and the store is idempotent (same address, same
value), so racing writes across token-blocks are benign — matching the
protocol's "at least once per slot" semantics without atomics.
"""

import os

import torch

if not torch.cuda.is_available():
    # Interpreter mode must be set before importing triton.
    os.environ.setdefault("TRITON_INTERPRET", "1")

import triton
import triton.language as tl

from .spec import FMIX_M1, FMIX_M2, GOLDEN32, TILE_WORDS


def _wrap_i32(v: int) -> int:
    """Encode a u32 constant as the int32 with the same bit pattern."""
    v &= 0xFFFFFFFF
    return v - (1 << 32) if v >= (1 << 31) else v


def make_params(slot_seed: int, device) -> torch.Tensor:
    """int32 tensor [M1, M2, GOLDEN, seed] (u32 bit patterns)."""
    return torch.tensor(
        [_wrap_i32(FMIX_M1), _wrap_i32(FMIX_M2), _wrap_i32(GOLDEN32),
         _wrap_i32(slot_seed)],
        dtype=torch.int32,
        device=device,
    )


@triton.jit
def _fmix32u(h, m1, m2):
    """murmur3 finalizer, native u32 wrapping arithmetic."""
    h = h ^ (h >> 16)
    h = h * m1
    h = h ^ (h >> 13)
    h = h * m2
    h = h ^ (h >> 16)
    return h


@triton.jit
def sketch_sweep_kernel(
    buf_ptr,  # int32 view of the weight buffer (little-endian u32 words)
    out_ptr,  # int32 (u32 bit patterns), one sketch per tile
    tile_ids_ptr,  # int64: canonical tile index of each swept tile (coverage set)
    params_ptr,  # int32[4]: fmix m1, m2, golden, slot_seed (u32 bit patterns)
    n_tiles,
    TILE_WORDS_C: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(axis=0).to(tl.int64)
    if pid >= n_tiles:
        return
    m1 = tl.load(params_ptr + 0).to(tl.uint32, bitcast=True)
    m2 = tl.load(params_ptr + 1).to(tl.uint32, bitcast=True)
    golden = tl.load(params_ptr + 2).to(tl.uint32, bitcast=True)
    seed = tl.load(params_ptr + 3).to(tl.uint32, bitcast=True)

    tile_id = tl.load(tile_ids_ptr + pid)
    r_tile = _fmix32u(_fmix32u(seed ^ tile_id.to(tl.uint32), m1, m2), m1, m2)

    s = tl.zeros((BLOCK,), dtype=tl.uint32)
    base = tile_id * TILE_WORDS_C
    for start in range(0, TILE_WORDS_C, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        w = tl.load(buf_ptr + base + offs).to(tl.uint32, bitcast=True)
        c = _fmix32u(r_tile + offs.to(tl.uint32) * golden, m1, m2) | 1
        s += c * w
    tl.store(out_ptr + pid, tl.sum(s, axis=0).to(tl.int32, bitcast=True))


@triton.jit
def moe_gemm_sketch_kernel(
    a_ptr,
    b_ptr,
    b32_ptr,  # int32 view of b: [E, N, K//2], contiguous
    c_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    partials_ptr,  # int32 (u32 bit patterns), one per canonical tile
    coverage_ptr,
    params_ptr,  # int32[4], as in sketch_sweep_kernel
    N,
    K,
    EM,
    num_valid_tokens,
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

    # ---- PoRW additions (native u32) ----
    K2 = K // 2  # u32 words per (expert, n) row; row == canonical tile
    m1 = tl.load(params_ptr + 0).to(tl.uint32, bitcast=True)
    m2 = tl.load(params_ptr + 1).to(tl.uint32, bitcast=True)
    golden = tl.load(params_ptr + 2).to(tl.uint32, bitcast=True)
    seed = tl.load(params_ptr + 3).to(tl.uint32, bitcast=True)
    tile_idx = off_expert * N + offs_bn
    r_tile = _fmix32u(_fmix32u(seed ^ tile_idx.to(tl.uint32), m1, m2), m1, m2)
    sk = tl.zeros((BLOCK_SIZE_N,), dtype=tl.uint32)
    offs_k2 = tl.arange(0, BLOCK_SIZE_K // 2).to(tl.int64)
    b32_ptrs = (
        b32_ptr
        + off_expert * N * K2
        + offs_k2[:, None]
        + offs_bn[None, :] * K2
    )

    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        k_rem = K - k * BLOCK_SIZE_K
        a = tl.load(
            a_ptrs,
            mask=token_mask[:, None] & (offs_k[None, :] < k_rem),
            other=0.0,
        )
        b = tl.load(b_ptrs, mask=offs_k[:, None] < k_rem, other=0.0)
        # (PoC keeps fp32 dot for interpreter/GPU parity; the production
        # kernel uses the native fp16 tensor-core path.)
        accumulator += tl.dot(a.to(tl.float32), b.to(tl.float32))

        if ENABLE_SKETCH:
            # Second view of the just-loaded tile as u32 words (L1/L2 hit).
            w = tl.load(b32_ptrs).to(tl.uint32, bitcast=True)  # [K2_blk, N_blk]
            j = (k * (BLOCK_SIZE_K // 2) + offs_k2).to(tl.uint32)
            c = _fmix32u(r_tile[None, :] + j[:, None] * golden, m1, m2) | 1
            sk += tl.sum(c * w, axis=0)
            b32_ptrs += BLOCK_SIZE_K // 2

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if ENABLE_SKETCH:
        # Idempotent per-tile store + coverage flag.
        out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        n_mask = out_n < N
        tl.store(
            partials_ptr + tile_idx, sk.to(tl.int32, bitcast=True), mask=n_mask
        )
        tl.store(
            coverage_ptr + tile_idx,
            tl.full((BLOCK_SIZE_N,), 1, tl.int8),
            mask=n_mask,
        )

    c_out = accumulator.to(tl.float16)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N).to(tl.int64)
    c_ptrs = c_ptr + offs_token[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    tl.store(c_ptrs, c_out, mask=token_mask[:, None] & (offs_cn[None, :] < N))


# ---------------------------------------------------------------------------
# Host wrappers
# ---------------------------------------------------------------------------


def _u32_np(t: torch.Tensor):
    """int32 tensor (u32 bit patterns) -> numpy uint32 array."""
    import numpy as np

    return t.cpu().numpy().view(np.uint32)


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
    """Launch the fused kernel; returns (c [M*top_k, N], partials u32 np,
    coverage np)."""
    from .reference import moe_align

    M, K = a.shape
    E, N, Kb = b.shape
    assert K == Kb and b.is_contiguous() and K % 2 == 0 and block_k % 2 == 0
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
    n_tiles = E * N  # 2*K bytes == TILE_BYTES: one tile per (expert, n) row
    partials = torch.zeros(n_tiles, dtype=torch.int32, device=dev)
    coverage = torch.zeros(n_tiles, dtype=torch.int8, device=dev)
    b32 = b.view(torch.int32)
    params = make_params(slot_seed, dev)

    grid = (triton.cdiv(EM, block_m) * triton.cdiv(N, block_n),)
    moe_gemm_sketch_kernel[grid](
        a,
        b,
        b32,
        c,
        sorted_token_ids,
        expert_ids_t,
        num_post_padded_t,
        partials,
        coverage,
        params,
        N,
        K,
        EM,
        num_valid_tokens,
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
    return c, _u32_np(partials), coverage.cpu().numpy()


def run_sketch_sweep(
    buf_bytes: torch.Tensor,
    slot_seed: int,
    tile_ids: torch.Tensor | None = None,
    block: int = 512,
):
    """Sweep over a uint8 buffer; returns per-swept-tile sketches (numpy
    uint32, in tile_ids order).  ``tile_ids`` (int64) selects a coverage
    subset; default = all tiles (S1-over-coverage with full coverage)."""
    assert buf_bytes.dtype == torch.uint8 and buf_bytes.numel() % (TILE_WORDS * 4) == 0
    words = buf_bytes.view(torch.int32)
    total_tiles = words.numel() // TILE_WORDS
    if tile_ids is None:
        tile_ids = torch.arange(total_tiles, dtype=torch.int64, device=buf_bytes.device)
    n_tiles = tile_ids.numel()
    out = torch.zeros(n_tiles, dtype=torch.int32, device=buf_bytes.device)
    params = make_params(slot_seed, buf_bytes.device)
    sketch_sweep_kernel[(n_tiles,)](
        words, out, tile_ids, params, n_tiles,
        TILE_WORDS_C=TILE_WORDS, BLOCK=block,
    )
    return _u32_np(out)
