"""Host-side helpers and numpy references for the PoC."""

import numpy as np


def moe_align(topk_ids: np.ndarray, num_experts: int, block_m: int):
    """Minimal reimplementation of vLLM's ``moe_align_block_size``.

    topk_ids: [M, top_k] expert assignment per (token, slot).
    Returns (sorted_token_ids [EM] int32, expert_ids [EM/block_m] int32,
    num_tokens_post_padded int). Padding entries are ``M*top_k`` (masked out
    in-kernel via ``num_valid_tokens``).
    """
    flat = topk_ids.reshape(-1)
    num_valid = flat.size
    sorted_ids, expert_ids = [], []
    for e in range(num_experts):
        idxs = np.nonzero(flat == e)[0].astype(np.int32)
        if idxs.size == 0:
            continue
        pad = (-idxs.size) % block_m
        padded = np.concatenate([idxs, np.full(pad, num_valid, dtype=np.int32)])
        sorted_ids.append(padded)
        expert_ids.extend([e] * (padded.size // block_m))
    sorted_token_ids = (
        np.concatenate(sorted_ids)
        if sorted_ids
        else np.zeros(0, dtype=np.int32)
    )
    return (
        sorted_token_ids.astype(np.int32),
        np.asarray(expert_ids, dtype=np.int32),
        int(sorted_token_ids.size),
    )


def moe_gemm_reference(a: np.ndarray, b: np.ndarray, topk_ids: np.ndarray):
    """out[i] = a[i // top_k] @ b[expert(i)].T for flat slot index i."""
    M, K = a.shape
    top_k = topk_ids.shape[1]
    E, N, _ = b.shape
    out = np.zeros((M * top_k, N), dtype=np.float32)
    flat = topk_ids.reshape(-1)
    a32, b32 = a.astype(np.float32), b.astype(np.float32)
    for i in range(M * top_k):
        out[i] = a32[i // top_k] @ b32[flat[i]].T
    return out


def covered_experts(topk_ids: np.ndarray) -> set[int]:
    return set(int(e) for e in np.unique(topk_ids.reshape(-1)))
