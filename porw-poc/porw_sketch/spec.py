"""PoRW sketch — canonical specification and numpy reference implementation.

Spec (v0, PoC parameters):

- The registered weight buffer is a byte array (the bytes as stored in HBM:
  fp16/bf16/int8/fp4 — the sketch is over stored bytes, not decoded values).
- Canonical word: 16-bit little-endian word ``w_j`` (index j over the buffer).
- Canonical tile: TILE_BYTES = 4096 bytes = TILE_WORDS = 2048 words.
- Per-slot randomness: ``slot_seed`` (u32 in the PoC; production: u64 derived
  from PoT global challenge + device_id).

Coefficients (the crypto-critical part — see docs/porw-p1-feasibility.md §3
for why per-WORD slot-fresh coefficients are mandatory; per-tile constants
are compressible to 8 bytes/tile and completely break the scheme):

    r_tile     = fmix32(fmix32(slot_seed ^ tile_idx))
    c_j        = fmix32(r_tile + (j_in_tile * GOLDEN32))      # per word
    s_tile     = sum_j c_j * u32(w_j)   mod 2^32              # per tile
    sketch     = blake-like fold of all covered s_tile         # global

Order-independence: s_tile is a sum mod 2^32, so any kernel decomposition
(any block shape, any launch order, atomic or idempotent accumulation)
yields the same value as long as every covered word contributes exactly once
per slot.

All arithmetic is integer and exact — independent of GPU FP behavior.
The PoC uses 32-bit accumulation for GPU cheapness; forgery probability per
tile per slot is ~2^-32, amplified across tiles/slots by fraud-proof
cross-checks. A 64-bit variant is a straightforward parameter change.
"""

import numpy as np

TILE_BYTES = 4096
TILE_WORDS = TILE_BYTES // 2
GOLDEN32 = 0x9E3779B9
M32 = 0xFFFFFFFF


def fmix32(h: np.ndarray) -> np.ndarray:
    """murmur3 32-bit finalizer, vectorized, on uint64 arrays masked to u32.

    Matches the Triton kernel implementation op-for-op (int64 + mask, so the
    interpreter/GPU/numpy all agree bit-exactly).
    """
    h = h & M32
    h = (h ^ (h >> 16)) & M32
    h = (h * 0x85EBCA6B) & M32
    h = (h ^ (h >> 13)) & M32
    h = (h * 0xC2B2AE35) & M32
    h = (h ^ (h >> 16)) & M32
    return h


def tile_coeffs(slot_seed: int, tile_idx: np.ndarray | int) -> np.ndarray:
    """Per-word coefficients for one or more tiles.

    Returns shape (..., TILE_WORDS) uint64 (values < 2^32).
    """
    tile_idx = np.asarray(tile_idx, dtype=np.uint64)
    r_tile = fmix32(fmix32((slot_seed & M32) ^ tile_idx))
    j = np.arange(TILE_WORDS, dtype=np.uint64)
    return fmix32(r_tile[..., None] + (j * GOLDEN32 & M32))


def sketch_tiles(slot_seed: int, buf: np.ndarray) -> np.ndarray:
    """Reference sketch: per-tile s_tile over a contiguous byte buffer.

    buf: uint8 array, length a multiple of TILE_BYTES.
    Returns uint64 array (values < 2^32), one entry per tile.
    """
    assert buf.dtype == np.uint8 and buf.size % TILE_BYTES == 0
    words = buf.view("<u2").astype(np.uint64).reshape(-1, TILE_WORDS)
    n_tiles = words.shape[0]
    coeffs = tile_coeffs(slot_seed, np.arange(n_tiles, dtype=np.uint64))
    return (coeffs * words).sum(axis=1) & M32


def sketch_tiles_broken_per_tile_coeff(slot_seed: int, buf: np.ndarray) -> np.ndarray:
    """The BROKEN variant (per-tile constant coefficient) used by the attack
    demo in tests: s_tile = r_tile * sum_j(w_j).  An adversary who stores only
    sum_j(w_j) (4 bytes per 4096-byte tile) reproduces this for every slot.
    """
    assert buf.dtype == np.uint8 and buf.size % TILE_BYTES == 0
    words = buf.view("<u2").astype(np.uint64).reshape(-1, TILE_WORDS)
    n_tiles = words.shape[0]
    r_tile = fmix32(
        fmix32((slot_seed & M32) ^ np.arange(n_tiles, dtype=np.uint64))
    )
    return (r_tile * (words.sum(axis=1) & M32)) & M32
