"""PoC validation: sketch spec, fused Triton kernel (CPU interpreter or GPU),
coverage semantics, and the compression-attack demo that motivates per-word
slot-fresh coefficients."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from porw_sketch import spec
from porw_sketch.kernels import run_moe_gemm, run_sketch_sweep
from porw_sketch.reference import covered_experts, moe_gemm_reference

RNG = np.random.default_rng(7)

# PoC dims: 2*K bytes == TILE_BYTES so one 4 KiB tile == one (expert, n) row.
E, N, K = 4, 128, spec.TILE_BYTES // 2
M, TOP_K = 24, 2
SLOT_SEEDS = [0x00000001, 0xDEADBEEF, 0x9E3779B9]


def random_weights() -> np.ndarray:
    w = RNG.standard_normal((E, N, K), dtype=np.float32).astype(np.float16)
    return w


def weight_bytes(b: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(b).view(np.uint8).reshape(-1)


# ---------------------------------------------------------------------------
# Spec properties
# ---------------------------------------------------------------------------


def test_spec_deterministic_and_slot_sensitive():
    buf = weight_bytes(random_weights())
    s1 = spec.sketch_tiles(SLOT_SEEDS[0], buf)
    s2 = spec.sketch_tiles(SLOT_SEEDS[0], buf)
    s3 = spec.sketch_tiles(SLOT_SEEDS[1], buf)
    assert np.array_equal(s1, s2)
    assert (s1 != s3).mean() > 0.99  # fresh coefficients per slot


def test_spec_partition_independence():
    """Any decomposition of the word sum gives the same tile sketch —
    the property that makes kernel block shape / launch order irrelevant."""
    buf = weight_bytes(random_weights())
    ref = spec.sketch_tiles(SLOT_SEEDS[1], buf)
    words = buf.view("<u4").astype(np.uint64).reshape(-1, spec.TILE_WORDS)
    coeffs = spec.tile_coeffs(
        SLOT_SEEDS[1], np.arange(words.shape[0], dtype=np.uint64)
    )
    perm = RNG.permutation(spec.TILE_WORDS)
    chunks = np.array_split(perm, 13)
    acc = np.zeros(words.shape[0], dtype=np.uint64)
    for ch in chunks:  # arbitrary partition, arbitrary order
        acc = (acc + (coeffs[:, ch] * words[:, ch]).sum(axis=1)) & spec.M32
    assert np.array_equal(acc, ref)


def test_single_bit_corruption_detected():
    buf = weight_bytes(random_weights())
    ref = spec.sketch_tiles(SLOT_SEEDS[2], buf)
    for _ in range(64):
        i = int(RNG.integers(buf.size))
        bad = buf.copy()
        bad[i] ^= 1 << int(RNG.integers(8))
        got = spec.sketch_tiles(SLOT_SEEDS[2], bad)
        assert not np.array_equal(got, ref)


# ---------------------------------------------------------------------------
# Compression attack demo (why per-word slot-fresh coefficients are mandatory)
# ---------------------------------------------------------------------------


def test_per_tile_coeff_scheme_is_broken():
    """With per-tile constant coefficients, storing 4 bytes per 4096-byte
    tile (the word sum) reproduces the sketch for EVERY slot: a 1024x
    compression that fully defeats the residency proof."""
    buf = weight_bytes(random_weights())
    words = buf.view("<u4").astype(np.uint64).reshape(-1, spec.TILE_WORDS)
    stolen_summary = words.sum(axis=1) & spec.M32  # 4 bytes/tile
    n_tiles = words.shape[0]
    for seed in range(100):
        honest = spec.sketch_tiles_broken_per_tile_coeff(seed, buf)
        r_tile = spec.fmix32(
            spec.fmix32(np.uint64(seed) ^ np.arange(n_tiles, dtype=np.uint64))
        )
        forged = (r_tile * stolen_summary) & spec.M32
        assert np.array_equal(forged, honest)  # attack succeeds every slot


def test_per_word_coeff_scheme_resists_compression():
    """Against the real scheme, the same 4-byte-per-tile adversary (and a
    stronger 64-functional one) fails on every slot tried."""
    buf = weight_bytes(random_weights())
    words = buf.view("<u4").astype(np.uint64).reshape(-1, spec.TILE_WORDS)
    n_tiles = words.shape[0]
    # Adversary A: stores word sums only, forges sketch as r_tile * sum.
    stolen_summary = words.sum(axis=1) & spec.M32
    # Adversary B: stores 64 fixed random linear functionals per tile
    # (256 bytes per 4096-byte tile) and forges via least-squares
    # reconstruction of the tile from them.
    F = RNG.integers(0, 4, size=(64, spec.TILE_WORDS)).astype(np.float64)
    stored = F @ words.T.astype(np.float64)  # 64 values per tile
    recon, *_ = np.linalg.lstsq(F, stored, rcond=None)
    recon_words = np.clip(np.round(recon.T), 0, spec.M32).astype(np.uint64)
    for seed in SLOT_SEEDS:
        honest = spec.sketch_tiles(seed, buf)
        r_tile = spec.fmix32(
            spec.fmix32(np.uint64(seed) ^ np.arange(n_tiles, dtype=np.uint64))
        )
        forged_a = (r_tile * stolen_summary) & spec.M32
        coeffs = spec.tile_coeffs(seed, np.arange(n_tiles, dtype=np.uint64))
        forged_b = (coeffs * recon_words).sum(axis=1) & spec.M32
        assert not np.array_equal(forged_a, honest)
        assert not np.array_equal(forged_b, honest)
        # per-tile: essentially every tile mismatches
        assert (forged_a == honest).mean() < 0.01
        assert (forged_b == honest).mean() < 0.01


# ---------------------------------------------------------------------------
# Triton kernels (CPU interpreter without GPU; native backend with one)
# ---------------------------------------------------------------------------


def test_sweep_kernel_matches_reference():
    b = random_weights()
    buf = weight_bytes(b)
    for seed in SLOT_SEEDS[:2]:
        got = run_sketch_sweep(torch.from_numpy(buf.copy()), seed)
        ref = spec.sketch_tiles(seed, buf)
        assert np.array_equal(got.astype(np.uint64), ref)


def test_sweep_kernel_coverage_subset():
    """S1-over-coverage: sweeping only the tiles in a coverage set yields
    the same per-tile values as a full sweep, in tile_ids order."""
    b = random_weights()
    buf = weight_bytes(b)
    seed = SLOT_SEEDS[2]
    full = spec.sketch_tiles(seed, buf)
    rng = np.random.default_rng(3)
    subset = np.sort(rng.choice(full.size, size=full.size // 3, replace=False))
    got = run_sketch_sweep(
        torch.from_numpy(buf.copy()), seed,
        tile_ids=torch.from_numpy(subset.astype(np.int64)),
    )
    assert np.array_equal(got.astype(np.uint64), full[subset])


@pytest.fixture(scope="module")
def moe_run():
    b = random_weights()
    a = RNG.standard_normal((M, K), dtype=np.float32).astype(np.float16)
    # Route to experts {0, 2, 3} only — expert 1 stays cold.
    topk_ids = RNG.choice([0, 2, 3], size=(M, TOP_K)).astype(np.int32)
    seed = SLOT_SEEDS[1]
    c, partials, coverage = run_moe_gemm(
        torch.from_numpy(a.copy()),
        torch.from_numpy(b.copy()),
        torch.from_numpy(topk_ids.copy()),
        seed,
    )
    return a, b, topk_ids, seed, c.numpy(), partials, coverage


def test_moe_kernel_gemm_correct(moe_run):
    a, b, topk_ids, _, c, _, _ = moe_run
    ref = moe_gemm_reference(a, b, topk_ids)
    np.testing.assert_allclose(c.astype(np.float32), ref, rtol=2e-2, atol=2e-2)


def test_moe_kernel_sketch_matches_spec(moe_run):
    _, b, topk_ids, seed, _, partials, coverage = moe_run
    ref_tiles = spec.sketch_tiles(seed, weight_bytes(b))  # all E*N tiles
    hot = covered_experts(topk_ids)
    for e in range(E):
        sl = slice(e * N, (e + 1) * N)
        if e in hot:
            assert coverage[sl].all(), f"expert {e} should be covered"
            assert np.array_equal(
                partials[sl].astype(np.uint64), ref_tiles[sl]
            ), f"expert {e} sketch mismatch"
        else:
            assert not coverage[sl].any(), f"cold expert {e} must stay uncovered"


def test_moe_kernel_batch_invariance():
    """Different batch compositions touching the same experts produce
    identical per-tile sketches (idempotent-store semantics)."""
    b = random_weights()
    seed = SLOT_SEEDS[2]
    outs = []
    for m in (8, 24):
        a = RNG.standard_normal((m, K), dtype=np.float32).astype(np.float16)
        topk_ids = np.full((m, TOP_K), 0, dtype=np.int32)
        topk_ids[:, 1] = 2
        _, partials, coverage = run_moe_gemm(
            torch.from_numpy(a),
            torch.from_numpy(b.copy()),
            torch.from_numpy(topk_ids),
            seed,
        )
        outs.append((partials.copy(), coverage.copy()))
    assert np.array_equal(outs[0][0], outs[1][0])
    assert np.array_equal(outs[0][1], outs[1][1])


def test_fused_equals_sweep_on_covered_tiles(moe_run):
    """Cross-check: the fused kernel and the standalone sweep kernel agree
    tile-for-tile — so dense layers (S1 sweep) and MoE layers (S2 fused)
    can share one verifier."""
    _, b, topk_ids, seed, _, partials, coverage = moe_run
    sweep = run_sketch_sweep(
        torch.from_numpy(weight_bytes(b).copy()), seed
    )
    mask = coverage.astype(bool)
    assert np.array_equal(partials[mask], sweep[mask])
