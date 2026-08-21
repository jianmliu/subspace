//! The pluggable sketch backend.
//!
//! The agent is backend-agnostic: it hands a coverage set and a slot seed to a
//! [`SketchBackend`] and gets back one per-tile sketch value per covered tile.
//! The CPU backend ([`crate::cpu::CpuSketchBackend`]) computes them with the
//! canonical [`subspace_proof_of_residency::sketch_tile`] — bit-identical to
//! the GPU Triton kernels — so the whole agent runs and is validated without a
//! GPU. A production GPU backend implements the same trait over an
//! HBM-resident buffer with the S1-over-coverage sweep kernel; nothing else in
//! the agent changes.

use subspace_proof_of_residency::{Hash32, TILE_BYTES};

/// A model whose raw weight bytes the backend can sketch. Backends hold their
/// own residency (host RAM for CPU, HBM for GPU); the agent addresses tiles by
/// canonical index.
pub trait SketchBackend {
    /// Number of canonical 4 KiB tiles in the resident model.
    fn tile_count(&self) -> u64;

    /// Compute the per-tile sketch for each tile in `coverage`, in the same
    /// order, under `slot_seed`. Each entry is the canonical `sketch_tile`
    /// value for that tile. `coverage` indices are `< tile_count()`.
    fn sketch_coverage(&self, slot_seed: u32, coverage: &[u64]) -> Vec<u32>;

    /// Canonical bytes of one tile (`TILE_BYTES` long). Cross-audit path:
    /// the auditor recomputes a peer's sketch from these bytes and, on a
    /// mismatch, embeds them in the fraud proof. Host-side and cold — a GPU
    /// backend serves this from its staging copy, not the HBM hot path.
    fn tile_bytes(&self, idx: u64) -> Vec<u8>;

    /// The model's weight-commitment leaves (`weights_leaf` per tile, in
    /// order) — the tree under `R_W`. Used to build the `weights_proof`
    /// opening of a fraud proof. Cold path, host-side.
    fn weights_leaves(&self) -> Vec<Hash32>;
}

/// Bytes per canonical tile (re-exported for backend implementors).
pub const TILE: usize = TILE_BYTES;
