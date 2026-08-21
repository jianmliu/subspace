//! CPU sketch backend: holds the raw model bytes in host memory and computes
//! sketches with the canonical reference implementation. Slow but exact and
//! GPU-free — it lets the whole agent and a devnet run and be validated on
//! any machine, and it is the correctness oracle a GPU backend is checked
//! against (both call `sketch_tile`).

use crate::backend::{SketchBackend, TILE};
use subspace_proof_of_residency::{Hash32, merkle_root, sketch_tile, weights_leaf};

/// A model resident in host RAM.
pub struct CpuSketchBackend {
    /// Raw weight bytes, length a multiple of `TILE`.
    weights: Vec<u8>,
    tile_count: u64,
}

impl CpuSketchBackend {
    /// Build from raw weight bytes (must be a whole number of tiles).
    pub fn new(weights: Vec<u8>) -> Result<Self, &'static str> {
        if weights.is_empty() || weights.len() % TILE != 0 {
            return Err("weights length must be a non-zero multiple of the tile size");
        }
        let tile_count = (weights.len() / TILE) as u64;
        Ok(Self {
            weights,
            tile_count,
        })
    }

    fn tile(&self, idx: u64) -> &[u8; TILE] {
        let start = idx as usize * TILE;
        self.weights[start..start + TILE]
            .try_into()
            .expect("slice is exactly one tile; qed")
    }

    /// The model's `R_W` commitment: Merkle root over per-tile weight leaves.
    /// This is the `model_id` a device announces and the registry stores.
    pub fn model_root(&self) -> Hash32 {
        let leaves: Vec<Hash32> = (0..self.tile_count)
            .map(|i| weights_leaf(i, self.tile(i)))
            .collect();
        merkle_root(&leaves)
    }
}

impl SketchBackend for CpuSketchBackend {
    fn tile_count(&self) -> u64 {
        self.tile_count
    }

    fn sketch_coverage(&self, slot_seed: u32, coverage: &[u64]) -> Vec<u32> {
        coverage
            .iter()
            .map(|&idx| sketch_tile(slot_seed, idx, self.tile(idx)))
            .collect()
    }
}
