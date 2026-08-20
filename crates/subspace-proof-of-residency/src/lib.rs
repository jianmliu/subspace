//! Proof-of-Resident-Weights (PoRW) consensus primitives.
//!
//! This crate is the canonical Rust implementation of the PoRW residency
//! sketch (spec v2) and its supporting structures:
//!
//! - the per-tile **sketch**: a challenge-randomized, word-granular linear
//!   digest over raw weight bytes, computable only by reading every covered
//!   word in the current slot (see `docs/porw-p1-feasibility.md` §3 for why
//!   word-granular slot-fresh coefficients are mandatory);
//! - **tile Merkle commitments** for both the registered model weights
//!   (`R_W`, root over weight tiles) and the per-slot per-tile sketch values
//!   (`partials_root`), enabling O(tile) fraud proofs;
//! - **ticket expansion** turning a solution's sketch commitment into a
//!   stream of 32-byte audit chunks (one lottery ticket each), whose length
//!   is proportional to `coverage × service multiplier`;
//! - the **hardware envelope check** capping claimed work at the device's
//!   physical bandwidth, so a compromised TEE yields bounded inflation.
//!
//! Bit-compatibility: `sketch_tile` matches the Python/numpy reference and
//! the Triton kernels in `porw-poc/` bit-for-bit; cross-language test
//! vectors generated from the Python spec are checked in `tests`.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use parity_scale_codec::{Decode, Encode};
use scale_info::TypeInfo;

/// Canonical tile size in bytes.
pub const TILE_BYTES: usize = 4096;
/// 32-bit little-endian words per tile.
pub const TILE_WORDS: usize = TILE_BYTES / 4;
/// Coefficient index stride (golden ratio, murmur-style).
pub const GOLDEN32: u32 = 0x9E37_79B9;

const FMIX_M1: u32 = 0x85EB_CA6B;
const FMIX_M2: u32 = 0xC2B2_AE35;

/// murmur3 32-bit finalizer.
#[inline]
pub fn fmix32(mut h: u32) -> u32 {
    h ^= h >> 16;
    h = h.wrapping_mul(FMIX_M1);
    h ^= h >> 13;
    h = h.wrapping_mul(FMIX_M2);
    h ^= h >> 16;
    h
}

/// Per-tile coefficient seed.
#[inline]
pub fn tile_seed(slot_seed: u32, tile_idx: u64) -> u32 {
    fmix32(fmix32(slot_seed ^ (tile_idx as u32)))
}

/// Per-word coefficient. Forced odd: odd multipliers are bijective mod 2^32,
/// binding every bit of the word including the MSB (an even coefficient would
/// let a bit-31 flip vanish, since c * 2^31 mod 2^32 == 0 for even c).
#[inline]
pub fn word_coeff(r_tile: u32, j: u32) -> u32 {
    fmix32(r_tile.wrapping_add(j.wrapping_mul(GOLDEN32))) | 1
}

/// Sketch of one canonical tile: sum over 32-bit LE words of
/// `coeff(j) * word(j) mod 2^32`. Order/partition independent (modular sum),
/// so any kernel decomposition that touches each word exactly once agrees.
pub fn sketch_tile(slot_seed: u32, tile_idx: u64, tile: &[u8; TILE_BYTES]) -> u32 {
    let r_tile = tile_seed(slot_seed, tile_idx);
    let mut acc = 0u32;
    for (j, word) in tile.chunks_exact(4).enumerate() {
        let w = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
        acc = acc.wrapping_add(word_coeff(r_tile, j as u32).wrapping_mul(w));
    }
    acc
}

/// Per-device slot seed: first 4 LE bytes of blake3(global_challenge || device_id).
/// Mixing the device id makes sketches non-transferable between devices.
pub fn derive_slot_seed(global_challenge: &[u8; 32], device_id: &[u8; 32]) -> u32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(global_challenge);
    hasher.update(device_id);
    let hash = hasher.finalize();
    let bytes = hash.as_bytes();
    u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

// ---------------------------------------------------------------------------
// Tile Merkle commitments (binary blake3 tree, duplicate-last padding)
// ---------------------------------------------------------------------------

/// 32-byte Merkle node/root.
pub type Hash32 = [u8; 32];

/// Leaf for the weights commitment `R_W`: blake3(LE64 tile_idx || tile bytes).
pub fn weights_leaf(tile_idx: u64, tile: &[u8; TILE_BYTES]) -> Hash32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&tile_idx.to_le_bytes());
    hasher.update(tile);
    *hasher.finalize().as_bytes()
}

/// Leaf for the per-slot sketch commitment `partials_root`:
/// blake3(LE64 tile_idx || LE32 s_tile).
pub fn partials_leaf(tile_idx: u64, s_tile: u32) -> Hash32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&tile_idx.to_le_bytes());
    hasher.update(&s_tile.to_le_bytes());
    *hasher.finalize().as_bytes()
}

fn merkle_parent(left: &Hash32, right: &Hash32) -> Hash32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(left);
    hasher.update(right);
    *hasher.finalize().as_bytes()
}

/// Merkle root over leaves (duplicate-last padding at each level).
/// Empty input yields the hash of the empty string.
pub fn merkle_root(leaves: &[Hash32]) -> Hash32 {
    if leaves.is_empty() {
        return *blake3::hash(&[]).as_bytes();
    }
    let mut level: Vec<Hash32> = leaves.to_vec();
    while level.len() > 1 {
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let right = pair.get(1).unwrap_or(&pair[0]);
            next.push(merkle_parent(&pair[0], right));
        }
        level = next;
    }
    level[0]
}

/// Inclusion proof: sibling hashes from leaf level to root.
pub fn merkle_proof(leaves: &[Hash32], mut index: usize) -> Vec<Hash32> {
    let mut proof = Vec::new();
    let mut level: Vec<Hash32> = leaves.to_vec();
    while level.len() > 1 {
        let sibling = if index % 2 == 0 {
            *level.get(index + 1).unwrap_or(&level[index])
        } else {
            level[index - 1]
        };
        proof.push(sibling);
        let mut next = Vec::with_capacity(level.len().div_ceil(2));
        for pair in level.chunks(2) {
            let right = pair.get(1).unwrap_or(&pair[0]);
            next.push(merkle_parent(&pair[0], right));
        }
        level = next;
        index /= 2;
    }
    proof
}

/// Verify an inclusion proof produced by [`merkle_proof`].
pub fn merkle_verify(root: &Hash32, leaf: &Hash32, mut index: usize, proof: &[Hash32]) -> bool {
    let mut acc = *leaf;
    for sibling in proof {
        acc = if index % 2 == 0 {
            merkle_parent(&acc, sibling)
        } else {
            merkle_parent(sibling, &acc)
        };
        index /= 2;
    }
    acc == *root
}

// ---------------------------------------------------------------------------
// Tickets and envelope
// ---------------------------------------------------------------------------

/// Number of lottery tickets for a solution: covered bytes swept per slot,
/// in ticket units (one ticket per `ticket_unit` bytes of audited traffic).
/// `m_t_millis` is the service multiplier in thousandths of a full coverage
/// sweep (>= 1000 means the coverage set was swept at least once).
pub fn ticket_count(coverage_bytes: u64, m_t_millis: u64, ticket_unit: u64) -> u64 {
    (coverage_bytes.saturating_mul(m_t_millis) / 1000) / ticket_unit.max(1)
}

/// Hardware envelope: claimed audited traffic must not exceed what the
/// device's registered bandwidth can physically move in one slot. This is
/// the bound that turns a fully compromised TEE into bounded inflation.
pub fn check_envelope(coverage_bytes: u64, m_t_millis: u64, bandwidth_bytes_per_slot: u64) -> bool {
    // coverage_bytes * m_t_millis / 1000 <= bandwidth_bytes_per_slot
    coverage_bytes.saturating_mul(m_t_millis) <= bandwidth_bytes_per_slot.saturating_mul(1000)
}

/// Derive the `chunk_index`-th 32-byte audit chunk (lottery ticket) from a
/// solution commitment, via blake3 XOF over (partials_root || slot_seed).
pub fn ticket_chunk(partials_root: &Hash32, slot_seed: u32, chunk_index: u64) -> Hash32 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(partials_root);
    hasher.update(&slot_seed.to_le_bytes());
    let mut reader = hasher.finalize_xof();
    let mut out = [0u8; 32];
    reader.set_position(chunk_index.saturating_mul(32));
    reader.fill(&mut out);
    out
}

// ---------------------------------------------------------------------------
// Solution and fraud proof types
// ---------------------------------------------------------------------------

/// PoRW solution: what a winning device submits alongside its signature.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, TypeInfo)]
pub struct PorwSolution {
    /// Attested physical device (one card, one identity).
    pub device_id: [u8; 32],
    /// Registered model commitment root (`R_W`) this solution audits.
    pub model_id: [u8; 32],
    /// Folded sketch over the coverage set (fast consistency check).
    pub sketch: u32,
    /// Merkle root of per-tile sketch values — anchor for tile-granular
    /// fraud proofs.
    pub partials_root: [u8; 32],
    /// Covered bytes (size of the coverage set) this slot.
    pub coverage_bytes: u64,
    /// Service multiplier in thousandths of a full coverage sweep.
    pub m_t_millis: u64,
    /// Index of the winning ticket chunk.
    pub chunk_index: u64,
}

/// Tile-granular fraud proof against a committed solution: shows that the
/// per-tile sketch value committed under `partials_root` disagrees with the
/// value recomputed from the canonical weight bytes committed under `R_W`.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, TypeInfo)]
pub struct TileFraudProof {
    /// The disputed tile.
    pub tile_idx: u64,
    /// Claimed per-tile sketch value (as committed by the accused).
    pub claimed_s_tile: u32,
    /// Inclusion proof of `(tile_idx, claimed_s_tile)` under `partials_root`.
    pub partials_proof: Vec<[u8; 32]>,
    /// Canonical tile bytes (retrieved from the DSN / a resident replica).
    pub tile_bytes: Vec<u8>,
    /// Inclusion proof of `(tile_idx, tile_bytes)` under the model's `R_W`.
    pub weights_proof: Vec<[u8; 32]>,
}

/// Outcome of verifying a [`TileFraudProof`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FraudVerdict {
    /// Proof is valid and demonstrates fraud: claimed value is committed,
    /// tile bytes are canonical, and the recomputed sketch disagrees.
    Fraud,
    /// Proof is valid but the recomputed sketch agrees — no fraud shown.
    NoFraud,
    /// Proof is malformed (bad lengths or Merkle paths do not verify).
    Invalid,
}

/// Verify a tile fraud proof against a solution's commitments.
///
/// `global_challenge` is the PoT challenge of the disputed slot; the
/// per-device slot seed is derived internally so the verifier needs no
/// knowledge of the accused's workload.
pub fn verify_tile_fraud_proof(
    solution: &PorwSolution,
    global_challenge: &[u8; 32],
    model_root: &Hash32,
    proof: &TileFraudProof,
) -> FraudVerdict {
    if proof.tile_bytes.len() != TILE_BYTES {
        return FraudVerdict::Invalid;
    }
    // 1. The claimed per-tile value must be committed under partials_root.
    let claimed_leaf = partials_leaf(proof.tile_idx, proof.claimed_s_tile);
    if !merkle_verify(
        &solution.partials_root,
        &claimed_leaf,
        proof.tile_idx as usize,
        &proof.partials_proof,
    ) {
        return FraudVerdict::Invalid;
    }
    // 2. The tile bytes must be the canonical bytes committed under R_W.
    let tile: &[u8; TILE_BYTES] = proof
        .tile_bytes
        .as_slice()
        .try_into()
        .expect("length checked above; qed");
    let weights_leaf_hash = weights_leaf(proof.tile_idx, tile);
    if !merkle_verify(
        model_root,
        &weights_leaf_hash,
        proof.tile_idx as usize,
        &proof.weights_proof,
    ) {
        return FraudVerdict::Invalid;
    }
    // 3. Recompute the true sketch and compare.
    let slot_seed = derive_slot_seed(global_challenge, &solution.device_id);
    let true_s_tile = sketch_tile(slot_seed, proof.tile_idx, tile);
    if true_s_tile == proof.claimed_s_tile {
        FraudVerdict::NoFraud
    } else {
        FraudVerdict::Fraud
    }
}

#[cfg(test)]
mod tests;
