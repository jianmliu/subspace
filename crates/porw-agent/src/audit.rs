//! Cross-audit: the replica-side of epoch replica cross-verification.
//!
//! Replicas are the only parties holding a model's canonical bytes, so only
//! replicas can audit replicas — and any replica can audit any peer, because
//! the peer's sketch seed is public (`derive_slot_seed(challenge, device)`)
//! and the sketch is deterministic over the shared bytes.
//!
//! Each epoch, [`audit_duties`] tells this agent which peers to audit and
//! which tiles to sample (a pure function of the on-chain beacon — every
//! honest node computes the same schedule). For each sampled tile the target
//! committed, the agent obtains the target's partials opening (off-chain: the
//! target must serve openings on request; refusal is its own offense) and
//! runs [`cross_check`]: recompute the sketch from local canonical bytes and
//! compare. A mismatch yields a ready-to-submit [`TileFraudProof`] — the
//! reporter takes the accused's bond and the accused forfeits its escrowed
//! rewards.
//!
//! Auditors are not paid for clean audits and acknowledge nothing on chain:
//! the schedule directs honest effort and bounds its bandwidth (k peers × t
//! tiles ≈ megabytes per epoch against TB/s memory), while enforcement stays
//! with the permissionless fraud path.

use crate::backend::SketchBackend;
use subspace_proof_of_residency::{
    Hash32, PorwSolution, TileFraudProof, audit_tile_sample, derive_slot_seed, merkle_proof,
    merkle_verify, partials_leaf, select_auditors, sketch_tile,
};

/// One audit duty: sample these tiles of this peer's commitments this epoch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuditTask {
    /// The peer device to audit.
    pub target_device: Hash32,
    /// Canonical tile indices to sample. The auditor checks the intersection
    /// with the tiles the target actually committed this epoch (uncommitted
    /// tiles have nothing to compare against).
    pub tiles: Vec<u64>,
}

/// This agent's audit duties for an epoch: for every peer replica of the
/// model whose beacon-selected auditor panel includes this device, the tile
/// sample to check. Pure function of the beacon — chain, peers, and this
/// agent all derive the identical schedule.
///
/// `k` is the auditor fan-out per target and `t` the tiles sampled per
/// (auditor, target) pair; both are protocol parameters.
pub fn audit_duties(
    beacon: &Hash32,
    model_id: &Hash32,
    my_device: &Hash32,
    replicas: &[Hash32],
    k: usize,
    t: usize,
    n_tiles: u64,
) -> Vec<AuditTask> {
    replicas
        .iter()
        .filter(|target| *target != my_device)
        .filter(|target| select_auditors(beacon, model_id, target, replicas, k).contains(my_device))
        .map(|target| AuditTask {
            target_device: *target,
            tiles: audit_tile_sample(beacon, model_id, target, my_device, n_tiles, t),
        })
        .collect()
}

/// A target's committed per-tile value with its opening, as served by the
/// target on request (off-chain data-availability duty).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommittedTile {
    /// The tile the value was committed for.
    pub tile_idx: u64,
    /// The per-tile sketch value the target committed.
    pub claimed_s_tile: u32,
    /// Position of the leaf in the target's partials tree (coverage order).
    pub partials_index: u64,
    /// Inclusion proof under the solution's `partials_root`.
    pub partials_proof: Vec<Hash32>,
}

/// Outcome of cross-checking one committed tile.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrossCheckOutcome {
    /// The committed value matches the canonical bytes.
    Consistent,
    /// The committed value is wrong: a ready-to-submit fraud proof.
    Fraud(Box<TileFraudProof>),
    /// The served opening does not verify against the solution's
    /// `partials_root` (or the tile is out of range) — nothing is proven
    /// either way. Persistent refusal to serve valid openings is the
    /// target's own (data-availability) offense.
    Unverifiable,
}

/// Cross-check one committed tile of a peer's solution against this
/// replica's canonical bytes.
pub fn cross_check<B: SketchBackend>(
    backend: &B,
    solution: &PorwSolution,
    global_challenge: &Hash32,
    committed: &CommittedTile,
) -> CrossCheckOutcome {
    if committed.tile_idx >= backend.tile_count() {
        return CrossCheckOutcome::Unverifiable;
    }
    // The served opening must actually commit (tile_idx, claimed) under the
    // solution's partials_root, else there is nothing to dispute.
    let leaf = partials_leaf(committed.tile_idx, committed.claimed_s_tile);
    if !merkle_verify(
        &solution.partials_root,
        &leaf,
        committed.partials_index as usize,
        &committed.partials_proof,
    ) {
        return CrossCheckOutcome::Unverifiable;
    }

    // Recompute the true sketch under the target's public slot seed from our
    // own canonical bytes.
    let slot_seed = derive_slot_seed(global_challenge, &solution.device_id);
    let tile_bytes = backend.tile_bytes(committed.tile_idx);
    let tile: &[u8; subspace_proof_of_residency::TILE_BYTES] = tile_bytes
        .as_slice()
        .try_into()
        .expect("backend serves whole tiles; qed");
    let true_s_tile = sketch_tile(slot_seed, committed.tile_idx, tile);
    if true_s_tile == committed.claimed_s_tile {
        return CrossCheckOutcome::Consistent;
    }

    // Mismatch: assemble the fraud proof from our weights tree and the
    // target's own opening.
    let weights_leaves = backend.weights_leaves();
    CrossCheckOutcome::Fraud(Box::new(TileFraudProof {
        tile_idx: committed.tile_idx,
        claimed_s_tile: committed.claimed_s_tile,
        partials_index: committed.partials_index,
        partials_proof: committed.partials_proof.clone(),
        tile_bytes,
        weights_proof: merkle_proof(&weights_leaves, committed.tile_idx as usize),
    }))
}
