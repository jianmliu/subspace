//! Per-slot solution assembly: turn a coverage set + backend into a
//! device-signed [`PorwSolution`] ready to hand to the node's
//! `claim_porw_slot`.

use crate::backend::{SketchBackend, TILE};
use sp_core::{Pair, ed25519};
use subspace_proof_of_residency::{
    Hash32, PorwSolution, derive_slot_seed, merkle_root, partials_leaf, ticket_chunk, ticket_count,
};

/// What the agent knows about a slot: the PoT global challenge and the
/// coverage set (the tiles inference actually touched this slot; on the CPU
/// devnet the driver supplies it, in production it comes from router
/// telemetry).
pub struct SlotContext {
    /// 32-byte global challenge for the slot (from the node's PoT).
    pub global_challenge: [u8; 32],
    /// Canonical tile indices covered this slot.
    pub coverage: Vec<u64>,
    /// Service multiplier in thousandths of a full coverage sweep.
    pub m_t_millis: u64,
}

/// Static per-device parameters for assembly.
pub struct SolutionParams {
    /// Attested physical device id.
    pub device_id: [u8; 32],
    /// Registered model commitment (`R_W`).
    pub model_id: [u8; 32],
    /// Bytes of audited traffic per lottery ticket (runtime constant,
    /// `PORW_TICKET_UNIT`).
    pub ticket_unit: u64,
}

/// SolutionRange is u64 in Subspace; the bidirectional (wrap-around) distance
/// between two points on the ring, matching
/// `subspace_core_primitives::solutions::bidirectional_distance`.
fn ring_distance(a: u64, b: u64) -> u64 {
    let d = a.wrapping_sub(b);
    d.min(b.wrapping_sub(a))
}

fn le_u64_prefix(bytes: &[u8; 32]) -> u64 {
    u64::from_le_bytes(bytes[..8].try_into().expect("32 >= 8; qed"))
}

/// Assemble a signed solution for the slot, choosing the best (lowest
/// ring-distance to the challenge) ticket among those the coverage/multiplier
/// authorize. Returns the solution and its distance (the node decides whether
/// it clears the current solution range), or `None` when the coverage and
/// service multiplier earn zero tickets — the exact same `ticket_count` the
/// chain computes, so the agent never emits a solution the chain would reject
/// as `ChunkOutOfRange`.
pub fn assemble_solution<B: SketchBackend>(
    backend: &B,
    node_key: &ed25519::Pair,
    params: &SolutionParams,
    ctx: &SlotContext,
) -> Option<(PorwSolution, u64)> {
    let slot_seed = derive_slot_seed(&ctx.global_challenge, &params.device_id);

    // Per-tile sketches over the coverage set, and their Merkle commitment.
    let partials = backend.sketch_coverage(slot_seed, &ctx.coverage);
    let partial_leaves: Vec<Hash32> = ctx
        .coverage
        .iter()
        .zip(&partials)
        .map(|(&tile_idx, &s)| partials_leaf(tile_idx, s))
        .collect();
    let partials_root = merkle_root(&partial_leaves);
    let folded = partials.iter().fold(0u32, |a, s| a.wrapping_add(*s));

    let coverage_bytes = (ctx.coverage.len() * TILE) as u64;
    // Raw ticket count — identical to the chain's. Zero tickets means the slot
    // earned no lottery entry, so there is nothing to author.
    let tickets = ticket_count(coverage_bytes, ctx.m_t_millis, params.ticket_unit);
    if tickets == 0 {
        return None;
    }

    // Pick the best ticket (lowest distance to the challenge). This mirrors the
    // farmer picking its best audit chunk; the node re-derives and checks it.
    let challenge_point = le_u64_prefix(&ctx.global_challenge);
    let (chunk_index, _best) = (0..tickets)
        .map(|i| {
            let chunk = ticket_chunk(&params.model_id, &partials_root, slot_seed, i);
            (i, ring_distance(challenge_point, le_u64_prefix(&chunk)))
        })
        .min_by_key(|(_, d)| *d)
        .expect("tickets >= 1; qed");
    let best_chunk = ticket_chunk(&params.model_id, &partials_root, slot_seed, chunk_index);
    let distance = ring_distance(challenge_point, le_u64_prefix(&best_chunk));

    let mut solution = PorwSolution {
        device_id: params.device_id,
        model_id: params.model_id,
        sketch: folded,
        partials_root,
        coverage_bytes,
        m_t_millis: ctx.m_t_millis,
        chunk_index,
        signature: [0u8; 64],
    };
    solution.signature = node_key
        .sign(&solution.signing_payload(&ctx.global_challenge))
        .0;
    Some((solution, distance))
}
