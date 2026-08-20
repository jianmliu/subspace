//! Proof-of-Resident-Weights (PoRW) solution validation for block authorship.
//!
//! This module is the client-side glue between a [`PorwSolution`] produced by
//! an attested PoRW agent and the chain's lottery: fast-path registry checks
//! run through the [`PorwApi`] runtime API (device, activation delay,
//! measurement, model announcement, hardware envelope — returning the ticket
//! count), the winning ticket chunk is re-derived and its distance to the
//! PoT global challenge is checked against the stake-scaled solution range
//! (composing [`scale_solution_range`], mirroring the farming path in
//! [`crate::slot_worker`]).
//!
//! Wiring these checks into `slot_worker`/`block_import` block production
//! (PreDigest carriage, reward signing) is the P3 milestone; the validation
//! semantics live here so both sides share one implementation.

use sp_api::ProvideRuntimeApi;
use sp_consensus_subspace::{PorwApi, scale_solution_range};
use sp_runtime::traits::Block as BlockT;
use subspace_core_primitives::solutions::SolutionRange;
use subspace_proof_of_residency::{PorwSolution, derive_slot_seed, ticket_chunk};

/// Why a PoRW solution was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PorwSolutionError {
    /// Runtime API call failed.
    RuntimeApi,
    /// Registry fast path rejected the solution (unknown/inactive device,
    /// revoked measurement, unannounced model, or envelope exceeded).
    Rejected,
    /// The claimed chunk index is not within the solution's ticket count.
    ChunkOutOfRange,
    /// The winning chunk's distance exceeds the (stake-scaled) target range.
    OutsideSolutionRange,
}

/// Bidirectional distance between the derived ticket chunk and the global
/// challenge, in [`SolutionRange`] units — same semantics as the farming
/// path's audit-chunk distance.
pub fn porw_solution_distance(
    solution: &PorwSolution,
    global_challenge: &[u8; 32],
) -> SolutionRange {
    let slot_seed = derive_slot_seed(global_challenge, &solution.device_id);
    let chunk = ticket_chunk(&solution.partials_root, slot_seed, solution.chunk_index);
    let chunk_as_range = SolutionRange::from_le_bytes(
        chunk[..size_of::<SolutionRange>()]
            .try_into()
            .expect("Chunk is larger than solution range; qed"),
    );
    let challenge_as_range = SolutionRange::from_le_bytes(
        global_challenge[..size_of::<SolutionRange>()]
            .try_into()
            .expect("Challenge is larger than solution range; qed"),
    );
    subspace_core_primitives::solutions::bidirectional_distance(
        &challenge_as_range,
        &chunk_as_range,
    )
}

/// Validate a PoRW solution against the parent state: registry fast path via
/// [`PorwApi`], ticket-range check, and the stake-scaled solution-range
/// check. Returns the solution distance (for best-solution selection).
pub fn verify_porw_solution<Block, Client>(
    client: &Client,
    parent_hash: Block::Hash,
    solution: &PorwSolution,
    global_challenge: &[u8; 32],
    solution_range: SolutionRange,
    voter_weight: u128,
    max_voter_weight: u128,
) -> Result<SolutionRange, PorwSolutionError>
where
    Block: BlockT,
    Client: ProvideRuntimeApi<Block>,
    Client::Api: PorwApi<Block>,
{
    let tickets = client
        .runtime_api()
        .porw_solution_tickets(parent_hash, solution.clone())
        .map_err(|_| PorwSolutionError::RuntimeApi)?
        .ok_or(PorwSolutionError::Rejected)?;
    if solution.chunk_index >= tickets {
        return Err(PorwSolutionError::ChunkOutOfRange);
    }

    let distance = porw_solution_distance(solution, global_challenge);
    let scaled_solution_range =
        scale_solution_range(solution_range, voter_weight, max_voter_weight);
    if distance <= scaled_solution_range / 2 {
        Ok(distance)
    } else {
        Err(PorwSolutionError::OutsideSolutionRange)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_deterministic_and_seed_sensitive() {
        let solution = PorwSolution {
            device_id: [3; 32],
            model_id: [5; 32],
            sketch: 42,
            partials_root: [7; 32],
            coverage_bytes: 1 << 30,
            m_t_millis: 1000,
            chunk_index: 0,
        };
        let challenge_a = [1u8; 32];
        let challenge_b = [2u8; 32];
        assert_eq!(
            porw_solution_distance(&solution, &challenge_a),
            porw_solution_distance(&solution, &challenge_a),
        );
        assert_ne!(
            porw_solution_distance(&solution, &challenge_a),
            porw_solution_distance(&solution, &challenge_b),
        );
        // A different device derives a different chunk for the same slot.
        let mut other = solution.clone();
        other.device_id = [4; 32];
        assert_ne!(
            porw_solution_distance(&solution, &challenge_a),
            porw_solution_distance(&other, &challenge_a),
        );
    }
}
