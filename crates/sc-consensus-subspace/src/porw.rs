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
//! P3 status: both sides of the authorship path are implemented as
//! self-contained, tested logic sharing one validation and digest carriage —
//! authorship ([`claim_porw_slot`] selects the best qualifying solution and
//! builds the [`PorwPreDigest`]; [`porw_pre_digest_logs`] / [`porw_seal_digest`]
//! emit the header logs) and import ([`verify_porw_block`] extracts, derives
//! the slot challenge and validates; [`verify_porw_seal`] checks the device
//! seal over the pre-hash). What remains is node-service integration: driving
//! [`claim_porw_slot`] from the live slot loop with candidate solutions
//! streamed from the attested PoRW agent (the `porw-agent` component), which
//! does not exist yet.

use sp_api::ProvideRuntimeApi;
use sp_consensus_slots::Slot;
use sp_consensus_subspace::digests::{
    CompatiblePorwDigestItem, PorwPreDigest, extract_porw_pre_digest,
};
use sp_consensus_subspace::{PorwApi, scale_solution_range};
use sp_core::ed25519;
use sp_runtime::DigestItem;
use sp_runtime::traits::{Block as BlockT, Header as HeaderT};
use subspace_core_primitives::pot::PotOutput;
use subspace_core_primitives::solutions::SolutionRange;
use subspace_proof_of_residency::{PorwSolution, derive_slot_seed, ticket_chunk};

/// Derive a slot's 32-byte global challenge from its proof of time, matching
/// the farming path (`PotOutput -> global randomness -> global challenge`).
pub fn global_challenge_for_slot(proof_of_time: PotOutput, slot: Slot) -> [u8; 32] {
    *proof_of_time
        .derive_global_randomness()
        .derive_global_challenge(slot.into())
}

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
    /// The block header has no (or a duplicate) PoRW pre-digest.
    MissingPreDigest,
}

/// Bidirectional distance between the derived ticket chunk and the global
/// challenge, in [`SolutionRange`] units — same semantics as the farming
/// path's audit-chunk distance.
pub fn porw_solution_distance(
    solution: &PorwSolution,
    global_challenge: &[u8; 32],
) -> SolutionRange {
    let slot_seed = derive_slot_seed(global_challenge, &solution.device_id);
    let chunk = ticket_chunk(
        &solution.model_id,
        &solution.partials_root,
        slot_seed,
        solution.chunk_index,
    );
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
        .porw_solution_tickets(parent_hash, solution.clone(), *global_challenge)
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

/// Block-import entry point for PoRW blocks: extract the PoRW pre-digest from
/// a block header, derive the slot's global challenge from its proof of time,
/// and run the full [`verify_porw_solution`] against the parent state.
///
/// This is the verification half of the P3 authorship path. Producing the
/// pre-digest in `slot_worker` and sealing the block are the remaining P3
/// pieces; both sides share this validation and the digest carriage.
pub fn verify_porw_block<Block, Client, RewardAddress>(
    client: &Client,
    parent_hash: Block::Hash,
    header: &Block::Header,
    solution_range: SolutionRange,
    voter_weight: u128,
    max_voter_weight: u128,
) -> Result<(PorwPreDigest<RewardAddress>, SolutionRange), PorwSolutionError>
where
    Block: BlockT,
    Client: ProvideRuntimeApi<Block>,
    Client::Api: PorwApi<Block>,
    RewardAddress: parity_scale_codec::Decode,
{
    let pre_digest: PorwPreDigest<RewardAddress> =
        extract_porw_pre_digest(header).map_err(|_| PorwSolutionError::MissingPreDigest)?;
    let global_challenge = global_challenge_for_slot(pre_digest.proof_of_time(), pre_digest.slot());
    let distance = verify_porw_solution(
        client,
        parent_hash,
        pre_digest.solution(),
        &global_challenge,
        solution_range,
        voter_weight,
        max_voter_weight,
    )?;
    Ok((pre_digest, distance))
}

// ---------------------------------------------------------------------------
// Authorship side (the produce half, symmetric to verify_porw_block)
// ---------------------------------------------------------------------------

/// Pick the best (lowest-distance = highest-quality) solution from candidates
/// that already passed [`verify_porw_solution`]. Mirrors the farming path's
/// best-solution selection.
pub fn select_best_solution(
    candidates: impl IntoIterator<Item = (PorwSolution, SolutionRange)>,
) -> Option<(PorwSolution, SolutionRange)> {
    candidates.into_iter().min_by_key(|(_, distance)| *distance)
}

/// Build the PoRW pre-digest an author writes into its block for a slot.
pub fn build_porw_pre_digest<RewardAddress>(
    slot: Slot,
    reward_address: RewardAddress,
    solution: PorwSolution,
    proof_of_time: PotOutput,
) -> PorwPreDigest<RewardAddress> {
    PorwPreDigest::V0 {
        slot,
        reward_address,
        solution,
        proof_of_time,
    }
}

/// The pre-digest log(s) an author appends to the block header (counterpart of
/// the farming `pre_digest_data`).
pub fn porw_pre_digest_logs<RewardAddress: parity_scale_codec::Encode>(
    pre_digest: &PorwPreDigest<RewardAddress>,
) -> Vec<DigestItem> {
    vec![DigestItem::porw_pre_digest(pre_digest)]
}

/// The seal digest an author appends after signing the block pre-hash with the
/// device node key (counterpart of the farming reward seal).
pub fn porw_seal_digest(signature: [u8; 64]) -> DigestItem {
    DigestItem::porw_seal(signature)
}

/// Claim a slot for PoRW: derive the slot challenge, verify each candidate
/// solution against the parent state, and build the pre-digest from the best
/// qualifying one. Returns `None` if no candidate qualifies. This is the
/// authorship counterpart of [`verify_porw_block`]; the node service supplies
/// candidates from the attested agent and does the actual sealing.
#[allow(clippy::too_many_arguments)]
pub fn claim_porw_slot<Block, Client, RewardAddress>(
    client: &Client,
    parent_hash: Block::Hash,
    slot: Slot,
    reward_address: RewardAddress,
    proof_of_time: PotOutput,
    solution_range: SolutionRange,
    voter_weight: u128,
    max_voter_weight: u128,
    candidates: impl IntoIterator<Item = PorwSolution>,
) -> Option<PorwPreDigest<RewardAddress>>
where
    Block: BlockT,
    Client: ProvideRuntimeApi<Block>,
    Client::Api: PorwApi<Block>,
{
    let global_challenge = global_challenge_for_slot(proof_of_time, slot);
    let verified = candidates.into_iter().filter_map(|solution| {
        let distance = verify_porw_solution(
            client,
            parent_hash,
            &solution,
            &global_challenge,
            solution_range,
            voter_weight,
            max_voter_weight,
        )
        .ok()?;
        Some((solution, distance))
    });
    let (solution, _distance) = select_best_solution(verified)?;
    Some(build_porw_pre_digest(
        slot,
        reward_address,
        solution,
        proof_of_time,
    ))
}

/// Verify a PoRW block seal: the device node key signed the block pre-hash.
/// `pubkey` is the registered node key of the pre-digest's `device_id`.
pub fn verify_porw_seal(pre_hash: &[u8], pubkey: &[u8; 32], signature: &[u8; 64]) -> bool {
    use sp_core::Pair;
    <ed25519::Pair as Pair>::verify(
        &ed25519::Signature::from_raw(*signature),
        pre_hash,
        &ed25519::Public::from_raw(*pubkey),
    )
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
            signature: [0u8; 64],
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

    fn sol(device: u8, chunk: u64) -> PorwSolution {
        PorwSolution {
            device_id: [device; 32],
            model_id: [5; 32],
            sketch: 1,
            partials_root: [7; 32],
            coverage_bytes: 1 << 30,
            m_t_millis: 1000,
            chunk_index: chunk,
            signature: [0u8; 64],
        }
    }

    #[test]
    fn select_best_picks_lowest_distance() {
        assert_eq!(select_best_solution(std::iter::empty()), None);
        let best = select_best_solution([
            (sol(1, 0), 300),
            (sol(2, 1), 100), // lowest distance = best quality
            (sol(3, 2), 200),
        ])
        .unwrap();
        assert_eq!(best.0.device_id, [2; 32]);
        assert_eq!(best.1, 100);
    }

    #[test]
    fn pre_digest_logs_round_trip_through_the_header() {
        use sp_consensus_subspace::digests::extract_porw_pre_digest;
        use sp_runtime::testing::Header as TestHeader;
        use sp_runtime::traits::Header as _;

        let pre_digest =
            build_porw_pre_digest(Slot::from(7), 42u64, sol(1, 3), PotOutput::default());
        let mut header = TestHeader::new_from_number(0);
        for log in porw_pre_digest_logs(&pre_digest) {
            header.digest_mut().push(log);
        }
        let extracted: PorwPreDigest<u64> = extract_porw_pre_digest(&header).unwrap();
        assert_eq!(extracted, pre_digest);
    }

    #[test]
    fn seal_signs_and_verifies_over_the_pre_hash() {
        use sp_core::Pair;

        let pair = ed25519::Pair::from_seed(&[11u8; 32]);
        let pubkey = pair.public().0;
        let pre_hash = [0xABu8; 32];
        let sig = pair.sign(&pre_hash).0;

        // Seal digest carries the signature and round-trips.
        let seal = porw_seal_digest(sig);
        assert_eq!(seal.as_porw_seal(), Some(sig));

        // Correct pubkey + pre-hash verifies; a tampered pre-hash does not.
        assert!(verify_porw_seal(&pre_hash, &pubkey, &sig));
        let mut bad_hash = pre_hash;
        bad_hash[0] ^= 1;
        assert!(!verify_porw_seal(&bad_hash, &pubkey, &sig));
        // A different device key does not verify.
        let other = ed25519::Pair::from_seed(&[22u8; 32]).public().0;
        assert!(!verify_porw_seal(&pre_hash, &other, &sig));
    }
}
