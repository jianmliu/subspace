use super::*;
use crate::cpu::CpuSketchBackend;
use sp_core::Pair;
use subspace_proof_of_residency::{
    TILE_BYTES, derive_slot_seed, merkle_root, partials_leaf, sketch_tile,
};

const N_TILES: u64 = 8;
/// One ticket per tile of audited traffic (test scale), matching the devnet.
const TICKET_UNIT: u64 = TILE_BYTES as u64;

fn model_bytes() -> Vec<u8> {
    (0..(N_TILES as usize * TILE_BYTES) as u64)
        .map(|i| ((i.wrapping_mul(2654435761) >> 7) & 0xFF) as u8)
        .collect()
}

fn active_agent() -> (PorwAgent<CpuSketchBackend>, [u8; 32]) {
    let backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let model_id = backend.model_root();
    let device_id = [0xD1; 32];
    let mut agent = PorwAgent::new([0x11; 32], device_id, model_id, TICKET_UNIT, backend);
    assert_eq!(agent.state(), AgentState::Unregistered);
    agent.on_registered();
    assert_eq!(agent.state(), AgentState::Registered);
    agent.on_activated();
    assert_eq!(agent.state(), AgentState::Active);
    (agent, model_id)
}

fn ctx(coverage: Vec<u64>) -> SlotContext {
    SlotContext {
        global_challenge: [0x9E; 32],
        coverage,
        m_t_millis: 1000,
    }
}

#[test]
fn inactive_agent_refuses_to_author() {
    let backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let model_id = backend.model_root();
    let agent = PorwAgent::new([1; 32], [2; 32], model_id, TICKET_UNIT, backend);
    assert!(matches!(
        agent.author_slot(&ctx(vec![0, 1])),
        Err(AgentError::NotActive(AgentState::Unregistered))
    ));
}

#[test]
fn coverage_out_of_range_is_rejected() {
    let (agent, _) = active_agent();
    assert!(matches!(
        agent.author_slot(&ctx(vec![0, N_TILES])), // N_TILES is out of range
        Err(AgentError::CoverageOutOfRange { tile, tile_count })
            if tile == N_TILES && tile_count == N_TILES
    ));
}

#[test]
fn authored_solution_is_well_formed_and_device_signed() {
    let (agent, model_id) = active_agent();
    let coverage = vec![0u64, 2, 5, 7];
    let context = ctx(coverage.clone());
    let (solution, distance) = agent.author_slot(&context).unwrap();

    // Identity fields.
    assert_eq!(solution.device_id, agent.device_id());
    assert_eq!(solution.model_id, model_id);
    assert_eq!(
        solution.coverage_bytes,
        (coverage.len() * TILE_BYTES) as u64
    );
    assert_eq!(solution.m_t_millis, 1000);

    // partials_root and folded sketch match an independent recomputation.
    let seed = derive_slot_seed(&context.global_challenge, &solution.device_id);
    let tiles = model_bytes();
    let mut folded = 0u32;
    let leaves: Vec<_> = coverage
        .iter()
        .map(|&i| {
            let tile: &[u8; TILE_BYTES] = tiles
                [i as usize * TILE_BYTES..(i as usize + 1) * TILE_BYTES]
                .try_into()
                .unwrap();
            let s = sketch_tile(seed, i, tile);
            folded = folded.wrapping_add(s);
            partials_leaf(i, s)
        })
        .collect();
    assert_eq!(solution.partials_root, merkle_root(&leaves));
    assert_eq!(solution.sketch, folded);

    // The device node key signed the solution over the slot challenge.
    let ok = sp_io::crypto::ed25519_verify(
        &sp_core::ed25519::Signature::from_raw(solution.signature),
        &solution.signing_payload(&context.global_challenge),
        &sp_core::ed25519::Public::from_raw(agent.node_pubkey()),
    );
    assert!(ok, "solution must be signed by the device node key");

    // Distance is the ring distance of the chosen ticket; sanity: < u64::MAX.
    assert!(distance < u64::MAX);
}

#[test]
fn zero_ticket_slot_is_refused_not_faked() {
    // Coverage earning zero tickets (here: no tiles ⇒ zero audited bytes) must
    // surface as `NoTickets`, never a chunk-0 solution the chain would reject.
    let (agent, _) = active_agent();
    assert!(matches!(
        agent.author_slot(&ctx(vec![])),
        Err(AgentError::NoTickets)
    ));
}

#[test]
fn different_challenges_yield_different_solutions() {
    let (agent, _) = active_agent();
    let (a, _) = agent.author_slot(&ctx(vec![0, 1, 2])).unwrap();
    let b_ctx = SlotContext {
        global_challenge: [0x11; 32],
        coverage: vec![0, 1, 2],
        m_t_millis: 1000,
    };
    let (b, _) = agent.author_slot(&b_ctx).unwrap();
    // Fresh challenge ⇒ fresh slot seed ⇒ different sketches and root.
    assert_ne!(a.partials_root, b.partials_root);
    assert_ne!(a.sketch, b.sketch);
}

#[test]
fn audit_duties_cover_every_peer_deterministically() {
    let backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let model_id = backend.model_root();
    let beacon = subspace_proof_of_residency::audit_beacon(3, &[0x77; 32]);
    let me = [0xD1u8; 32];
    let peer = [0xD2u8; 32];
    let replicas = [me, peer];

    // Two replicas, fan-out k=1: each is the other's only possible auditor,
    // so my duties are exactly the peer, never myself.
    let duties = audit_duties(&beacon, &model_id, &me, &replicas, 1, 4, N_TILES);
    assert_eq!(duties.len(), 1);
    assert_eq!(duties[0].target_device, peer);
    assert_eq!(duties[0].tiles.len(), 4);
    assert!(duties[0].tiles.iter().all(|&t| t < N_TILES));
    // Same beacon ⇒ same schedule (what makes it a schedule at all).
    assert_eq!(
        duties,
        audit_duties(&beacon, &model_id, &me, &replicas, 1, 4, N_TILES)
    );
}

#[test]
fn cross_check_clears_honest_commitments_and_convicts_tampered_ones() {
    use subspace_proof_of_residency::{FraudVerdict, verify_tile_fraud_proof};

    // Target and auditor are two replicas of the same model.
    let target_backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let auditor_backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let model_id = target_backend.model_root();
    let target_device = [0xD2u8; 32];
    let challenge = [0x9Eu8; 32];
    let slot_seed = derive_slot_seed(&challenge, &target_device);

    // The target commits a sparse (MoE-style) coverage set, with tile 5's
    // value tampered — it does not really hold those bytes.
    let coverage: [u64; 3] = [1, 5, 7];
    let mut s_tiles: Vec<u32> = target_backend.sketch_coverage(slot_seed, &coverage);
    s_tiles[1] ^= 0xBAD;
    let partial_leaves: Vec<_> = coverage
        .iter()
        .zip(&s_tiles)
        .map(|(&i, &s)| partials_leaf(i, s))
        .collect();
    let solution = subspace_proof_of_residency::PorwSolution {
        device_id: target_device,
        model_id,
        sketch: s_tiles.iter().fold(0u32, |a, s| a.wrapping_add(*s)),
        partials_root: merkle_root(&partial_leaves),
        coverage_bytes: (coverage.len() * TILE_BYTES) as u64,
        m_t_millis: 1000,
        chunk_index: 0,
        signature: [0u8; 64],
    };
    let opening = |pos: usize| CommittedTile {
        tile_idx: coverage[pos],
        claimed_s_tile: s_tiles[pos],
        partials_index: pos as u64,
        partials_proof: subspace_proof_of_residency::merkle_proof(&partial_leaves, pos),
    };

    // Honest tile: consistent.
    assert_eq!(
        cross_check(&auditor_backend, &solution, &challenge, &opening(0)),
        CrossCheckOutcome::Consistent
    );

    // Tampered tile: the auditor produces a fraud proof that verifies as
    // Fraud with the exact on-chain verifier — ready to submit.
    match cross_check(&auditor_backend, &solution, &challenge, &opening(1)) {
        CrossCheckOutcome::Fraud(proof) => {
            assert_eq!(
                verify_tile_fraud_proof(&solution, &challenge, &model_id, &proof),
                FraudVerdict::Fraud
            );
        }
        other => panic!("expected Fraud, got {other:?}"),
    }

    // A mangled opening proves nothing either way.
    let mut bad = opening(0);
    bad.partials_proof[0][0] ^= 1;
    assert_eq!(
        cross_check(&auditor_backend, &solution, &challenge, &bad),
        CrossCheckOutcome::Unverifiable
    );
}

#[test]
fn testkit_evidence_verifies_against_registered_root() {
    // The evidence testkit builds must verify with the same crate the runtime
    // uses, binding this agent's device id and node key.
    let (agent, _) = active_agent();
    let root = testkit::test_vendor_root(0x55);
    let measurement = [0xAA; 32];
    let evidence_bytes = testkit::build_evidence(
        &root,
        0x66,
        agent.device_id(),
        agent.node_pubkey(),
        measurement,
    );
    let evidence = porw_attestation::Evidence::decode(&evidence_bytes).unwrap();
    assert_eq!(
        porw_attestation::verify_evidence(
            &[root.public().0],
            &agent.device_id(),
            &agent.node_pubkey(),
            &evidence,
        ),
        Ok(measurement)
    );
}
