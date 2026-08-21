//! End-to-end CPU devnet: a PoRW agent produces a device-signed solution, the
//! device is registered on chain with real attestation evidence, and the
//! on-chain fast path plus the consensus distance check accept the agent's
//! solution — the whole authorship→acceptance loop, GPU- and TEE-free.

mod mock;

use mock::*;
use pallet_porw_registry::SolutionRejection;
use porw_agent::cpu::CpuSketchBackend;
use porw_agent::{AgentState, PorwAgent, SlotContext, testkit};
use sp_core::Pair;
use subspace_proof_of_residency::TILE_BYTES;

type Registry = pallet_porw_registry::Pallet<Test>;

const N_TILES: usize = 16;
const MEASUREMENT: [u8; 32] = [0xAA; 32];
const DEVICE_ID: [u8; 32] = [0xD1; 32];
const GLOBAL_CHALLENGE: [u8; 32] = [0x9E; 32];
const VENDOR_ROOT_SEED: u8 = 0x55;

fn model_bytes() -> Vec<u8> {
    (0..(N_TILES * TILE_BYTES) as u64)
        .map(|i| ((i.wrapping_mul(2654435761) >> 7) & 0xFF) as u8)
        .collect()
}

fn build_agent() -> PorwAgent<CpuSketchBackend> {
    let backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let model_id = backend.model_root();
    PorwAgent::new([0x11; 32], DEVICE_ID, model_id, TICKET_UNIT, backend)
}

/// Governance + owner steps that bring a device to `Active` on chain, using
/// attestation evidence the agent's testkit synthesizes for this node key.
fn register_and_announce(agent: &PorwAgent<CpuSketchBackend>) {
    let root = testkit::test_vendor_root(VENDOR_ROOT_SEED);
    // Governance: whitelist the measurement and trust the (test) vendor root.
    assert_ok(Registry::register_measurement(
        RuntimeOrigin::root(),
        MEASUREMENT,
    ));
    assert_ok(Registry::add_trusted_root(
        RuntimeOrigin::root(),
        root.public().0,
    ));
    assert_ok(Registry::register_model(
        RuntimeOrigin::root(),
        agent.model_id(),
        (N_TILES * TILE_BYTES) as u64,
        2,
        1000,
    ));
    // Owner registers the attested device and announces the resident model.
    let evidence = testkit::build_evidence(
        &root,
        0x66,
        agent.device_id(),
        agent.node_pubkey(),
        MEASUREMENT,
    );
    assert_ok(Registry::register_device(
        RuntimeOrigin::signed(1),
        agent.device_id(),
        agent.node_pubkey(),
        1 << 40, // generous envelope
        evidence,
    ));
    assert_ok(Registry::announce_model(
        RuntimeOrigin::signed(1),
        agent.device_id(),
        agent.model_id(),
    ));
}

fn assert_ok(r: sp_runtime::DispatchResult) {
    r.expect("dispatch should succeed");
}

#[test]
fn agent_solution_is_accepted_by_the_chain_fast_path() {
    new_test_ext().execute_with(|| {
        let mut agent = build_agent();
        register_and_announce(&agent);
        agent.on_registered();

        // Author a slot with a coverage set (a MoE-ish subset of tiles).
        let ctx = SlotContext {
            global_challenge: GLOBAL_CHALLENGE,
            coverage: vec![0, 3, 4, 9, 15],
            m_t_millis: 2000,
        };

        // Before activation, the chain rejects the solution as inactive even
        // though it is well-formed and correctly signed.
        agent.on_activated(); // agent-side lifecycle (mirrors chain)
        assert_eq!(agent.state(), AgentState::Active);
        let (solution, distance) = agent.author_slot(&ctx).unwrap();
        assert_eq!(
            Registry::check_solution_signed(&solution, &GLOBAL_CHALLENGE),
            Err(SolutionRejection::DeviceInactive)
        );

        // Advance past the on-chain activation delay.
        System::set_block_number(1 + ACTIVATION_DELAY);

        // The chain fast path now accepts the agent's signed solution:
        // registered + activated device, valid device signature over the slot
        // challenge, announced model, within the hardware envelope.
        assert_eq!(
            Registry::check_solution_signed(&solution, &GLOBAL_CHALLENGE),
            Ok(())
        );

        // Consensus math: with a full-range target every solution qualifies,
        // and the chosen ticket is within the tickets the coverage authorizes.
        let solution_range = u64::MAX;
        assert!(distance <= solution_range / 2);
        // The agent only authors when the coverage earns at least one ticket,
        // and never picks a chunk beyond that count (the fast path enforces the
        // same bound on import).
        let tickets = subspace_proof_of_residency::ticket_count(
            solution.coverage_bytes,
            solution.m_t_millis,
            TICKET_UNIT,
        );
        assert!(tickets >= 1);
        assert!(solution.chunk_index < tickets);
    });
}

#[test]
fn tampered_solution_is_rejected_by_the_chain() {
    new_test_ext().execute_with(|| {
        let mut agent = build_agent();
        register_and_announce(&agent);
        agent.on_registered();
        agent.on_activated();
        System::set_block_number(1 + ACTIVATION_DELAY);

        let ctx = SlotContext {
            global_challenge: GLOBAL_CHALLENGE,
            coverage: vec![0, 1, 2],
            m_t_millis: 1000,
        };
        let (mut solution, _) = agent.author_slot(&ctx).unwrap();

        // Tamper with the sketch after signing: the device signature no longer
        // covers it, so the chain fast path rejects the bad signature.
        solution.sketch ^= 0xDEAD;
        assert_eq!(
            Registry::check_solution_signed(&solution, &GLOBAL_CHALLENGE),
            Err(SolutionRejection::BadSignature)
        );

        // Over-claiming the service multiplier busts the hardware envelope.
        let (mut greedy, _) = agent.author_slot(&ctx).unwrap();
        greedy.m_t_millis = u64::MAX / 2;
        // Re-sign so the signature is valid and only the envelope check bites.
        let key = sp_core::ed25519::Pair::from_seed(&[0x11; 32]);
        greedy.signature = key.sign(&greedy.signing_payload(&GLOBAL_CHALLENGE)).0;
        assert_eq!(
            Registry::check_solution_signed(&greedy, &GLOBAL_CHALLENGE),
            Err(SolutionRejection::EnvelopeExceeded)
        );
    });
}

#[test]
fn cross_audit_catches_a_lying_replica_end_to_end() {
    new_test_ext().execute_with(|| {
        // The registered target device commits a coverage set with one tile's
        // sketch value tampered (it does not really hold those bytes), signed
        // by its real node key so the commitment is chain-valid on its face.
        let agent = build_agent();
        register_and_announce(&agent);
        System::set_block_number(1 + ACTIVATION_DELAY);
        let model_id = agent.model_id();
        let slot_seed =
            subspace_proof_of_residency::derive_slot_seed(&GLOBAL_CHALLENGE, &DEVICE_ID);

        let target_backend = CpuSketchBackend::new(model_bytes()).unwrap();
        let coverage: [u64; 4] = [0, 3, 9, 15];
        let mut s_tiles =
            porw_agent::SketchBackend::sketch_coverage(&target_backend, slot_seed, &coverage);
        s_tiles[2] ^= 0xBAD; // tile 9 is a lie
        let partial_leaves: Vec<_> = coverage
            .iter()
            .zip(&s_tiles)
            .map(|(&i, &s)| subspace_proof_of_residency::partials_leaf(i, s))
            .collect();
        let mut solution = subspace_proof_of_residency::PorwSolution {
            device_id: DEVICE_ID,
            model_id,
            sketch: s_tiles.iter().fold(0u32, |a, s| a.wrapping_add(*s)),
            partials_root: subspace_proof_of_residency::merkle_root(&partial_leaves),
            coverage_bytes: (coverage.len() * TILE_BYTES) as u64,
            m_t_millis: 1000,
            chunk_index: 0,
            signature: [0u8; 64],
        };
        let device_key = sp_core::ed25519::Pair::from_seed(&[0x11; 32]);
        solution.signature = device_key
            .sign(&solution.signing_payload(&GLOBAL_CHALLENGE))
            .0;
        // The fast path accepts it: the lie is invisible without the bytes.
        assert_ok(
            Registry::check_solution_signed(&solution, &GLOBAL_CHALLENGE).map_err(|_| {
                sp_runtime::DispatchError::Other("fast path should accept the signed commitment")
            }),
        );

        // The target earns a block reward this epoch — held in escrow.
        Registry::note_block_reward(DEVICE_ID, 500);
        let owner_before = Balances::free_balance(1);

        // An auditor replica (holding the same canonical bytes) draws its
        // beacon duties, obtains the target's opening for a sampled tile,
        // and cross-checks it against local bytes.
        let auditor_backend = CpuSketchBackend::new(model_bytes()).unwrap();
        let auditor_device = [0xD9u8; 32];
        let beacon = subspace_proof_of_residency::audit_beacon(1, &[0x42; 32]);
        let duties = porw_agent::audit_duties(
            &beacon,
            &model_id,
            &auditor_device,
            &[DEVICE_ID, auditor_device],
            1,
            N_TILES,
            N_TILES as u64,
        );
        assert_eq!(duties[0].target_device, DEVICE_ID);
        // Tile 9 is in the sample (t = N_TILES samples everything committed).
        assert!(duties[0].tiles.contains(&9));
        let opening = porw_agent::CommittedTile {
            tile_idx: 9,
            claimed_s_tile: s_tiles[2],
            partials_index: 2,
            partials_proof: subspace_proof_of_residency::merkle_proof(&partial_leaves, 2),
        };
        let proof =
            match porw_agent::cross_check(&auditor_backend, &solution, &GLOBAL_CHALLENGE, &opening)
            {
                porw_agent::CrossCheckOutcome::Fraud(proof) => *proof,
                other => panic!("auditor must convict the lying tile, got {other:?}"),
            };

        // The auditor submits the proof as reporter (account 2): bond moves
        // to the reporter, the device is revoked, and the escrowed reward is
        // forfeited — never minted.
        let reporter_before = Balances::free_balance(2);
        assert_ok(Registry::report_fraud(
            RuntimeOrigin::signed(2),
            solution,
            GLOBAL_CHALLENGE,
            proof,
        ));
        assert_eq!(Balances::free_balance(2), reporter_before + BOND);
        assert!(pallet_porw_registry::Devices::<Test>::get(DEVICE_ID).is_none());
        assert!(
            !pallet_porw_registry::EscrowedRewards::<Test>::contains_key(
                System::block_number() / 10, // EpochLength = 10 in the mock
                DEVICE_ID
            )
        );
        assert_eq!(Balances::free_balance(1), owner_before);
    });
}
