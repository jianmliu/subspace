use crate::mock::*;
use crate::{Devices, Error, Id32, Models, ReplicaCount, SolutionRejection};
use frame_support::traits::fungible::InspectHold;
use frame_support::{assert_noop, assert_ok};
use subspace_proof_of_residency::{
    Hash32, PorwSolution, TILE_BYTES, TileFraudProof, derive_slot_seed, merkle_proof, merkle_root,
    partials_leaf, sketch_tile, weights_leaf,
};

const DEVICE: Id32 = [1; 32];
const CHALLENGE: Id32 = [9; 32];
const N_TILES: usize = 4;

type Registry = crate::Pallet<Test>;

/// Cross into the next epoch (EpochLength = 1 in the mock) and settle it.
/// `settle_epoch` is idempotent per epoch, so a fresh block is required for
/// each fold — this mirrors how the `on_initialize` hook settles in production.
fn advance_and_settle() {
    let now = System::block_number();
    System::set_block_number(now + 1);
    assert_ok!(Registry::settle_epoch(RuntimeOrigin::signed(1)));
}

fn model_tiles() -> Vec<[u8; TILE_BYTES]> {
    (0..(N_TILES * TILE_BYTES) as u64)
        .map(|i| ((i.wrapping_mul(2654435761) >> 7) & 0xFF) as u8)
        .collect::<Vec<u8>>()
        .chunks_exact(TILE_BYTES)
        .map(|c| c.try_into().unwrap())
        .collect()
}

fn model_root(tiles: &[[u8; TILE_BYTES]]) -> (Id32, Vec<Hash32>) {
    let leaves: Vec<Hash32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| weights_leaf(i as u64, t))
        .collect();
    (merkle_root(&leaves), leaves)
}

/// Register measurement + model + device (owner=1) and announce the model.
fn setup_registered_device() -> (Id32, Vec<[u8; TILE_BYTES]>, Vec<Hash32>) {
    let tiles = model_tiles();
    let (model_id, weight_leaves) = model_root(&tiles);
    assert_ok!(Registry::register_measurement(
        RuntimeOrigin::root(),
        MEASUREMENT
    ));
    assert_ok!(Registry::add_trusted_root(
        RuntimeOrigin::root(),
        root_pubkey()
    ));
    assert_ok!(Registry::register_model(
        RuntimeOrigin::root(),
        model_id,
        (N_TILES * TILE_BYTES) as u64,
        2,
        1000,
    ));
    assert_ok!(Registry::register_device(
        RuntimeOrigin::signed(1),
        DEVICE,
        device_pubkey(),
        1 << 30, // 1 GiB/slot envelope
        build_evidence(DEVICE, device_pubkey()),
    ));
    assert_ok!(Registry::announce_model(
        RuntimeOrigin::signed(1),
        DEVICE,
        model_id
    ));
    (model_id, tiles, weight_leaves)
}

fn build_solution(
    model_id: Id32,
    tiles: &[[u8; TILE_BYTES]],
    tamper_tile: Option<usize>,
) -> (PorwSolution, Vec<u32>, Vec<Hash32>) {
    let slot_seed = derive_slot_seed(&CHALLENGE, &DEVICE);
    let mut s_tiles: Vec<u32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| sketch_tile(slot_seed, i as u64, t))
        .collect();
    if let Some(i) = tamper_tile {
        s_tiles[i] ^= 0xBAD;
    }
    let partial_leaves: Vec<Hash32> = s_tiles
        .iter()
        .enumerate()
        .map(|(i, s)| partials_leaf(i as u64, *s))
        .collect();
    let mut solution = PorwSolution {
        device_id: DEVICE,
        model_id,
        sketch: s_tiles.iter().fold(0u32, |a, s| a.wrapping_add(*s)),
        partials_root: merkle_root(&partial_leaves),
        coverage_bytes: (N_TILES * TILE_BYTES) as u64,
        m_t_millis: 1000,
        chunk_index: 0,
        signature: [0u8; 64],
    };
    solution.signature = sign_solution(&solution, &CHALLENGE);
    (solution, s_tiles, partial_leaves)
}

#[test]
fn register_device_holds_bond_and_activates_after_delay() {
    new_test_ext().execute_with(|| {
        let (model_id, tiles, _) = setup_registered_device();
        assert_eq!(Balances::balance_on_hold(&HoldReason::get(), &1), BOND);
        assert_eq!(ReplicaCount::<Test>::get(model_id), 1);

        let (solution, ..) = build_solution(model_id, &tiles, None);
        // Before the activation delay: inactive.
        assert_eq!(
            Registry::check_solution(&solution),
            Err(SolutionRejection::DeviceInactive)
        );
        System::set_block_number(11);
        assert_ok!(Registry::check_solution(&solution));

        // Envelope: claiming more swept bytes than the device can move fails.
        let mut greedy = solution.clone();
        greedy.m_t_millis = 100_000_000;
        assert_eq!(
            Registry::check_solution(&greedy),
            Err(SolutionRejection::EnvelopeExceeded)
        );

        // Revoking the measurement kills the fast path immediately.
        assert_ok!(Registry::revoke_measurement(
            RuntimeOrigin::root(),
            MEASUREMENT
        ));
        assert_eq!(
            Registry::check_solution(&solution),
            Err(SolutionRejection::MeasurementRevoked)
        );
    });
}

#[test]
fn registration_requires_valid_attestation_and_whitelisted_measurement() {
    new_test_ext().execute_with(|| {
        let pk = device_pubkey();
        let ev = build_evidence(DEVICE, pk);

        // No trusted root yet: even valid evidence fails attestation.
        assert_noop!(
            Registry::register_device(RuntimeOrigin::signed(1), DEVICE, pk, 1, ev.clone()),
            Error::<Test>::AttestationInvalid
        );
        assert_ok!(Registry::add_trusted_root(
            RuntimeOrigin::root(),
            root_pubkey()
        ));

        // Root trusted, but the attested measurement is not whitelisted.
        assert_noop!(
            Registry::register_device(RuntimeOrigin::signed(1), DEVICE, pk, 1, ev.clone()),
            Error::<Test>::UnknownMeasurement
        );
        assert_ok!(Registry::register_measurement(
            RuntimeOrigin::root(),
            MEASUREMENT
        ));

        // Malformed evidence bytes: attestation fails.
        assert_noop!(
            Registry::register_device(RuntimeOrigin::signed(1), DEVICE, pk, 1, vec![0xFF; 4]),
            Error::<Test>::AttestationInvalid
        );
        // Evidence bound to a different node key than the one being
        // registered: rejected (the substitution gap).
        let wrong_key = [0x99; 32];
        assert_noop!(
            Registry::register_device(
                RuntimeOrigin::signed(1),
                DEVICE,
                wrong_key,
                1,
                build_evidence(DEVICE, pk)
            ),
            Error::<Test>::AttestationInvalid
        );

        // Correct evidence registers; the device id cannot be re-registered.
        assert_ok!(Registry::register_device(
            RuntimeOrigin::signed(1),
            DEVICE,
            pk,
            1,
            ev.clone()
        ));
        assert_noop!(
            Registry::register_device(RuntimeOrigin::signed(2), DEVICE, pk, 1, ev),
            Error::<Test>::DeviceExists
        );
    });
}

#[test]
fn deregistration_releases_bond() {
    new_test_ext().execute_with(|| {
        let (model_id, ..) = setup_registered_device();
        assert_noop!(
            Registry::deregister_device(RuntimeOrigin::signed(2), DEVICE),
            Error::<Test>::NotOwner
        );
        assert_ok!(Registry::deregister_device(
            RuntimeOrigin::signed(1),
            DEVICE
        ));
        assert_eq!(Balances::balance_on_hold(&HoldReason::get(), &1), 0);
        assert!(Devices::<Test>::get(DEVICE).is_none());
        assert_eq!(ReplicaCount::<Test>::get(model_id), 0);
    });
}

#[test]
fn fraud_proof_slashes_bond_to_reporter() {
    new_test_ext().execute_with(|| {
        let (model_id, tiles, weight_leaves) = setup_registered_device();
        // The accused commits a wrong per-tile value for tile 2.
        let (solution, s_tiles, partial_leaves) = build_solution(model_id, &tiles, Some(2));

        let proof = TileFraudProof {
            tile_idx: 2,
            partials_index: 2,
            claimed_s_tile: s_tiles[2],
            partials_proof: merkle_proof(&partial_leaves, 2),
            tile_bytes: tiles[2].to_vec(),
            weights_proof: merkle_proof(&weight_leaves, 2),
        };
        let reporter_before = Balances::free_balance(2);
        assert_ok!(Registry::report_fraud(
            RuntimeOrigin::signed(2),
            solution,
            CHALLENGE,
            proof
        ));
        // Bond moved to the reporter; device revoked; replicas decremented.
        assert_eq!(Balances::free_balance(2), reporter_before + BOND);
        assert_eq!(Balances::balance_on_hold(&HoldReason::get(), &1), 0);
        assert!(Devices::<Test>::get(DEVICE).is_none());
        assert_eq!(ReplicaCount::<Test>::get(model_id), 0);
    });
}

#[test]
fn honest_solution_cannot_be_slashed() {
    new_test_ext().execute_with(|| {
        let (model_id, tiles, weight_leaves) = setup_registered_device();
        let (solution, s_tiles, partial_leaves) = build_solution(model_id, &tiles, None);

        let honest = TileFraudProof {
            tile_idx: 1,
            partials_index: 1,
            claimed_s_tile: s_tiles[1],
            partials_proof: merkle_proof(&partial_leaves, 1),
            tile_bytes: tiles[1].to_vec(),
            weights_proof: merkle_proof(&weight_leaves, 1),
        };
        assert_noop!(
            Registry::report_fraud(
                RuntimeOrigin::signed(2),
                solution.clone(),
                CHALLENGE,
                honest
            ),
            Error::<Test>::NotFraud
        );

        // Malformed evidence (non-canonical tile bytes) is Invalid, not Fraud.
        let mut bogus_tile = tiles[1];
        bogus_tile[0] ^= 1;
        let bogus = TileFraudProof {
            tile_idx: 1,
            partials_index: 1,
            claimed_s_tile: s_tiles[1],
            partials_proof: merkle_proof(&partial_leaves, 1),
            tile_bytes: bogus_tile.to_vec(),
            weights_proof: merkle_proof(&weight_leaves, 1),
        };
        assert_noop!(
            Registry::report_fraud(RuntimeOrigin::signed(2), solution, CHALLENGE, bogus),
            Error::<Test>::FraudProofInvalid
        );
        // Bond untouched.
        assert_eq!(Balances::balance_on_hold(&HoldReason::get(), &1), BOND);
    });
}

#[test]
fn fabricated_unsigned_solution_cannot_slash() {
    new_test_ext().execute_with(|| {
        let (model_id, tiles, weight_leaves) = setup_registered_device();
        // Attacker fabricates a wrong solution for the victim device using
        // only public data (tiles + Merkle paths) and does NOT sign it with
        // the device key (they can't).
        let (mut solution, s_tiles, partial_leaves) = build_solution(model_id, &tiles, Some(2));
        solution.signature = [0u8; 64]; // no valid device signature

        let proof = TileFraudProof {
            tile_idx: 2,
            partials_index: 2,
            claimed_s_tile: s_tiles[2],
            partials_proof: merkle_proof(&partial_leaves, 2),
            tile_bytes: tiles[2].to_vec(),
            weights_proof: merkle_proof(&weight_leaves, 2),
        };
        assert_noop!(
            Registry::report_fraud(RuntimeOrigin::signed(2), solution, CHALLENGE, proof),
            Error::<Test>::BadSolutionSignature
        );
        // Victim's bond is safe and device still registered.
        assert_eq!(Balances::balance_on_hold(&HoldReason::get(), &1), BOND);
        assert!(Devices::<Test>::get(DEVICE).is_some());
    });
}

#[test]
fn withdraw_model_decrements_replicas_and_gates_solutions() {
    new_test_ext().execute_with(|| {
        let (model_id, tiles, _) = setup_registered_device();
        System::set_block_number(11);
        let (solution, ..) = build_solution(model_id, &tiles, None);
        assert_ok!(Registry::check_solution(&solution));
        assert_eq!(ReplicaCount::<Test>::get(model_id), 1);

        assert_ok!(Registry::withdraw_model(
            RuntimeOrigin::signed(1),
            DEVICE,
            model_id
        ));
        assert_eq!(ReplicaCount::<Test>::get(model_id), 0);
        assert_eq!(
            Registry::check_solution(&solution),
            Err(SolutionRejection::ModelNotAnnounced)
        );
        // Withdrawing again fails; device stays registered.
        assert_noop!(
            Registry::withdraw_model(RuntimeOrigin::signed(1), DEVICE, model_id),
            Error::<Test>::ModelNotAnnounced
        );
    });
}

// ---------------------------------------------------------------------------
// Tokenomics: demand-following model reward weight
// ---------------------------------------------------------------------------

fn tiny_model(id: Id32, floor: u32) {
    assert_ok!(Registry::register_model(
        RuntimeOrigin::root(),
        id,
        TILE_BYTES as u64,
        1,
        floor,
    ));
}

#[test]
fn model_starts_at_floor_weight() {
    new_test_ext().execute_with(|| {
        let id = [0x01; 32];
        tiny_model(id, 50);
        assert_eq!(Registry::model_reward_weight(&id), 50);
    });
}

#[test]
fn burning_fees_raises_weight_and_supply_drops() {
    new_test_ext().execute_with(|| {
        let id = [0x02; 32];
        tiny_model(id, 50);
        let supply_before = Balances::total_issuance();

        // Account 1 burns 800 units of inference fees for the model.
        // FeePerWeightUnit = 10, EMA smoothing N = 4.
        assert_ok!(Registry::record_inference_fee(
            RuntimeOrigin::signed(1),
            id,
            800
        ));
        // The fee is really burned (removed from supply).
        assert_eq!(Balances::total_issuance(), supply_before - 800);
        // Pending, not yet reflected in weight until settlement.
        assert_eq!(Registry::model_reward_weight(&id), 50);

        // Settle: ema = (0*3 + 800)/4 = 200 ; demand weight = 200/10 = 20.
        // max(floor 50, 20) = 50 — demand still below floor.
        advance_and_settle();
        assert_eq!(Models::<Test>::get(id).unwrap().demand_ema, 200);
        assert_eq!(Registry::model_reward_weight(&id), 50);

        // Sustained demand pushes the EMA up and weight above the floor.
        for _ in 0..6 {
            assert_ok!(Registry::record_inference_fee(
                RuntimeOrigin::signed(1),
                id,
                800
            ));
            advance_and_settle();
        }
        // EMA converges toward 800 ⇒ demand weight toward 80 > floor 50.
        assert!(Registry::model_reward_weight(&id) > 50);
        assert_eq!(
            Registry::model_reward_weight(&id),
            (Models::<Test>::get(id).unwrap().demand_ema / 10) as u32
        );
    });
}

#[test]
fn demand_decays_back_to_floor_without_fees() {
    new_test_ext().execute_with(|| {
        let id = [0x03; 32];
        tiny_model(id, 5);
        // Build up demand.
        for _ in 0..8 {
            assert_ok!(Registry::record_inference_fee(
                RuntimeOrigin::signed(1),
                id,
                400
            ));
            advance_and_settle();
        }
        let hot = Registry::model_reward_weight(&id);
        assert!(hot > 5);
        // Demand stops: EMA decays each epoch toward zero, weight toward floor.
        for _ in 0..20 {
            advance_and_settle();
        }
        assert_eq!(Registry::model_reward_weight(&id), 5); // back to floor
    });
}

#[test]
fn recording_fees_for_unknown_model_fails() {
    new_test_ext().execute_with(|| {
        assert_noop!(
            Registry::record_inference_fee(RuntimeOrigin::signed(1), [0xFF; 32], 10),
            Error::<Test>::UnknownModel
        );
    });
}

#[test]
fn weight_is_capped_at_max() {
    new_test_ext().execute_with(|| {
        let id = [0x04; 32];
        // Floor already above the cap is clamped at registration.
        tiny_model(id, u32::MAX);
        assert_eq!(Registry::model_reward_weight(&id), MaxModelWeight::get());
    });
}

#[test]
fn block_reward_escrows_and_releases_after_audit_window() {
    new_test_ext().execute_with(|| {
        setup_registered_device();
        let supply_before = Balances::total_issuance();
        let owner_before = Balances::free_balance(1);

        // Reward earned in epoch 1 (EpochLength = 1 block in the mock):
        // nothing is minted at escrow time.
        crate::Pallet::<Test>::note_block_reward(DEVICE, 500);
        assert_eq!(Balances::total_issuance(), supply_before);
        assert_eq!(Balances::free_balance(1), owner_before);
        assert!(crate::EscrowedRewards::<Test>::contains_key(1, DEVICE));

        // Epoch 2 settles: epoch 1's audit window (epoch 2) has just opened,
        // so the reward stays in escrow.
        advance_and_settle();
        assert_eq!(Balances::free_balance(1), owner_before);
        assert!(crate::EscrowedRewards::<Test>::contains_key(1, DEVICE));

        // Epoch 3 settles: epoch 2 — the audit window — passed with no
        // confirmed fraud, so epoch 1's reward is minted to the owner.
        advance_and_settle();
        assert_eq!(Balances::free_balance(1), owner_before + 500);
        assert_eq!(Balances::total_issuance(), supply_before + 500);
        assert!(!crate::EscrowedRewards::<Test>::contains_key(1, DEVICE));

        // Two rewards in one epoch accumulate into one bucket.
        crate::Pallet::<Test>::note_block_reward(DEVICE, 100);
        crate::Pallet::<Test>::note_block_reward(DEVICE, 200);
        advance_and_settle();
        advance_and_settle();
        assert_eq!(Balances::free_balance(1), owner_before + 500 + 300);
    });
}

#[test]
fn fraud_within_the_audit_window_forfeits_escrowed_rewards() {
    new_test_ext().execute_with(|| {
        let (model_id, tiles, weight_leaves) = setup_registered_device();
        let supply_before = Balances::total_issuance();
        let owner_before = Balances::free_balance(1);

        // The cheat earns a reward in the current epoch, then its fraudulent
        // commitment is caught within the audit window.
        crate::Pallet::<Test>::note_block_reward(DEVICE, 500);
        let (solution, s_tiles, partial_leaves) = build_solution(model_id, &tiles, Some(2));
        let proof = TileFraudProof {
            tile_idx: 2,
            claimed_s_tile: s_tiles[2],
            partials_index: 2,
            partials_proof: merkle_proof(&partial_leaves, 2),
            tile_bytes: tiles[2].to_vec(),
            weights_proof: merkle_proof(&weight_leaves, 2),
        };
        assert_ok!(Registry::report_fraud(
            RuntimeOrigin::signed(2),
            solution,
            CHALLENGE,
            proof
        ));

        // Escrow clawed back at report time; settling past the window mints
        // nothing — the forfeited reward never enters supply.
        assert!(!crate::EscrowedRewards::<Test>::contains_key(1, DEVICE));
        advance_and_settle();
        advance_and_settle();
        assert_eq!(Balances::free_balance(1), owner_before);
        // Supply unchanged except the bond transfer (hold → reporter's free),
        // which does not mint.
        assert_eq!(Balances::total_issuance(), supply_before);
    });
}

#[test]
fn deregistration_waits_out_the_escrow_window() {
    new_test_ext().execute_with(|| {
        setup_registered_device();
        crate::Pallet::<Test>::note_block_reward(DEVICE, 500);

        // Escrowed pay cannot be walked out from under a pending audit.
        assert_noop!(
            Registry::deregister_device(RuntimeOrigin::signed(1), DEVICE),
            Error::<Test>::EscrowPending
        );

        // Once the window passes and the reward is released, exit is free.
        advance_and_settle();
        advance_and_settle();
        assert_ok!(Registry::deregister_device(
            RuntimeOrigin::signed(1),
            DEVICE
        ));
    });
}
