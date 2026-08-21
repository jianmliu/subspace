//! A real block-production loop, CPU- and TEE-free: the agent authors a
//! solution each slot, `claim_porw_slot` selects it and builds the pre-digest,
//! the block is sealed with the device key, and `verify_porw_block` imports it
//! — forming a growing chain of sealed, parent-linked blocks.
//!
//! The only stand-in is the runtime-API client: a `MockRuntime` implements
//! `PorwApi` by running the real fast-path checks (device registered + active,
//! device signature over the slot challenge, hardware envelope) directly,
//! rather than through Substrate storage. That is the accepted way to test
//! consensus-client code without standing up the full node; the on-chain fast
//! path itself is proven against the real pallet in `devnet.rs`.

mod mock;

use mock::Block;
use porw_agent::cpu::CpuSketchBackend;
use porw_agent::{PorwAgent, SlotContext};
use sc_consensus_subspace::porw::{
    claim_porw_slot, global_challenge_for_slot, porw_pre_digest_logs, porw_seal_digest,
    verify_porw_block, verify_porw_seal,
};
use sp_api::{ApiRef, ProvideRuntimeApi};
use sp_consensus_slots::Slot;
use sp_consensus_subspace::digests::CompatiblePorwDigestItem;
use sp_core::Pair;
use sp_runtime::traits::{Header as HeaderT, Zero};
use subspace_core_primitives::pot::PotOutput;
use subspace_proof_of_residency::{PorwSolution, TILE_BYTES, check_envelope, ticket_count};

type Header = <Block as sp_runtime::traits::Block>::Header;
type Hash = <Block as sp_runtime::traits::Block>::Hash;

const N_TILES: usize = 16;
const DEVICE_ID: [u8; 32] = [0xD1; 32];
const NODE_SEED: [u8; 32] = [0x11; 32];
const TICKET_UNIT: u64 = TILE_BYTES as u64; // one ticket per tile (test scale)

fn model_bytes() -> Vec<u8> {
    (0..(N_TILES * TILE_BYTES) as u64)
        .map(|i| ((i.wrapping_mul(2654435761) >> 7) & 0xFF) as u8)
        .collect()
}

// ---------------------------------------------------------------------------
// Mock runtime-API client: real fast-path checks, no Substrate storage.
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct MockRuntime {
    device_pubkey: [u8; 32],
    model_id: [u8; 32],
    bandwidth_bytes_per_slot: u64,
    active: bool,
}

impl ProvideRuntimeApi<Block> for MockRuntime {
    type Api = MockRuntime;
    fn runtime_api(&self) -> ApiRef<'_, Self::Api> {
        (*self).clone().into()
    }
}

sp_api::mock_impl_runtime_apis! {
    impl sp_consensus_subspace::PorwApi<Block> for MockRuntime {
        fn porw_solution_tickets(
            &self,
            solution: PorwSolution,
            global_challenge: [u8; 32],
        ) -> Option<u64> {
            // Faithful fast path: device known + active, model matches, device
            // signature valid over this slot's challenge, within the envelope.
            if !self.active
                || solution.device_id != DEVICE_ID
                || solution.model_id != self.model_id
            {
                return None;
            }
            let sig_ok = sp_io::crypto::ed25519_verify(
                &sp_core::ed25519::Signature::from_raw(solution.signature),
                &solution.signing_payload(&global_challenge),
                &sp_core::ed25519::Public::from_raw(self.device_pubkey),
            );
            if !sig_ok {
                return None;
            }
            if !check_envelope(
                solution.coverage_bytes,
                solution.m_t_millis,
                self.bandwidth_bytes_per_slot,
            ) {
                return None;
            }
            Some(ticket_count(solution.coverage_bytes, solution.m_t_millis, TICKET_UNIT))
        }
    }
}

// ---------------------------------------------------------------------------
// The loop
// ---------------------------------------------------------------------------

fn pot_for_slot(slot: u64) -> PotOutput {
    let mut bytes = [0u8; 16];
    bytes[..8].copy_from_slice(&slot.to_le_bytes());
    PotOutput::from(bytes)
}

/// Seal a header authored with a pre-digest: hash the header as-is (the
/// pre-hash), sign it with the device key, and append the seal digest.
fn seal_header(mut header: Header, node_key: &sp_core::ed25519::Pair) -> Header {
    let pre_hash = header.hash();
    let signature = node_key.sign(pre_hash.as_ref()).0;
    header.digest_mut().push(porw_seal_digest(signature));
    header
}

/// Import-side seal check: pop the seal, recompute the pre-hash, verify.
fn check_seal(header: &Header, pubkey: &[u8; 32]) -> bool {
    let mut logs = header.digest().logs().to_vec();
    let seal = match logs.pop().and_then(|l| l.as_porw_seal()) {
        Some(s) => s,
        None => return false,
    };
    let mut pre = header.clone();
    pre.digest_mut().logs.pop();
    verify_porw_seal(pre.hash().as_ref(), pubkey, &seal)
}

#[test]
fn produces_and_imports_a_chain_of_sealed_porw_blocks() {
    let backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let model_id = backend.model_root();
    let node_key = sp_core::ed25519::Pair::from_seed(&NODE_SEED);
    let node_pubkey = node_key.public().0;

    let mut agent = PorwAgent::new(NODE_SEED, DEVICE_ID, model_id, TICKET_UNIT, backend);
    agent.on_registered();
    agent.on_activated();

    let client = MockRuntime {
        device_pubkey: node_pubkey,
        model_id,
        bandwidth_bytes_per_slot: 1 << 40,
        active: true,
    };

    // Full solution range so every authored solution clears the lottery.
    let solution_range = u64::MAX;
    let mut parent_hash: Hash = Default::default();
    let mut parent_number: u64 = 0;
    let mut produced = 0u32;

    for slot in 1..=6u64 {
        let proof_of_time = pot_for_slot(slot);
        let challenge = global_challenge_for_slot(proof_of_time, Slot::from(slot));

        // Author: the agent produces a candidate solution for the slot.
        let (solution, _distance) = agent
            .author_slot(&SlotContext {
                global_challenge: challenge,
                coverage: vec![0, 3, 4, 9, 15],
                m_t_millis: 2000,
            })
            .unwrap();

        // Claim: the node selects the best qualifying solution and builds the
        // pre-digest via the real client-side authorship path.
        let pre_digest = claim_porw_slot::<Block, _, u64>(
            &client,
            parent_hash,
            Slot::from(slot),
            /* reward_address */ 1u64,
            proof_of_time,
            solution_range,
            /* voter_weight */ 0,
            /* max_voter_weight */ 0,
            vec![solution],
        )
        .expect("a qualifying solution should be claimed");

        // Build the block header carrying the pre-digest, then seal it.
        let mut digest = sp_runtime::Digest::default();
        for log in porw_pre_digest_logs(&pre_digest) {
            digest.push(log);
        }
        let header = Header::new(
            parent_number + 1,
            Default::default(),
            Default::default(),
            parent_hash,
            digest,
        );
        let header = seal_header(header, &node_key);

        // Import: verify the block's pre-digest against parent state, and its
        // seal against the device key. This is what a node does on import.
        let (imported_pre_digest, _distance) =
            verify_porw_block::<Block, _, u64>(&client, parent_hash, &header, solution_range, 0, 0)
                .expect("the produced block must import");
        assert_eq!(imported_pre_digest.slot(), Slot::from(slot));
        assert!(check_seal(&header, &node_pubkey), "seal must verify");

        // Chain linkage: this block builds on the previous one.
        assert_eq!(*header.number(), parent_number + 1);
        assert_eq!(*header.parent_hash(), parent_hash);
        parent_hash = header.hash();
        parent_number = *header.number();
        produced += 1;
    }

    assert_eq!(produced, 6, "six sealed, linked PoRW blocks were produced");
    assert!(!parent_number.is_zero());
}

#[test]
fn a_block_with_a_foreign_seal_is_rejected() {
    let backend = CpuSketchBackend::new(model_bytes()).unwrap();
    let model_id = backend.model_root();
    let node_key = sp_core::ed25519::Pair::from_seed(&NODE_SEED);
    let node_pubkey = node_key.public().0;
    let mut agent = PorwAgent::new(NODE_SEED, DEVICE_ID, model_id, TICKET_UNIT, backend);
    agent.on_registered();
    agent.on_activated();
    let client = MockRuntime {
        device_pubkey: node_pubkey,
        model_id,
        bandwidth_bytes_per_slot: 1 << 40,
        active: true,
    };

    let slot = 1u64;
    let proof_of_time = pot_for_slot(slot);
    let challenge = global_challenge_for_slot(proof_of_time, Slot::from(slot));
    let (solution, _) = agent
        .author_slot(&SlotContext {
            global_challenge: challenge,
            coverage: vec![0, 1, 2],
            m_t_millis: 1000,
        })
        .unwrap();
    let pre_digest = claim_porw_slot::<Block, _, u64>(
        &client,
        Default::default(),
        Slot::from(slot),
        1u64,
        proof_of_time,
        u64::MAX,
        0,
        0,
        vec![solution],
    )
    .unwrap();
    let mut digest = sp_runtime::Digest::default();
    for log in porw_pre_digest_logs(&pre_digest) {
        digest.push(log);
    }
    let header = Header::new(
        1,
        Default::default(),
        Default::default(),
        Default::default(),
        digest,
    );
    // Seal with a different key: the pre-digest still imports (it is device-
    // signed), but the block seal does not match the device node key.
    let foreign = sp_core::ed25519::Pair::from_seed(&[0x99; 32]);
    let header = seal_header(header, &foreign);
    assert!(
        !check_seal(&header, &node_pubkey),
        "a foreign seal must be rejected"
    );
}
