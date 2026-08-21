use crate as pallet_porw_registry;
use crate::{Id32, PorwAttestation};
use frame_support::traits::{ConstU32, ConstU64, ConstU128, VariantCount};
use frame_support::{derive_impl, parameter_types};
use parity_scale_codec::{Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::BuildStorage;

type Balance = u128;
type Block = frame_system::mocking::MockBlock<Test>;

pub const MEASUREMENT: Id32 = [0xAA; 32];
pub const BOND: Balance = 100;

/// A fixed ed25519 device key for tests. `pubkey()` / `sign(payload)` let
/// tests produce solutions the registry will accept as device-signed.
pub fn device_keypair() -> sp_core::ed25519::Pair {
    use sp_core::Pair;
    sp_core::ed25519::Pair::from_seed(&[7u8; 32])
}

pub fn device_pubkey() -> Id32 {
    use sp_core::Pair;
    device_keypair().public().0
}

pub fn sign_solution(
    solution: &subspace_proof_of_residency::PorwSolution,
    global_challenge: &Id32,
) -> [u8; 64] {
    use sp_core::Pair;
    device_keypair()
        .sign(&solution.signing_payload(global_challenge))
        .0
}

#[derive(
    Encode, Decode, MaxEncodedLen, TypeInfo, Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd,
)]
pub enum MockHoldReason {
    PorwBond,
}

impl VariantCount for MockHoldReason {
    const VARIANT_COUNT: u32 = 1;
}

/// Test attestation vendor root (the governance-trusted key on a testnet).
pub fn attestation_root() -> sp_core::ed25519::Pair {
    use sp_core::Pair;
    sp_core::ed25519::Pair::from_seed(&[0x55; 32])
}

pub fn root_pubkey() -> Id32 {
    use sp_core::Pair;
    attestation_root().public().0
}

/// Build valid attestation evidence for `device_id` binding `node_pubkey` and
/// declaring `MEASUREMENT`, signed by the test root's device-identity chain.
pub fn build_evidence(device_id: Id32, node_pubkey: Id32) -> Vec<u8> {
    use parity_scale_codec::Encode;
    use porw_attestation::{AttestationReport, DeviceCert, Evidence};
    use sp_core::Pair;

    let root = attestation_root();
    let device_identity = sp_core::ed25519::Pair::from_seed(&[0x66; 32]);

    let mut cert = DeviceCert {
        device_id,
        device_pubkey: device_identity.public().0,
        vendor_sig: [0; 64],
    };
    cert.vendor_sig = root.sign(&cert.body()).0;

    let mut report = AttestationReport {
        device_id,
        measurement: MEASUREMENT,
        node_pubkey,
        report_sig: [0; 64],
    };
    report.report_sig = device_identity.sign(&report.body()).0;

    Evidence { cert, report }.encode()
}

frame_support::construct_runtime!(
    pub struct Test {
        System: frame_system,
        Balances: pallet_balances,
        PorwRegistry: pallet_porw_registry,
    }
);

#[derive_impl(frame_system::config_preludes::TestDefaultConfig)]
impl frame_system::Config for Test {
    type Block = Block;
    type AccountData = pallet_balances::AccountData<Balance>;
}

#[derive_impl(pallet_balances::config_preludes::TestDefaultConfig as pallet_balances::DefaultConfig)]
impl pallet_balances::Config for Test {
    type Balance = Balance;
    type ExistentialDeposit = ConstU128<1>;
    type AccountStore = System;
    type RuntimeHoldReason = MockHoldReason;
    type DustRemoval = ();
}

parameter_types! {
    pub const HoldReason: MockHoldReason = MockHoldReason::PorwBond;
    pub const BondAmount: Balance = BOND;
    pub const FeePerWeightUnit: Balance = 10;
    pub const MaxModelWeight: u32 = 1_000_000;
}

impl pallet_porw_registry::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type Balance = Balance;
    type Currency = Balances;
    type HoldReason = HoldReason;
    type DemandEmaSmoothing = ConstU32<4>;
    type FeePerWeightUnit = FeePerWeightUnit;
    type MaxModelWeight = MaxModelWeight;
    type Attestation = PorwAttestation;
    type BondAmount = BondAmount;
    type ActivationDelay = ConstU64<10>;
    // One block per epoch keeps the tokenomics tests legible: advancing the
    // block number by one between settlements crosses exactly one epoch.
    type EpochLength = ConstU64<1>;
}

pub fn new_test_ext() -> sp_io::TestExternalities {
    let mut storage = frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap();
    pallet_balances::GenesisConfig::<Test> {
        balances: vec![(1, 10_000_000), (2, 10_000_000)],
    }
    .assimilate_storage(&mut storage)
    .unwrap();
    let mut ext: sp_io::TestExternalities = storage.into();
    ext.execute_with(|| System::set_block_number(1));
    ext
}
