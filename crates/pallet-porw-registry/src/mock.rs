use crate as pallet_porw_registry;
use crate::{AttestationVerifier, Id32};
use frame_support::traits::{ConstU64, ConstU128, VariantCount};
use frame_support::{derive_impl, parameter_types};
use parity_scale_codec::{Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::BuildStorage;

type Balance = u128;
type Block = frame_system::mocking::MockBlock<Test>;

pub const MEASUREMENT: Id32 = [0xAA; 32];
pub const BOND: Balance = 100;

#[derive(
    Encode, Decode, MaxEncodedLen, TypeInfo, Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd,
)]
pub enum MockHoldReason {
    PorwBond,
}

impl VariantCount for MockHoldReason {
    const VARIANT_COUNT: u32 = 1;
}

/// Stub verifier: evidence must equal the device id; measurement is fixed.
pub struct StubAttestation;

impl AttestationVerifier for StubAttestation {
    fn verify(device_id: &Id32, evidence: &[u8]) -> Option<Id32> {
        (evidence == device_id).then_some(MEASUREMENT)
    }
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
}

impl pallet_porw_registry::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type Balance = Balance;
    type Currency = Balances;
    type HoldReason = HoldReason;
    type Attestation = StubAttestation;
    type BondAmount = BondAmount;
    type ActivationDelay = ConstU64<10>;
}

pub fn new_test_ext() -> sp_io::TestExternalities {
    let mut storage = frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap();
    pallet_balances::GenesisConfig::<Test> {
        balances: vec![(1, 1000), (2, 1000)],
    }
    .assimilate_storage(&mut storage)
    .unwrap();
    let mut ext: sp_io::TestExternalities = storage.into();
    ext.execute_with(|| System::set_block_number(1));
    ext
}
