//! Minimal mock runtime for the devnet: frame-system + balances +
//! pallet-porw-registry wired to the real `PorwAttestation` verifier.

use frame_support::traits::{ConstU64, ConstU128, VariantCount};
use frame_support::{derive_impl, parameter_types};
use parity_scale_codec::{Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::BuildStorage;

pub type Balance = u128;
pub type Block = frame_system::mocking::MockBlock<Test>;

pub const BOND: Balance = 100;
/// Bytes of audited traffic per lottery ticket (matches the runtime constant).
pub const TICKET_UNIT: u64 = 1 << 30;

#[derive(
    Encode, Decode, MaxEncodedLen, TypeInfo, Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd,
)]
pub enum MockHoldReason {
    PorwBond,
}

impl VariantCount for MockHoldReason {
    const VARIANT_COUNT: u32 = 1;
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

/// Activation delay in blocks (kept small for the test).
pub const ACTIVATION_DELAY: u64 = 10;

impl pallet_porw_registry::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type Balance = Balance;
    type Currency = Balances;
    type HoldReason = HoldReason;
    type DemandEmaSmoothing = frame_support::traits::ConstU32<8>;
    type FeePerWeightUnit = frame_support::traits::ConstU128<100>;
    type MaxModelWeight = frame_support::traits::ConstU32<1_000_000>;
    type Attestation = pallet_porw_registry::PorwAttestation;
    type BondAmount = BondAmount;
    type ActivationDelay = ConstU64<ACTIVATION_DELAY>;
}

pub fn new_test_ext() -> sp_io::TestExternalities {
    let mut storage = frame_system::GenesisConfig::<Test>::default()
        .build_storage()
        .unwrap();
    pallet_balances::GenesisConfig::<Test> {
        balances: vec![(1, 1_000_000)],
    }
    .assimilate_storage(&mut storage)
    .unwrap();
    let mut ext: sp_io::TestExternalities = storage.into();
    ext.execute_with(|| System::set_block_number(1));
    ext
}
