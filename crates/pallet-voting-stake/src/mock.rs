use crate as pallet_voting_stake;
use frame_support::derive_impl;
use frame_support::parameter_types;
use frame_support::traits::VariantCount;
use frame_support::traits::ConstU128;
use parity_scale_codec::{Decode, Encode, MaxEncodedLen};
use scale_info::TypeInfo;
use sp_runtime::BuildStorage;

type Balance = u128;
type AccountId = u64;
type Block = frame_system::mocking::MockBlock<Test>;

#[derive(
    Encode, Decode, MaxEncodedLen, TypeInfo, Clone, Copy, Debug, PartialEq, Eq, Ord, PartialOrd,
)]
pub enum MockHoldReason {
    VotingStake,
}

impl VariantCount for MockHoldReason {
    const VARIANT_COUNT: u32 = 1;
}

frame_support::construct_runtime!(
    pub struct Test {
        System: frame_system,
        Balances: pallet_balances,
        VotingStake: pallet_voting_stake,
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
    pub const MinStake: Balance = 5;
    pub const MaxStake: Balance = 1000;
    pub const HoldReason: MockHoldReason = MockHoldReason::VotingStake;
}

impl pallet_voting_stake::Config for Test {
    type RuntimeEvent = RuntimeEvent;
    type Balance = Balance;
    type Currency = Balances;
    type HoldReason = HoldReason;
    type MinStake = MinStake;
    type MaxStake = MaxStake;
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
    storage.into()
}
