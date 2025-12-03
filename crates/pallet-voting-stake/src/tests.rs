use crate::mock::{new_test_ext, RuntimeOrigin, Test, VotingStake};
use frame_support::assert_ok;
use frame_support::traits::fungible::InspectHold;
use pallet_balances::Pallet as Balances;
use frame_system::Pallet as System;

#[test]
fn stake_and_unstake_updates_hold() {
    new_test_ext().execute_with(|| {
        System::<Test>::inc_providers(&1);

        // stake up
        assert_ok!(VotingStake::set_voting_stake(RuntimeOrigin::signed(1), 10));
        assert_eq!(VotingStake::stake_of(1), 10);
        assert_eq!(
            Balances::<Test>::balance_on_hold(&crate::mock::HoldReason::get(), &1),
            10
        );

        // reduce stake
        assert_ok!(VotingStake::set_voting_stake(RuntimeOrigin::signed(1), 6));
        assert_eq!(VotingStake::stake_of(1), 6);
        assert_eq!(
            Balances::<Test>::balance_on_hold(&crate::mock::HoldReason::get(), &1),
            6
        );

        // unstake fully
        assert_ok!(VotingStake::set_voting_stake(RuntimeOrigin::signed(1), 0));
        assert_eq!(VotingStake::stake_of(1), 0);
        assert_eq!(
            Balances::<Test>::balance_on_hold(&crate::mock::HoldReason::get(), &1),
            0
        );
    });
}
