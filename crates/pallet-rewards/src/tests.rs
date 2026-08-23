use crate::mock::Test;
use crate::{RemainingIssuance, RewardPoint, VoterSubsidyPoints};
use frame_support::BoundedVec;

type Pallet = crate::Pallet<Test>;

#[test]
fn avg_blockspace_updated_correctly() {
    // For `block_height` <= `avg_blockspace_usage_num_blocks` we just take average between currently used blockspace and old average
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 400, 100, 99), 250);
    assert_eq!(Pallet::update_avg_blockspace_usage(0, 400, 100, 99), 200);
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 0, 100, 99), 50);
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 400, 100, 100), 250);
    assert_eq!(Pallet::update_avg_blockspace_usage(0, 400, 100, 100), 200);
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 0, 100, 100), 50);

    // For `block_height` > `avg_blockspace_usage_num_blocks` we use the correct formula
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 400, 100, 101), 394);
    assert_eq!(Pallet::update_avg_blockspace_usage(0, 400, 100, 101), 392);
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 0, 100, 101), 1);

    // Special case check for no averaging blocks
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 400, 0, 100), 100);
    assert_eq!(Pallet::update_avg_blockspace_usage(0, 400, 0, 100), 0);
    assert_eq!(Pallet::update_avg_blockspace_usage(100, 0, 0, 100), 100);
}

#[test]
fn correct_block_vote_reward() {
    // No reward unless parameters are set
    assert_eq!(Pallet::block_reward(&[], 0, 0), 0);
    assert_eq!(Pallet::vote_reward(&[], 0), 0);

    let reward_start_block = 10;
    let mut points = vec![
        RewardPoint {
            block: 0_u64,
            subsidy: 100000000000000000_u128,
        },
        RewardPoint {
            block: 201600,
            subsidy: 99989921015995728,
        },
        RewardPoint {
            block: 79041600,
            subsidy: 92408728791312960,
        },
        RewardPoint {
            block: 779041600,
            subsidy: 45885578019877912,
        },
        RewardPoint {
            block: 2443104160,
            subsidy: 8687806947398648,
        },
    ];
    points.iter_mut().for_each(|point| {
        point.block += reward_start_block;
    });

    // No reward before initial subsidy start block
    assert_eq!(Pallet::block_reward(&points, reward_start_block - 1, 0), 0);
    assert_eq!(Pallet::vote_reward(&points, reward_start_block - 1), 0);
    // Rewards starts at initial subsidy start block
    assert_ne!(Pallet::block_reward(&points, reward_start_block, 0), 0);
    assert_ne!(Pallet::vote_reward(&points, reward_start_block), 0);
    // Blockspace usage (and storage fees as the result) mean lower block reward
    assert!(
        Pallet::block_reward(&points, reward_start_block, 1000)
            < Pallet::block_reward(&points, reward_start_block, 0)
    );

    // Reward at points should match expectations
    for point in &points {
        assert_eq!(
            Pallet::block_reward(&points, point.block, 0),
            point.subsidy,
            "Block {}",
            point.block
        );
        assert_eq!(
            Pallet::vote_reward(&points, point.block),
            point.subsidy,
            "Block {}",
            point.block
        );
        // Should decrease for subsequent blocks
        assert!(
            Pallet::block_reward(&points, point.block + 1, 0) <= point.subsidy,
            "Block {}",
            point.block
        );
        assert!(
            Pallet::vote_reward(&points, point.block + 1) <= point.subsidy,
            "Block {}",
            point.block
        );
    }

    let last_point = points.last().unwrap();
    assert_eq!(
        Pallet::block_reward(&points, last_point.block, 0),
        last_point.subsidy
    );
    assert_eq!(
        Pallet::vote_reward(&points, last_point.block),
        last_point.subsidy
    );

    assert_eq!(
        Pallet::block_reward(&points, last_point.block + 1, 0),
        last_point.subsidy
    );
    assert_eq!(
        Pallet::vote_reward(&points, last_point.block + 1),
        last_point.subsidy
    );
}

#[test]
fn vote_rewards_are_weighted() {
    use crate::mock::{MOCK_VOTERS, new_test_ext};
    use frame_support::traits::Hooks;
    use pallet_balances::Pallet as Balances;

    new_test_ext().execute_with(|| {
        // Two voters with different balances acting as weights
        MOCK_VOTERS.with(|v| {
            *v.borrow_mut() = vec![(1, 1), (2, 3)];
        });

        // Configure subsidy so vote reward is non-zero
        let points: BoundedVec<_, _> = vec![RewardPoint {
            block: 0,
            subsidy: 100,
        }]
        .try_into()
        .unwrap();
        VoterSubsidyPoints::<Test>::put(points);
        RemainingIssuance::<Test>::put(1_000);

        // Run finalization to issue rewards
        frame_system::Pallet::<Test>::set_block_number(1);
        Pallet::on_finalize(1);

        // Total vote reward per vote is 100 with 10% proposer tax; no block author so tax is added
        // back to voters. Total voter pool: 200, split 1:3.
        assert_eq!(Balances::<Test>::free_balance(&1), 50);
        assert_eq!(Balances::<Test>::free_balance(&2), 150);

        // Clean up for other tests
        MOCK_VOTERS.with(|v| v.borrow_mut().clear());
    });
}

#[test]
fn stake_scale_weights_split_rewards_without_saturation() {
    use crate::mock::{MOCK_VOTERS, new_test_ext};
    use frame_support::traits::Hooks;

    new_test_ext().execute_with(|| {
        // Stake-derived weights at realistic scale (~10^25, i.e. millions of
        // 10^18-precision tokens) against an AI3-scale reward pool (10^18).
        // The naive u128 `pool * weight` is ~10^43 and saturates, silently
        // skewing the split; the widened math must keep the exact 1:3 ratio.
        let w: u128 = 10u128.pow(25);
        MOCK_VOTERS.with(|v| {
            *v.borrow_mut() = vec![(1, w), (2, 3 * w)];
        });
        let points: BoundedVec<_, _> = vec![RewardPoint {
            block: 0,
            subsidy: 10u128.pow(18),
        }]
        .try_into()
        .unwrap();
        VoterSubsidyPoints::<Test>::put(points);
        RemainingIssuance::<Test>::put(10u128.pow(19));

        frame_system::Pallet::<Test>::set_block_number(1);
        Pallet::on_finalize(1);

        let v1 = pallet_balances::Pallet::<Test>::free_balance(&1);
        let v2 = pallet_balances::Pallet::<Test>::free_balance(&2);
        assert!(v1 > 0);
        assert_eq!(v2, 3 * v1, "weighted split must stay exactly 1:3");

        MOCK_VOTERS.with(|v| v.borrow_mut().clear());
    });
}
