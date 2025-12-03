#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(rust_2018_idioms)]

use frame_support::pallet_prelude::*;
use frame_support::traits::fungible::{Inspect, InspectHold, MutateHold};
use frame_support::traits::tokens::Precision;
use frame_system::pallet_prelude::*;
use sp_runtime::traits::{AtLeast32BitUnsigned, Zero};
use subspace_runtime_primitives::VotingStakeProvider;

pub use pallet::*;

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

#[frame_support::pallet]
pub mod pallet {
    use super::*;

    #[pallet::pallet]
    pub struct Pallet<T>(_);

    #[pallet::config]
    pub trait Config: frame_system::Config {
        type RuntimeEvent: From<Event<Self>> + IsType<<Self as frame_system::Config>::RuntimeEvent>;

        type Balance: Parameter + AtLeast32BitUnsigned + Default + Copy + MaxEncodedLen;

        type Currency: Inspect<Self::AccountId, Balance = Self::Balance>
            + InspectHold<Self::AccountId, Balance = Self::Balance>
            + MutateHold<Self::AccountId, Balance = Self::Balance>;

        /// Hold reason used when staking for voting weight.
        type HoldReason: Get<<Self::Currency as InspectHold<Self::AccountId>>::Reason>;

        /// Minimum stake (unless unstaking to zero).
        #[pallet::constant]
        type MinStake: Get<Self::Balance>;

        /// Maximum stake allowed per account.
        #[pallet::constant]
        type MaxStake: Get<Self::Balance>;
    }

    #[pallet::storage]
    #[pallet::getter(fn stake_of)]
    pub type VotingStake<T: Config> =
        StorageMap<_, Twox64Concat, T::AccountId, T::Balance, ValueQuery>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        StakeUpdated { who: T::AccountId, amount: T::Balance },
    }

    #[pallet::error]
    pub enum Error<T> {
        StakeTooLow,
        StakeTooHigh,
        HoldFailed,
        ReleaseFailed,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Set the voting stake for the caller. Moves funds into/out of a hold.
        #[pallet::call_index(0)]
        #[pallet::weight(Weight::zero())]
        pub fn set_voting_stake(origin: OriginFor<T>, amount: T::Balance) -> DispatchResult {
            let who = ensure_signed(origin)?;

            if !amount.is_zero() {
                ensure!(amount >= T::MinStake::get(), Error::<T>::StakeTooLow);
                ensure!(amount <= T::MaxStake::get(), Error::<T>::StakeTooHigh);
            }

            let current = VotingStake::<T>::get(&who);
            let hold_reason = T::HoldReason::get();

            match amount.cmp(&current) {
                core::cmp::Ordering::Greater => {
                    let delta = amount - current;
                    T::Currency::hold(&hold_reason, &who, delta).map_err(|_| Error::<T>::HoldFailed)?;
                }
                core::cmp::Ordering::Less => {
                    let delta = current - amount;
                    T::Currency::release(&hold_reason, &who, delta, Precision::Exact)
                        .map_err(|_| Error::<T>::ReleaseFailed)?;
                }
                core::cmp::Ordering::Equal => return Ok(()),
            }

            if amount.is_zero() {
                VotingStake::<T>::remove(&who);
            } else {
                VotingStake::<T>::insert(&who, amount);
            }

            Self::deposit_event(Event::StakeUpdated { who, amount });
            Ok(())
        }
    }
}

impl<T: Config> VotingStakeProvider<T::AccountId, T::Balance> for pallet::Pallet<T> {
    fn voting_stake(account: &T::AccountId) -> T::Balance {
        pallet::VotingStake::<T>::get(account)
    }
}
