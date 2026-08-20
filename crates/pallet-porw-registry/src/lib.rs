//! PoRW registry pallet: attested devices, registered models, measurement
//! whitelist, fidelity bond and tile-granular fraud-proof slashing.
//!
//! Consensus weight in PoRW comes from residency (capacity), never from the
//! bond — the bond exists only to deter the protocol's two trusted
//! assertions (agent-reported service multiplier; sketch integrity under a
//! compromised TEE). See `docs/proof-of-resident-weights.md` §6.4.
//!
//! Verification split:
//! - [`Pallet::check_solution`] is the per-block fast path (registry,
//!   activation delay, model announcement, hardware envelope);
//! - [`Call::report_fraud`] is the slow path: anyone may submit a
//!   [`TileFraudProof`] showing that a per-tile sketch value committed under
//!   a solution's `partials_root` disagrees with the canonical weight bytes
//!   committed under the model's `R_W` root. A confirmed fraud slashes the
//!   device's bond to the reporter and revokes the device.

#![cfg_attr(not(feature = "std"), no_std)]
#![forbid(unsafe_code)]
#![warn(rust_2018_idioms)]

extern crate alloc;

use frame_support::pallet_prelude::*;
use frame_support::traits::fungible::{Inspect, InspectHold, MutateHold};
use frame_support::traits::tokens::Precision;
use frame_system::pallet_prelude::*;
use sp_runtime::traits::{AtLeast32BitUnsigned, Saturating};
use subspace_proof_of_residency::{
    FraudVerdict, PorwSolution, TileFraudProof, check_envelope, verify_tile_fraud_proof,
};

pub use pallet::*;

#[cfg(test)]
mod mock;
#[cfg(test)]
mod tests;

/// 32-byte identifier (device id, model root `R_W`, measurement digest).
pub type Id32 = [u8; 32];

/// Pluggable attestation verifier. Production wires NVIDIA CC + TDX/SNP
/// evidence verification (native or optimistic); tests use a stub.
pub trait AttestationVerifier {
    /// Verify `evidence` for `device_id`; on success return the measured
    /// agent-code digest (checked against the on-chain whitelist).
    fn verify(device_id: &Id32, evidence: &[u8]) -> Option<Id32>;
}

/// TESTNET-ONLY attestation stub: treats 32-byte evidence as the claimed
/// measurement digest itself, verifying nothing. Real deployments implement
/// [`AttestationVerifier`] over NVIDIA CC / TDX / SNP evidence (P4).
pub struct InsecureEvidenceAsMeasurement;

impl AttestationVerifier for InsecureEvidenceAsMeasurement {
    fn verify(_device_id: &Id32, evidence: &[u8]) -> Option<Id32> {
        evidence.try_into().ok()
    }
}

/// Registered model metadata.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, MaxEncodedLen, TypeInfo)]
pub struct ModelInfo {
    /// Total weight bytes (multiple of the canonical tile size).
    pub size_bytes: u64,
    /// Replicas below which the model relies on storage-track arbitration
    /// only (service parameter, not a safety one).
    pub min_replicas: u32,
    /// Relative reward weight (demand-following in production; static here).
    pub reward_weight: u32,
}

/// Registered device metadata.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, MaxEncodedLen, TypeInfo)]
pub struct DeviceInfo<AccountId, Balance, BlockNumber> {
    /// Account that bonded and controls this device.
    pub owner: AccountId,
    /// Ed25519 public key of the device node key (generated in the CVM).
    /// Solutions must be signed by the matching secret key — this is what
    /// binds a solution to the device in the fraud path and in P3 block
    /// production.
    pub pubkey: Id32,
    /// Whitelisted agent-code measurement this device attested to.
    pub measurement: Id32,
    /// Registered hardware envelope: bytes the device can physically move
    /// through HBM in one slot. Caps claimed work (bounded-inflation bound).
    pub bandwidth_bytes_per_slot: u64,
    /// Bond actually held at registration. Stored per-device so a later
    /// change to the `BondAmount` constant cannot strand an existing bond.
    pub bond: Balance,
    /// Registration block; the device joins the lottery only after the
    /// activation delay (the instant-rental deterrent).
    pub registered_at: BlockNumber,
}

/// Why a solution failed the fast path.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SolutionRejection {
    UnknownDevice,
    DeviceInactive,
    MeasurementRevoked,
    UnknownModel,
    ModelNotAnnounced,
    EnvelopeExceeded,
    BadSignature,
}

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

        /// Hold reason for the fidelity bond.
        type HoldReason: Get<<Self::Currency as InspectHold<Self::AccountId>>::Reason>;

        /// Attestation evidence verifier.
        type Attestation: AttestationVerifier;

        /// Fidelity bond per device. Sized to the fraud opportunity (a
        /// bounded multiple of epoch revenue), never to capacity.
        #[pallet::constant]
        type BondAmount: Get<Self::Balance>;

        /// Blocks between registration and lottery eligibility.
        #[pallet::constant]
        type ActivationDelay: Get<BlockNumberFor<Self>>;
    }

    /// Whitelisted agent-code measurements (governance-managed).
    #[pallet::storage]
    pub type Measurements<T: Config> = StorageMap<_, Twox64Concat, Id32, (), OptionQuery>;

    /// Registered models by `R_W` root.
    #[pallet::storage]
    pub type Models<T: Config> = StorageMap<_, Twox64Concat, Id32, ModelInfo, OptionQuery>;

    /// Registered devices.
    #[pallet::storage]
    pub type Devices<T: Config> = StorageMap<
        _,
        Twox64Concat,
        Id32,
        DeviceInfo<T::AccountId, T::Balance, BlockNumberFor<T>>,
        OptionQuery,
    >;

    /// Which models a device has announced residency for.
    #[pallet::storage]
    pub type DeviceModels<T: Config> =
        StorageDoubleMap<_, Twox64Concat, Id32, Twox64Concat, Id32, (), OptionQuery>;

    /// Announced replica count per model (service-tier signal).
    #[pallet::storage]
    pub type ReplicaCount<T: Config> = StorageMap<_, Twox64Concat, Id32, u32, ValueQuery>;

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        MeasurementRegistered {
            measurement: Id32,
        },
        MeasurementRevoked {
            measurement: Id32,
        },
        ModelRegistered {
            model_id: Id32,
        },
        DeviceRegistered {
            device_id: Id32,
            owner: T::AccountId,
        },
        DeviceDeregistered {
            device_id: Id32,
        },
        ModelAnnounced {
            device_id: Id32,
            model_id: Id32,
        },
        ModelWithdrawn {
            device_id: Id32,
            model_id: Id32,
        },
        FraudConfirmed {
            device_id: Id32,
            reporter: T::AccountId,
            tile_idx: u64,
        },
    }

    #[pallet::error]
    pub enum Error<T> {
        /// Attestation evidence did not verify.
        AttestationInvalid,
        /// Attested measurement is not whitelisted.
        UnknownMeasurement,
        /// Device id already registered.
        DeviceExists,
        /// Device id not registered.
        UnknownDevice,
        /// Caller does not own the device.
        NotOwner,
        /// Model not registered.
        UnknownModel,
        /// Model size is not a multiple of the tile size.
        BadModelSize,
        /// Bond could not be held.
        BondFailed,
        /// Bond could not be released.
        ReleaseFailed,
        /// Fraud proof is malformed (Merkle paths / lengths do not verify).
        FraudProofInvalid,
        /// Fraud proof verified but shows agreement — no fraud.
        NotFraud,
        /// Solution's claimed model is not announced by the device.
        ModelNotAnnounced,
        /// Solution is not signed by the accused device's node key.
        BadSolutionSignature,
    }

    #[pallet::call]
    impl<T: Config> Pallet<T> {
        /// Whitelist an agent-code measurement. Governance only.
        #[pallet::call_index(0)]
        #[pallet::weight(Weight::zero())]
        pub fn register_measurement(origin: OriginFor<T>, measurement: Id32) -> DispatchResult {
            ensure_root(origin)?;
            Measurements::<T>::insert(measurement, ());
            Self::deposit_event(Event::MeasurementRegistered { measurement });
            Ok(())
        }

        /// Revoke a measurement (e.g. a compromised agent build). Devices
        /// attested to it fail the fast path immediately.
        #[pallet::call_index(1)]
        #[pallet::weight(Weight::zero())]
        pub fn revoke_measurement(origin: OriginFor<T>, measurement: Id32) -> DispatchResult {
            ensure_root(origin)?;
            Measurements::<T>::remove(measurement);
            Self::deposit_event(Event::MeasurementRevoked { measurement });
            Ok(())
        }

        /// Register a model by its `R_W` tile-Merkle root. Governance in
        /// this milestone; the stake-to-list admission market replaces this.
        #[pallet::call_index(2)]
        #[pallet::weight(Weight::zero())]
        pub fn register_model(
            origin: OriginFor<T>,
            model_id: Id32,
            size_bytes: u64,
            min_replicas: u32,
            reward_weight: u32,
        ) -> DispatchResult {
            ensure_root(origin)?;
            ensure!(
                size_bytes > 0
                    && size_bytes % (subspace_proof_of_residency::TILE_BYTES as u64) == 0,
                Error::<T>::BadModelSize
            );
            Models::<T>::insert(
                model_id,
                ModelInfo {
                    size_bytes,
                    min_replicas,
                    reward_weight,
                },
            );
            Self::deposit_event(Event::ModelRegistered { model_id });
            Ok(())
        }

        /// Register an attested device: verify evidence, check the measured
        /// agent build against the whitelist, hold the fidelity bond.
        #[pallet::call_index(3)]
        #[pallet::weight(Weight::zero())]
        pub fn register_device(
            origin: OriginFor<T>,
            device_id: Id32,
            pubkey: Id32,
            bandwidth_bytes_per_slot: u64,
            evidence: alloc::vec::Vec<u8>,
        ) -> DispatchResult {
            let owner = ensure_signed(origin)?;
            ensure!(
                !Devices::<T>::contains_key(device_id),
                Error::<T>::DeviceExists
            );
            let measurement = T::Attestation::verify(&device_id, &evidence)
                .ok_or(Error::<T>::AttestationInvalid)?;
            ensure!(
                Measurements::<T>::contains_key(measurement),
                Error::<T>::UnknownMeasurement
            );
            let bond = T::BondAmount::get();
            T::Currency::hold(&T::HoldReason::get(), &owner, bond)
                .map_err(|_| Error::<T>::BondFailed)?;
            Devices::<T>::insert(
                device_id,
                DeviceInfo {
                    owner: owner.clone(),
                    pubkey,
                    measurement,
                    bandwidth_bytes_per_slot,
                    bond,
                    registered_at: frame_system::Pallet::<T>::block_number(),
                },
            );
            Self::deposit_event(Event::DeviceRegistered { device_id, owner });
            Ok(())
        }

        /// Deregister an owned device and release its (stored) bond.
        #[pallet::call_index(4)]
        #[pallet::weight(Weight::zero())]
        pub fn deregister_device(origin: OriginFor<T>, device_id: Id32) -> DispatchResult {
            let who = ensure_signed(origin)?;
            let info = Devices::<T>::get(device_id).ok_or(Error::<T>::UnknownDevice)?;
            ensure!(info.owner == who, Error::<T>::NotOwner);
            T::Currency::release(
                &T::HoldReason::get(),
                &who,
                info.bond,
                Precision::BestEffort,
            )
            .map_err(|_| Error::<T>::ReleaseFailed)?;
            Self::remove_device(&device_id);
            Self::deposit_event(Event::DeviceDeregistered { device_id });
            Ok(())
        }

        /// Announce that a device holds a registered model resident.
        #[pallet::call_index(5)]
        #[pallet::weight(Weight::zero())]
        pub fn announce_model(
            origin: OriginFor<T>,
            device_id: Id32,
            model_id: Id32,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            let info = Devices::<T>::get(device_id).ok_or(Error::<T>::UnknownDevice)?;
            ensure!(info.owner == who, Error::<T>::NotOwner);
            ensure!(
                Models::<T>::contains_key(model_id),
                Error::<T>::UnknownModel
            );
            if !DeviceModels::<T>::contains_key(device_id, model_id) {
                DeviceModels::<T>::insert(device_id, model_id, ());
                ReplicaCount::<T>::mutate(model_id, |c| *c = c.saturating_add(1));
            }
            Self::deposit_event(Event::ModelAnnounced {
                device_id,
                model_id,
            });
            Ok(())
        }

        /// Report a tile-granular fraud proof against a committed solution.
        /// On confirmed fraud: the device's bond is transferred to the
        /// reporter and the device is revoked.
        #[pallet::call_index(6)]
        #[pallet::weight(Weight::zero())]
        pub fn report_fraud(
            origin: OriginFor<T>,
            solution: PorwSolution,
            global_challenge: Id32,
            proof: TileFraudProof,
        ) -> DispatchResult {
            let reporter = ensure_signed(origin)?;
            let device = Devices::<T>::get(solution.device_id).ok_or(Error::<T>::UnknownDevice)?;
            ensure!(
                Models::<T>::contains_key(solution.model_id),
                Error::<T>::UnknownModel
            );
            // Bind the solution to the accused device: it can only be
            // slashed for a solution its node key actually signed. Without
            // this, anyone could fabricate a wrong solution for any device
            // (tiles and Merkle paths are public) and steal its bond.
            ensure!(
                Self::verify_device_signature(&device.pubkey, &solution, &global_challenge),
                Error::<T>::BadSolutionSignature
            );
            let tile_idx = proof.tile_idx;
            match verify_tile_fraud_proof(&solution, &global_challenge, &solution.model_id, &proof)
            {
                FraudVerdict::Fraud => {
                    // Slash the device's stored bond to the reporter.
                    let _ = T::Currency::transfer_on_hold(
                        &T::HoldReason::get(),
                        &device.owner,
                        &reporter,
                        device.bond,
                        Precision::BestEffort,
                        frame_support::traits::tokens::Restriction::Free,
                        frame_support::traits::tokens::Fortitude::Force,
                    );
                    Self::remove_device(&solution.device_id);
                    Self::deposit_event(Event::FraudConfirmed {
                        device_id: solution.device_id,
                        reporter,
                        tile_idx,
                    });
                    Ok(())
                }
                FraudVerdict::NoFraud => Err(Error::<T>::NotFraud.into()),
                FraudVerdict::Invalid => Err(Error::<T>::FraudProofInvalid.into()),
            }
        }

        /// Withdraw a model announcement (device no longer holds it
        /// resident). Decrements the replica count; the model can no longer
        /// win the lottery from this device until re-announced.
        #[pallet::call_index(7)]
        #[pallet::weight(Weight::zero())]
        pub fn withdraw_model(
            origin: OriginFor<T>,
            device_id: Id32,
            model_id: Id32,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            let info = Devices::<T>::get(device_id).ok_or(Error::<T>::UnknownDevice)?;
            ensure!(info.owner == who, Error::<T>::NotOwner);
            ensure!(
                DeviceModels::<T>::take(device_id, model_id).is_some(),
                Error::<T>::ModelNotAnnounced
            );
            ReplicaCount::<T>::mutate(model_id, |c| *c = c.saturating_sub(1));
            Self::deposit_event(Event::ModelWithdrawn {
                device_id,
                model_id,
            });
            Ok(())
        }
    }

    impl<T: Config> Pallet<T> {
        /// Per-block fast-path validation of a PoRW solution (registry,
        /// activation delay, measurement, model announcement, envelope).
        /// Ticket-count / solution-range math composes on top of this in
        /// the client (`sc-consensus-subspace` integration).
        pub fn check_solution(solution: &PorwSolution) -> Result<(), SolutionRejection> {
            let device =
                Devices::<T>::get(solution.device_id).ok_or(SolutionRejection::UnknownDevice)?;
            let now = frame_system::Pallet::<T>::block_number();
            if now
                < device
                    .registered_at
                    .saturating_add(T::ActivationDelay::get())
            {
                return Err(SolutionRejection::DeviceInactive);
            }
            if !Measurements::<T>::contains_key(device.measurement) {
                return Err(SolutionRejection::MeasurementRevoked);
            }
            if !Models::<T>::contains_key(solution.model_id) {
                return Err(SolutionRejection::UnknownModel);
            }
            if !DeviceModels::<T>::contains_key(solution.device_id, solution.model_id) {
                return Err(SolutionRejection::ModelNotAnnounced);
            }
            if !check_envelope(
                solution.coverage_bytes,
                solution.m_t_millis,
                device.bandwidth_bytes_per_slot,
            ) {
                return Err(SolutionRejection::EnvelopeExceeded);
            }
            Ok(())
        }

        /// Full block-authorship validation: [`Self::check_solution`] plus a
        /// check that the solution is signed by the device's node key over
        /// this slot's `global_challenge`. This is what block import runs, so
        /// an unsigned or forged solution can never author a block.
        pub fn check_solution_signed(
            solution: &PorwSolution,
            global_challenge: &Id32,
        ) -> Result<(), SolutionRejection> {
            Self::check_solution(solution)?;
            let device =
                Devices::<T>::get(solution.device_id).ok_or(SolutionRejection::UnknownDevice)?;
            if !Self::verify_device_signature(&device.pubkey, solution, global_challenge) {
                return Err(SolutionRejection::BadSignature);
            }
            Ok(())
        }

        fn remove_device(device_id: &Id32) {
            for (model_id, ()) in DeviceModels::<T>::drain_prefix(device_id) {
                ReplicaCount::<T>::mutate(model_id, |c| *c = c.saturating_sub(1));
            }
            Devices::<T>::remove(device_id);
        }

        /// Verify the device node key's ed25519 signature over the solution's
        /// signing payload (all fields except the signature, plus the slot's
        /// global challenge).
        fn verify_device_signature(
            pubkey: &Id32,
            solution: &PorwSolution,
            global_challenge: &Id32,
        ) -> bool {
            let payload = solution.signing_payload(global_challenge);
            sp_io::crypto::ed25519_verify(
                &sp_core::ed25519::Signature::from_raw(solution.signature),
                &payload,
                &sp_core::ed25519::Public::from_raw(*pubkey),
            )
        }
    }
}
