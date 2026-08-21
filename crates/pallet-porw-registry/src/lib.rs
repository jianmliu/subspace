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
use frame_support::traits::fungible::{Inspect, InspectHold, Mutate, MutateHold};
use frame_support::traits::tokens::Precision;
use frame_system::pallet_prelude::*;
use sp_runtime::SaturatedConversion;
use sp_runtime::traits::{AtLeast32BitUnsigned, Saturating, Zero};
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

/// Pluggable attestation verifier: parse `evidence` and verify its signature
/// chain against the governance-configured `trusted_roots`, binding
/// `device_id` and `node_pubkey`, returning the measured agent-code digest.
///
/// The roots are passed in (from on-chain [`TrustedRoots`] storage) so the
/// verifier stays stateless while the trust anchors remain governance state.
pub trait AttestationVerifier {
    /// Verify `evidence` and return the measured agent-code digest on success.
    fn verify(
        trusted_roots: &[Id32],
        device_id: &Id32,
        node_pubkey: &Id32,
        evidence: &[u8],
    ) -> Option<Id32>;
}

/// Production attestation verifier: decodes [`porw_attestation::Evidence`] and
/// verifies the vendor-root -> device-identity -> report signature chain.
/// The only testnet-specific part is which roots governance trusts.
pub struct PorwAttestation;

impl AttestationVerifier for PorwAttestation {
    fn verify(
        trusted_roots: &[Id32],
        device_id: &Id32,
        node_pubkey: &Id32,
        evidence: &[u8],
    ) -> Option<Id32> {
        let evidence = porw_attestation::Evidence::decode(evidence)?;
        porw_attestation::verify_evidence(trusted_roots, device_id, node_pubkey, &evidence).ok()
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
    /// Floor reward weight: the model keeps at least this weight regardless of
    /// demand (whitepaper §5.6 — stops a model with a demand lull from being
    /// evicted; paid for by the lister in the open-listing follow-on).
    pub floor_weight: u32,
    /// EMA of per-epoch burned inference fees for this model — the demand
    /// signal. Updated at each epoch settlement (§6.2).
    pub demand_ema: u128,
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
            + Mutate<Self::AccountId, Balance = Self::Balance>
            + InspectHold<Self::AccountId, Balance = Self::Balance>
            + MutateHold<Self::AccountId, Balance = Self::Balance>;

        /// Hold reason for the fidelity bond.
        type HoldReason: Get<<Self::Currency as InspectHold<Self::AccountId>>::Reason>;

        /// EMA smoothing factor `N` for demand: each epoch,
        /// `ema = (ema*(N-1) + pending) / N`. Larger = smoother / more
        /// hysteresis (§10.1), damping the demand-feedback loop (§5.6).
        #[pallet::constant]
        type DemandEmaSmoothing: Get<u32>;

        /// Burned-fee amount that maps to one unit of reward weight. The
        /// demand component of a model's weight is `demand_ema / this`.
        #[pallet::constant]
        type FeePerWeightUnit: Get<Self::Balance>;

        /// Maximum effective reward weight (normalization ceiling).
        #[pallet::constant]
        type MaxModelWeight: Get<u32>;

        /// Attestation evidence verifier.
        type Attestation: AttestationVerifier;

        /// Fidelity bond per device. Sized to the fraud opportunity (a
        /// bounded multiple of epoch revenue), never to capacity.
        #[pallet::constant]
        type BondAmount: Get<Self::Balance>;

        /// Blocks between registration and lottery eligibility.
        #[pallet::constant]
        type ActivationDelay: Get<BlockNumberFor<Self>>;

        /// Blocks per settlement epoch. Demand EMA folds and reward-weight
        /// recomputation happen at most once per epoch (`block / EpochLength`),
        /// which bounds the per-block settlement cost and stops the demand EMA
        /// from being decayed more than once per epoch. Must be non-zero.
        #[pallet::constant]
        type EpochLength: Get<BlockNumberFor<Self>>;
    }

    /// Whitelisted agent-code measurements (governance-managed).
    #[pallet::storage]
    pub type Measurements<T: Config> = StorageMap<_, Twox64Concat, Id32, (), OptionQuery>;

    /// Trusted attestation vendor root public keys (governance-managed):
    /// NVIDIA / Intel / AMD roots in production, a test key on a testnet.
    #[pallet::storage]
    pub type TrustedRoots<T: Config> = StorageMap<_, Twox64Concat, Id32, (), OptionQuery>;

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

    /// Burned inference fees accrued to each model since the last epoch
    /// settlement. Folded into the demand EMA by `settle_epoch`.
    #[pallet::storage]
    pub type PendingFees<T: Config> = StorageMap<_, Twox64Concat, Id32, u128, ValueQuery>;

    /// Current effective reward weight per model: `max(floor, demand)`, capped
    /// at `MaxModelWeight`. This is what the reward-distribution layer reads.
    #[pallet::storage]
    pub type ModelWeight<T: Config> = StorageMap<_, Twox64Concat, Id32, u32, ValueQuery>;

    /// Highest epoch index that has been settled. `None` = never settled.
    /// Settlement is idempotent per epoch: a settle whose target epoch is not
    /// strictly greater than this value is a no-op.
    #[pallet::storage]
    pub type SettledThroughEpoch<T: Config> = StorageValue<_, u64, OptionQuery>;

    /// Block rewards earned by a device during an epoch, held in escrow until
    /// the epoch's cross-audit window closes. Keyed `(epoch, device)` so the
    /// per-epoch release drains one bucket; a fraud clawback only ever probes
    /// the (at most two) still-unreleased epochs for the accused device.
    /// Nothing is minted until release, so a forfeit is simply a deletion —
    /// the reward never enters supply.
    #[pallet::storage]
    pub type EscrowedRewards<T: Config> = StorageDoubleMap<
        _,
        Twox64Concat,
        u64,
        Twox64Concat,
        Id32,
        (T::AccountId, T::Balance),
        OptionQuery,
    >;

    /// Cross-audit beacon for the current epoch: fixed at the epoch boundary
    /// (so it is unknowable during the preceding commit epoch) and the sole
    /// input, with the replica set, to the pure audit-assignment functions in
    /// `subspace-proof-of-residency`. Directs this epoch's audits of the
    /// previous epoch's commitments.
    #[pallet::storage]
    pub type AuditBeaconValue<T: Config> = StorageValue<_, Id32, OptionQuery>;

    #[pallet::hooks]
    impl<T: Config> Hooks<BlockNumberFor<T>> for Pallet<T> {
        /// Settle any elapsed epoch once per block. Cheap in the common case:
        /// [`Self::try_settle_epoch`] short-circuits when the current epoch is
        /// already settled, so the fold loop only runs on epoch boundaries.
        fn on_initialize(_now: BlockNumberFor<T>) -> Weight {
            // Always at least one read (SettledThroughEpoch); the per-model
            // fold sweep runs only when crossing into a new epoch.
            match Self::try_settle_epoch() {
                Some(models) => {
                    let m = u64::from(models);
                    // read+write per model, plus the epoch-marker read+write.
                    T::DbWeight::get().reads_writes(m + 1, m + 1)
                }
                None => T::DbWeight::get().reads(1),
            }
        }
    }

    #[pallet::event]
    #[pallet::generate_deposit(pub(super) fn deposit_event)]
    pub enum Event<T: Config> {
        MeasurementRegistered {
            measurement: Id32,
        },
        MeasurementRevoked {
            measurement: Id32,
        },
        TrustedRootAdded {
            root: Id32,
        },
        TrustedRootRemoved {
            root: Id32,
        },
        InferenceFeeBurned {
            model_id: Id32,
            amount: u128,
        },
        EpochSettled {
            models: u32,
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
        /// A block reward was placed in escrow pending the audit window.
        RewardEscrowed {
            device_id: Id32,
            epoch: u64,
            amount: T::Balance,
        },
        /// An escrowed reward survived its audit window and was minted.
        RewardReleased {
            device_id: Id32,
            owner: T::AccountId,
            epoch: u64,
            amount: T::Balance,
        },
        /// Escrowed rewards of a fraudulent device were forfeited (never
        /// minted — the reward simply does not enter supply).
        RewardForfeited {
            device_id: Id32,
            amount: T::Balance,
        },
        /// The cross-audit beacon for a new epoch was fixed.
        AuditBeaconSet {
            epoch: u64,
            beacon: Id32,
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
        /// The fee to burn could not be taken from the caller.
        FeeBurnFailed,
        /// Solution is not signed by the accused device's node key.
        BadSolutionSignature,
        /// Device still has rewards in escrow; exit must wait out the
        /// cross-audit window (retry after ~2 epochs).
        EscrowPending,
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
            floor_weight: u32,
        ) -> DispatchResult {
            ensure_root(origin)?;
            ensure!(
                size_bytes > 0
                    && size_bytes % (subspace_proof_of_residency::TILE_BYTES as u64) == 0,
                Error::<T>::BadModelSize
            );
            let floor_weight = floor_weight.min(T::MaxModelWeight::get());
            Models::<T>::insert(
                model_id,
                ModelInfo {
                    size_bytes,
                    min_replicas,
                    floor_weight,
                    demand_ema: 0,
                },
            );
            // Start at the floor until demand accrues.
            ModelWeight::<T>::insert(model_id, floor_weight);
            Self::deposit_event(Event::ModelRegistered { model_id });
            Ok(())
        }

        /// Whitelist an attestation vendor root public key. Governance only.
        #[pallet::call_index(8)]
        #[pallet::weight(Weight::zero())]
        pub fn add_trusted_root(origin: OriginFor<T>, root: Id32) -> DispatchResult {
            ensure_root(origin)?;
            TrustedRoots::<T>::insert(root, ());
            Self::deposit_event(Event::TrustedRootAdded { root });
            Ok(())
        }

        /// Remove a trusted attestation vendor root. Governance only.
        #[pallet::call_index(9)]
        #[pallet::weight(Weight::zero())]
        pub fn remove_trusted_root(origin: OriginFor<T>, root: Id32) -> DispatchResult {
            ensure_root(origin)?;
            TrustedRoots::<T>::remove(root);
            Self::deposit_event(Event::TrustedRootRemoved { root });
            Ok(())
        }

        /// Record burned inference fees for a model: the caller's `amount` is
        /// **actually burned** (removed from supply) and accrued to the
        /// model's demand signal. Burning is what makes demand a costly,
        /// unfakeable signal — inflating a model's weight by self-dealing costs
        /// real tokens (whitepaper §5.1). In production the inference fee
        /// handler calls [`Pallet::note_inference_fee`] directly; this
        /// extrinsic is the permissionless facilitator entry point.
        #[pallet::call_index(10)]
        #[pallet::weight(Weight::zero())]
        pub fn record_inference_fee(
            origin: OriginFor<T>,
            model_id: Id32,
            amount: T::Balance,
        ) -> DispatchResult {
            let who = ensure_signed(origin)?;
            ensure!(
                Models::<T>::contains_key(model_id),
                Error::<T>::UnknownModel
            );
            T::Currency::burn_from(
                &who,
                amount,
                frame_support::traits::tokens::Preservation::Preserve,
                Precision::Exact,
                frame_support::traits::tokens::Fortitude::Polite,
            )
            .map_err(|_| Error::<T>::FeeBurnFailed)?;
            Self::note_inference_fee(model_id, amount.saturated_into::<u128>());
            Ok(())
        }

        /// Manually trigger epoch settlement. **Idempotent per epoch**: if the
        /// current epoch has already been settled (by a prior call or the
        /// `on_initialize` hook) this is a no-op, so it cannot be spammed to
        /// repeatedly decay the demand EMA. Settlement also runs automatically
        /// once per epoch from `on_initialize`; this call only advances it
        /// early within an unsettled epoch.
        #[pallet::call_index(11)]
        #[pallet::weight(Weight::zero())]
        pub fn settle_epoch(origin: OriginFor<T>) -> DispatchResult {
            ensure_signed(origin)?;
            Self::try_settle_epoch();
            Ok(())
        }

        /// Register an attested device: verify evidence against the trusted
        /// roots (binding the device id and node key), check the measured
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
            let roots: alloc::vec::Vec<Id32> = TrustedRoots::<T>::iter_keys().collect();
            let measurement = T::Attestation::verify(&roots, &device_id, &pubkey, &evidence)
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
        ///
        /// Refused while the device still has rewards in escrow: exit must
        /// wait out the cross-audit window (~2 epochs after the last authored
        /// block), so escrowed pay cannot be walked out from under a pending
        /// audit. (A full exit-delay on the bond itself — a pending-exit
        /// state — is future work; today a cheat that deregisters before
        /// being reported escapes the bond but never its unreleased pay.)
        #[pallet::call_index(4)]
        #[pallet::weight(Weight::zero())]
        pub fn deregister_device(origin: OriginFor<T>, device_id: Id32) -> DispatchResult {
            let who = ensure_signed(origin)?;
            let info = Devices::<T>::get(device_id).ok_or(Error::<T>::UnknownDevice)?;
            ensure!(info.owner == who, Error::<T>::NotOwner);
            ensure!(
                !Self::has_pending_escrow(&device_id),
                Error::<T>::EscrowPending
            );
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
                    // Claw back every reward still in escrow: fraud caught
                    // within the audit window costs the cheat its pending
                    // pay, not just its bond.
                    Self::forfeit_escrow(&solution.device_id);
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

        /// Accrue burned inference fees to a model's demand signal. Internal
        /// entry point for the runtime fee handler (the extrinsic wraps this
        /// after burning). No-op for an unknown model.
        pub fn note_inference_fee(model_id: Id32, amount: u128) {
            if Models::<T>::contains_key(model_id) {
                PendingFees::<T>::mutate(model_id, |p| *p = p.saturating_add(amount));
                Self::deposit_event(Event::InferenceFeeBurned { model_id, amount });
            }
        }

        /// Current effective reward weight of a model (what the reward-
        /// distribution layer reads). Zero for an unknown model.
        pub fn model_reward_weight(model_id: &Id32) -> u32 {
            ModelWeight::<T>::get(model_id)
        }

        /// Escrow a block reward earned by a device this epoch instead of
        /// paying it immediately. Entry point for the runtime's block-reward
        /// hook when a PoRW block author is rewarded. Nothing is minted here:
        /// the reward enters supply only when its escrow bucket is released
        /// (audit window passed), so a fraud-triggered forfeit is a pure
        /// deletion. No-op for an unregistered device or zero epoch length.
        pub fn note_block_reward(device_id: Id32, amount: T::Balance) {
            if amount.is_zero() {
                return;
            }
            let Some(device) = Devices::<T>::get(device_id) else {
                return;
            };
            let epoch_len = T::EpochLength::get();
            if epoch_len.is_zero() {
                return;
            }
            let now = frame_system::Pallet::<T>::block_number();
            let epoch: u64 = (now / epoch_len).saturated_into();
            EscrowedRewards::<T>::mutate(epoch, device_id, |entry| match entry {
                Some((_, total)) => *total = total.saturating_add(amount),
                None => *entry = Some((device.owner, amount)),
            });
            Self::deposit_event(Event::RewardEscrowed {
                device_id,
                epoch,
                amount,
            });
        }

        /// Whether the device has any reward still in escrow (same probe
        /// range as [`Self::forfeit_escrow`]).
        fn has_pending_escrow(device_id: &Id32) -> bool {
            let epoch_len = T::EpochLength::get();
            if epoch_len.is_zero() {
                return false;
            }
            let now = frame_system::Pallet::<T>::block_number();
            let current_epoch: u64 = (now / epoch_len).saturated_into();
            (current_epoch.saturating_sub(2)..=current_epoch)
                .any(|epoch| EscrowedRewards::<T>::contains_key(epoch, device_id))
        }

        /// Forfeit every still-escrowed reward of a device (fraud clawback).
        /// Escrow buckets older than `current_epoch - 1` were already
        /// released by settlement (which runs in `on_initialize`, before any
        /// extrinsic), so probing the last three epochs covers every entry
        /// that can exist. Forfeited rewards were never minted — they simply
        /// never enter supply.
        fn forfeit_escrow(device_id: &Id32) {
            let epoch_len = T::EpochLength::get();
            if epoch_len.is_zero() {
                return;
            }
            let now = frame_system::Pallet::<T>::block_number();
            let current_epoch: u64 = (now / epoch_len).saturated_into();
            let mut forfeited = T::Balance::zero();
            for epoch in current_epoch.saturating_sub(2)..=current_epoch {
                if let Some((_, amount)) = EscrowedRewards::<T>::take(epoch, device_id) {
                    forfeited = forfeited.saturating_add(amount);
                }
            }
            if !forfeited.is_zero() {
                Self::deposit_event(Event::RewardForfeited {
                    device_id: *device_id,
                    amount: forfeited,
                });
            }
        }

        /// Registered ed25519 node public key of a device, or `None` if the
        /// device is not registered. Block import reads this to verify the
        /// block seal against the solution's device.
        pub fn device_node_key(device_id: &Id32) -> Option<Id32> {
            Devices::<T>::get(device_id).map(|d| d.pubkey)
        }

        /// Settle the current epoch if it has not been settled yet. Idempotent
        /// per epoch: folding runs at most once per epoch index, so neither a
        /// spammed extrinsic nor repeated hook invocations within one epoch can
        /// decay the demand EMA more than once.
        ///
        /// Folds exactly once when crossing into a new epoch (never a catch-up
        /// sweep of skipped epochs): the `on_initialize` hook runs every block,
        /// so in production no epoch boundary is ever missed, and folding only
        /// the newest epoch keeps the per-block cost `O(models)` and bounded.
        ///
        /// Returns `Some(models_folded)` when a fold ran (for weight
        /// accounting), or `None` when the current epoch was already settled.
        fn try_settle_epoch() -> Option<u32> {
            let epoch_len = T::EpochLength::get();
            if epoch_len.is_zero() {
                return None;
            }
            let now = frame_system::Pallet::<T>::block_number();
            let current_epoch: u64 = (now / epoch_len).saturated_into();
            if let Some(e) = SettledThroughEpoch::<T>::get() {
                if e >= current_epoch {
                    return None;
                }
            }
            let folded = Self::fold_demand_epoch();

            // Fix this epoch's cross-audit beacon. Derived from the parent
            // block hash at the boundary so it was unknowable during the
            // commit epoch it audits (production should feed PoT-derived
            // randomness here instead — the parent hash is grindable by the
            // boundary-block author within its solution set).
            let parent = frame_system::Pallet::<T>::parent_hash();
            let mut entropy = [0u8; 32];
            let bytes = parent.as_ref();
            entropy[..bytes.len().min(32)].copy_from_slice(&bytes[..bytes.len().min(32)]);
            let beacon = subspace_proof_of_residency::audit_beacon(current_epoch, &entropy);
            AuditBeaconValue::<T>::put(beacon);
            Self::deposit_event(Event::AuditBeaconSet {
                epoch: current_epoch,
                beacon,
            });

            // Release escrow whose audit window has fully passed: rewards
            // earned in epoch e are paid at the settle of e + 2, after all of
            // e + 1 (the audit window) produced no confirmed fraud. Nothing
            // was minted at escrow time, so payment mints here. Iterating the
            // live map is O(unreleased entries) — at most the last two
            // epochs' authors, since every earlier bucket was drained by an
            // earlier settle.
            if let Some(release_through) = current_epoch.checked_sub(2) {
                let due: alloc::vec::Vec<(u64, Id32, (T::AccountId, T::Balance))> =
                    EscrowedRewards::<T>::iter()
                        .filter(|(epoch, _, _)| *epoch <= release_through)
                        .collect();
                for (epoch, device_id, (owner, amount)) in due {
                    EscrowedRewards::<T>::remove(epoch, device_id);
                    if T::Currency::mint_into(&owner, amount).is_ok() {
                        Self::deposit_event(Event::RewardReleased {
                            device_id,
                            owner,
                            epoch,
                            amount,
                        });
                    }
                }
            }

            SettledThroughEpoch::<T>::put(current_epoch);
            Some(folded)
        }

        /// Fold one epoch's pending fees into each model's demand EMA and
        /// recompute its effective reward weight
        /// `clamp(demand_ema / FeePerWeightUnit, floor, MaxModelWeight)`.
        /// Returns the number of models folded.
        fn fold_demand_epoch() -> u32 {
            let n = u128::from(T::DemandEmaSmoothing::get().max(1));
            let unit = T::FeePerWeightUnit::get().saturated_into::<u128>().max(1);
            let max_weight = T::MaxModelWeight::get();
            let mut count = 0u32;
            let model_ids: alloc::vec::Vec<Id32> = Models::<T>::iter_keys().collect();
            for model_id in model_ids {
                Models::<T>::mutate(model_id, |maybe| {
                    if let Some(info) = maybe {
                        let pending = PendingFees::<T>::take(model_id);
                        // ema = (ema*(n-1) + pending) / n
                        info.demand_ema = info
                            .demand_ema
                            .saturating_mul(n - 1)
                            .saturating_add(pending)
                            / n;
                        let demand_weight =
                            (info.demand_ema / unit).min(u128::from(max_weight)) as u32;
                        let effective = demand_weight.max(info.floor_weight).min(max_weight);
                        ModelWeight::<T>::insert(model_id, effective);
                        count += 1;
                    }
                });
            }
            Self::deposit_event(Event::EpochSettled { models: count });
            count
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
