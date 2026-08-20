# PoRW code-review follow-ups

Findings from the `/code-review` pass over `origin/main...HEAD`, with
resolution status.

## Fixed in the PoRW code (this branch)

| # | File | Issue | Fix |
|---|------|-------|-----|
| 1 | `pallet-porw-registry` `report_fraud` | **Critical:** slashed a device on an entirely caller-fabricated `PorwSolution` — tiles and Merkle paths are public, so anyone could forge a wrong solution for any device and steal its bond. | `PorwSolution` now carries an ed25519 `signature`; devices register a `pubkey`; `report_fraud` verifies the accused device actually signed the solution (over a payload that includes the slot challenge) before slashing. New test `fabricated_unsigned_solution_cannot_slash`. |
| 2 | `subspace-proof-of-residency` `ticket_chunk` | Ticket stream derived from `partials_root + slot_seed` only, so a device announcing several models replayed one ticket stream across all of them. | `model_id` mixed into the XOF. Ticket test asserts per-model distinctness. |
| 3 | `pallet-porw-registry` `deregister_device` | Released `BondAmount::get()` with `Precision::Exact`; a later governance change to the constant would strand existing bonds. | Bond amount stored per-device in `DeviceInfo.bond`; released with `BestEffort`. |
| 4 | `pallet-porw-registry` `ModelWithdrawn` | Event declared but no extrinsic — announcements could only be dropped by full deregistration; `ReplicaCount` overstated live replicas. | Added `withdraw_model` extrinsic (decrements replicas, gates further solutions). New test `withdraw_model_decrements_replicas_and_gates_solutions`. |
| 5 | `pallet-porw-registry` `check_solution` | Activation-delay check used unchecked `+`. | `saturating_add`, consistent with the rest of the pallet. |

## Fixed — chain-halting stake bugs (this branch)

1. **`MaxVoterBalance = Balance::MAX` halted the chain once anyone staked.**
   `max_voting_stake_weight` was `sqrt(u128::MAX) ≈ 1.8e19`; the first
   non-zero stake collapsed every scaled solution range by ~1e5–1e9× and
   stopped block production unrecoverably. **Fixed:** `MaxVoterBalance`
   (subspace-runtime) and `MaxVotingBalance` (test-runtime) are now finite
   governance placeholders (`10_000_000 * AI3` / equivalent SHANNON), and
   `VotingStakeMax` matches.

2. **Zero-stake→max, first-stake→zero exclusion cliff.**
   `voting_stake_weight` returned `max_weight` at `TotalStake == 0` but
   dropped every non-staker to `sqrt(0) = 0` (scaled range 0, permanent
   exclusion) the instant anyone staked. **Fixed:** a shared
   `Pallet::voter_weight(stake, max_weight)` maps stake to a bounded
   continuous weight in `[floor, max_weight]` where
   `floor = max_weight * MIN_VOTER_WEIGHT_BPS / 10000` (50%). No stake
   anywhere is still a full-weight no-op; once staking is active a
   non-staker drops only to the floor (a bounded ~2× range change), never
   to zero. Stake is a bounded bonus, never a participation gate — matching
   the intent that capacity gates and stake only shifts share. New test
   `voter_weight_floors_and_never_excludes`; all 27 existing pallet-subspace
   tests still pass; both runtimes' wasm builds pass.

   Note (transient): when the very first stake appears, non-stakers' ranges
   halve for up to one difficulty-adjustment era before the solution-range
   adjustment re-centres block time. `MIN_VOTER_WEIGHT_BPS` and
   `MaxVoterBalance` are the two knobs governance should calibrate against
   real tokenomics.

## Open — in the merged PoS branch (author decision required)

3. **Reward split saturates with realistic stakes.** `pallet-rewards`
   computes `total_voter_pool.saturating_mul(weight) / total_weight` in
   u128; with production-scale stakes the product saturates, shares
   collapse, and the `is_last` remainder branch hands the final voter
   (by `BTreeMap` order) almost the whole pool. **Suggested fix:**
   `U256` intermediate, or normalize weights before multiplying.

4. **New runtime-API calls not `api_version`-gated.**
   `slot_worker.rs` / `block_import.rs` call `max_voting_stake_weight` /
   `voting_stake_weight` unconditionally; against a runtime exposing
   `SubspaceApi < v3` (during upgrade rollout or historical sync) these
   error — `.ok()?` silently aborts authorship, `?` hard-fails import.
   `block_import.rs:763` already shows the gating pattern to follow.
