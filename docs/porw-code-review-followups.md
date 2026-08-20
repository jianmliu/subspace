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

## Open — in the merged PoS branch (author decision required)

These pre-date the PoRW work; they live in the `origin/PoS` code merged
into this branch and touch consensus-critical stake math. Flagging rather
than changing unilaterally.

1. **`MaxVoterBalance = Balance::MAX` halts the chain once anyone stakes.**
   `crates/subspace-runtime/src/lib.rs` (and the same in
   `test/subspace-test-runtime`). `max_voting_stake_weight` becomes
   `sqrt(u128::MAX) ≈ 1.8e19`; the first non-zero stake collapses every
   farmer's scaled solution range by ~1e5–1e9×, so no solution qualifies
   and block production stops — and since unstaking needs a block, it is
   unrecoverable. **Suggested fix:** set `MaxVotingBalance` to a realistic
   cap (e.g. total issuance or a governance parameter), not `Balance::MAX`.

2. **Zero-stake grants maximum weight, not a neutral baseline.**
   `pallet-subspace` `voting_stake_weight` returns `max_weight` for every
   voter while `TotalStake == 0`. The instant one account stakes the
   minimum, all non-stakers drop to `sqrt(0) = 0` weight (scaled range 0,
   permanent exclusion) — a discontinuous cliff a single actor can trigger
   to monopolize production. Ties into #1.

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
