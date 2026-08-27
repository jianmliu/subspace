# PoRW ↔ aigg-spec Alignment

**Status:** living mapping document (2026-08)

**Upstream spec:** [`jianmliu/aigg-spec`](https://github.com/jianmliu/aigg-spec)
— `docs/architecture/mep-porw-modular-interfaces.md` (interface spec),
`docs/architecture/evm-compute-market-protocol.md` (EVM deployment),
`proposals/ai3/ai3-domain-incentive-proposal.md` (AI3 route, which names this
branch as the *Consensus and PoRW reference branch*).

`aigg-spec` is the canonical home of the chain-neutral MEP/PoRW
specifications. Per its §10.9, **this repository is the PoRW research and
implementation reference** — consensus-specific code here is not imported
into the shared application interfaces unless a deployment explicitly adopts
it. This document maps what exists here onto the spec's interfaces, records
where the semantics already conform, and lists the gaps.

## 1. The two deployment postures

The spec generalizes PoRW away from any single chain. Two postures coexist:

| Posture | Where PoRW output goes | Status |
|---|---|---|
| **L1 dual-track consensus** (this branch's research vehicle) | PoRW solutions author blocks on a Subspace-derived chain (VRAM track beside PoAS) | Implemented end-to-end on CPU; documented in `proof-of-resident-weights.md` |
| **GPU Domain / EVM Worker protocol** (aigg-spec productization) | Signed solutions aggregate into an `EpochPoRWRoot`; capacity is an epoch-scoped *fact* consumed by incentive/eligibility policy — **never block authorship, never consumed by execution** | Specified in aigg-spec; reuses this branch's verifier, registry, audit, and escrow machinery per AI3 proposal §14.2 |

The AI3 proposal is explicit: *"Do not modify PoAS or PoAS rewards for this
plan"* and *"[the experimental PoRW branch] must be adapted from alternative
main-chain block authorship into a GPU Domain Worker protocol."* The
consensus experiment stays valuable as the research testbed in which the
primitives are hardened; the productized deployment consumes the same
primitives behind the spec's interfaces.

## 2. Verifier surface: `IPoRWVerifier`

Spec (§6.4):

```text
IPoRWVerifier {
    schemeId() -> PoRWScheme_ID
    verifyCommitment(claim, proof) -> VerificationResult
    verifyOpening(claim_id, challenge, opening) -> VerificationResult
    verifyFraudProof(claim_id, fraud_proof) -> VerificationResult
}
```

This branch (`crates/subspace-proof-of-residency`, no_std, deterministic):

| Spec method | Implementation | Notes |
|---|---|---|
| `schemeId()` | `PORW_SCHEME_ID = "aigg:porw:sketch-tile:v2"` / `porw_scheme_digest()` | v2 = 4 KiB tiles, per-word slot-fresh odd u32 coefficients (fmix32), blake3 Merkle, strictly-ascending coverage |
| `verifyCommitment` | `PorwSolution::signing_payload` + ed25519 check, `check_envelope`, `ticket_count` (fast path composed in `pallet-porw-registry::check_solution_signed`) | Device signature binds solution to device; hardware envelope caps claimable work |
| `verifyOpening` | `verify_opening_response` (`OpeningResponse::Committed` / `NotCommitted`, `LeafWitness`) | Includes adjacent-leaf **non-inclusion** proofs; leaf count pinned by signed `coverage_bytes` |
| `verifyFraudProof` | `verify_tile_fraud_proof` (`TileFraudProof`, `FraudVerdict`) | Single-tile granularity: O(4 KiB) + two Merkle paths |

The spec's dispute-path evidence list (EVM doc §9.4) is covered one-for-one:

| Spec dispute evidence | Here |
|---|---|
| a signed Worker solution | `PorwSolution` + `signing_payload` |
| a commitment opening or proof of non-inclusion | `OpeningResponse::{Committed, NotCommitted}` |
| a canonical model tile + Merkle proof against the model root | `TileFraudProof::{tile_bytes, weights_proof}` vs `R_W` |
| a recomputed sketch fraud proof | `verify_tile_fraud_proof` |
| an expired-response claim | `claim_expired_challenge` (pallet) |
| a succinct proof accepted by the registered verifier | future scheme version (new `PORW_SCHEME_ID`) |

## 3. Challenge lifecycle: `IPoRWChallengeManager`

Spec (§6.5): `openChallenge(claim, challenge_ref, evidence_ref,
challenger_bond)` → `respond` → `resolve`; challenger escrow; one terminal
resolution; resolution only through the claim-pinned verifier.

Here (`pallet-porw-registry`):

| Spec | Implementation |
|---|---|
| `openChallenge` + challenger bond | `challenge_opening` + `OpeningChallengeDeposit` (held) |
| `respond` | `respond_opening` (verifies via `verify_opening_response`; valid answer forfeits the deposit to the device owner — the spec's *"honest Workers receive compensation for forced valid responses"*) |
| `resolve` (expiry branch) | `claim_expired_challenge` — fraud-grade: bond to challenger, escrow forfeited, device revoked, deposit returned |
| fraud-proof branch | `report_fraud` (permissionless, bounty to reporter) |
| one terminal resolution / idempotent | challenge keyed `(device, (partials_root, tile_idx))`; removed on resolution; duplicates rejected (`ChallengeExists`) |
| `submitAuthorizedResolution` (async verifier) | N/A — the runtime verifies synchronously on chain; the pinned-verifier requirement is met structurally (see §5) |

## 4. Epoch machinery: `IPoRWEpochRegistry` semantics

Spec: claims are **epoch-scoped**; capacity derives only from finalized
claims; execution never consumes capacity; finalization follows a challenge
window.

Here, the same lifecycle exists with consensus-flavored naming:

| Spec concept | Here |
|---|---|
| epoch-scoped claim | per-slot `PorwSolution`s accrue per-epoch; rewards bucket per `(epoch, device)` |
| challenge window before finality | `EscrowedRewards`: epoch-e rewards mint only at the e+2 settle, after the audit window (epoch e+1) passes without fraud |
| `finalizeClaim` | escrow release (mint) at settlement |
| `invalidateClaim` | `forfeit_escrow` on confirmed fraud / expired challenge (never minted — never enters supply) |
| audit scheduling | `audit_beacon` (PoT-derived entropy) + `select_auditors` + `audit_tile_sample` — pure functions, zero on-chain schedule state |
| *"task execution does not consume capacity"* | holds by construction: the lottery/capacity reads coverage; receipts and inference never decrement anything |
| Aggregator `EpochPoRWRoot` (EVM §9.3) | not needed on the Substrate L1 (solutions land directly); required for the EVM deployment — an aggregation of the same signed `PorwSolution`s this crate already defines |

Exit safety (spec Bond Vault §10.1: *"withdrawals cannot complete while
challenges … or reward escrows remain open"*): implemented as the two-step
`deregister_device` → `ExitDelay` → `finalize_deregistration`, gated on empty
escrow and no open challenge.

## 5. Verifier pinning (`ComponentPin`)

The spec pins every claim to an exact verifier implementation. On the
Substrate L1 the pin is structural: a claim's verifier is the runtime Wasm
that was active at the block that accepted it, and runtime upgrades are
prospective — old blocks were verified by the old code, matching the spec's
*"new verifier applies only to claims that pin it"*. For the EVM/Domain
deployment, `PORW_SCHEME_ID` + `porw_scheme_digest()` provide the scheme
half of the pin; the contract `VerifierRegistry` provides the implementation
half.

## 6. Identity mapping: `model_id` vs `MEP_ID`

The spec's `MEP_ID = hash(ModelProfile, ExecutionProfile,
DeploymentConstraints)`. This branch's `model_id` is exactly
`ModelProfile.weight_root` (`R_W`) — the consensus-critical component: the
sketch, fraud proofs, and storage-track arbitration all bind to raw weight
bytes only. Runtime digest, quantization, tokenizer, and sampling envelope
(the rest of the MEP) do not affect residency verification and live at the
application layer. A deployment maps `MEP_ID → weight_root` when submitting
PoRW claims; two MEPs sharing one weight root share residency proofs by
design (same bytes resident), while conformance to the *execution* profile
is attested separately (spec `IConformanceRegistry` / pillar D — never by
PoRW, per spec invariant 5/6: execution attestation and PoRW cannot
substitute for each other, which matches this design's C/D pillar split).

## 7. Conformance: spec PoRW suite ↔ tests here

Spec §15.2 required PoRW fixtures, mapped to existing tests:

| Spec fixture | Test |
|---|---|
| valid commitment | `devnet::agent_solution_is_accepted_by_the_chain_fast_path`; `blockloop::produces_and_imports_a_chain_of_sealed_porw_blocks` |
| valid opening | `opening_challenge_is_answered_with_a_commitment_opening`, `..._with_non_inclusion` (pallet); `opening_responses_prove_commitment_and_non_commitment` (primitives) |
| invalid tile | `fraud_proof_catches_tampered_commitment`, `fraud_proof_works_for_non_contiguous_coverage` (primitives); `cross_audit_catches_a_lying_replica_end_to_end` (devnet) |
| missing response | `unanswered_opening_challenge_slashes_like_fraud` (pallet) |
| forged resolution rejection | `fabricated_unsigned_solution_cannot_slash`, `honest_solution_cannot_be_slashed` (pallet); `a_block_with_a_foreign_seal_is_rejected` (blockloop) |
| exact verifier-pin enforcement | structural on Substrate (§5); contract-side fixture belongs to the EVM implementation |
| capacity-view: receipts do not decrement capacity | holds by construction here; fixture belongs to the EVM implementation |

Cross-language vectors (spec §15.1): the sketch already ships bit-identical
vectors across numpy / Triton / Rust (`porw-poc/` and
`cross_language_sketch_vectors`); these are the natural seed for the spec's
`conformance/cross-language/` fixtures.

## 8. Gaps / next adaptations (per AI3 proposal §14.2)

1. **GPU Domain runtime**: register a Domain under the existing
   `pallet-domains` framework and move Worker-facing PoRW there, leaving
   PoAS untouched. The registry/attestation/audit/escrow pallets port with
   naming changes; block authorship (`sc-consensus-subspace::porw`,
   `PorwPreDigest`, seals) stays in the L1 research track only.
2. **Epoch claim aggregation**: an `EpochPoRWRoot` builder over signed
   `PorwSolution`s (for the EVM/Domain path) with openings compatible with
   `verify_opening_response`.
3. **MEP primitives**: ModelManifest / MEP / MEP-vote / incentive-vault
   objects (spec §6.2, §6.11–6.13) — currently only `weight_root`-keyed
   `ModelInfo` with demand-EMA weights exists here.
4. **Capacity non-duplication**: the simultaneous-residency rule of
   `CapacityUnitDefinition` — one device claiming the same VRAM bytes under
   several MEPs/pools needs an explicit rule (today: one `device_id`, per-model
   announcements, envelope-capped totals; a cross-model total-VRAM cap is not
   yet enforced).
5. **Worker pools / delegation**: spec Worker Pool policy objects; this
   branch has only the bounded sqrt stake scaling on the L1.
6. **Read-only facts API**: events/APIs for an external incentive controller
   (spec fact envelopes with provenance and expiry).
