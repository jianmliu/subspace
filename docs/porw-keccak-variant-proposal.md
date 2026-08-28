# Scheme registration draft: `aigg:porw:sketch-tile-keccak:v1`

**Status:** submission draft for the aigg-spec process (this repository is
the reference-implementation home; the normative registration belongs in
`jianmliu/aigg-spec`). Prepared following the adoption decision: *the
direct verifier is adopted now; this keccak variant is the designated
cost-reduction step.*

## 1. What is being registered

A new PoRW verification scheme for EVM deployments:

| | `sketch-tile:v2` (existing) | `sketch-tile-keccak:v1` (this registration) |
|---|---|---|
| Scheme id | `aigg:porw:sketch-tile:v2` | `aigg:porw:sketch-tile-keccak:v1` |
| Sketch math | word-granular u32, fmix32 odd coefficients | **identical** |
| Tile size / coverage order | 4 KiB, strictly ascending | **identical** |
| All hashes (leaves, nodes, slot seed, scheme digest) | blake3 | **keccak256** |
| Ticket expansion (`ticket_chunk`) | blake3 XOF (L1 lottery) | **not part of the variant** (the EVM pilot consumes capacity facts, not a lottery) |
| Conformance vectors | `conformance/sketch-tile-v2.json` | `conformance/sketch-tile-keccak-v1.json` |
| Reference implementation | `subspace-proof-of-residency` | `subspace_proof_of_residency::keccak` (same crate, shared logic) |
| Primary deployment | Substrate L1 research track | Auto EVM contracts (pilot) |

Per aigg-spec §2.6/§12.3 this is a **new scheme id**, never a
reinterpretation: commitments from one scheme do not verify under the
other (covered by tests), the per-device slot seed — and therefore every
committed sketch value — differs by construction, and each scheme pins its
own conformance vectors.

## 2. Why (measured, not estimated)

From `docs/porw-evm-feasibility.md` (same Solidity harness, same tree
depths — weights 25 / partials 21 — against Auto EVM's 52M block gas
limit; reproduce with `cd porw-evm-bench && forge test -vv`):

| Dispute object | blake3 v2 | keccak v1 (measured) | Reduction |
|---|---:|---:|---:|
| Full tile fraud proof | 8,618,217 (16.6% block) | **1,106,534 (2.1% block)** | **7.8×** |
| Committed opening, depth 21 | 1,442,506 | **70,586** | 20× |
| 4,104 B tile hash | 4,439,420 | **292,189** | 15× |
| Merkle verify, depth 25 | 1,646,827 | 75,303 | 22× |

The remaining fraud-proof cost is dominated by the sketch recomputation
(967k gas), which is the scheme's irreducible math and hash-independent.
Under the keccak variant, a 52M block absorbs ~23 full fraud proofs at 50%
headroom (vs ~3 under blake3): the worst-case dispute-congestion margin
widens by the same 7.8×.

## 3. Semantics (normative summary)

All definitions are those of `sketch-tile:v2` with the hash function
replaced by keccak256:

- `weights_leaf = keccak256(LE64 tile_idx || tile_bytes)`; `R_W` is the
  duplicate-last-padding binary keccak256 tree over the leaves in tile
  order.
- `partials_leaf = keccak256(LE64 tile_idx || LE32 s_tile)`;
  `partials_root` is the same tree over coverage-ordered leaves
  (coverage strictly ascending; leaf count pinned by the signed
  `coverage_bytes / 4096`).
- `slot_seed = LE32(keccak256(global_challenge || device_id)[0..4])`.
- Sketch: unchanged — `s_tile = Σ_j coeff(j) · word_j mod 2^32`,
  `coeff(j) = fmix32(tile_seed + j·GOLDEN32) | 1`,
  `tile_seed = fmix32(fmix32(slot_seed ^ u32(tile_idx)))`.
- Fraud proof / opening / non-inclusion verification: logic identical to
  v2 (`verify_tile_fraud_proof`, `verify_opening_response`), over the
  keccak commitments.
- Scheme digest (verifier pinning): `keccak256(scheme_id)` =
  `0x718b2eb3b33a6d18d904363ee1cc2fb797e344af10132ebfd93aef8d5d72e4b4`.
- Signature suite remains outside the scheme id (secp256k1/`ecrecover`
  on EVM deployments; ed25519 on Substrate).
- Audit-schedule derivation (beacon/auditor/tile sampling) is deployment
  infrastructure, not part of the verifier scheme; a deployment states its
  own choice.

## 4. Conformance

`conformance/sketch-tile-keccak-v1.json` (drift-tested against the Rust
reference; regeneration gated behind `UPDATE_CONFORMANCE=1` and a scheme
bump) covers: scheme digest, slot-seed derivation, per-tile sketch
vectors under the keccak-derived seed, both trees over the reference
buffer, and the full tampered-commitment scenario (committed opening,
non-inclusion bracket, fraud proof with expected verdict). Two
independent implementations already reproduce it bit-for-bit: the Rust
module `subspace_proof_of_residency::keccak` and the Solidity verifier
(`porw-evm-bench/test/KeccakVariant.t.sol`, 9 tests).

## 5. Registration and migration rules (per aigg-spec §12)

- New `PoRWScheme_ID` registered alongside v2; `ExecutionProfile`s choose
  their scheme via `porw_scheme_id`. Existing v2 claims and MEPs are
  untouched.
- A model listed under both schemes has two independent `R_W` roots (the
  trees differ); the weight bytes are identical, so DSN storage is shared.
- Claims pin the exact verifier (`ComponentPin`); a claim under one scheme
  can only ever be resolved by that scheme's pinned verifier.
- Recommended deployment posture: the Auto EVM pilot registers MEPs with
  `porw_scheme_id = aigg:porw:sketch-tile-keccak:v1`; the Substrate L1
  research track keeps `sketch-tile:v2`.

## 6. What the aigg-spec PR should contain

1. Scheme entry (id, digest, normative summary of §3) in the modular
   interfaces document or a `specs/` scheme registry.
2. The conformance fixture (copied at an identified revision from this
   repository, per the spec's "identify the exact upstream revision"
   rule).
3. A pointer to the measured feasibility numbers (§2) as the registration
   rationale.
