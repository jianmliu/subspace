# PoRW EVM Feasibility Report

**Status:** measured benchmarks v1 (2026-08) — the feasibility-gate evidence
required by the AI3 Verifiable Compute Market Pilot proposal §8.4 before
production PoRW incentives.

**Method:** the dispute-path verifier for scheme `aigg:porw:sketch-tile:v2`
was ported to Solidity (`porw-evm-bench/`, Foundry) — full BLAKE3 (chunk
tree included), the word-granular sketch, both Merkle trees, and the three
dispute entry points. **Semantic fidelity is proven, not assumed**: 13
differential tests reproduce the Rust reference's
`conformance/sketch-tile-v2.json` bit-for-bit (blake3 single- and
multi-chunk, sketch vectors under three seeds, slot-seed derivation, both
trees, committed opening, non-inclusion bracket, fraud-proof verdicts).
Reproduce with:

```
cd porw-evm-bench
forge test --match-contract ConformanceTest        # 13 differential tests
forge test --match-contract GasBench -vv           # the numbers below
```

**Target chain:** Auto EVM. Block gas limit derived from this repo's
`domains/primitives/evm-tracker` + `domains/primitives/runtime`:
`GAS_PER_SECOND = 40M`, `WEIGHT_PER_GAS = 25_000`, block weight =
65% × 2 s ⇒ **BlockGasLimit = 52,000,000 gas** per domain block.

The Solidity is a clarity-first port (no assembly, no via-IR hand-tuning
beyond the default optimizer): every number below is an **upper bound**
with known optimization headroom.

## 1. Direct EVM verification cost (measured)

Realistic tree depths: a 70 GB model = ~17M 4 KiB tiles ⇒ weights-tree
depth 25; a 2M-tile per-slot coverage ⇒ partials depth 21.

| Operation | Gas | % of a 52M block |
|---|---:|---:|
| Device signature (`ecrecover`) | 3,216 | ~0% |
| BLAKE3, one compression (12 B leaf hash) | 64,876 | 0.12% |
| BLAKE3 Merkle verify, depth 25 | 1,646,827 | 3.2% |
| keccak256 Merkle verify, depth 25 (variant) | 75,303 | 0.14% |
| Sketch recomputation, one tile (1024 words) | 966,980 | 1.9% |
| `weightsLeaf` = BLAKE3 over 4,104 B (~69 compressions) | 4,439,420 | 8.5% |
| Opening response, committed leaf, depth 21 | 1,442,506 | 2.8% |
| **Full tile fraud proof** (partials d21 + weights d25 + tile hash + sketch) | **8,618,217** | **16.6%** |

Findings:

- **The normal path costs nothing per solution.** Only the
  `EpochPoRWRoot` and bounded metadata go on chain (one storage write +
  event, ~50–100k gas per epoch per aggregator submission). All numbers
  above are dispute-only, exactly as the spec's stance requires
  ("expensive computation occurs only during disputes").
- **The scheme's irreducible on-chain math — the sketch — is cheap**:
  967k gas (1.9% of a block). It is pure u32 arithmetic and needs no
  precompile.
- **BLAKE3 is the dominant cost**, ~65k gas per compression in clean
  Solidity vs ~2.5k-equivalent for keccak. The 4 KiB tile hash (69
  compressions) alone is 4.4M of the 8.6M fraud-proof total.
- An assembly BLAKE3 (the usual 5–10× for hash ports) would put the full
  fraud proof at roughly 1.5–2.5M gas without any scheme change.

## 2. Optimistic response and challenge cost (measured + on-chain flow)

The challenge lifecycle itself (post challenge → respond → expire) is
storage-light: a challenge record write (~50–100k gas), a response
verification (the opening rows above), and an expiry claim (~100k plus
slash bookkeeping). The binding costs are the response verifications:

- committed-leaf opening, depth 21: **1.44M gas** (2.8% of a block);
- non-inclusion (two adjacent openings): ~**2.9M gas** (5.6%);
- calldata: an opening response is ~864 B ≈ 14k gas; a fraud proof is
  ~5,888 B ≈ 94k gas worst-case (EIP-2028, all-nonzero bytes).

A forced honest response therefore costs the responder roughly
1.5–3M gas plus a transaction — bounded, and compensated by the
challenger's deposit under the pallet-mirrored rules (deposit to the
device owner on a valid answer).

## 3. Proof size and calldata (measured)

| Object | Bytes | Calldata gas (worst case) |
|---|---:|---:|
| Opening response (leaf + depth-21 proof) | 864 | 13,824 |
| Tile fraud proof (4 KiB tile + d21 + d25 proofs + fields) | 5,888 | 94,208 |

Both are far below any practical calldata ceiling; the 4 KiB tile is the
floor for byte-level fraud attribution and is inherent to the scheme.

## 4. Worst-case dispute congestion (derived from measurements)

Per 52M-gas Auto EVM block, reserving 50% headroom for ordinary traffic:

- ~3 full fraud proofs, or
- ~18 committed-opening responses, or
- ~9 non-inclusion responses.

Over a 600-block challenge window (~1 hour at ~6 s domain blocks) at 50%
utilization: ~1,800 fraud proofs or ~10,800 opening responses can clear.
Mass-challenge griefing is therefore priced, not free: forcing one
response costs the attacker a deposit per challenge (returned only on
default), while the window provides three orders of magnitude more
capacity than any honest dispute rate. Residual risk — an attacker
willing to burn deposits to congest the window — is handled by (a)
deposit sizing as a governance parameter, (b) per-device challenge
dedup (one open challenge per `(device, partials_root, tile)`), and
(c) window extension on high utilization if adopted at the contract
layer. With the keccak scheme variant (§6) the same window clears ~7×
more disputes.

## 5. Verifier upgrade and emergency-disable behavior (design, per spec)

- Every claim pins its verifier (`ComponentPin` / `verifier_pin` in the
  spec; on this repo's Substrate reference the pin is the runtime version
  at acceptance). A new verifier version applies only to claims that pin
  it; old claims stay evaluated by the implementation they accepted
  (aigg-spec §12.3).
- **Emergency disable** marks one exact verifier implementation revoked
  *prospectively*: new claim acceptance and reward finalization pause;
  finalized history and user balances are untouched (pilot proposal
  §15.1: a verifier defect "pauses new proof acceptance and related
  rewards without freezing ordinary user withdrawals").
- The escalation path if costs must fall further is the spec's §9.5
  ordering — batching → proof redesign → succinct verifier → optional
  precompile — each entering as a NEW pinned verifier version, never a
  reinterpretation.

## 6. Alternative verifiers and adapters (analysis)

| Alternative | On-chain cost | Trade-off |
|---|---|---|
| **keccak-Merkle scheme variant** | Fraud proof ≈ **1.2M gas** (sketch 967k + keccak paths ~80k + keccak tile hash ~10k + overhead) — 7× cheaper | A NEW `PORW_SCHEME_ID` with new conformance vectors; the Rust side gains a keccak tree mode; blake3 stays in the L1 research scheme. The cheapest fully-EVM-native route. |
| **Assembly BLAKE3** | Fraud proof ≈ 1.5–2.5M gas (est.) | Same scheme id (pure implementation change); higher audit burden for the hash port. |
| **ZK verifier** (SNARK over sketch + blake3 paths) | ~300k–500k gas verify (Groth16/PLONK class) | Off-chain proving infrastructure and latency; circuit for 69 blake3 compressions + 1024-word sketch is moderate; enters as a new pinned verifier per §12.3. Justified only if dispute volume makes 1–8M-gas disputes material. |
| **TEE-vendor / approved attestation adapters** | ~3–10k gas (signature checks) | Different confidence class (`TEE-attested`, spec §4.4) — not a byte-level fraud proof; acceptable only where the deployment's policy accepts that trust class, and orthogonal to pillar-C disputes. |
| **BLAKE3 precompile on Auto EVM** | Fraud proof ≈ ~1.1M gas | The §9.5 last escalation before any new-Domain discussion; needs an Auto EVM runtime change, so it competes with simply adopting the keccak variant. |

**Signature suite** (outside the scheme id): solutions on the EVM
deployment bind with secp256k1/`ecrecover` at 3,216 gas — a non-issue.
The Substrate reference keeps ed25519; both are deployment choices under
the spec's `attestation_scheme` registration.

## 7. Verdict against the gate

The fail-closed rule is: *if no verifier meets the cost and trust
requirements, production PoRW rewards remain disabled.*

**A verifier meets the requirements today, unoptimized.** The direct
Solidity verifier — semantics proven bit-identical to the reference —
verifies the worst dispute object at 8.6M gas (16.6% of one Auto EVM
block), on a path that executes only during disputes, with spam priced by
challenger deposits and congestion capacity three orders of magnitude
above honest dispute rates. Two independent, already-specified routes
(keccak scheme variant; assembly BLAKE3) reduce the worst case to
~1.2–2.5M gas if dispute volume ever warrants it, and a ZK verifier
remains available behind the same pinning discipline.

Recommended for the pilot's months 3–4 decision: adopt the **direct
verifier now** (numbers above), and take the **keccak-Merkle scheme
variant** to the aigg-spec process as the designated cost-reduction step —
it is the only alternative that is simultaneously 7× cheaper, fully
EVM-native, and free of new infrastructure or new trust classes.

## Appendix: environment

- forge 1.5.1-stable, solc 0.8.33, optimizer on (200 runs), via-IR.
- Gas numbers are external-call measurements including calldata copy;
  test harness overhead is excluded via `gasleft()` deltas.
- Tree depths: weights 25 (17M tiles ≈ 70 GB), partials 21 (2M covered
  tiles). Costs scale linearly in depth at ~66k gas (blake3) / ~3k gas
  (keccak) per level.
- Block gas limit derivation: `maximum_domain_block_weight()` =
  65% × 2×10¹² ref-time; `WEIGHT_PER_GAS` = 25,000 ⇒ 52M gas.
