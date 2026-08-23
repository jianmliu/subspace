# Proof of Resident Weights (PoRW) — Design Document

**VRAM-residency consensus via TEE hardware attestation + inference-piggybacked auditing**

Status: research draft v0.4 (2026-08)

> This is the English edition of `proof-of-resident-weights.md`. The Chinese
> original is canonical; when the two diverge, the Chinese text wins.

> v0.2 changes: the "dedicated sweep" mode is removed — auditing rides
> entirely on real inference (a single execution mode); lottery weight
> becomes "unique-coverage tickets × TEE-metered service multiplier (capped
> by the hardware envelope)", i.e. earn-by-serving; MoE is supported
> natively (coverage set = the experts actually activated); the coverage
> bitmap lets replica cross-verification recompute commitments without
> knowing the inference inputs, making v0.2 strictly more verifiable than
> v0.1.

---

## 0. One-sentence overview

Keep Subspace's Proof-of-Archival-Storage skeleton (PoT clock + per-slot
challenge + capacity-weighted lottery + solution-range difficulty
adjustment), and swap the "scarce resource" from uniquely-encoded plots on
SSD to **raw large-model weights proven by TEE to be resident in GPU
VRAM**. Replica uniqueness is no longer solved by cryptographic sealing:
remote attestation of GPU/CPU confidential computing makes "one physical
card" the Sybil-resistant counting unit. Auditing introduces zero extra
VRAM reads — the commitment accumulates on the weight sweeps that inference
decode is already doing, and lottery weight is proportional to the service
actually delivered (earn-by-serving): **running inference is mining**.

## 1. Background and motivation

### 1.1 Subspace as it stands (this repository)

Subspace consensus is three cooperating layers:

1. **Resource = the amount of history stored (a stock)**. Farmers plot
   erasure-coded history into sectors (≤1000 pieces per sector,
   `MAX_PIECES_IN_SECTOR` in `crates/subspace-runtime/src/lib.rs`); win
   probability is proportional to storage.
2. **Auditing**: each slot (1 s, `SLOT_DURATION = 1000`) the PoT challenge
   derives an s-bucket per sector; the farmer reads a few tens of KB and
   checks whether a 32-byte chunk falls within `solution_range`
   (`crates/subspace-farmer-components/src/auditing.rs`).
3. **Preventing compute-for-storage substitution**: plot encoding uses
   Chia-style PoS tables + KZG (`subspace-proof-of-space`), so on-demand
   recomputation is far more expensive than reading disk; PoT (sequential
   AES, `subspace-proof-of-time`) is an unaccelerable hard clock that
   pins the response window shut.

The key property: **every farmer's plot bytes are unique** —
`SectorId::new(public_key_hash, sector_index, history_size)` seeds the
encoding with the farmer's public key. That is the Sybil-resistance
premise: without unique encoding, one physical copy of the data could
answer audits for arbitrarily many identities.

### 1.2 Why change it

AI inference is a bandwidth/VRAM-bound workload. We want the consensus
scarce resource to coincide with AI hardware: **model weights resident in
GPU VRAM**. That raises a fundamental conflict:

- Consensus needs unique replicas (else Sybil) → per-farmer sealed
  encoding;
- Inference needs weights in raw format → sealed bytes cannot be fed to an
  inference framework;
- "Cheaply reversible" sealing doesn't work either: cheap decode ⇒ cheap
  on-demand encode ⇒ one raw copy + a little compute can forge a "unique
  replica" for any identity on the spot.

The impossibility triangle: **inference-usable (fast decode) /
replica-unique (expensive on-demand re-encode) / no extra space** — pick
at most two. Unsolvable in a purely cryptographic frame. This design swaps
out the "cryptographic uniqueness" corner for a TEE.

### 1.3 Design goals

| # | Goal | Measure |
|---|------|---------|
| G1 | The same VRAM bytes are simultaneously consensus collateral and inference weights | No second copy, no sealed encoding |
| G2 | Sybil resistance: rewards proportional to physical hardware, independent of identity count | One card, one registration |
| G3 | Zero extra audit bandwidth: meter only the reads inference actually performs; no dedicated sweep mode | Single execution mode |
| G4 | Earn-by-serving: lottery weight proportional to actual service; partially-activated models (MoE) supported natively | Coverage tickets × service multiplier |
| G5 | Keep Subspace's lottery / difficulty adjustment / PoT skeleton | Minimal consensus-layer changes |
| G6 | Trust dependencies explicit, bounded, governable | Measurement whitelist on chain; TEE compromise bounded by hardware envelope |

**Explicitly accepted trust dependencies**: NVIDIA (GPU device identity
and CC firmware), Intel TDX / AMD SEV-SNP (CVMs), and
governance-whitelisted agent measurements. This is a premise of the
design and is not re-argued below.

## 2. Architecture overview

```
                          ┌────────────────────────────────────┐
                          │  On chain (Substrate runtime)      │
                          │  pallet-subspace (modified)        │
                          │   ├─ DeviceRegistry                │
                          │   ├─ ModelRegistry                 │
                          │   ├─ MeasurementSet (whitelist)    │
                          │   ├─ Lottery check (solution range)│
                          │   └─ Bond & slashing               │
                          └───────────▲────────────────────────┘
                                      │ Solution{sketch, sig, device_id}
        PoT chain (unchanged) ──challenge──►
                          ┌───────────┴────────────────────────┐
                          │  Node-side CVM (TDX/SEV-SNP)       │
                          │  PoRW Agent (measured, open-source, │
                          │              reproducibly built)    │
                          │   ├─ Holds node key (born in CVM)  │
                          │   ├─ Manages weight load/residency │
                          │   ├─ Inference service (ai3-inference) │
                          │   └─ Sketch collection & signing   │
                          └───────────▲────────────────────────┘
                                      │ CC-encrypted PCIe / NVLink
                          ┌───────────┴────────────────────────┐
                          │  GPU (H100/H200/B200, CC mode)     │
                          │   HBM: model weights (raw, Merkleized) │
                          │   Kernel: inference + fused sketch │
                          └────────────────────────────────────┘
```

Four trust pillars — none dispensable, mutually redundant:

1. **Pillar B (device attestation)**: unique device identity + code
   measurement ⇒ Sybil resistance and protocol compliance.
2. **Pillar C (sketch)**: a challenge-randomized, word-granular linear
   digest ⇒ proves the bytes are genuinely **resident** and read; the
   second line of defense when the TEE is compromised (timing and replica
   cross-verification still hold).
3. **Pillar D (inference attestation)**: a per-request attestation quote ⇒
   proves **output authenticity**, `O = M_registered(I)` produced inside
   an attested TEE (§4.7).
4. **PoT (unchanged)**: an unaccelerable clock ⇒ the sketch's response
   deadline has an objective basis.

## 3. Pillar B: device attestation and registration

### 3.1 Hardware basis

- **GPU**: NVIDIA supports confidential computing (CC mode) from Hopper
  on. A unique identity key is fused into the device; the certificate
  chain anchors to NVIDIA's root CA. The attestation report covers
  VBIOS/firmware/driver measurements and CC state, obtained over an SPDM
  session. Blackwell improves CC overhead and supports multi-GPU CC
  domains over protected NVLink.
- **CPU/VM**: GPU CC requires the host to be a confidential VM (Intel TDX
  or AMD SEV-SNP). The CVM launch measurement covers the PoRW Agent
  image.
- **Limitation**: consumer cards (RTX 40/50) have no CC. The B+C route is
  datacenter-card exclusive; consumer cards participate via the fallback
  path (§8.3).

### 3.2 Registration flow (one card, one identity)

```
1. Agent generates a node keypair (sk, pk) inside the CVM; sk never leaves.
2. Agent collects: CVM attestation (incl. agent measurement, pk binding)
                 + GPU attestation report (incl. device_id, CC state).
3. On chain, register_device(evidence):
   a. Verify the NVIDIA / Intel / AMD certificate chains and report signatures.
   b. Check agent measurement ∈ MeasurementSet (governance-maintained whitelist).
   c. Check device_id is unregistered (no double registration of one card).
   d. Require a bond (slashing collateral).
   e. Write DeviceRegistry: device_id → (pk, vram_capacity, epoch).
4. Periodic re-attestation (per epoch); expiry drops the device from the
   lottery set automatically.
```

Certificate-chain verification is heavy and need not run fully on chain:
optimistic verification + a fraud-proof window, or a lightweight
verification committee pre-validating with the chain checking only the
committee's aggregate signature. The prototype simply verifies natively.

### 3.3 What attestation provides — and what it doesn't

| Provides | Does not provide (pillar C fills in) |
|----------|--------------------------------------|
| This is a real, unique physical GPU | The weights are in HBM right now |
| The running agent code is a whitelisted build | The weights are genuinely read (not merely claimed) |
| The node key is held by the correct code | Timeliness of the response |
| PCIe/VRAM contents invisible to the host | That the TEE itself is uncompromised |

Because the agent code is measured and honest, cheats like "keep the
weights in host RAM and page them in" are **excluded by code
construction** while the TEE holds — a cheater cannot run modified agent
code (the measurement won't match). The sketch deadline of pillar C is
defense-in-depth for the TEE-compromised case.

## 4. Pillar C: inference-following residency audit and service metering

### 4.1 Core observation

LLM decode sweeps the entire weights out of HBM once per generated token
(that is exactly why inference is bandwidth-bound). The behavior consensus
wants to force — "high-intensity reads of resident data every slot" —
**inference is already doing**. The audit only needs to accumulate a
commitment on this already-paid-for data stream.

### 4.2 Sketch definition (integer domain, fully decoupled from inference numerics)

Partition the weights into 4 KiB tiles: `W = {T_0, T_1, ..., T_{n-1}}`,
with Merkle root `R_W` registered in the ModelRegistry. Each slot, PoT
yields the global challenge `c_t` (reusing the existing
`global_challenge` derivation path). Let `C_t ⊆ {0..n-1}` be the set of
tiles inference **actually read** this slot (the coverage set; ≈ the full
set under busy dense-model decode, = the routed experts + shared layers
under MoE). Define:

```
r_i      = PRF(c_t, device_id, i)              // per-tile seed
c_{i,j}  = PRF'(r_i, j)                        // per-WORD coefficient, generated in-register
sketch_t = Σ_{i ∈ C_t} Σ_j  c_{i,j} ⊙ w_{i,j}  // multiply-add mod 2^32/2^64
```

> **Coefficients must be word-granular and fresh every slot** (v0.2.1 fix; see
> `porw-p1-feasibility.md` §3): with one coefficient shared across a tile,
> the sketch collapses to `r_i · L(T_i)` (L a fixed linear functional), and
> a cheater can store the 4-byte L value per tile and pass all future
> audits (1024× compression). The PoC demonstrates this attack executably;
> with word-granular PRF coefficients (murmur3 fmix32, ~6 integer
> instructions per word) the attack fails.

Key points:

- **Accumulate only on reads inference already performs.** The
  multiply-add fuses into the decode kernel's K-loop (the weight tile is
  already in registers/SMEM at that moment; note it must be the mainloop,
  not the epilogue — a GEMM epilogue sees only the output C tile, never
  the weight byte stream). No extra HBM reads are issued; whatever the
  card is doing is what gets metered. Actual deployment is an **S1+S2
  hybrid** (`porw-p1-feasibility.md` §2): fusion surgery is only necessary
  for coverage-dependent MoE Triton kernels; dense/closed-source (cuBLAS)
  paths always have full coverage and are backstopped, with zero semantic
  loss, by a standalone once-per-slot sweep (~2.4% bandwidth tax).
- **Integer multiply-add over the raw bytes**, not floating point — fully
  deterministic, reproducible across cards and drivers, independent of the
  inference framework's numerics.
- Coefficients rotate every slot ⇒ `sketch_t` cannot be assembled from old
  slots' results; every tile in `C_t` must have been genuinely read **within
  this slot** at least once (a Freivalds-style random linear sketch:
  cheaters storing low-precision/low-rank approximations necessarily get
  it wrong).
- `device_id` is mixed into the PRF ⇒ different cards produce different
  sketches; forwarding another card's sketch is useless.

### 4.3 The cryptographic hard boundary: a sketch can only prove "at least once"

A fact that must be faced squarely: coefficients are slot-granular, so a
tile read once versus k times within a slot contributes to the sketch in a
way derivable from one read (multiply by k). Therefore **the physical
quantity a sketch can prove is capped at "at least one genuine read per
tile per slot"** — unique coverage `C_t`. Read counts beyond one sweep,
token counts, and other "labor intensity" cannot be cryptographically
proven by the sketch.

Lottery weight therefore splits into two factors, each backed by the
mechanism that can actually carry it:

| Factor | Meaning | Backing mechanism |
|--------|---------|-------------------|
| **Coverage tickets** `|C_t|` (in bytes) | These weight bytes are resident and were genuinely read this slot | Sketch (cryptographic, cross-verifiable) |
| **Service multiplier** `m_t` | Service actually delivered this slot (§4.4) | TEE metering + hardware-envelope cap (§4.6) |

### 4.4 Earn-by-serving: service metering

The agent (measured, trusted code) counts the slot's service inside the
CVM, truthfully:

```
m_t = (decode steps completed this slot × weight bytes actually read per step) / |C_t| bytes
```

i.e. "how many full sweeps of the coverage set the weights underwent".
Properties:

- **Dense model, busy**: one full-weight sweep per token ⇒ m_t ≈ token
  steps in the slot (multiple requests in a batch share one sweep; m_t
  counts bandwidth sweeps, not tokens — see §10 on whether to introduce
  token weighting).
- **MoE**: cold experts outside C_t earn no tickets; hot experts earn
  what they read. No make-up sweep is needed — earn-by-serving semantics
  hold natively.
- **Idle cards**: no inference, no tickets. A rational farmer will run
  self-generated load (batch=1 decode tight loop, physically converging
  to v0.1's sweep kernel). The protocol does not distinguish real from
  self-generated load, and does not need to: this forms the GPU revenue
  floor, while the premium for real service is carried by the inference
  fee market (§5.1). The protocol thereby drops one execution mode.

### 4.5 Timing and lottery (reusing the Subspace skeleton)

```
slot t:  PoT ──► c_t
         │
         ├─ GPU: decode proceeds as usual; sketch_t and coverage bitmap C_t
         │       accumulate on the side
         ├─ Agent: tickets = hash_expand(sketch_t, |C_t| × m_t)   // stream length ∝ tickets
         │         check is_within_solution_range(...) per 32B chunk  // existing logic
         └─ Win ⇒ Solution {
               device_id, model_id, slot,
               sketch_t, coverage_bitmap, partials_root, m_t, chunk_index,
               //        ^ Merkle root of per-tile sketch values (the kernel
               //          already produces per-tile partials) — enables
               //          single-tile spot checks (§4.6)
               sig = Sign_sk(...)            // node key, attestation-backed
            } ──► author block
```

- **Work weighting**: the chunk stream length ∝ `|C_t| × m_t` (coverage
  bytes × sweep count = weight bytes genuinely streamed this slot),
  matching Subspace's "one ticket per chunk" semantics; the difficulty
  adjustment (pallet-subspace solution-range era logic) is reused as-is,
  so network difficulty automatically tracks the network's real inference
  throughput.
- **Response deadline**: `BLOCK_AUTHORING_DELAY` is kept, as
  defense-in-depth for TEE compromise (PCIe/network streaming cheats
  can't make the deadline, §6.2).

### 4.6 On-chain verification

Fast path (every block, ~a few signature verifications):

1. `device_id ∈ DeviceRegistry` and attestation unexpired;
2. `sig` matches the registered `pk`;
3. `model_id ∈ ModelRegistry` and the device has announced this model;
4. **Hardware envelope check**: `|C_t| × m_t ≤ registered HBM bandwidth
   of this card model × slot duration`. This puts a physical ceiling on
   the TEE meter: even with the agent fully compromised, a single card
   cannot inflate its tickets beyond a constant multiple of its true
   bandwidth — the failure mode of TEE trust is **bounded inflation**,
   not unbounded forgery;
5. Chunk within solution range (same as existing `subspace-verification`
   logic).

Deep path (spot checks + fraud proofs) — two tiers, both anchored at
`partials_root` (the Merkle root of per-tile sketch values), which makes
fraud proofs **single-tile-granular**: fetch one tile's bytes → recompute
`s_tile` → compare against the committed value on the Merkle path —
O(4 KiB) plus one path, not a whole-model recomputation:

- **Fast catch = VRAM replica cross-verification**: each epoch, other
  registered devices holding the same model are randomly assigned to spot
  check a few tiles of the claimed `C_t` (the weights are already in
  their own HBM, so a spot check is nearly free; even a full
  recomputation is ~24 ms). **The verifier does not need the target's
  inference inputs** (the sketch depends only on coefficients, tile
  bytes, and the coverage set — not on activations). Mismatch ⇒ fraud
  proof ⇒ bond slashed, device_id revoked, reporter takes the bounty.
  The honest-majority assumption is only needed within "the replica set
  of one model" — hot models have many replicas, so it holds naturally,
  and the vast majority of ticket weight is covered by this tier.
- **Final arbitration = storage track** (§5.6 risk 2): for single-replica
  models, or when cross-verification ends in a stand-off, the DSN shards
  are the canonical bytes for `R_W`. Any PoAS farmer can retrieve the
  piece containing the disputed tile (~MiB), recompute, and submit a
  fraud proof — the final ruling depends on no VRAM replica existing, and
  it also adjudicates misbehavior by the verifiers themselves. The
  challenge-period length is parameterized by DSN retrieval latency.

#### 4.6.1 Epoch scheduling of replica cross-verification (implemented)

Cross-verification is possible because of one asymmetry: the sketch's
slot seed is `derive_slot_seed(global_challenge, device_id)` — **public**
and **per-device**. Per-device guarantees two replicas of the same model
produce different sketches for the same tile (required for replica
uniqueness); public means **any replica holding the model's true bytes
can recompute any peer's committed per-tile values using the target's
seed**. Hence "only replicas can audit replicas, and any replica can
audit any same-model peer" — scheduling is just rostering this existing
fraud-proof path.

**Two-epoch pipeline + reward escrow**:

```
epoch e      Commit period: author blocks normally, accumulate partials_root commitments
e boundary   Fix beacon B_e (epoch-boundary randomness) → whole network derives assignments locally
epoch e+1    Audit window: assigned replicas recompute & compare e's commitments;
             mismatch ⇒ TileFraudProof
e+2 settle   e's block rewards are only now released (minted) from escrow
```

The crux is **reward escrow** (`EscrowedRewards`): block rewards earned in
epoch e are **not minted**; they are held until the settle at e+2, after
the audit window passed without fraud, and only then minted to the device
owner. Fraud proven within the window ⇒ the escrow entry is simply deleted
(it never entered supply) + the bond goes to the reporter + the device is
revoked. Deregistration is refused while escrow is pending
(`EscrowPending`), closing the "walk away with pay still under audit"
exit; an exit delay on the bond itself (a pending-exit state machine)
remains future work.

**Assignment is a pure function, with near-zero on-chain state**
(`subspace-proof-of-residency`):

- `audit_beacon(epoch, entropy)`: the epoch beacon. The entropy must be
  unknowable before the epoch boundary (production: PoT-derived
  randomness; the pallet currently uses the boundary block's parent hash
  as a placeholder, documented as grindable by the boundary-block author
  within its solution set and to be replaced before launch) — otherwise a
  liar could predict the sampled tiles and keep true values ready for
  just those;
- `select_auditors(B, model, target, replicas, k)`: for each target, take
  the k lowest rank-hashes within the replica set (excluding the target
  itself); fan-out k ≈ 3;
- `audit_tile_sample(B, model, target, auditor, n_tiles, t)`: each
  (auditor, target) pair independently samples t distinct tiles; different
  auditors' samples differ, widening combined coverage.

The chain stores only the beacon (`AuditBeaconValue`, written at each
epoch-boundary settlement). No assignment table, no acks, no pay for
clean audits — the schedule exists to **direct honest effort and bound its
bandwidth**; all enforcement lives in the permissionless fraud-proof path
(report-for-bounty).

**Sample size and cost**: with forgery fraction f and total sample N, the
miss rate is (1−f)^N. At f = 1%, N ≈ 690 reaches 99.9% detection; at
4 KiB per tile against TB/s-class HBM, recomputation cost is negligible —
each replica audits on average k ≈ 3 peers per epoch, MB-scale traffic.
The real constraint is not compute but **opening availability**: the
auditor needs the target's Merkle opening for a committed tile (the
partials tree is built in coverage order, so the fraud proof carries
`partials_index` to locate the leaf — the leaf hash itself binds
`tile_idx`, so lying about the position merely fails verification and can
never shift blame across tiles). The target must serve openings on
request within the audit window; refusal means it cannot substantiate its
own commitment and is treated as unavailability — a small
data-availability sub-problem.

**Single-replica models** (`ReplicaCount == 1`) have no peer auditors:
they fall back to storage-track arbitration (the final-arbitration path
above) — precisely what `min_replicas` signifies as a service parameter.
For replicas joining/leaving mid-stream, only commitments made while both
parties were in the replica set are audited, bounded by `registered_at` /
announce blocks.

Implementation map: the assignment pure functions and the
`partials_index` fix live in `subspace-proof-of-residency`; escrow,
beacon, forfeiture, and the deregistration gate live in
`pallet-porw-registry` (`note_block_reward` / `EscrowedRewards` /
`AuditBeaconValue` / `forfeit_escrow`); the auditor side (`audit_duties`
/ `cross_check`, producing ready-to-submit fraud proofs) lives in
`porw-agent::audit`; the end-to-end (a lying replica caught by an
auditor → bond slashed + escrow clawed back) is
`cross_audit_catches_a_lying_replica_end_to_end` in `porw-devnet`.

**The stack's boundary (honest restatement)**: everything above catches
"residency and coverage forgery". In-envelope inflation of `m_t` is
cryptographically uncatchable (§4.3 hard boundary); its defenses are TEE
measurement + envelope cap + statistical anomaly detection.

But this soft spot's economic exposure is much smaller than it looks, and
it shrinks over time:

- **`m_t` affects only block rewards, never service revenue.** Inference
  fees settle per request, are endorsed per quote by pillar D (§4.7), and
  outputs are verified by the user — inflating `m_t` earns not a cent of
  service fees; farming service fees requires genuinely burning fees
  (§5.1), which is payment, not cheating.
- **Cheating gains are precisely bounded to "saving electricity".** The
  busy farmer (real load), the idle honest farmer (junk decode saturating
  bandwidth at real power cost), and the `m_t` liar are all capped by the
  same envelope — lying buys no extra tickets, only the power saved by
  not running the sweep (~hundreds of watts per card), while getting
  caught costs the bond plus revocation: naturally negative expected
  value.
- **The soft spot's weight decays with the revenue mix**: `m_t` only acts
  on the inflation subsidy; as the network matures and (cryptographically
  hard) fee revenue grows (§5.2 subsidy→fee transition), the sole soft
  spot's share of total income declines monotonically.
- `m_t` itself cannot be externally recomputed (it counts real load); its
  defense is the fast-path envelope cap + attestation measurement +
  statistical anomaly detection (a device pinned at its envelope ceiling
  with no matching inference revenue invites governance investigation).

### 4.7 Pillar D: inference attestation and the end-to-end agent lifecycle

Today's TEE-ML ecosystem (Phala, Atoma, NVIDIA CC, etc.) can already
issue an attestation quote **per inference request**, binding
`(input_hash, output_hash, model_measurement, cvm_measurement)` to the
device key. Adopting it as the fourth pillar makes the fact that
`O = M_registered(I)` was produced inside an attested TEE **verifiable**.

**Why both C and D are needed — not redundant** (a key clarification):

| | Pillar C: sketch | Pillar D: inference attestation |
|---|---|---|
| Proves | Weights are **resident** and read (capacity) | **Output authenticity** `O=M(I)` (usefulness) |
| When | Challenge-driven, **every slot**, even with no requests | Request-driven, **only when requests exist** |
| Serves | Consensus lottery (scarce-resource metering) | Fee market (verifiable receipts) |

Empty slots: C only (self-generated load / sweep) → the revenue floor
(§4.4, §5.1). Slots with requests: C+D → fee-market settlement and
fee-burn weighting (§5.1) gain a verifiable basis. D cannot prove
"X bytes continuously resident right now" — the continuous capacity
commitment the lottery needs — so D cannot replace C. Both reuse **the
same `R_W` root** for model identity, reinforcing each other.

**D sidesteps the determinism problem**: the hardest part of verifying
inference is floating-point irreproducibility across nodes (ZKML is
expensive; TOPLOC-style recomputation is brittle). TEE attestation needs
no recomputation and no bit-identity — it only attests "this attested CVM
ran the measured code and produced O from I"; the trust is in the TEE,
not in reproduction. This is exactly why, once this design accepts a TEE
trust root, proof-of-useful-work goes from "hard" to "shippable".

**Closing the loop into an end-to-end agent lifecycle**, every step
attested:

| Lifecycle step | Carried by | Proof |
|---|---|---|
| Perceive (read memory) | Storage track (§5.4) | Permanence + envelope encryption; plaintext only opens inside an attested CVM |
| Think (inference on resident weights) | VRAM track | Residency (C: sketch) + device (B: attestation) |
| Act (produce output) | Inference attestation | Pillar D: `O=M(I)` inside an attested TEE |
| Remember (write memory) | Storage track | Permanence + memory root on chain |
| Identity / keys | CVM | Node key generated in-CVM, never leaves the TEE |

The four pillars share one TEE trust root and cover the agent's full
perceive→think→act→remember loop. Concretely, pillar D interfaces with
`ai3-inference` (inference layer), memory with `aigg-memory`, settlement
and fee burning with `aigg-facilitator` / `aigg-wallet` — PoRW consensus
plus the four-pillar proofs form a chain-native, verifiable, immortal
agent lifecycle substrate.

## 5. Economics and governance

### 5.1 Self-generated load (wash trading): accept it, and price it

Once lottery weight follows actual inference, the question must be
answered: what if farmers send themselves requests? This design's stance
is **do not block it at the consensus layer**, because:

- Self-generated load must also genuinely consume HBM bandwidth (sketch +
  envelope guarantee this); it cannot buy tickets beyond the physical
  ceiling — this is "junk mining = bandwidth PoW", the same thing as idle
  sweeping, a revenue floor rather than a loophole;
- Distinguishing "real users" from "yourself" is undecidable in a
  permissionless network;
- The real stratification belongs on the fee side: **consensus rewards**
  pay for physical resources (residency + bandwidth; junk and real load
  at the same price), **inference fees** pay for usefulness (only real
  users pay). If consensus rewards should also tilt toward real service,
  fee-burn weighting can be introduced (fees burned by paying requests
  boost tickets proportionally — self-farming then requires burning real
  money, giving wash trading a cost), as a governance-tunable parameter.
  That the "paying request was real inference" is endorsed by pillar D's
  attestation (§4.7), no longer trust in agent self-reporting.

### 5.2 Block rewards as the native currency for inference fees

Block rewards go precisely to the resource that provides inference
capacity (GPUs with resident weights), so denominating AI agents'
inference fees in the same token closes a loop rather than forcing one —
this is the standard shape of resource networks (Filecoin/Render). The
flywheel:

```
Inflation rewards → subsidize standby capacity (revenue floor, §5.1)
Agents pay for inference → fees partly burned + partly to providers
Demand ↑ → burn ↑ → net inflation ↓ → token appreciates → GPU revenue ↑
        → capacity ↑ → service better/cheaper
```

Three reasons it holds:

1. **Endogenous settlement**: agents (especially on-chain agents) need
   programmable, streamable, per-token micropayment settlement — a native
   token satisfies this naturally, and it composes with fee-burn
   weighting (§5.1) without exchange friction: one asset is
   simultaneously the reward, the fee, and the anti-wash burn target.
2. **Demand-driven security**: fee burning converts inference demand
   directly into consensus security budget (EIP-1559-style); the more
   useful the network, the more secure it is.
3. **Supply-side bootstrapping**: early on, with no demand, inflation
   keeps capacity online (standby subsidy); once demand arrives, the
   revenue mix naturally shifts from inflation to fees (Bitcoin's
   subsidy→fee transition curve, but driven here by real service
   revenue).

Three design points that must be faced:

- **Denomination ≠ settlement.** Inference has real fiat costs (power,
  depreciation); token volatility can crush provider margins. Quotes
  should anchor to compute cost (fiat or compute units), with the token
  only as the settlement asset (oracle conversion); using a volatile
  asset directly as the unit of account is the classic death of resource
  networks.
- **Death spiral**: token falls → GPUs exit → capacity drops → service
  degrades → demand falls further. Mitigations: registration-bond exit
  delay (already in this design), long-residency reward boosts, a
  protocol treasury buying capacity in downturns.
- **Pure-inflation risk**: with no demand the token has mining output and
  no sink. Hedge: besides burning, require agents/aggregators to stake
  for service quota and priority (work-token model), giving the token a
  usage-proportional locked demand.

### 5.3 Model listing: staked listing + demand-following weight

> ★ Demand-following weight is implemented in `pallet-porw-registry`: per
> model `demand_ema` (an EMA of burned fees) + `floor_weight`;
> `record_inference_fee` **really burns** the caller's tokens before
> accumulating the demand signal (wash trading costs real money, §5.1);
> `settle_epoch` folds per epoch into `effective weight =
> clamp(demand_ema / fee-unit, floor, cap)`; `model_reward_weight` is read
> by the reward layer. The EMA smoothing provides §10.1's hysteresis.
> Wired into the runtime; covered by tests.

- **ModelRegistry**: `model_id → (R_W merkle root, size, version,
  min_replicas, reward_weight)`.
- **Three-phase listing**:
  1. Early governance whitelist — concentrate limited VRAM on few models
     to reach replica counts (`min_replicas` is a safety parameter:
     replica cross-verification §4.6 relies on the same-model replica
     set).
  2. Mature-phase open listing: anyone can propose with stake + listing
     fee + license declaration; governance retains the power to remove
     reward weight (ending the subsidy ≠ erasing the data, see §5.4).
  3. **Reward weight follows demand**: each model's weight ∝ a moving
     average of its recent burned inference fees. Unused → weight decays
     → farmers free VRAM for hot models; wash traders must genuinely burn
     fees (= paying to reserve capacity: market behavior, not attack,
     consistent with §5.1); the inflation subsidy flows precisely to
     verified usefulness.
- **Cold-start subsidy**: new model → no demand → no weight → nobody
  loads it. Broken by converting part of the listing stake into bootstrap
  weight for the first N epochs (the proposer buys initial capacity;
  sustained community staking can keep long-tail models alive).
- The expected equilibrium is a power law: a few open-source flagships
  hold most replicas; the long tail lives on stake. The reward-weight
  curve is the network's capacity planner.
- **Weight distribution**: the weight bytes are uploaded into the
  Subspace DSN (the existing archiver/gateway path works as-is; the
  incentive comes from the storage track, §5.4); new nodes pull from the
  DSN and verify against `R_W`.

### 5.4 Data permanence: dual-track consensus (v0.3 revision)

**The PoRW VRAM tier provides no permanence and should not** — it is an
economically-anchored hot cache: when a model's reward weight decays to
zero, farmers evict it from VRAM. Permanence must be promised by the
DSN/archival layer, and this exposed a structural gap in v0.2: a
single-track PoRW cut out the PoAS lottery that used to pay the storage
layer, leaving "weights uploaded into history" without an incentive
source. The revision is **dual-track consensus**:

| Track | Reward share | Mechanism | What it buys |
|-------|-------------|-----------|--------------|
| VRAM track (PoRW) | X% | This design (§3–4) | Inference capacity for hot models |
| Storage track (PoAS) | Y% | Original Subspace (uniquely-encoded plots + s-bucket audits, SSD) | Permanence of all history |

X:Y is a governance parameter. The dual track resolves two leftovers at
once: history and retired weights live on cheap SSD (dissolving "VRAM is
too expensive to hold history"), and chiapos/KZG/plotting survive
unchanged on the storage track — the "no longer needed" row in §7 refers
only to the VRAM track not needing sealed encoding.

**Permanence semantics (whitepaper-grade wording)**: all ever-listed
model weights enter archival history, erasure-coded shards scattered
across all SSD farmers; a model retired from the VRAM tier still has its
bytes in the archive and can be re-listed when demand returns. This is an
**economic-probabilistic** guarantee (storage-track inflation persists +
storage cost falls faster than history grows ⇒ replica counts hold), not
a cryptographic absolute — data lives as long as the chain lives; do not
promise "absolute permanence" externally.

**The permanence-vs-delisting tension dissolves in the layering**:
governance can only touch reward weight (stop subsidizing a model's VRAM
replicas); it cannot touch the archival layer — DSN shards are
content-agnostic, and a single farmer holds uninterpretable fragments.
Censorship resistance stays in the archive; compliance acts on the
subsidy. Retired model versions have long-term value for reproducibility
research and model genealogy — "a national library of models" is one
legitimizing narrative for storage-track inflation.

**Narrative layering: memory to the storage track, thinking to the VRAM
track.** An agent = memory (data) + thinking (inference over shared
models), which maps onto the two tracks exactly:

- The storage track carries **permanent agent memory** (conversation
  history, experience, knowledge bases, embeddings, LoRA deltas; memory
  root on chain, content erasure-coded into the archive) — the agent's
  "soul": written once, paid once, persists at near-zero marginal cost;
- The VRAM track carries the **compute market** — the agent's "brain",
  rented per token.

This yields a property centralized platforms cannot offer: **agents can
pause and resurrect** — state is independent of any provider; after years
of dormancy, any VRAM-track node can pull the memory + load the
registered model and revive the agent in place. "Permanent memory +
on-demand compute = an immortal on-chain agent" is the headline
narrative; the model library becomes supporting infrastructure.
Economically, an agent's token consumption widens from inference fees to
a "cost of living" (writing memory + running inference), with each
track's fee flow closing its own loop. The technical boundary lands
naturally in the right place: the sketch counts tickets only over
registered weights; memory/KV/RAG reads earn nothing (§6.1) — the two
tracks' audit targets have zero overlap. **Memory privacy**: memory
ciphertext on chain (envelope-encrypted under the holder's key),
plaintext opened only inside an attestation-passing CVM — the privacy
narrative and the compute narrative share one TEE trust root.

### 5.5 Model lifecycle

- **Version updates**: a new version = a new `model_id`. The old version
  gets a deprecation epoch with double-counted rewards, then leaves the
  lottery set. No re-plot cost (there is no sealed encoding); switching
  cost = download + load into VRAM.
- **Multi-model / sharding**: one card may register several models
  (7B+13B); a multi-GPU NVLink CC domain may register a tensor-parallel
  large model — sketches accumulate per shard under an aggregate
  signature.

### 5.6 Hot-tier eviction: three risks for non-hot models, each with a reinforcement

The VRAM tier is a demand-driven hot cache (§5.4); non-hot models are not
guaranteed a VRAM replica. No data is lost (storage track), but three
real risks remain, each paired with a mechanism:

**Clarification first: the three storage forms of one set of weights.**
The storage track's uniform random sampling (farmers cannot choose their
shards) does not conflict with inference needing complete weights —
because the inference hot path never reads the storage track directly;
the bridge is "DSN gather-and-rebuild + full local cache":

| Form | Content | Who chooses | Incentive | Purpose |
|------|---------|-------------|-----------|---------|
| Raw weights in VRAM | Complete, self-chosen model | Farmer (demand-following, §5.3) | Pillar-C lottery + inference fees | Inference hot path |
| Full local SSD copy | Complete, raw format | Farmer hoards freely | Indirect: seconds-level reload + resurrection bounties (warm tier below) | VRAM load source |
| DSN unique-encoded shards | Uniform sample, unchoosable | Protocol-assigned | PoAS storage-track rewards | Permanence + final arbitration (risk 2) |

Listing flow: DSN gather (reassemble full weights from the erasure
threshold, minutes) → verify per tile against the `R_W` Merkle root →
keep a full local SSD copy → load into VRAM. All subsequent reloads use
the local copy (seconds); gather recurs only on first load or copy loss.
The full local copy is not a uniquely-encoded plot and earns no consensus
tickets — it is pure utility storage, incentivized indirectly by reload
speed and resurrection bounties.

**Risk 1: cold-start latency (service availability).** An evicted model
gets a request and must "resurrect": without a local copy, DSN-gathering
tens of GB + loading VRAM takes minutes. **Reinforcement: tiered service
levels + a pre-warming market.** Model tier is queryable and latency
predictable: hot (≥ min_replicas, instant) / warm (farmers' speculative
local-SSD caches, seconds–minutes) / cold (DSN only, minutes–hours).
Paying requests may attach a **resurrection bounty**: the first node to
load the model and deliver attested service claims it — cold starts
become market behavior, and farmers have an incentive to hoard "likely to
resurrect" retired models on cheap local SSD to race for bounties; the
warm tier emerges by itself.

**Risk 2: verification failure for near-singleton models (safety-grade,
most important).** §4.6's replica cross-verification needs an honest
majority within a model's replica set; a model down to 1–2 replicas has
no one to verify its sketches, and the TEE-compromise defense-in-depth
fails for singletons. **Reinforcement: the storage track as the court of
arbitration.** The sketch depends only on (challenge coefficients ×
weight bytes × coverage set), and the storage track holds all the weight
bytes — **any storage-track farmer can recompute any model's sketch for
any slot from the DSN** and submit a fraud proof: slow (DSN reads) but
entirely feasible within the challenge period. Final adjudication of
sketch disputes therefore **depends on no VRAM replica existing**:
cross-verification is merely the "fast catch"; the storage track is the
court of final appeal. `min_replicas` demotes from a safety parameter to
a service parameter — the second structural synergy of the dual-track
design (the first being permanence, §5.4).

**Risk 3: unstable demand-feedback loop.** Weight-follows-demand is a
positive feedback: demand falls → weight falls → eviction → worse service
→ demand falls further (single-model death spiral); and in reverse
(burst demand → no replicas → cannot serve → demand evaporates).
**Reinforcement: floor weight + hysteresis.** A model still in the
registered set receives `max(demand-following weight, floor)`, with the
floor paid by the lister's ongoing stake (residency rent) — keeping a
model in the hot tier costs somebody money, demand-side or
proposer-side, at a posted price; weight adjustment gets a hysteresis
window (EMA + eviction cooldown) to damp oscillation. Agents can make
SLA decisions or pre-pay warming based on the tier.

## 6. Threat model

### 6.1 TEE intact

| Attack | Defense |
|--------|---------|
| Sybil (one card, many identities) | Unique device_id registration |
| Claim residency but keep weights in host RAM/SSD | Agent code is measured and won't cooperate; sketch deadline regardless |
| Forge the sketch | Agent won't sign it; coverage bitmap + replica cross-verification backstop |
| Inflate service m_t | Measured agent won't; envelope cap backstops |
| Self-generated wash load | Not treated as an attack (§5.1): consumes real bandwidth; forms the revenue floor |
| Forward another device's sketch | device_id mixed into the PRF |
| Store compressed/low-rank weights | The sketch is a word-granular random linear combination over raw bytes; approximations necessarily err |
| Long-context junk requests | KV cache is not registered weights; attention reads earn no tickets |
| Instant card-rental attack | Registration needs bond + attestation + epoch activation delay (playing the role plotting slowness used to); a rented card must also first obtain a weight replica (DSN download is bandwidth-bound) |

### 6.2 TEE compromised (defense in depth)

Assume the attacker can forge agent behavior but not the device
certificate chain:

- Inflated m_t ⇒ capped by the hardware envelope; inflation bounded (≤ the
  card's bandwidth ceiling over actual usage);
- Forged sketch values ⇒ caught by replica cross-verification, slashed
  (economic defense);
- Weights in host RAM with sketch streamed ⇒ PCIe 5.0 x16 is only
  64 GB/s: 80 GB of weights takes 1.25 s > the response deadline, while
  direct HBM reads take 24 ms — a 50× timing gap (physical defense);
- Weights on a remote host ⇒ network bandwidth is even more hopeless
  (100 Gbps ≈ 12.5 GB/s);
- The device certificate chain itself forged (NVIDIA root-CA-leak grade)
  ⇒ systemic risk: governance revokes the affected measurement versions
  and transitions to patched firmware (§8.3 fallback path).

### 6.3 Economic security

Attack cost = acquiring a majority of "registered, activation-delay-aged
VRAM bytes". Since that requires genuine CC hardware + bonds + delays, it
is equivalent to "buying/controlling a majority of the H100 VRAM
participating in consensus". As in PoS, bond slashing gives registered
capacity a direct economic loss for misbehavior; as in Subspace, PoT
prevents long-range attacks via fast-computing future challenges.

### 6.4 The three roles of stake: gate preserved, bounded modulation, fidelity bond

Guiding principle: **capacity (residency) is always the hard gate; stake
never replaces it**. Within that constraint, stake has three legitimate
roles. Distinguishing them keeps PoRW from becoming plain PoS.

**Must reject: pure PoS (tickets ∝ stake, no capacity gate)** — the rich
would author blocks without residency and the VRAM proof would be
decorative. This is the only red line.

**Adopted: capacity gate + bounded sqrt stake modulation** (the mechanism
of this repository's `PoS` branch). `scale_solution_range` scales the
range in the win predicate `solution_distance ≤ scaled_range/2` by
`sqrt(stake)/sqrt(MaxVotingBalance)`:

```
P(win) ∝ |C_t| (residency) × m_t (service) × sqrt(effective stake)/sqrt(cap)
          └──── capacity gate; zero ⇒ out ────┘  └─ bounded economic alignment ─┘
```

Three design choices of that branch hold the red line exactly and are
adopted as-is:

- **Capacity remains the gate**: zero capacity = zero solution candidates
  = zero weight; no amount of stake helps ⇒ the VRAM residency proof is
  not decorative;
- **`sqrt` sublinearity + `MaxVotingBalance` hard cap**: doubling
  influence takes 4× the stake, and a whale beyond the cap ties with
  whoever just reached it ⇒ strongly anti-plutocratic; stake cannot buy
  unbounded influence;
- **Zero total stake → degrades to pure capacity consensus**
  (`voting_stake_weight` returns max_weight) ⇒ staking is an overlay, not
  a network prerequisite, and can be enabled smoothly.

**Fidelity bond — unifiable with the modulation stake above.** PoRW swaps
cryptographic uniqueness for a TEE, introducing two trusted assertions
that need a slashable deterrent: (1) the service multiplier `m_t` (§4.4)
is agent-reported and inflatable within the envelope; (2) TEE-compromise
sketch forgery (§6.2) needs a slashing target. One stake serves both as
slashing collateral and (through sqrt) as a reward boost — bond and
modulation unified, one stone two birds.

Two synergies and one tension:

- **Synergy 1**: stake incidentally gives the token usage-proportional
  locked demand, hedging §5.2's "pure inflation risk".
- **Synergy 2**: `domain operator` staking (Subspace's execution layer
  already has it) is orthogonal and stacks directly.
- **Tension**: PoRW wants rewards to follow useful service (`m_t`, fee
  burn); stake modulation makes rewards also (sublinearly) follow
  capital. `sqrt` + `cap` is precisely the knob that keeps capital from
  overpowering residency and service — a governance trade-off;
  `MaxVotingBalance` / `MinVotingBalance` / the sqrt curvature are
  tunable parameters.

**Long-range / nothing-at-stake** is absorbed by the inherited PoT hard
clock, as in Subspace — no reliance on stake.

**Implementation**: reuse the `PoS` branch directly —
`pallet-voting-stake` + `scale_solution_range` +
`voting_stake_weight` / `max_voting_stake_weight` runtime APIs. The only
substitution on the PoRW side is the semantics of the scaled "base
capacity": from Subspace's plot capacity to `|C_t| × m_t` (residency ×
service, §4.3–4.4); the stake-scaling layer applies unchanged.

## 7. Mapping onto the existing code

| Existing component | Fate under PoRW |
|--------------------|-----------------|
| `subspace-proof-of-time` | **Kept as-is** (clock and challenge source) |
| `pallet-subspace` difficulty adjustment / solution range | **Kept as-is** (the metered object becomes resident bytes) |
| `subspace-verification::is_within_solution_range` | **Reused** (applied to the sketch-expanded chunk stream) |
| Challenge-derivation skeleton in `auditing.rs` | **Rewritten** as sketch coefficient derivation (PRF(c_t, device_id, i)) |
| `subspace-proof-of-space` (chiapos) / KZG chunk witness / unique-encoded plotting | **No longer needed on the VRAM track** (TEE replaces cryptographic uniqueness); **kept as-is on the storage track** (§5.4 dual track) for history and retired-weight permanence |
| archiver / DSN / gateway | **Kept**; repurposed for model-weight storage and distribution (plus the archival dual track) |
| `subspace-farmer` | Replaced by the **PoRW Agent**: runs inside the CVM, manages attestation, weights, sketch kernels, and the inference-service interface |
| `shared/subspace-proof-of-space-gpu` (CUDA/ROCm) | Starting reference for the GPU kernel engineering |
| `Solution` struct (`subspace-core-primitives`) | Reworked: `{device_id, model_id, sketch, chunk_index, sig}` replaces the KZG witness fields |

New components (★ = already landed on this branch):

1. ★ `crates/subspace-proof-of-residency`: the PoRW consensus
   primitives — the canonical Rust implementation of the sketch spec
   (bit-identical with Python/Triton cross-language test vectors), tile
   Merkle commitments (`R_W` and `partials_root`), ticket expansion,
   envelope check, `verify_tile_fraud_proof`, and the cross-audit
   assignment pure functions (§4.6.1). no_std.
2. ★ `crates/pallet-porw-registry`: device/model/measurement registries +
   fidelity bond (fungible holds) + single-tile fraud-proof slashing
   (bounty to the reporter) + the `check_solution` fast path
   (registration / activation delay / measurement revocation / model
   announcement / envelope) + demand-following model weights + reward
   escrow with audit-window release (§4.6.1). Attestation goes through
   the pluggable `AttestationVerifier` trait (P4 connects real NVIDIA
   CC/TDX/SNP verification). Compiles to wasm; covered by mock-runtime
   tests.
3. ★ Landed: `crates/porw-agent` — the node-side agent (state machine
   Unregistered→Registered→Active, solution assembly + device signing,
   pluggable `SketchBackend` trait swapping GPU/CPU, CPU backend using
   the canonical sketch, cross-audit duties + `cross_check`) and
   `crates/porw-devnet` — pure-CPU end-to-end integration tests: agent
   produces a signed solution → registered with real attestation
   evidence → activated → accepted by the on-chain fast path + consensus
   distance check, all green with zero GPU and zero TEE. Remaining: a
   GPU `SketchBackend`, wiring into a live node service.
4. `porw-sketch-gpu`: the S1-over-coverage targeted sweep (a Triton
   version exists in the PoC) + the optional fused hardening mode.
5. ★ Landed: `crates/porw-attestation` — the attestation-evidence
   verification library, with a real ed25519 signature chain (vendor
   root → device identity certificate → report) binding device_id + node
   key + measurement. Pallet integration: `TrustedRoots` governance
   storage + the `PorwAttestation` implementation; `register_device`
   performs real verification (the runtime has switched off the insecure
   stub). **Remaining hardware-specific work**: parsing the NVIDIA
   NRAS/SPDM and TDX/SNP wire formats (replacing `Evidence::decode`) and
   the real NVIDIA/Intel/AMD roots (replacing governance test roots) —
   the verification core is format-agnostic and reused as-is.

## 8. Phased roadmap

### P0 — mechanism validation (no GPU, no TEE, pure DRAM)

- Implement "audit amplification" on the existing farmer: the challenge
  derives N chained s-buckets with a tunable per-slot audit fraction f;
  matching verification. Purpose: validate solution-range statistics and
  difficulty-adjustment stability under amplified auditing. **The shared
  foundation of all subsequent routes.**

### P1 — fused sketch kernel (GPU, no TEE)

- First, standalone validation tools: cross-card determinism tests of
  integer-domain tile multiply-add, coverage-bitmap recomputation checks
  (this is also the verifier of §4.6's deep path — the deliverable is
  reused directly).
- Core: the vLLM/SGLang fusion PoC (custom epilogue or a side-channel
  kernel riding L2/SMEM), quantifying the net overhead of sketch
  accumulation on inference throughput (target <2%) and measuring
  coverage-set statistics under real MoE load. **This is the design's
  highest-risk engineering assumption and should be falsified/confirmed
  first.**

### P2 — attestation closed loop (with TEE)

- TDX/SNP CVM + H100 CC environment; agent prototype: key generation,
  dual attestation collection, off-chain verifier; minimal
  `pallet-porw-registry` (native certificate-chain verification).

### P3 — consensus integration

- ★ Landed: the `PorwApi` runtime API (declared in
  sp-consensus-subspace, implemented in subspace-runtime — registry fast
  path + ticket computation); both pallets mounted in subspace-runtime
  (`PorwRegistry` = index 10, `PorwBond` hold reason, wasm builds);
  `sc-consensus-subspace::porw` client-side verification glue (ticket
  expansion → bidirectional distance to the PoT challenge →
  stake-scaled solution range, same semantics as the farming path).
- ★ Landed (P3 stage 1): the PoRW pre-digest carrier (`PorwPreDigest`,
  its own `PORW` engine ID, coexisting with the farming pre-digest) +
  header extraction + the import-side entry `verify_porw_block` (extract
  pre-digest → derive the slot challenge from PoT → full validation);
  fast path upgraded to `check_solution_signed` (registration + envelope
  + **device signature**), so unsigned/forged solutions can never author
  a block.
- ★ Landed (P3 authorship): `claim_porw_slot` (pick the best solution +
  build the pre-digest), `porw_pre_digest_logs` / `porw_seal_digest`
  (header logs), `verify_porw_seal` (the device node key's seal over the
  block pre-hash, preventing a solution from being grafted onto a
  different block body). Authorship and import now share one validation
  and digest carrier, each unit-tested.
- ★ Landed (`porw-devnet` block loop): a real produce→seal→link→import
  loop driven by `claim_porw_slot`, forming a growing chain of
  device-key-sealed, parent-linked blocks (a mock runtime-API client
  performs the real fast-path checks without Substrate storage). Foreign
  seals are rejected. All green on CPU.
- ★ Landed: replica cross-verification scheduling (§4.6.1) — beacon,
  assignment pure functions, reward escrow with clawback, auditor-side
  duties and cross_check, end-to-end conviction test.
- Remaining (needs real infrastructure): wiring into the real
  `subspace-node` service (PoT gadget, networking, multiple nodes); a
  GPU `SketchBackend`; a testnet.

### 8.3 Fallback paths (design insurance)

- **Consumer cards without CC**: excluded from the main PoRW lottery;
  may join a parallel lottery of P0-style DRAM/VRAM audit amplification
  at lower reward weight, preserving long-tail decentralization.
- **TEE trust collapse**: a governance switch slides consensus weight
  toward the purely economic/physical mode ("replica cross-verification
  + deadlines") — degraded security without downtime — while retaining
  the ultimate option of reverting to classic PoAS.

## 9. Related work (positioning)

- **Filecoin**: cryptographic sealing + double copies in exchange for
  retrievable data — we use a TEE to eliminate sealing.
- **Bittensor / io.net class**: no hardware proof or soft proof only;
  metering relies on subjective committee scoring — our win probability
  is physically bound to byte residency.
- **Prime Intellect TOPLOC (2025)**: locality-sensitive hashing of
  activations to verify "the inference result is genuine" — orthogonal
  and complementary: PoRW proves "weights resident and read"; TOPLOC-
  style schemes prove "the output really came from those weights"; the
  two stack (the fused sketch layer can commit activations in passing).
- **The NVIDIA CC ecosystem (Phala, Atoma, etc.)**: has validated the
  engineering feasibility of "TEE-GPU runs inference and emits proofs";
  our difference is attaching that to a Nakamoto-style capacity lottery
  rather than stopping at per-task verification.

## 10. Open questions

1. **Whether m_t should be token-weighted**: m_t currently counts
   "bandwidth sweeps", which is batching-neutral (batch 128 and batch 1
   earn the same for one sweep) — physically clean, but it does not
   reward the real throughput of efficient batching. Token weighting
   (one sweep × tokens in the batch) moves the ticket ceiling from the
   bandwidth envelope to a compute envelope and amplifies self-generated
   load, requiring §5.1's fee-burn weighting as a companion. Suggested:
   replay real workloads under both weightings during P0.
2. Coverage-bitmap correctness and overhead of the fused kernel under
   speculative decoding, prefix caching, and quantized kernels (weights
   dequantized in SMEM).
3. PRF and hash_expand choices (BLAKE3 XOF vs ChaCha) and their GPU
   throughput trade-offs; compact coverage-bitmap encoding (highly
   structured under MoE; compressible at expert granularity).
4. On-chain certificate-chain verification cost and the game parameters
   of optimistic verification (challenge period, deposits).
5. Residency-metering semantics under multi-tenant inference (one card,
   multiple models dynamically swapped).
6. ~~Stacking with inference-authenticity proofs~~ → absorbed as pillar
   D (§4.7). Remaining open sub-questions: on-chain verification /
   sampling strategy and cost of pillar-D quotes, and batching them with
   pillar-B device-certificate verification (item 4 above).

---

*This document is a research draft; all parameters (tile size, deadlines,
bond amounts, epoch length) are placeholders pending P0/P1 calibration.*
