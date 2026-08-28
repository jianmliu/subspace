# porw-evm-bench

EVM feasibility benchmarks for the PoRW `aigg:porw:sketch-tile:v2` dispute
path — the measured evidence behind `docs/porw-evm-feasibility.md` (the
feasibility gate of the AI3 Verifiable Compute Market Pilot proposal §8.4).

- `src/Blake3.sol` — full BLAKE3 (hash mode, chunk tree) in clarity-first
  Solidity.
- `src/PorwVerifier.sol` — exact port of the scheme's dispute math: sketch
  recomputation, blake3 Merkle commitments (+ keccak variant for
  comparison), fraud-proof / opening / non-inclusion verification.
- `test/Conformance.t.sol` — 13 differential tests against the Rust
  reference via `../crates/subspace-proof-of-residency/conformance/`
  (bit-identical or it is not the same scheme).
- `test/Gas.t.sol` — the gas measurements at realistic tree depths.

```
forge test --match-contract ConformanceTest
forge test --match-contract GasBench -vv
```

`lib/forge-std` is vendored for reproducibility (installed by
`forge init`). This project is intentionally outside the cargo workspace.
