//! Tests, including cross-language vectors generated from the Python
//! reference spec in `porw-poc/porw_sketch/spec.py` (see repo docs). The
//! deterministic buffer is `buf[i] = ((i * 2654435761) >> 7) & 0xFF` over
//! wrapping u64 arithmetic, 4 tiles.

use super::*;

const N_TILES: usize = 4;

/// Per-tile sketches for slot seeds [1, 0xDEADBEEF, 0x9E3779B9], generated
/// by the Python/numpy reference implementation.
const VECTOR_CASES: [(u32, [u32; N_TILES]); 3] = [
    (1, [3485902744, 372182208, 1349964192, 38446344]),
    (
        0xDEAD_BEEF,
        [1557940684, 3400778530, 3931927942, 1241214034],
    ),
    (0x9E37_79B9, [1823348526, 1479624626, 958815834, 3798367726]),
];

/// First four per-word coefficients of tiles 0 and 3 at slot seed 1.
const COEFF_PROBE_TILE0: [u32; 4] = [1461123477, 2317529113, 1004244359, 3102047685];
const COEFF_PROBE_TILE3: [u32; 4] = [2272955061, 2986084877, 4130036541, 2293564949];

fn reference_buffer() -> Vec<u8> {
    (0..(N_TILES * TILE_BYTES) as u64)
        .map(|i| ((i.wrapping_mul(2654435761) >> 7) & 0xFF) as u8)
        .collect()
}

fn buffer_tiles(buf: &[u8]) -> Vec<[u8; TILE_BYTES]> {
    buf.chunks_exact(TILE_BYTES)
        .map(|c| c.try_into().unwrap())
        .collect()
}

#[test]
fn cross_language_sketch_vectors() {
    let tiles = buffer_tiles(&reference_buffer());
    for (seed, expected) in VECTOR_CASES {
        for (idx, tile) in tiles.iter().enumerate() {
            assert_eq!(
                sketch_tile(seed, idx as u64, tile),
                expected[idx],
                "seed {seed:#x} tile {idx}"
            );
        }
    }
}

#[test]
fn cross_language_coeff_vectors() {
    let r0 = tile_seed(1, 0);
    let r3 = tile_seed(1, 3);
    for j in 0..4u32 {
        assert_eq!(word_coeff(r0, j), COEFF_PROBE_TILE0[j as usize]);
        assert_eq!(word_coeff(r3, j), COEFF_PROBE_TILE3[j as usize]);
    }
}

#[test]
fn coefficients_are_odd() {
    let r = tile_seed(0xDEAD_BEEF, 7);
    for j in 0..TILE_WORDS as u32 {
        assert_eq!(word_coeff(r, j) & 1, 1);
    }
}

#[test]
fn single_bit_corruption_always_detected() {
    let tiles = buffer_tiles(&reference_buffer());
    let baseline = sketch_tile(42, 0, &tiles[0]);
    // Odd coefficients are bijective mod 2^32: any single-bit flip must
    // change the sketch. Try every bit position of a few words plus a
    // pseudo-random sample across the tile.
    let mut lcg = 0x1234_5678_u64;
    for trial in 0..256 {
        let (byte, bit) = if trial < 32 {
            (trial / 8, trial % 8)
        } else {
            lcg = lcg
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (
                ((lcg >> 33) as usize) % TILE_BYTES,
                ((lcg >> 29) as usize) % 8,
            )
        };
        let mut bad = tiles[0];
        bad[byte] ^= 1 << bit;
        assert_ne!(sketch_tile(42, 0, &bad), baseline, "byte {byte} bit {bit}");
    }
}

#[test]
fn merkle_proofs_roundtrip() {
    let tiles = buffer_tiles(&reference_buffer());
    let leaves: Vec<Hash32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| weights_leaf(i as u64, t))
        .collect();
    let root = merkle_root(&leaves);
    for (i, leaf) in leaves.iter().enumerate() {
        let proof = merkle_proof(&leaves, i);
        assert!(merkle_verify(&root, leaf, i, &proof));
        // Wrong index or tampered leaf must fail.
        assert!(!merkle_verify(&root, leaf, i + 1, &proof) || leaves.len() == 1);
        let mut bad = *leaf;
        bad[0] ^= 1;
        assert!(!merkle_verify(&root, &bad, i, &proof));
    }
}

#[test]
fn envelope_and_tickets() {
    // 70 GB coverage swept 3.2x against a 240 GB/slot envelope: allowed.
    let cov = 70_u64 * 1 << 30;
    assert!(check_envelope(cov, 3200, 240 * (1 << 30)));
    // Claiming 4x against the same envelope: rejected.
    assert!(!check_envelope(cov, 4000, 240 * (1 << 30)));
    // Tickets scale linearly with coverage and multiplier.
    let unit = 1 << 30;
    assert_eq!(ticket_count(cov, 1000, unit), 70);
    assert_eq!(ticket_count(cov, 2000, unit), 140);
    // Distinct chunk indexes, slot seeds and models yield distinct tickets.
    let model = [3u8; 32];
    let root = [7u8; 32];
    assert_ne!(
        ticket_chunk(&model, &root, 1, 0),
        ticket_chunk(&model, &root, 1, 1)
    );
    assert_ne!(
        ticket_chunk(&model, &root, 1, 0),
        ticket_chunk(&model, &root, 2, 0)
    );
    let model2 = [4u8; 32];
    assert_ne!(
        ticket_chunk(&model, &root, 1, 0),
        ticket_chunk(&model2, &root, 1, 0)
    );
}

fn build_solution_and_proofs(
    tamper_tile: Option<usize>,
) -> (PorwSolution, [u8; 32], Hash32, Vec<TileFraudProof>) {
    let challenge = [9u8; 32];
    let device_id = [3u8; 32];
    let slot_seed = derive_slot_seed(&challenge, &device_id);
    let tiles = buffer_tiles(&reference_buffer());

    let weight_leaves: Vec<Hash32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| weights_leaf(i as u64, t))
        .collect();
    let model_root = merkle_root(&weight_leaves);

    let mut s_tiles: Vec<u32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| sketch_tile(slot_seed, i as u64, t))
        .collect();
    if let Some(i) = tamper_tile {
        s_tiles[i] ^= 0xBAD; // the accused commits a wrong per-tile value
    }
    let partial_leaves: Vec<Hash32> = s_tiles
        .iter()
        .enumerate()
        .map(|(i, s)| partials_leaf(i as u64, *s))
        .collect();
    let partials_root = merkle_root(&partial_leaves);

    let solution = PorwSolution {
        device_id,
        model_id: model_root,
        sketch: s_tiles.iter().fold(0u32, |a, s| a.wrapping_add(*s)),
        partials_root,
        coverage_bytes: (N_TILES * TILE_BYTES) as u64,
        m_t_millis: 1000,
        chunk_index: 0,
        signature: [0u8; 64],
    };

    let proofs = (0..N_TILES)
        .map(|i| TileFraudProof {
            tile_idx: i as u64,
            claimed_s_tile: s_tiles[i],
            partials_index: i as u64,
            partials_proof: merkle_proof(&partial_leaves, i),
            tile_bytes: tiles[i].to_vec(),
            weights_proof: merkle_proof(&weight_leaves, i),
        })
        .collect();

    (solution, challenge, model_root, proofs)
}

#[test]
fn fraud_proof_honest_solution_shows_no_fraud() {
    let (solution, challenge, model_root, proofs) = build_solution_and_proofs(None);
    for proof in &proofs {
        assert_eq!(
            verify_tile_fraud_proof(&solution, &challenge, &model_root, proof),
            FraudVerdict::NoFraud
        );
    }
}

#[test]
fn fraud_proof_catches_tampered_commitment() {
    let (solution, challenge, model_root, proofs) = build_solution_and_proofs(Some(2));
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &model_root, &proofs[2]),
        FraudVerdict::Fraud
    );
    // Untampered tiles remain clean.
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &model_root, &proofs[0]),
        FraudVerdict::NoFraud
    );
}

#[test]
fn fraud_proof_rejects_malformed_evidence() {
    let (solution, challenge, model_root, mut proofs) = build_solution_and_proofs(None);
    // Non-canonical tile bytes (not committed under R_W).
    proofs[1].tile_bytes[0] ^= 1;
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &model_root, &proofs[1]),
        FraudVerdict::Invalid
    );
    // Wrong length.
    proofs[0].tile_bytes.pop();
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &model_root, &proofs[0]),
        FraudVerdict::Invalid
    );
    // Broken partials path.
    proofs[3].partials_proof[0][0] ^= 1;
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &model_root, &proofs[3]),
        FraudVerdict::Invalid
    );
}

#[test]
fn fraud_proof_works_for_non_contiguous_coverage() {
    // MoE-style coverage: tile 3 committed at partials position 1. Before
    // `partials_index` was added the verifier used tile_idx as the leaf
    // position, so an honest proof against a sparse coverage set could not
    // verify at all.
    let challenge = [9u8; 32];
    let device_id = [3u8; 32];
    let slot_seed = derive_slot_seed(&challenge, &device_id);
    let tiles = buffer_tiles(&reference_buffer());
    let weight_leaves: Vec<Hash32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| weights_leaf(i as u64, t))
        .collect();
    let model_root = merkle_root(&weight_leaves);

    let coverage: [u64; 2] = [1, 3];
    let mut s_tiles: Vec<u32> = coverage
        .iter()
        .map(|&i| sketch_tile(slot_seed, i, &tiles[i as usize]))
        .collect();
    s_tiles[1] ^= 0xBAD; // tamper the value committed for tile 3
    let partial_leaves: Vec<Hash32> = coverage
        .iter()
        .zip(&s_tiles)
        .map(|(&i, &s)| partials_leaf(i, s))
        .collect();
    let solution = PorwSolution {
        device_id,
        model_id: model_root,
        sketch: s_tiles.iter().fold(0u32, |a, s| a.wrapping_add(*s)),
        partials_root: merkle_root(&partial_leaves),
        coverage_bytes: (coverage.len() * TILE_BYTES) as u64,
        m_t_millis: 1000,
        chunk_index: 0,
        signature: [0u8; 64],
    };

    let proof = TileFraudProof {
        tile_idx: 3,
        claimed_s_tile: s_tiles[1],
        partials_index: 1, // coverage-order position, not the tile index
        partials_proof: merkle_proof(&partial_leaves, 1),
        tile_bytes: tiles[3].to_vec(),
        weights_proof: merkle_proof(&weight_leaves, 3),
    };
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &model_root, &proof),
        FraudVerdict::Fraud
    );

    // Lying about the position: the leaf hash binds tile_idx, so the proof
    // simply fails to verify — it cannot shift blame across tiles.
    let mut shifted = proof.clone();
    shifted.partials_index = 0;
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &model_root, &shifted),
        FraudVerdict::Invalid
    );
}

#[test]
fn audit_assignment_is_deterministic_and_excludes_target() {
    let beacon = audit_beacon(7, &[0xAB; 32]);
    let model = [5u8; 32];
    let replicas: Vec<Hash32> = (0u8..6).map(|i| [i; 32]).collect();
    let target = replicas[2];

    let a = select_auditors(&beacon, &model, &target, &replicas, 3);
    let b = select_auditors(&beacon, &model, &target, &replicas, 3);
    assert_eq!(a, b, "assignment must be a pure function of the beacon");
    assert_eq!(a.len(), 3);
    assert!(!a.contains(&target), "a device never audits itself");
    // A different beacon reshuffles the panel (overwhelmingly likely).
    let other = select_auditors(&audit_beacon(8, &[0xAB; 32]), &model, &target, &replicas, 3);
    assert_ne!(a, other);
    // Fewer peers than k: everyone else is assigned.
    let small = select_auditors(&beacon, &model, &target, &replicas[2..4], 3);
    assert_eq!(small.len(), 1, "target excluded, one peer remains");
    // No peers at all (single-replica model): empty panel.
    let none = select_auditors(&beacon, &model, &target, &[target], 3);
    assert!(none.is_empty());
}

#[test]
fn audit_tile_sample_is_deterministic_distinct_and_bounded() {
    let beacon = audit_beacon(7, &[0xAB; 32]);
    let (model, target, auditor) = ([5u8; 32], [2u8; 32], [1u8; 32]);

    let s = audit_tile_sample(&beacon, &model, &target, &auditor, 1000, 32);
    assert_eq!(
        s,
        audit_tile_sample(&beacon, &model, &target, &auditor, 1000, 32)
    );
    assert_eq!(s.len(), 32);
    assert!(s.iter().all(|&i| i < 1000));
    let mut dedup = s.clone();
    dedup.sort_unstable();
    dedup.dedup();
    assert_eq!(dedup.len(), 32, "sampled tiles must be distinct");

    // Different auditors of the same target sample different tiles
    // (overwhelmingly likely), widening combined coverage.
    let other = audit_tile_sample(&beacon, &model, &target, &[9u8; 32], 1000, 32);
    assert_ne!(s, other);

    // Requesting at least as many tiles as exist audits everything.
    assert_eq!(
        audit_tile_sample(&beacon, &model, &target, &auditor, 8, 32),
        (0..8).collect::<Vec<u64>>()
    );
    assert!(audit_tile_sample(&beacon, &model, &target, &auditor, 0, 4).is_empty());
}

#[test]
fn opening_responses_prove_commitment_and_non_commitment() {
    // Strictly ascending sparse coverage over tiles {1, 3}; challenged tiles
    // 0 (before), 2 (between), 3 (committed), 5 (after).
    let challenge = [9u8; 32];
    let device_id = [3u8; 32];
    let slot_seed = derive_slot_seed(&challenge, &device_id);
    let tiles = buffer_tiles(&reference_buffer());
    let coverage: [u64; 2] = [1, 3];
    let s_tiles: Vec<u32> = coverage
        .iter()
        .map(|&i| sketch_tile(slot_seed, i, &tiles[i as usize]))
        .collect();
    let leaves: Vec<Hash32> = coverage
        .iter()
        .zip(&s_tiles)
        .map(|(&i, &s)| partials_leaf(i, s))
        .collect();
    let root = merkle_root(&leaves);
    let n_leaves = leaves.len() as u64;
    let wit = |pos: usize| LeafWitness {
        tile_idx: coverage[pos],
        s_tile: s_tiles[pos],
        index: pos as u64,
        proof: merkle_proof(&leaves, pos),
    };

    // Committed tile: opening verifies and returns the committed value.
    assert_eq!(
        verify_opening_response(&root, n_leaves, 3, &OpeningResponse::Committed(wit(1))),
        Ok(Some(s_tiles[1]))
    );
    // Claiming the wrong tile with a real leaf fails.
    assert_eq!(
        verify_opening_response(&root, n_leaves, 2, &OpeningResponse::Committed(wit(1))),
        Err(())
    );

    // Between two committed leaves: bracketed non-inclusion.
    assert_eq!(
        verify_opening_response(
            &root,
            n_leaves,
            2,
            &OpeningResponse::NotCommitted {
                left: Some(wit(0)),
                right: Some(wit(1)),
            }
        ),
        Ok(None)
    );
    // Before the first leaf.
    assert_eq!(
        verify_opening_response(
            &root,
            n_leaves,
            0,
            &OpeningResponse::NotCommitted {
                left: None,
                right: Some(wit(0)),
            }
        ),
        Ok(None)
    );
    // After the last leaf.
    assert_eq!(
        verify_opening_response(
            &root,
            n_leaves,
            5,
            &OpeningResponse::NotCommitted {
                left: Some(wit(1)),
                right: None,
            }
        ),
        Ok(None)
    );

    // A committed tile cannot be denied: any non-inclusion shape around it
    // fails (tile 3 IS the last leaf; claiming "after last" needs
    // l.tile_idx < challenged which fails, bracketing fails adjacency/order).
    assert_eq!(
        verify_opening_response(
            &root,
            n_leaves,
            3,
            &OpeningResponse::NotCommitted {
                left: Some(wit(1)),
                right: None,
            }
        ),
        Err(())
    );
    assert_eq!(
        verify_opening_response(
            &root,
            n_leaves,
            3,
            &OpeningResponse::NotCommitted {
                left: Some(wit(0)),
                right: Some(wit(1)),
            }
        ),
        Err(())
    );
    // Non-adjacent bracket is rejected (hiding a leaf between them).
    assert_eq!(
        verify_opening_response(
            &root,
            n_leaves,
            2,
            &OpeningResponse::NotCommitted {
                left: Some(wit(0)),
                right: Some(LeafWitness { index: 2, ..wit(1) }),
            }
        ),
        Err(())
    );
    // Empty answer never verifies.
    assert_eq!(
        verify_opening_response(
            &root,
            n_leaves,
            2,
            &OpeningResponse::NotCommitted {
                left: None,
                right: None,
            }
        ),
        Err(())
    );
}

#[test]
fn scheme_id_is_stable() {
    // Pinned by ExecutionProfile.porw_scheme_id / IPoRWVerifier.schemeId()
    // in aigg-spec; a semantic change to the sketch is a NEW id, so this
    // constant must never drift for the v2 semantics.
    assert_eq!(PORW_SCHEME_ID, "aigg:porw:sketch-tile:v2");
    assert_eq!(
        porw_scheme_digest(),
        *blake3::hash(PORW_SCHEME_ID.as_bytes()).as_bytes()
    );
}

// ---------------------------------------------------------------------------
// Cross-language conformance fixtures (aigg-spec §15)
// ---------------------------------------------------------------------------
//
// `conformance/sketch-tile-v2.json` is the vector set an independent
// implementation (e.g. the Solidity verifier of the EVM deployment) must
// reproduce bit-for-bit. This test regenerates the fixture content from the
// canonical implementation and fails if the committed file drifts.
// To update after an INTENTIONAL semantic change (which is a new scheme id):
// `UPDATE_CONFORMANCE=1 cargo test -p subspace-proof-of-residency conformance`.

fn hex_bytes(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(2 + bytes.len() * 2);
    s.push_str("0x");
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

fn json_hash_list(hashes: &[Hash32], indent: &str) -> String {
    hashes
        .iter()
        .map(|h| format!("{indent}\"{}\"", hex_bytes(h)))
        .collect::<Vec<_>>()
        .join(",\n")
}

fn generate_conformance_fixture() -> String {
    let tiles = buffer_tiles(&reference_buffer());
    let buffer = reference_buffer();

    // Weights tree over the full 4-tile reference model.
    let weight_leaves: Vec<Hash32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| weights_leaf(i as u64, t))
        .collect();
    let weights_root = merkle_root(&weight_leaves);

    // Scenario: device commits sparse ascending coverage {1, 3} with the
    // value for tile 3 tampered (matches the fraud-proof unit tests).
    let challenge = [9u8; 32];
    let device_id = [3u8; 32];
    let slot_seed = derive_slot_seed(&challenge, &device_id);
    let coverage: [u64; 2] = [1, 3];
    let mut s_tiles: Vec<u32> = coverage
        .iter()
        .map(|&i| sketch_tile(slot_seed, i, &tiles[i as usize]))
        .collect();
    let honest_tile3 = s_tiles[1];
    s_tiles[1] ^= 0xBAD;
    let partial_leaves: Vec<Hash32> = coverage
        .iter()
        .zip(&s_tiles)
        .map(|(&i, &s)| partials_leaf(i, s))
        .collect();
    let partials_root = merkle_root(&partial_leaves);

    let sketch_cases = VECTOR_CASES
        .iter()
        .map(|(seed, values)| {
            format!(
                "    {{ \"slot_seed\": {seed}, \"per_tile\": [{}] }}",
                values
                    .iter()
                    .map(|v| v.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        })
        .collect::<Vec<_>>()
        .join(",\n");

    let proof_of = |leaves: &[Hash32], i: usize| {
        let p = merkle_proof(leaves, i);
        json_hash_list(&p, "        ")
    };

    format!(
        r#"{{
  "scheme": {{
    "id": "{scheme_id}",
    "digest": "{scheme_digest}"
  }},
  "params": {{
    "tile_bytes": {tile_bytes},
    "tile_words": {tile_words},
    "golden32": "0x9e3779b9",
    "coverage_order": "strictly ascending tile index",
    "hash": "blake3",
    "note": "signature suite is a deployment choice outside the scheme id (ed25519 on Substrate, secp256k1/ecrecover on EVM)"
  }},
  "reference_buffer": {{
    "formula": "buf[i] = ((i * 2654435761) >> 7) & 0xFF, wrapping u64 arithmetic, i in 0..n_tiles*tile_bytes",
    "n_tiles": 4,
    "blake3": "{buffer_hash}"
  }},
  "coefficients": {{
    "note": "word_coeff(tile_seed(slot_seed, tile_idx), j); always odd",
    "slot_seed": 1,
    "tile0_first4": [{c0}],
    "tile3_first4": [{c3}]
  }},
  "sketches": [
{sketch_cases}
  ],
  "slot_seed_derivation": {{
    "global_challenge": "{challenge_hex}",
    "device_id": "{device_hex}",
    "slot_seed": {slot_seed}
  }},
  "weights_tree": {{
    "leaves": [
{weight_leaves_json}
    ],
    "root": "{weights_root_hex}"
  }},
  "ticket_chunks": {{
    "note": "ticket_chunk(model_id=weights_root, partials_root, slot_seed, index)",
    "index_0": "{chunk0}",
    "index_1": "{chunk1}"
  }},
  "audit_beacon": {{
    "epoch": 7,
    "entropy": "{beacon_entropy}",
    "beacon": "{beacon}"
  }},
  "tampered_commitment_scenario": {{
    "coverage": [1, 3],
    "honest_s_tile_for_tile_3": {honest_tile3},
    "committed_s_tiles": [{committed0}, {committed1}],
    "partials_leaves": [
{partials_leaves_json}
    ],
    "partials_root": "{partials_root_hex}",
    "opening_committed_tile_3": {{
      "leaf_index": 1,
      "proof": [
{opening_proof}
      ],
      "expected": "verifies; opened value {committed1} != recomputed {honest_tile3} => TileFraudProof verdict Fraud"
    }},
    "non_inclusion_tile_2": {{
      "left":  {{ "tile_idx": 1, "s_tile": {committed0}, "index": 0 }},
      "right": {{ "tile_idx": 3, "s_tile": {committed1}, "index": 1 }},
      "expected": "adjacent bracket verifies => proven not committed"
    }},
    "fraud_proof_tile_3": {{
      "tile_idx": 3,
      "claimed_s_tile": {committed1},
      "partials_index": 1,
      "partials_proof": [
{fraud_partials_proof}
      ],
      "tile_bytes": "generate tile 3 from reference_buffer.formula",
      "weights_proof": [
{fraud_weights_proof}
      ],
      "expected_verdict": "Fraud"
    }}
  }}
}}
"#,
        scheme_id = PORW_SCHEME_ID,
        scheme_digest = hex_bytes(&porw_scheme_digest()),
        tile_bytes = TILE_BYTES,
        tile_words = TILE_WORDS,
        buffer_hash = hex_bytes(blake3::hash(&buffer).as_bytes()),
        c0 = COEFF_PROBE_TILE0
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        c3 = COEFF_PROBE_TILE3
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        challenge_hex = hex_bytes(&challenge),
        device_hex = hex_bytes(&device_id),
        weight_leaves_json = json_hash_list(&weight_leaves, "      "),
        weights_root_hex = hex_bytes(&weights_root),
        chunk0 = hex_bytes(&ticket_chunk(&weights_root, &partials_root, slot_seed, 0)),
        chunk1 = hex_bytes(&ticket_chunk(&weights_root, &partials_root, slot_seed, 1)),
        beacon_entropy = hex_bytes(&[0xAB; 32]),
        beacon = hex_bytes(&audit_beacon(7, &[0xAB; 32])),
        committed0 = s_tiles[0],
        committed1 = s_tiles[1],
        partials_leaves_json = json_hash_list(&partial_leaves, "      "),
        partials_root_hex = hex_bytes(&partials_root),
        opening_proof = proof_of(&partial_leaves, 1),
        fraud_partials_proof = proof_of(&partial_leaves, 1),
        fraud_weights_proof = proof_of(&weight_leaves, 3),
    )
}

#[test]
fn conformance_fixture_is_current() {
    let generated = generate_conformance_fixture();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/conformance/sketch-tile-v2.json"
    );
    if std::env::var("UPDATE_CONFORMANCE").is_ok() {
        std::fs::write(path, &generated).expect("write fixture");
        return;
    }
    let committed = std::fs::read_to_string(path)
        .expect("fixture missing; run with UPDATE_CONFORMANCE=1 to create");
    assert_eq!(
        committed, generated,
        "conformance fixture drifted from the canonical implementation; an \
         intentional semantic change requires a NEW scheme id and fixture, \
         then UPDATE_CONFORMANCE=1"
    );
}

// ---------------------------------------------------------------------------
// keccak-variant conformance fixture (aigg:porw:sketch-tile-keccak:v1)
// ---------------------------------------------------------------------------

#[test]
fn keccak_variant_basics() {
    // Distinct scheme, distinct commitments: nothing from one scheme
    // verifies under the other.
    assert_eq!(keccak::PORW_SCHEME_ID, "aigg:porw:sketch-tile-keccak:v1");
    assert_ne!(keccak::porw_scheme_digest(), porw_scheme_digest());
    let tiles = buffer_tiles(&reference_buffer());
    assert_ne!(
        keccak::weights_leaf(0, &tiles[0]),
        weights_leaf(0, &tiles[0])
    );
    let challenge = [9u8; 32];
    let device = [3u8; 32];
    assert_ne!(
        keccak::derive_slot_seed(&challenge, &device),
        derive_slot_seed(&challenge, &device)
    );

    // Same dispute logic end-to-end under keccak commitments: build the
    // tampered sparse scenario and check all three verdict classes.
    let slot_seed = keccak::derive_slot_seed(&challenge, &device);
    let coverage: [u64; 2] = [1, 3];
    let mut s_tiles: Vec<u32> = coverage
        .iter()
        .map(|&i| sketch_tile(slot_seed, i, &tiles[i as usize]))
        .collect();
    s_tiles[1] ^= 0xBAD;
    let partial_leaves: Vec<Hash32> = coverage
        .iter()
        .zip(&s_tiles)
        .map(|(&i, &s)| keccak::partials_leaf(i, s))
        .collect();
    let weight_leaves: Vec<Hash32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| keccak::weights_leaf(i as u64, t))
        .collect();
    let solution = PorwSolution {
        device_id: device,
        model_id: keccak::merkle_root(&weight_leaves),
        sketch: s_tiles.iter().fold(0u32, |a, s| a.wrapping_add(*s)),
        partials_root: keccak::merkle_root(&partial_leaves),
        coverage_bytes: (coverage.len() * TILE_BYTES) as u64,
        m_t_millis: 1000,
        chunk_index: 0,
        signature: [0u8; 64],
    };
    let proof = TileFraudProof {
        tile_idx: 3,
        claimed_s_tile: s_tiles[1],
        partials_index: 1,
        partials_proof: keccak::merkle_proof(&partial_leaves, 1),
        tile_bytes: tiles[3].to_vec(),
        weights_proof: keccak::merkle_proof(&weight_leaves, 3),
    };
    assert_eq!(
        keccak::verify_tile_fraud_proof(&solution, &challenge, &solution.model_id, &proof),
        FraudVerdict::Fraud
    );
    // The blake3 verifier must NOT accept keccak commitments.
    assert_eq!(
        verify_tile_fraud_proof(&solution, &challenge, &solution.model_id, &proof),
        FraudVerdict::Invalid
    );
    // Non-inclusion under keccak.
    assert_eq!(
        keccak::verify_opening_response(
            &solution.partials_root,
            2,
            2,
            &OpeningResponse::NotCommitted {
                left: Some(LeafWitness {
                    tile_idx: 1,
                    s_tile: s_tiles[0],
                    index: 0,
                    proof: keccak::merkle_proof(&partial_leaves, 0),
                }),
                right: Some(LeafWitness {
                    tile_idx: 3,
                    s_tile: s_tiles[1],
                    index: 1,
                    proof: keccak::merkle_proof(&partial_leaves, 1),
                }),
            }
        ),
        Ok(None)
    );
}

fn generate_keccak_conformance_fixture() -> String {
    let tiles = buffer_tiles(&reference_buffer());
    let buffer = reference_buffer();

    let weight_leaves: Vec<Hash32> = tiles
        .iter()
        .enumerate()
        .map(|(i, t)| keccak::weights_leaf(i as u64, t))
        .collect();
    let weights_root = keccak::merkle_root(&weight_leaves);

    let challenge = [9u8; 32];
    let device_id = [3u8; 32];
    let slot_seed = keccak::derive_slot_seed(&challenge, &device_id);

    // Sketch vectors under the keccak-derived slot seed (the sketch math is
    // shared with the blake3 scheme; the seed differs).
    let sketches: Vec<u32> = (0..4u64)
        .map(|i| sketch_tile(slot_seed, i, &tiles[i as usize]))
        .collect();

    let coverage: [u64; 2] = [1, 3];
    let mut s_tiles: Vec<u32> = coverage
        .iter()
        .map(|&i| sketch_tile(slot_seed, i, &tiles[i as usize]))
        .collect();
    let honest_tile3 = s_tiles[1];
    s_tiles[1] ^= 0xBAD;
    let partial_leaves: Vec<Hash32> = coverage
        .iter()
        .zip(&s_tiles)
        .map(|(&i, &s)| keccak::partials_leaf(i, s))
        .collect();
    let partials_root = keccak::merkle_root(&partial_leaves);

    let proof_of = |leaves: &[Hash32], i: usize| {
        let p = keccak::merkle_proof(leaves, i);
        json_hash_list(&p, "        ")
    };

    format!(
        r#"{{
  "scheme": {{
    "id": "{scheme_id}",
    "digest_keccak256": "{scheme_digest}"
  }},
  "params": {{
    "tile_bytes": {tile_bytes},
    "tile_words": {tile_words},
    "golden32": "0x9e3779b9",
    "coverage_order": "strictly ascending tile index",
    "hash": "keccak256",
    "note": "sketch math identical to sketch-tile:v2; every hash (leaves, nodes, slot seed, scheme digest) is keccak256; ticket expansion is not part of this variant"
  }},
  "reference_buffer": {{
    "formula": "buf[i] = ((i * 2654435761) >> 7) & 0xFF, wrapping u64 arithmetic, i in 0..n_tiles*tile_bytes",
    "n_tiles": 4,
    "keccak256": "{buffer_hash}"
  }},
  "slot_seed_derivation": {{
    "global_challenge": "{challenge_hex}",
    "device_id": "{device_hex}",
    "slot_seed": {slot_seed}
  }},
  "sketches": [
    {{ "slot_seed": {slot_seed}, "per_tile": [{sketch_list}] }}
  ],
  "weights_tree": {{
    "leaves": [
{weight_leaves_json}
    ],
    "root": "{weights_root_hex}"
  }},
  "tampered_commitment_scenario": {{
    "coverage": [1, 3],
    "honest_s_tile_for_tile_3": {honest_tile3},
    "committed_s_tiles": [{committed0}, {committed1}],
    "partials_leaves": [
{partials_leaves_json}
    ],
    "partials_root": "{partials_root_hex}",
    "opening_committed_tile_3": {{
      "leaf_index": 1,
      "proof": [
{opening_proof}
      ],
      "expected": "verifies; opened value {committed1} != recomputed {honest_tile3} => TileFraudProof verdict Fraud"
    }},
    "non_inclusion_tile_2": {{
      "left":  {{ "tile_idx": 1, "s_tile": {committed0}, "index": 0 }},
      "right": {{ "tile_idx": 3, "s_tile": {committed1}, "index": 1 }},
      "expected": "adjacent bracket verifies => proven not committed"
    }},
    "fraud_proof_tile_3": {{
      "tile_idx": 3,
      "claimed_s_tile": {committed1},
      "partials_index": 1,
      "partials_proof": [
{fraud_partials_proof}
      ],
      "tile_bytes": "generate tile 3 from reference_buffer.formula",
      "weights_proof": [
{fraud_weights_proof}
      ],
      "expected_verdict": "Fraud"
    }}
  }}
}}
"#,
        scheme_id = keccak::PORW_SCHEME_ID,
        scheme_digest = hex_bytes(&keccak::porw_scheme_digest()),
        tile_bytes = TILE_BYTES,
        tile_words = TILE_WORDS,
        buffer_hash = {
            use sha3::Digest;
            let mut out = [0u8; 32];
            out.copy_from_slice(&sha3::Keccak256::digest(&buffer));
            hex_bytes(&out)
        },
        challenge_hex = hex_bytes(&challenge),
        device_hex = hex_bytes(&device_id),
        sketch_list = sketches
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join(", "),
        weight_leaves_json = json_hash_list(&weight_leaves, "      "),
        weights_root_hex = hex_bytes(&weights_root),
        honest_tile3 = honest_tile3,
        committed0 = s_tiles[0],
        committed1 = s_tiles[1],
        partials_leaves_json = json_hash_list(&partial_leaves, "      "),
        partials_root_hex = hex_bytes(&partials_root),
        opening_proof = proof_of(&partial_leaves, 1),
        fraud_partials_proof = proof_of(&partial_leaves, 1),
        fraud_weights_proof = proof_of(&weight_leaves, 3),
    )
}

#[test]
fn keccak_conformance_fixture_is_current() {
    let generated = generate_keccak_conformance_fixture();
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/conformance/sketch-tile-keccak-v1.json"
    );
    if std::env::var("UPDATE_CONFORMANCE").is_ok() {
        std::fs::write(path, &generated).expect("write fixture");
        return;
    }
    let committed = std::fs::read_to_string(path)
        .expect("fixture missing; run with UPDATE_CONFORMANCE=1 to create");
    assert_eq!(
        committed, generated,
        "keccak-variant fixture drifted; a semantic change requires a NEW \
         scheme id and fixture, then UPDATE_CONFORMANCE=1"
    );
}
