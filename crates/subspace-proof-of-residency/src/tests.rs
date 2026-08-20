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
