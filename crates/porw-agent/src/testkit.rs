//! Devnet test helpers. In production the attestation evidence comes from the
//! TEE (NVIDIA CC + TDX/SNP); on a CPU devnet we synthesize it with a test
//! vendor root so the full register→author→verify path runs GPU- and TEE-free.
//! These helpers are the counterpart of the runtime's governance-added test
//! root: the same key must be added via `add_trusted_root`.

use parity_scale_codec::Encode;
use porw_attestation::{AttestationReport, DeviceCert, Evidence, Id32};
use sp_core::{Pair, ed25519};

/// Deterministic test vendor root keypair. Its public key must be registered
/// on chain via `add_trusted_root`.
pub fn test_vendor_root(seed: u8) -> ed25519::Pair {
    ed25519::Pair::from_seed(&[seed; 32])
}

/// Build attestation evidence binding `device_id`, `node_pubkey` and
/// `measurement`, signed by `vendor_root`'s device-identity chain. Encodes to
/// the bytes a device passes to `register_device`.
pub fn build_evidence(
    vendor_root: &ed25519::Pair,
    device_identity_seed: u8,
    device_id: Id32,
    node_pubkey: Id32,
    measurement: Id32,
) -> Vec<u8> {
    let device_identity = ed25519::Pair::from_seed(&[device_identity_seed; 32]);

    let mut cert = DeviceCert {
        device_id,
        device_pubkey: device_identity.public().0,
        vendor_sig: [0; 64],
    };
    cert.vendor_sig = vendor_root.sign(&cert.body()).0;

    let mut report = AttestationReport {
        device_id,
        measurement,
        node_pubkey,
        report_sig: [0; 64],
    };
    report.report_sig = device_identity.sign(&report.body()).0;

    Evidence { cert, report }.encode()
}
