//! End-to-end tests over a real ed25519 signature chain. Only the root of
//! trust and the wire format are test stand-ins; the chain verification,
//! binding and measurement extraction exercised here are production logic.

use super::*;
use sp_core::{Pair, ed25519};

struct World {
    root: ed25519::Pair,
    device_identity: ed25519::Pair,
    device_id: Id32,
    node_pubkey: Id32,
    measurement: Id32,
}

fn world() -> World {
    World {
        root: ed25519::Pair::from_seed(&[1u8; 32]),
        device_identity: ed25519::Pair::from_seed(&[2u8; 32]),
        device_id: [0xD1; 32],
        node_pubkey: ed25519::Pair::from_seed(&[3u8; 32]).public().0,
        measurement: [0xAA; 32],
    }
}

/// Build valid evidence for `w`.
fn good_evidence(w: &World) -> Evidence {
    let mut cert = DeviceCert {
        device_id: w.device_id,
        device_pubkey: w.device_identity.public().0,
        vendor_sig: [0; 64],
    };
    cert.vendor_sig = w.root.sign(&cert.body()).0;

    let mut report = AttestationReport {
        device_id: w.device_id,
        measurement: w.measurement,
        node_pubkey: w.node_pubkey,
        report_sig: [0; 64],
    };
    report.report_sig = w.device_identity.sign(&report.body()).0;

    Evidence { cert, report }
}

#[test]
fn valid_evidence_returns_measurement() {
    let w = world();
    let roots = [w.root.public().0];
    let ev = good_evidence(&w);
    assert_eq!(
        verify_evidence(&roots, &w.device_id, &w.node_pubkey, &ev),
        Ok(w.measurement)
    );
}

#[test]
fn evidence_round_trips_through_scale() {
    let w = world();
    let ev = good_evidence(&w);
    let bytes = ev.encode();
    assert_eq!(Evidence::decode(&bytes), Some(ev));
    assert_eq!(Evidence::decode(&[0xFF, 0x00]), None);
}

#[test]
fn untrusted_root_is_rejected() {
    let w = world();
    let ev = good_evidence(&w);
    // Empty root set.
    assert_eq!(
        verify_evidence(&[], &w.device_id, &w.node_pubkey, &ev),
        Err(AttestationError::UntrustedRoot)
    );
    // A different root than the one that signed the cert: no configured root
    // verifies the cert signature, so it is untrusted.
    let other_root = ed25519::Pair::from_seed(&[9u8; 32]).public().0;
    assert_eq!(
        verify_evidence(&[other_root], &w.device_id, &w.node_pubkey, &ev),
        Err(AttestationError::UntrustedRoot)
    );
}

#[test]
fn forged_device_cert_is_rejected() {
    let w = world();
    let roots = [w.root.public().0];
    let mut ev = good_evidence(&w);
    // A device identity key the root never certified: re-sign the report with
    // an attacker key and swap the cert's device_pubkey to match, but the
    // vendor signature no longer covers it.
    let attacker = ed25519::Pair::from_seed(&[7u8; 32]);
    ev.cert.device_pubkey = attacker.public().0;
    ev.report.report_sig = attacker.sign(&ev.report.body()).0;
    // Swapping device_pubkey changes the cert body, so the trusted root's
    // vendor signature no longer verifies it — the cert is not vouched for.
    assert_eq!(
        verify_evidence(&roots, &w.device_id, &w.node_pubkey, &ev),
        Err(AttestationError::UntrustedRoot)
    );
}

#[test]
fn tampered_report_is_rejected() {
    let w = world();
    let roots = [w.root.public().0];
    let mut ev = good_evidence(&w);
    // Change the measurement without re-signing: report signature breaks.
    ev.report.measurement = [0xBB; 32];
    assert_eq!(
        verify_evidence(&roots, &w.device_id, &w.node_pubkey, &ev),
        Err(AttestationError::BadReport)
    );
}

#[test]
fn substituted_node_key_is_rejected() {
    let w = world();
    let roots = [w.root.public().0];
    let ev = good_evidence(&w);
    // Attacker presents genuine evidence but claims a different node key: the
    // report binds the real one, so registration for the attacker key fails.
    let attacker_node = ed25519::Pair::from_seed(&[8u8; 32]).public().0;
    assert_eq!(
        verify_evidence(&roots, &w.device_id, &attacker_node, &ev),
        Err(AttestationError::NodeKeyMismatch)
    );
}

#[test]
fn wrong_device_id_is_rejected() {
    let w = world();
    let roots = [w.root.public().0];
    let ev = good_evidence(&w);
    assert_eq!(
        verify_evidence(&roots, &[0xEE; 32], &w.node_pubkey, &ev),
        Err(AttestationError::DeviceIdMismatch)
    );
}

#[test]
fn report_device_id_must_match_cert() {
    let w = world();
    let roots = [w.root.public().0];
    let mut ev = good_evidence(&w);
    // Report claims a different device than the (validly signed) cert.
    ev.report.device_id = [0xEE; 32];
    ev.report.report_sig = w.device_identity.sign(&ev.report.body()).0;
    // Caller registers the cert's device id; the report's mismatch is caught.
    assert_eq!(
        verify_evidence(&roots, &w.device_id, &w.node_pubkey, &ev),
        Err(AttestationError::DeviceIdMismatch)
    );
}
