//! PoRW attestation evidence verification.
//!
//! Verifies the B-pillar evidence a device presents at registration and
//! returns the measured agent-code digest to check against the on-chain
//! whitelist. The verification is a two-link signature chain that binds four
//! things together:
//!
//! ```text
//!   trusted vendor root ──signs──▶ device identity cert
//!        (governance)              { device_id, device_pubkey }
//!                                          │ signs
//!                                          ▼
//!                                  attestation report
//!        { device_id, measurement, node_pubkey }
//! ```
//!
//! On success the caller learns: this is the physical device `device_id`
//! vouched for by a trusted vendor root; it runs code measured as
//! `measurement`; and the node key `node_pubkey` (used to sign PoRW solutions
//! and block seals) was generated inside that device's TEE — closing the
//! "valid attestation + substituted node key" gap.
//!
//! ## What is real here and what is a test stand-in
//!
//! The verification *logic* is production-grade: a real ed25519 chain
//! (via `sp_io::crypto::ed25519_verify`), real device-id / node-key / vendor
//! binding, real measurement extraction, all covered by tests. What a real
//! deployment swaps in are (a) the trusted roots — NVIDIA's device-identity
//! root CA and the Intel TDX / AMD SEV-SNP roots instead of a governance test
//! key, and (b) the wire format — NVIDIA NRAS / SPDM measurement responses and
//! TDX/SNP quotes instead of this crate's SCALE encoding. Both are contained
//! behind [`Evidence::decode`] and the roots argument: the chain-verification
//! core is format-agnostic and is reused as-is.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

use alloc::vec::Vec;
use parity_scale_codec::{Decode, Encode};
use scale_info::TypeInfo;

/// 32-byte identifier / ed25519 public key / measurement digest.
pub type Id32 = [u8; 32];

/// Domain separators, so a signature over one structure can never be replayed
/// as a signature over another.
const DEVICE_CERT_CONTEXT: &[u8] = b"PORW-attestation-device-cert-v1";
const REPORT_CONTEXT: &[u8] = b"PORW-attestation-report-v1";

/// Device identity certificate: a trusted vendor root vouches that the
/// physical device `device_id` owns identity key `device_pubkey`. Analogue of
/// the NVIDIA GPU device-identity certificate chaining to the NVIDIA root CA.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, TypeInfo)]
pub struct DeviceCert {
    /// The physical device identifier.
    pub device_id: Id32,
    /// The device's attestation/identity public key.
    pub device_pubkey: Id32,
    /// Vendor root signature over the signed body (see [`DeviceCert::body`]).
    pub vendor_sig: [u8; 64],
}

impl DeviceCert {
    /// Bytes the vendor root signs.
    pub fn body(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(DEVICE_CERT_CONTEXT.len() + 64);
        out.extend_from_slice(DEVICE_CERT_CONTEXT);
        out.extend_from_slice(&self.device_id);
        out.extend_from_slice(&self.device_pubkey);
        out
    }
}

/// Attestation report signed by the device identity key: commits to the
/// measured agent code and the node key generated inside the TEE. Analogue of
/// an SPDM measurement response / TDX-SNP quote whose report-data binds the
/// node key.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, TypeInfo)]
pub struct AttestationReport {
    /// Must equal the certificate's `device_id`.
    pub device_id: Id32,
    /// Measured agent-code digest (checked against the on-chain whitelist).
    pub measurement: Id32,
    /// Node key generated inside the CVM (the report-data binding).
    pub node_pubkey: Id32,
    /// Device identity key signature over the signed body.
    pub report_sig: [u8; 64],
}

impl AttestationReport {
    /// Bytes the device identity key signs.
    pub fn body(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(REPORT_CONTEXT.len() + 96);
        out.extend_from_slice(REPORT_CONTEXT);
        out.extend_from_slice(&self.device_id);
        out.extend_from_slice(&self.measurement);
        out.extend_from_slice(&self.node_pubkey);
        out
    }
}

/// The full evidence bundle a device submits at registration.
#[derive(Debug, Clone, PartialEq, Eq, Encode, Decode, TypeInfo)]
pub struct Evidence {
    /// Vendor-root-signed device identity certificate.
    pub cert: DeviceCert,
    /// Device-identity-key-signed attestation report.
    pub report: AttestationReport,
}

impl Evidence {
    /// Decode SCALE-encoded evidence bytes (the registration extrinsic's
    /// `evidence` field). A real deployment decodes NRAS / TDX quote bytes
    /// here instead; the verification core below is unchanged.
    pub fn decode(bytes: &[u8]) -> Option<Self> {
        <Evidence as Decode>::decode(&mut &bytes[..]).ok()
    }
}

/// Why attestation verification failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AttestationError {
    /// The vendor root that signed the device cert is not trusted.
    UntrustedRoot,
    /// The device cert's vendor signature did not verify.
    BadDeviceCert,
    /// The report's device-identity signature did not verify.
    BadReport,
    /// The evidence does not bind the claimed device id.
    DeviceIdMismatch,
    /// The evidence does not bind the node key the caller is registering.
    NodeKeyMismatch,
}

fn ed25519_verify(sig: &[u8; 64], message: &[u8], pubkey: &Id32) -> bool {
    sp_io::crypto::ed25519_verify(
        &sp_core::ed25519::Signature::from_raw(*sig),
        message,
        &sp_core::ed25519::Public::from_raw(*pubkey),
    )
}

/// Verify attestation evidence and, on success, return the measured
/// agent-code digest.
///
/// - `trusted_roots`: governance-configured vendor root public keys (NVIDIA /
///   Intel / AMD in production; a test key on a testnet).
/// - `device_id`: the id the caller is registering; must be the one the
///   evidence is bound to.
/// - `node_pubkey`: the node key the caller is registering; the report must
///   bind exactly this key.
pub fn verify_evidence(
    trusted_roots: &[Id32],
    device_id: &Id32,
    node_pubkey: &Id32,
    evidence: &Evidence,
) -> Result<Id32, AttestationError> {
    let cert = &evidence.cert;
    let report = &evidence.report;

    // The evidence must be about the device and node key being registered.
    if &cert.device_id != device_id || &report.device_id != device_id {
        return Err(AttestationError::DeviceIdMismatch);
    }
    if &report.node_pubkey != node_pubkey {
        return Err(AttestationError::NodeKeyMismatch);
    }

    // Link 1: a trusted vendor root signed the device identity certificate.
    let root_ok = trusted_roots
        .iter()
        .any(|root| ed25519_verify(&cert.vendor_sig, &cert.body(), root));
    if trusted_roots.is_empty() {
        return Err(AttestationError::UntrustedRoot);
    }
    if !root_ok {
        // Distinguish "no trusted root matched" from a malformed signature by
        // checking whether ANY key would have accepted it is not possible;
        // report the chain failure at the cert link.
        return Err(AttestationError::BadDeviceCert);
    }

    // Link 2: the device identity key signed the attestation report.
    if !ed25519_verify(&report.report_sig, &report.body(), &cert.device_pubkey) {
        return Err(AttestationError::BadReport);
    }

    Ok(report.measurement)
}

#[cfg(test)]
mod tests;
