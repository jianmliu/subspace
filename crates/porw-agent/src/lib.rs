//! PoRW node-side agent.
//!
//! The agent is the component that lives beside a node (in a CVM in
//! production) and does everything a PoRW block author needs that is not
//! consensus logic: it holds the device node key, manages the resident model,
//! computes per-slot residency sketches through a pluggable [`SketchBackend`],
//! and assembles device-signed [`PorwSolution`]s that the node feeds to
//! `sc_consensus_subspace::porw::claim_porw_slot`.
//!
//! Everything here is GPU- and TEE-free: the [`cpu::CpuSketchBackend`] uses the
//! canonical reference sketch (bit-identical to the GPU kernels), so the whole
//! agent — and a devnet driven by it — runs and is validated on any machine.
//! The two things a real deployment swaps in are the sketch backend (a
//! GPU/HBM implementation of the same trait) and attestation evidence (from
//! the TEE instead of the [`testkit`] helper). Neither changes the agent's
//! control flow.
//!
//! ## Lifecycle
//!
//! ```text
//!   Unregistered ──register()──▶ Registered ──activate()──▶ Active
//!        │  (device attested,          │  (activation delay        │
//!        │   node key bound)           │   elapsed on chain)       │
//!        ▼                             ▼                           ▼
//!   produces nothing            produces nothing        produces per-slot
//!                                                        signed solutions
//! ```

use sp_core::{Pair, ed25519};
use subspace_proof_of_residency::PorwSolution;

pub mod backend;
pub mod cpu;
pub mod solution;
pub mod testkit;

pub use backend::SketchBackend;
pub use solution::{SlotContext, SolutionParams, assemble_solution};

/// Agent lifecycle state (mirrors the on-chain device lifecycle).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AgentState {
    /// Device not yet registered on chain.
    Unregistered,
    /// Registered and model announced, but still within the activation delay.
    Registered,
    /// Past the activation delay: eligible to author, produces solutions.
    Active,
}

/// Errors the agent surfaces.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    /// A solution was requested while the agent is not `Active`.
    #[error("agent is not active (state: {0:?})")]
    NotActive(AgentState),
    /// The coverage set referenced a tile the resident model does not have.
    #[error("coverage tile {tile} out of range (model has {tile_count} tiles)")]
    CoverageOutOfRange { tile: u64, tile_count: u64 },
}

/// The PoRW agent: node key + resident model backend + lifecycle.
pub struct PorwAgent<B: SketchBackend> {
    node_key: ed25519::Pair,
    device_id: [u8; 32],
    model_id: [u8; 32],
    ticket_unit: u64,
    backend: B,
    state: AgentState,
}

impl<B: SketchBackend> PorwAgent<B> {
    /// Create an agent for a device. `node_seed` derives the device node key
    /// (in production this key is generated inside the CVM and never leaves);
    /// `model_id` must be the backend model's `R_W` root.
    pub fn new(
        node_seed: [u8; 32],
        device_id: [u8; 32],
        model_id: [u8; 32],
        ticket_unit: u64,
        backend: B,
    ) -> Self {
        Self {
            node_key: ed25519::Pair::from_seed(&node_seed),
            device_id,
            model_id,
            ticket_unit,
            backend,
            state: AgentState::Unregistered,
        }
    }

    /// The device node public key (bound by attestation, verifies solutions).
    pub fn node_pubkey(&self) -> [u8; 32] {
        self.node_key.public().0
    }

    /// The device id this agent authors for.
    pub fn device_id(&self) -> [u8; 32] {
        self.device_id
    }

    /// The registered model commitment.
    pub fn model_id(&self) -> [u8; 32] {
        self.model_id
    }

    /// Current lifecycle state.
    pub fn state(&self) -> AgentState {
        self.state
    }

    /// Mark the device registered + model announced on chain.
    pub fn on_registered(&mut self) {
        if self.state == AgentState::Unregistered {
            self.state = AgentState::Registered;
        }
    }

    /// Mark the activation delay elapsed: the agent may now author.
    pub fn on_activated(&mut self) {
        if self.state == AgentState::Registered {
            self.state = AgentState::Active;
        }
    }

    /// Produce a device-signed solution for a slot, or an error if the agent
    /// is not active or the coverage set is invalid. Returns the solution and
    /// its ring-distance (the node decides whether it clears the range).
    pub fn author_slot(&self, ctx: &SlotContext) -> Result<(PorwSolution, u64), AgentError> {
        if self.state != AgentState::Active {
            return Err(AgentError::NotActive(self.state));
        }
        let tile_count = self.backend.tile_count();
        if let Some(&tile) = ctx.coverage.iter().find(|&&t| t >= tile_count) {
            return Err(AgentError::CoverageOutOfRange { tile, tile_count });
        }
        let params = SolutionParams {
            device_id: self.device_id,
            model_id: self.model_id,
            ticket_unit: self.ticket_unit,
        };
        Ok(assemble_solution(
            &self.backend,
            &self.node_key,
            &params,
            ctx,
        ))
    }
}

#[cfg(test)]
mod tests;
