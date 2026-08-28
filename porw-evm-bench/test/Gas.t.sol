// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

import {Test} from "forge-std/Test.sol";
import {console2} from "forge-std/console2.sol";
import {Blake3} from "../src/Blake3.sol";
import {PorwVerifier} from "../src/PorwVerifier.sol";

/// Gas measurements for the EVM feasibility gate, at realistic tree depths:
/// a 70 GB model = ~17M 4 KiB tiles => weights-tree depth 25; a large
/// per-slot coverage set => partials depth up to 25 (21 used here for a 2M
/// tile coverage). Numbers are logged per case; run with `forge test -vv
/// --match-contract GasBench`.
contract GasBench is Test {
    PorwVerifier v;

    uint64 constant TILE_IDX = 3;
    uint32 constant SLOT_SEED = 1970174283;
    bytes32 constant CHALLENGE = 0x0909090909090909090909090909090909090909090909090909090909090909;
    bytes32 constant DEVICE = 0x0303030303030303030303030303030303030303030303030303030303030303;

    bytes tile;
    uint32 trueS;
    bytes32[] wProof; // depth 25 (17M-tile model)
    bytes32[] pProof; // depth 21 (2M-tile coverage)
    bytes32 modelRoot;
    bytes32 partialsRoot;

    function tileBytes(uint256 tileIdx) internal pure returns (bytes memory out) {
        out = new bytes(4096);
        unchecked {
            for (uint256 j = 0; j < 4096; j++) {
                uint64 x = uint64((tileIdx * 4096 + j) * 2654435761);
                out[j] = bytes1(uint8((x >> 7) & 0xFF));
            }
        }
    }

    function foldRoot(bytes32 leaf, bytes32[] memory proof, uint256 index) internal pure returns (bytes32 acc) {
        acc = leaf;
        for (uint256 i = 0; i < proof.length; i++) {
            acc = index % 2 == 0 ? Blake3.hash(bytes.concat(acc, proof[i])) : Blake3.hash(bytes.concat(proof[i], acc));
            index /= 2;
        }
    }

    function setUp() public {
        v = new PorwVerifier();
        tile = tileBytes(TILE_IDX);
        trueS = v.sketchTile(SLOT_SEED, TILE_IDX, tile);

        wProof = new bytes32[](25);
        for (uint256 i = 0; i < 25; i++) {
            wProof[i] = keccak256(abi.encode("w", i));
        }
        pProof = new bytes32[](21);
        for (uint256 i = 0; i < 21; i++) {
            pProof[i] = keccak256(abi.encode("p", i));
        }
        // Roots chosen so every check passes and the full path executes:
        // the committed value is trueS+1, so the verdict is Fraud.
        partialsRoot = foldRoot(v.partialsLeaf(TILE_IDX, trueS + 1), pProof, 1);
        modelRoot = foldRoot(v.weightsLeaf(TILE_IDX, tile), wProof, TILE_IDX);
    }

    function test_gas_sketch_tile() public view {
        uint256 g0 = gasleft();
        v.sketchTile(SLOT_SEED, TILE_IDX, tile);
        console2.log("sketchTile (4096B, 1024 words):", g0 - gasleft());
    }

    function test_gas_blake3_single_block() public view {
        uint256 g0 = gasleft();
        v.partialsLeaf(TILE_IDX, trueS);
        console2.log("partialsLeaf (12B blake3, 1 compression):", g0 - gasleft());
    }

    function test_gas_blake3_tile_hash() public view {
        uint256 g0 = gasleft();
        v.weightsLeaf(TILE_IDX, tile);
        console2.log("weightsLeaf (4104B blake3, ~69 compressions):", g0 - gasleft());
    }

    function test_gas_merkle_blake3_depth25() public view {
        bytes32 leaf = keccak256("leaf");
        uint256 g0 = gasleft();
        v.merkleVerify(modelRoot, leaf, TILE_IDX, wProof);
        console2.log("merkleVerify blake3 depth-25:", g0 - gasleft());
    }

    function test_gas_merkle_keccak_depth25() public view {
        bytes32 leaf = keccak256("leaf");
        uint256 g0 = gasleft();
        v.merkleVerifyKeccak(modelRoot, leaf, TILE_IDX, wProof);
        console2.log("merkleVerify keccak depth-25:", g0 - gasleft());
    }

    function test_gas_opening_committed_depth21() public view {
        uint256 g0 = gasleft();
        v.verifyOpeningCommitted(partialsRoot, 1 << 21, TILE_IDX, trueS + 1, 1, pProof);
        console2.log("verifyOpeningCommitted depth-21:", g0 - gasleft());
    }

    function test_gas_fraud_proof_full_path() public {
        uint256 g0 = gasleft();
        uint8 verdict = v.verifyTileFraudProof(
            partialsRoot, modelRoot, CHALLENGE, DEVICE, TILE_IDX, trueS + 1, 1, pProof, tile, wProof
        );
        uint256 used = g0 - gasleft();
        assertEq(verdict, 0); // Fraud: the full path executed
        console2.log("verifyTileFraudProof (p21 + w25 + tile hash + sketch):", used);
    }

    function test_gas_ecrecover_baseline() public {
        bytes32 h = keccak256("m");
        // Any valid signature; use a fixed known-good vector via vm.sign.
        (address a, uint256 pk) = makeAddrAndKey("worker");
        (uint8 vv, bytes32 r, bytes32 s) = vm.sign(pk, h);
        uint256 g0 = gasleft();
        address rec = ecrecover(h, vv, r, s);
        console2.log("ecrecover (device signature check):", g0 - gasleft());
        assertEq(rec, a);
    }

    function test_calldata_cost_estimate() public pure {
        // Fraud-proof calldata: 4096-byte tile (pseudo-random, ~all nonzero)
        // + 25*32 + 21*32 proof bytes + ~10 words of fixed fields.
        uint256 tileB = 4096;
        uint256 proofB = (25 + 21) * 32;
        uint256 fixedB = 10 * 32;
        uint256 total = tileB + proofB + fixedB;
        // EIP-2028: 16 gas per nonzero byte (worst case: all nonzero).
        console2.log("fraud-proof calldata bytes:", total);
        console2.log("fraud-proof calldata gas (worst case, 16/B):", total * 16);
        // Opening response: leaf fields + depth-21 proof.
        uint256 opening = 6 * 32 + 21 * 32;
        console2.log("opening calldata bytes:", opening);
        console2.log("opening calldata gas (16/B):", opening * 16);
    }
}
