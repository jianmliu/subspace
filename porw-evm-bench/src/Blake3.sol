// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/// @title BLAKE3 in Solidity (hash mode, 32-byte output)
/// @notice Faithful port of the BLAKE3 reference for arbitrary-length inputs
/// (chunk tree included), used to benchmark the PoRW `sketch-tile:v2`
/// dispute path on the EVM. Correctness is established by differential tests
/// against the Rust reference via `conformance/sketch-tile-v2.json`.
/// This is a benchmarking implementation: clarity first, no assembly.
library Blake3 {
    uint32 private constant IV0 = 0x6A09E667;
    uint32 private constant IV1 = 0xBB67AE85;
    uint32 private constant IV2 = 0x3C6EF372;
    uint32 private constant IV3 = 0xA54FF53A;
    uint32 private constant IV4 = 0x510E527F;
    uint32 private constant IV5 = 0x9B05688C;
    uint32 private constant IV6 = 0x1F83D9AB;
    uint32 private constant IV7 = 0x5BE0CD19;

    uint32 private constant CHUNK_START = 1;
    uint32 private constant CHUNK_END = 2;
    uint32 private constant PARENT = 4;
    uint32 private constant ROOT = 8;

    uint256 private constant CHUNK_LEN = 1024;
    uint256 private constant BLOCK_LEN = 64;

    function iv() private pure returns (uint32[8] memory cv) {
        cv[0] = IV0;
        cv[1] = IV1;
        cv[2] = IV2;
        cv[3] = IV3;
        cv[4] = IV4;
        cv[5] = IV5;
        cv[6] = IV6;
        cv[7] = IV7;
    }

    function rotr(uint32 x, uint32 n) private pure returns (uint32) {
        unchecked {
            return (x >> n) | (x << (32 - n));
        }
    }

    function g(uint32[16] memory s, uint256 a, uint256 b, uint256 c, uint256 d, uint32 mx, uint32 my) private pure {
        unchecked {
            s[a] = s[a] + s[b] + mx;
            s[d] = rotr(s[d] ^ s[a], 16);
            s[c] = s[c] + s[d];
            s[b] = rotr(s[b] ^ s[c], 12);
            s[a] = s[a] + s[b] + my;
            s[d] = rotr(s[d] ^ s[a], 8);
            s[c] = s[c] + s[d];
            s[b] = rotr(s[b] ^ s[c], 7);
        }
    }

    function round(uint32[16] memory s, uint32[16] memory m) private pure {
        g(s, 0, 4, 8, 12, m[0], m[1]);
        g(s, 1, 5, 9, 13, m[2], m[3]);
        g(s, 2, 6, 10, 14, m[4], m[5]);
        g(s, 3, 7, 11, 15, m[6], m[7]);
        g(s, 0, 5, 10, 15, m[8], m[9]);
        g(s, 1, 6, 11, 12, m[10], m[11]);
        g(s, 2, 7, 8, 13, m[12], m[13]);
        g(s, 3, 4, 9, 14, m[14], m[15]);
    }

    /// MSG_PERMUTATION = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8]
    function permute(uint32[16] memory m) private pure returns (uint32[16] memory p) {
        p[0] = m[2];
        p[1] = m[6];
        p[2] = m[3];
        p[3] = m[10];
        p[4] = m[7];
        p[5] = m[0];
        p[6] = m[4];
        p[7] = m[13];
        p[8] = m[1];
        p[9] = m[11];
        p[10] = m[12];
        p[11] = m[5];
        p[12] = m[9];
        p[13] = m[14];
        p[14] = m[15];
        p[15] = m[8];
    }

    /// One BLAKE3 compression; returns the 8-word chaining value.
    function compress(uint32[8] memory cv, uint32[16] memory m, uint64 counter, uint32 blockLen, uint32 flags)
        internal
        pure
        returns (uint32[8] memory out)
    {
        uint32[16] memory s;
        s[0] = cv[0];
        s[1] = cv[1];
        s[2] = cv[2];
        s[3] = cv[3];
        s[4] = cv[4];
        s[5] = cv[5];
        s[6] = cv[6];
        s[7] = cv[7];
        s[8] = IV0;
        s[9] = IV1;
        s[10] = IV2;
        s[11] = IV3;
        s[12] = uint32(counter);
        s[13] = uint32(counter >> 32);
        s[14] = blockLen;
        s[15] = flags;

        round(s, m); // 1
        m = permute(m);
        round(s, m); // 2
        m = permute(m);
        round(s, m); // 3
        m = permute(m);
        round(s, m); // 4
        m = permute(m);
        round(s, m); // 5
        m = permute(m);
        round(s, m); // 6
        m = permute(m);
        round(s, m); // 7

        for (uint256 i = 0; i < 8; i++) {
            out[i] = s[i] ^ s[i + 8];
        }
    }

    /// Load one 64-byte block (zero-padded past `len`) from `data` starting at
    /// `offset`, as 16 little-endian u32 words.
    function loadBlock(bytes memory data, uint256 offset, uint256 len) private pure returns (uint32[16] memory m) {
        unchecked {
            for (uint256 j = 0; j < 16; j++) {
                uint32 w = 0;
                for (uint256 b = 0; b < 4; b++) {
                    uint256 idx = j * 4 + b;
                    if (idx < len) {
                        w |= uint32(uint8(data[offset + idx])) << uint32(b * 8);
                    }
                }
                m[j] = w;
            }
        }
    }

    /// Chaining value of chunk `chunkIndex` covering data[offset .. offset+len)
    /// (len in 1..=1024). `rootFlags` is ROOT when this chunk IS the whole tree.
    function chunkCv(bytes memory data, uint256 offset, uint256 len, uint64 chunkIndex, uint32 rootFlags)
        private
        pure
        returns (uint32[8] memory cv)
    {
        cv = iv();
        uint256 nblocks = len == 0 ? 1 : (len + BLOCK_LEN - 1) / BLOCK_LEN;
        for (uint256 i = 0; i < nblocks; i++) {
            uint256 blockOffset = i * BLOCK_LEN;
            uint256 blockLen = len - blockOffset > BLOCK_LEN ? BLOCK_LEN : len - blockOffset;
            uint32 flags = 0;
            if (i == 0) flags |= CHUNK_START;
            if (i == nblocks - 1) flags |= CHUNK_END | rootFlags;
            uint32[16] memory m = loadBlock(data, offset + blockOffset, blockLen);
            cv = compress(cv, m, chunkIndex, uint32(blockLen), flags);
        }
    }

    /// Parent-node chaining value over two child CVs.
    function parentCv(uint32[8] memory left, uint32[8] memory right, uint32 rootFlags)
        private
        pure
        returns (uint32[8] memory)
    {
        uint32[16] memory m;
        for (uint256 i = 0; i < 8; i++) {
            m[i] = left[i];
            m[i + 8] = right[i];
        }
        return compress(iv(), m, 0, uint32(BLOCK_LEN), PARENT | rootFlags);
    }

    /// Largest power of two strictly less than n (n >= 2).
    function leftLen(uint256 n) private pure returns (uint256 p) {
        p = 1;
        while (p * 2 < n) {
            p *= 2;
        }
    }

    /// Subtree CV over chunks [firstChunk, firstChunk + nchunks) of data.
    function subtreeCv(bytes memory data, uint256 offset, uint256 len, uint64 firstChunk, uint32 rootFlags)
        private
        pure
        returns (uint32[8] memory)
    {
        uint256 nchunks = (len + CHUNK_LEN - 1) / CHUNK_LEN;
        if (nchunks <= 1) {
            return chunkCv(data, offset, len, firstChunk, rootFlags);
        }
        uint256 leftChunks = leftLen(nchunks);
        uint256 leftBytes = leftChunks * CHUNK_LEN;
        uint32[8] memory l = subtreeCv(data, offset, leftBytes, firstChunk, 0);
        uint32[8] memory r = subtreeCv(data, offset + leftBytes, len - leftBytes, firstChunk + uint64(leftChunks), 0);
        return parentCv(l, r, rootFlags);
    }

    function cvToBytes32(uint32[8] memory cv) private pure returns (bytes32 out) {
        uint256 acc = 0;
        unchecked {
            for (uint256 i = 0; i < 8; i++) {
                uint32 w = cv[i];
                // little-endian byte order within each word, words in order
                uint256 swapped = (uint256(w & 0xFF) << 24) | (uint256((w >> 8) & 0xFF) << 16)
                    | (uint256((w >> 16) & 0xFF) << 8) | uint256(w >> 24);
                acc = (acc << 32) | swapped;
            }
        }
        out = bytes32(acc);
    }

    /// BLAKE3 hash (hash mode, 32-byte output) of `data`.
    function hash(bytes memory data) internal pure returns (bytes32) {
        if (data.length == 0) {
            return cvToBytes32(chunkCv(data, 0, 0, 0, ROOT));
        }
        return cvToBytes32(subtreeCv(data, 0, data.length, 0, ROOT));
    }
}
