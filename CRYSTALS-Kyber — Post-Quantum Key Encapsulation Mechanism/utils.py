"""
Utility / Hash Functions for CRYSTALS-Kyber
============================================
Wraps SHA3 and SHAKE from Python's hashlib to provide the G, H, J, PRF, XOF
functions used in the Kyber specification.

Also provides byte-level encode/decode for polynomial coefficients.
"""

import hashlib
import os

from params import Q, N


# ── Hash function wrappers (per Kyber spec naming) ────────────────────────────

def hash_h(data: bytes) -> bytes:
    """H: SHA3-256.  Returns 32 bytes."""
    return hashlib.sha3_256(data).digest()


def hash_g(data: bytes) -> tuple[bytes, bytes]:
    """G: SHA3-512.  Returns (first 32 bytes, last 32 bytes)."""
    h = hashlib.sha3_512(data).digest()
    return h[:32], h[32:]


def prf(key: bytes, nonce: int, length: int) -> bytes:
    """PRF(s, b):  SHAKE-256(s || b) → length bytes.
    Used for deterministic noise sampling."""
    shake = hashlib.shake_256(key + nonce.to_bytes(1, 'little'))
    return shake.digest(length)


def xof(seed: bytes, i: int, j: int) -> bytes:
    """XOF(ρ, i, j): SHAKE-128(ρ || j || i) → stream bytes.
    Returns enough bytes for rejection sampling a polynomial (≤ 4·256 = 1024).
    We generate 840 bytes (spec uses up to ~672 for degree 256, with margin)."""
    data = seed + bytes([j, i])
    shake = hashlib.shake_128(data)
    return shake.digest(840)


def kdf(data: bytes) -> bytes:
    """KDF: SHAKE-256(data) → 32 bytes.  Final shared-secret derivation."""
    return hashlib.shake_256(data).digest(32)


def random_bytes(n: int) -> bytes:
    """Cryptographically secure random bytes."""
    return os.urandom(n)


# ── Byte encoding / decoding of polynomials ───────────────────────────────────

def encode_poly(coeffs: list[int], bits: int) -> bytes:
    """Encode a list of 256 coefficients, each in [0, 2^bits), into a byte array.
    This performs simple bit-packing (little-endian bit order)."""
    if bits == 12:
        return _encode_12(coeffs)

    buf = 0
    buf_len = 0
    out = bytearray()
    for c in coeffs:
        buf |= (c & ((1 << bits) - 1)) << buf_len
        buf_len += bits
        while buf_len >= 8:
            out.append(buf & 0xFF)
            buf >>= 8
            buf_len -= 8
    if buf_len > 0:
        out.append(buf & 0xFF)
    return bytes(out)


def decode_poly(data: bytes, bits: int) -> list[int]:
    """Decode a byte array back into 256 coefficients of `bits` each."""
    if bits == 12:
        return _decode_12(data)

    mask = (1 << bits) - 1
    coeffs = []
    buf = 0
    buf_len = 0
    idx = 0
    for _ in range(N):
        while buf_len < bits:
            buf |= data[idx] << buf_len
            idx += 1
            buf_len += 8
        coeffs.append(buf & mask)
        buf >>= bits
        buf_len -= bits
    return coeffs


def _encode_12(coeffs: list[int]) -> bytes:
    """Optimised 12-bit encoder: packs pairs of coefficients into 3 bytes."""
    out = bytearray()
    for i in range(0, N, 2):
        a = coeffs[i] & 0xFFF
        b = coeffs[i + 1] & 0xFFF
        out.append(a & 0xFF)
        out.append(((a >> 8) & 0x0F) | ((b & 0x0F) << 4))
        out.append((b >> 4) & 0xFF)
    return bytes(out)


def _decode_12(data: bytes) -> list[int]:
    """Optimised 12-bit decoder: unpacks 3 bytes into a pair of coefficients."""
    coeffs = []
    for i in range(0, len(data), 3):
        a = data[i] | ((data[i + 1] & 0x0F) << 8)
        b = ((data[i + 1] >> 4) & 0x0F) | (data[i + 2] << 4)
        coeffs.append(a)
        coeffs.append(b)
    return coeffs[:N]
