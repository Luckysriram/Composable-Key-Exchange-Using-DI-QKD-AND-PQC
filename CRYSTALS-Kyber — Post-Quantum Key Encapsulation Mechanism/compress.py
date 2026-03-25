"""
Compression and Decompression for CRYSTALS-Kyber
==================================================
Maps polynomial coefficients between full ℤ_q representation and
reduced d-bit representation for ciphertext size reduction.

Compress_d(x)   = ⌈(2^d / q) · x⌋ mod 2^d
Decompress_d(x) = ⌈(q / 2^d) · x⌋

These are lossy operations — decompress(compress(x)) ≈ x, not exact.
"""

from params import Q, N
from poly import Poly, PolyVec


def compress_int(x: int, d: int) -> int:
    """Compress a single coefficient x ∈ [0, q) to d bits."""
    # ⌈(2^d / q) · x⌋ mod 2^d
    return ((x << d) + Q // 2) // Q % (1 << d)


def decompress_int(x: int, d: int) -> int:
    """Decompress a d-bit value back to an approximation in [0, q)."""
    # ⌈(q / 2^d) · x⌋
    return (x * Q + (1 << (d - 1))) >> d


def compress_poly(p: Poly, d: int) -> list[int]:
    """Compress all 256 coefficients of a polynomial to d bits each."""
    return [compress_int(c, d) for c in p.coeffs]


def decompress_poly(coeffs: list[int], d: int) -> Poly:
    """Decompress a list of d-bit values back into a polynomial."""
    return Poly([decompress_int(c, d) for c in coeffs])


def compress_polyvec(pv: PolyVec, d: int) -> list[list[int]]:
    """Compress each polynomial in a PolyVec."""
    return [compress_poly(p, d) for p in pv.polys]


def decompress_polyvec(compressed: list[list[int]], d: int) -> PolyVec:
    """Decompress a list of coefficient lists back into a PolyVec."""
    return PolyVec([decompress_poly(c, d) for c in compressed])
