"""
Number Theoretic Transform (NTT) for CRYSTALS-Kyber
=====================================================
Implements forward NTT, inverse NTT, and base-case multiplication
in ℤ_3329[x]/(x^256 + 1) following the reference pqcrystals/kyber.

The NTT converts a polynomial of degree < 256 into 128 degree-1 factors,
enabling O(n log n) polynomial multiplication instead of O(n²).

ζ = 17 is a primitive 256th root of unity mod 3329.
"""

from params import Q, N

# ── Precompute zetas in the exact order the reference uses ────────────────────
# The reference enumerates zetas as pow(17, brv(k), Q) for k = 0..127
# where brv(k) is the 7-bit bit-reversal of k.

def _bit_rev_7(x: int) -> int:
    """Reverse the lower 7 bits of x."""
    result = 0
    for _ in range(7):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result

# Build the zetas table exactly matching the pqcrystals reference
_ZETAS = [pow(17, _bit_rev_7(i), Q) for i in range(128)]


# ── Forward NTT (Cooley-Tukey butterfly) ──────────────────────────────────────

def ntt(f: list[int]) -> list[int]:
    """
    In-place NTT. Converts polynomial from coefficient to NTT domain.
    
    Follows the reference implementation:
      k = 1
      for len = 128, 64, 32, ..., 2:
          for start = 0, 2*len, 4*len, ...:
              zeta = ZETAS[k++]
              for j = start .. start+len-1:
                  t = zeta * f[j+len]
                  f[j+len] = f[j] - t
                  f[j]     = f[j] + t
    """
    r = list(f)
    k = 1
    len_ = 128
    while len_ >= 2:
        start = 0
        while start < 256:
            zeta = _ZETAS[k]
            k += 1
            for j in range(start, start + len_):
                t = (zeta * r[j + len_]) % Q
                r[j + len_] = (r[j] - t) % Q
                r[j] = (r[j] + t) % Q
            start += 2 * len_
        len_ //= 2
    return r


# ── Inverse NTT (Gentleman-Sande butterfly) ──────────────────────────────────

def ntt_inv(f: list[int]) -> list[int]:
    """
    In-place inverse NTT. Converts from NTT domain back to coefficients.
    
    Follows the reference implementation:
      k = 127
      for len = 2, 4, 8, ..., 128:
          for start = 0, 2*len, 4*len, ...:
              zeta = ZETAS[k--]
              for j = start .. start+len-1:
                  t = f[j]
                  f[j]     = t + f[j+len]
                  f[j+len] = zeta * (f[j+len] - t)
      f *= 3303  (which is 128^{-1} mod 3329)
    """
    r = list(f)
    k = 127
    len_ = 2
    while len_ <= 128:
        start = 0
        while start < 256:
            zeta = _ZETAS[k]
            k -= 1
            for j in range(start, start + len_):
                t = r[j]
                r[j] = (t + r[j + len_]) % Q
                r[j + len_] = (zeta * (r[j + len_] - t)) % Q
            start += 2 * len_
        len_ *= 2

    # Multiply by n^{-1} mod q = 256^{-1} mod 3329
    # 256^{-1} mod 3329 = 3303  (since 256 * 3303 = 845568 = 254*3329 - 1 ≡ ... let's compute)
    f_inv = pow(128, -1, Q)  # The reference NTT is a 128-point NTT on pairs
    # Actually in the Kyber reference, the scaling factor is 3303 = pow(128, -1, Q)
    # because the NTT is over 128 butterflies
    for i in range(256):
        r[i] = (r[i] * f_inv) % Q
    return r


# ── Base-case multiplication (degree-1 × degree-1 mod x² − ζ) ────────────────

def basemul(a0: int, a1: int, b0: int, b1: int, zeta: int) -> tuple[int, int]:
    """
    Multiply (a0 + a1·x) × (b0 + b1·x) mod (x² − ζ).
    Result: (a0·b0 + a1·b1·ζ,  a0·b1 + a1·b0)
    """
    c0 = (a0 * b0 + a1 * b1 * zeta) % Q
    c1 = (a0 * b1 + a1 * b0) % Q
    return c0, c1


def poly_basemul(a: list[int], b: list[int]) -> list[int]:
    """
    Pointwise multiplication of two NTT-domain polynomials.
    
    The NTT representation consists of 128 pairs of coefficients.
    Each pair (a[2i], a[2i+1]) represents a degree-1 polynomial
    evaluated at a root of x² − ζ^{2·brv(i)+1}.
    
    Returns the product (still in NTT domain).
    """
    r = [0] * N
    for i in range(64):
        # Zeta for the first pair in each group of 4
        z0 = _ZETAS[64 + i]
        
        r0, r1 = basemul(a[4*i+0], a[4*i+1], b[4*i+0], b[4*i+1], z0)
        r[4*i+0] = r0
        r[4*i+1] = r1

        r2, r3 = basemul(a[4*i+2], a[4*i+3], b[4*i+2], b[4*i+3], (Q - z0) % Q)
        r[4*i+2] = r2
        r[4*i+3] = r3

    return r
