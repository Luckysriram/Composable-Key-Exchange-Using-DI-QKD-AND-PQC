"""
Sampling Functions for CRYSTALS-Kyber
======================================
Implements:
  1. Centered Binomial Distribution (CBD) — for secret/error polynomials
  2. Uniform rejection sampling — for the public matrix A from XOF stream
"""

from params import Q, N
from utils import prf, xof
from poly import Poly


# ── Centered Binomial Distribution (CBD) ──────────────────────────────────────

def cbd(eta: int, data: bytes) -> Poly:
    """
    Sample a polynomial with coefficients from the Centered Binomial
    Distribution CBD_η.

    For each coefficient:
        a = sum of η random bits
        b = sum of η random bits
        coefficient = (a - b) mod q

    This gives coefficients in {-η, ..., η}.

    Input: 64·η bytes of pseudorandom data.
    Output: Poly with coefficients in [0, q) (representing values in {-η,...,η}).
    """
    coeffs = [0] * N
    bits = _bytes_to_bits(data)

    for i in range(N):
        a = sum(bits[2 * i * eta + j] for j in range(eta))
        b = sum(bits[2 * i * eta + eta + j] for j in range(eta))
        coeffs[i] = (a - b) % Q

    return Poly(coeffs)


def _bytes_to_bits(data: bytes) -> list[int]:
    """Convert a byte string to a list of bits (LSB first per byte)."""
    bits = []
    for byte in data:
        for i in range(8):
            bits.append((byte >> i) & 1)
    return bits


def sample_noise(seed: bytes, eta: int, nonce: int, k: int = 1) -> list[Poly]:
    """
    Sample k polynomials from CBD_η using PRF(seed, nonce), PRF(seed, nonce+1), ...

    Used to generate secret vectors s, error vectors e, and encryption noise.
    """
    polys = []
    for i in range(k):
        buf = prf(seed, nonce + i, 64 * eta)
        polys.append(cbd(eta, buf))
    return polys


# ── Uniform Rejection Sampling (for matrix A) ────────────────────────────────

def sample_ntt(seed: bytes, i: int, j: int) -> Poly:
    """
    Sample a uniformly random polynomial in NTT domain by rejection
    sampling from a SHAKE-128 (XOF) stream.

    The XOF stream is parsed 3 bytes at a time, yielding two 12-bit
    candidates. Each candidate < q is accepted as a coefficient.

    This directly produces an NTT-domain polynomial (no NTT call needed).
    """
    stream = xof(seed, i, j)
    coeffs = []
    pos = 0

    while len(coeffs) < N:
        # Parse 3 bytes → 2 candidates
        b0 = stream[pos]
        b1 = stream[pos + 1]
        b2 = stream[pos + 2]
        pos += 3

        d1 = b0 | ((b1 & 0x0F) << 8)       # first 12-bit value
        d2 = (b1 >> 4) | (b2 << 4)          # second 12-bit value

        if d1 < Q:
            coeffs.append(d1)
        if d2 < Q and len(coeffs) < N:
            coeffs.append(d2)

    return Poly(coeffs[:N])


def generate_matrix(seed: bytes, k: int, transpose: bool = False) -> list[list[Poly]]:
    """
    Generate the k×k public matrix  in NTT domain.

    Each entry Â[i][j] is sampled via rejection sampling from XOF(ρ, i, j).
    If transpose=True, we sample Â[j][i] instead (used in encryption).

    ┌                    ┐
    │ Â[0][0]  … Â[0][k-1] │
    │   ⋮            ⋮      │
    │ Â[k-1][0] … Â[k-1][k-1] │
    └                    ┘
    """
    matrix = []
    for i in range(k):
        row = []
        for j in range(k):
            if transpose:
                row.append(sample_ntt(seed, j, i))
            else:
                row.append(sample_ntt(seed, i, j))
        matrix.append(row)
    return matrix
