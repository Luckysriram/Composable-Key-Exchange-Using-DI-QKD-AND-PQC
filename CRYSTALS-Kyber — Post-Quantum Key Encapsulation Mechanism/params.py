"""
CRYSTALS-Kyber Parameter Sets
==============================
Defines parameters for Kyber512, Kyber768, Kyber1024 as specified in the
CRYSTALS-Kyber / ML-KEM specification (FIPS 203).

Key constants:
  n = 256         — polynomial ring degree
  q = 3329        — modulus (prime, q ≡ 1 mod 256)
  ZETAS           — precomputed roots of unity for NTT in bit-reversed order
"""

from dataclasses import dataclass

# ── Global constants ──────────────────────────────────────────────────────────

N = 256        # Ring dimension (degree of x^n + 1)
Q = 3329       # Modulus
MONT = 2285    # Montgomery factor: 2^16 mod q
QINV = 62209   # q^{-1} mod 2^16  (for Montgomery reduction)

# ── Precomputed NTT zetas (bit-reversed order) ───────────────────────────────
# ζ = 17 is a primitive 256-th root of unity mod 3329.
# These are ζ^{br(i)} mod q for i = 0..127 where br is 7-bit bit-reversal.

def _precompute_zetas():
    """Compute zetas in bit-reversed order for the NTT."""
    zeta = 17
    zetas = [0] * 128
    for i in range(128):
        # bit-reverse i in 7 bits
        br = int(f'{i:07b}'[::-1], 2)
        zetas[i] = pow(zeta, br, Q)
    return zetas

ZETAS = _precompute_zetas()

# Montgomery form of zetas (zeta * 2^16 mod q)
ZETAS_MONT = [(z * MONT) % Q for z in ZETAS]


# ── Parameter sets ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class KyberParams:
    """Holds all parameters for a specific Kyber security level."""
    name: str
    k: int           # Module dimension (number of polynomials in vector)
    eta1: int         # Noise parameter for secret / first noise vector
    eta2: int         # Noise parameter for second noise vector / encryption noise
    du: int           # Compression bits for u (ciphertext component)
    dv: int           # Compression bits for v (ciphertext component)

    @property
    def n(self) -> int:
        return N

    @property
    def q(self) -> int:
        return Q

    @property
    def pk_size(self) -> int:
        """Public key size in bytes: 12*k*n/8 + 32."""
        return 12 * self.k * N // 8 + 32

    @property
    def sk_size(self) -> int:
        """Secret key size in bytes (IND-CCA): 24*k*n/8 + 96."""
        return 24 * self.k * N // 8 + 96

    @property
    def ct_size(self) -> int:
        """Ciphertext size in bytes: du*k*n/8 + dv*n/8."""
        return self.du * self.k * N // 8 + self.dv * N // 8

    @property
    def ss_size(self) -> int:
        """Shared secret size in bytes."""
        return 32


# The three standardized parameter sets
KYBER512 = KyberParams(name="Kyber512",   k=2, eta1=3, eta2=2, du=10, dv=4)
KYBER768 = KyberParams(name="Kyber768",   k=3, eta1=2, eta2=2, du=10, dv=4)
KYBER1024 = KyberParams(name="Kyber1024", k=4, eta1=2, eta2=2, du=11, dv=5)

# Lookup table
PARAM_SETS = {
    "kyber512":  KYBER512,
    "kyber768":  KYBER768,
    "kyber1024": KYBER1024,
}
