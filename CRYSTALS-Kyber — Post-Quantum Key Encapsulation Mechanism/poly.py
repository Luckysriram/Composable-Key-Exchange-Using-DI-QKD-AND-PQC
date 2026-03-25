"""
Polynomial Arithmetic for CRYSTALS-Kyber
=========================================
Provides Poly (single polynomial in ℤ_q[x]/(x^256+1)) and PolyVec
(vector of k polynomials) with all operations needed by the scheme.

Polynomials can be in either "normal" (coefficient) or "NTT" domain.
Most heavy lifting is done in NTT domain for efficiency.
"""

from params import Q, N
from ntt import ntt, ntt_inv, poly_basemul


class Poly:
    """A polynomial in ℤ_3329[x]/(x^256 + 1), stored as 256 coefficients."""

    __slots__ = ('coeffs',)

    def __init__(self, coeffs: list[int] | None = None):
        if coeffs is None:
            self.coeffs = [0] * N
        else:
            self.coeffs = list(coeffs)

    # ── Arithmetic ────────────────────────────────────────────────────────

    def add(self, other: 'Poly') -> 'Poly':
        """Coefficient-wise addition mod q."""
        return Poly([(a + b) % Q for a, b in zip(self.coeffs, other.coeffs)])

    def sub(self, other: 'Poly') -> 'Poly':
        """Coefficient-wise subtraction mod q."""
        return Poly([(a - b) % Q for a, b in zip(self.coeffs, other.coeffs)])

    def mul_ntt(self, other: 'Poly') -> 'Poly':
        """Pointwise (base-case) multiplication in NTT domain."""
        return Poly(poly_basemul(self.coeffs, other.coeffs))

    def reduce(self) -> 'Poly':
        """Reduce all coefficients into [0, q)."""
        return Poly([c % Q for c in self.coeffs])

    # ── NTT transforms ────────────────────────────────────────────────────

    def to_ntt(self) -> 'Poly':
        """Convert from coefficient domain to NTT domain."""
        return Poly(ntt(self.coeffs))

    def from_ntt(self) -> 'Poly':
        """Convert from NTT domain back to coefficient domain."""
        return Poly(ntt_inv(self.coeffs))

    # ── Utility ───────────────────────────────────────────────────────────

    def copy(self) -> 'Poly':
        return Poly(list(self.coeffs))

    def __repr__(self) -> str:
        preview = self.coeffs[:5]
        return f"Poly({preview}... [{N} coeffs])"


class PolyVec:
    """A vector of k Poly objects."""

    __slots__ = ('polys', 'k')

    def __init__(self, polys: list[Poly]):
        self.polys = polys
        self.k = len(polys)

    @classmethod
    def zeros(cls, k: int) -> 'PolyVec':
        return cls([Poly() for _ in range(k)])

    # ── Arithmetic ────────────────────────────────────────────────────────

    def add(self, other: 'PolyVec') -> 'PolyVec':
        return PolyVec([a.add(b) for a, b in zip(self.polys, other.polys)])

    def sub(self, other: 'PolyVec') -> 'PolyVec':
        return PolyVec([a.sub(b) for a, b in zip(self.polys, other.polys)])

    def reduce(self) -> 'PolyVec':
        return PolyVec([p.reduce() for p in self.polys])

    # ── NTT transforms ────────────────────────────────────────────────────

    def to_ntt(self) -> 'PolyVec':
        return PolyVec([p.to_ntt() for p in self.polys])

    def from_ntt(self) -> 'PolyVec':
        return PolyVec([p.from_ntt() for p in self.polys])

    # ── Inner product in NTT domain ───────────────────────────────────────

    def inner_product_ntt(self, other: 'PolyVec') -> Poly:
        """Compute <self, other> = Σ self[i] * other[i]  (in NTT domain).
        Returns a single polynomial (still in NTT domain)."""
        acc = Poly()
        for a, b in zip(self.polys, other.polys):
            acc = acc.add(a.mul_ntt(b))
        return acc.reduce()

    # ── Matrix-vector product ─────────────────────────────────────────────

    @staticmethod
    def matrix_vec_ntt(matrix: list[list[Poly]], vec: 'PolyVec') -> 'PolyVec':
        """Compute matrix × vector in NTT domain.
        matrix[i][j] and vec.polys[j] are all in NTT domain.
        Returns a PolyVec of the resulting polynomials (NTT domain)."""
        k = len(matrix)
        result = []
        for i in range(k):
            acc = Poly()
            for j in range(k):
                acc = acc.add(matrix[i][j].mul_ntt(vec.polys[j]))
            result.append(acc.reduce())
        return PolyVec(result)

    def __repr__(self) -> str:
        return f"PolyVec(k={self.k})"
