"""
poly.py - Polynomial arithmetic in Z_q[x]/(x^256 + 1)

This module implements:
- Basic polynomial operations (+, -, scalar multiply)
- Centered reduction (mapping to [-q/2, q/2])
- Power2Round: splits coefficient into high/low parts
- Decompose: more general splitting for hint generation
- HighBits / LowBits
- MakeHint / UseHint: for compression in signatures
- Vector and matrix operations over polynomials
"""

from ntt import Q, N, ntt, inv_ntt, poly_pointwise_mul, poly_mul_ntt


# ─── Basic polynomial operations ─────────────────────────────────────────────

def poly_add(a: list[int], b: list[int]) -> list[int]:
    """Add two polynomials coefficient-wise mod Q."""
    return [(x + y) % Q for x, y in zip(a, b)]


def poly_sub(a: list[int], b: list[int]) -> list[int]:
    """Subtract polynomial b from a, coefficient-wise mod Q."""
    return [(x - y) % Q for x, y in zip(a, b)]


def poly_negate(a: list[int]) -> list[int]:
    """Negate polynomial: -a mod Q."""
    return [(Q - x) % Q for x in a]


def poly_scalar_mul(a: list[int], s: int) -> list[int]:
    """Multiply polynomial by scalar s mod Q."""
    return [(x * s) % Q for x in a]


def poly_zero() -> list[int]:
    """Return the zero polynomial."""
    return [0] * N


def center_coeff(x: int) -> int:
    """
    Map coefficient x (mod Q) to centered representative in (-Q/2, Q/2].
    Used for computing norms.
    """
    x = x % Q
    if x > Q // 2:
        x -= Q
    return x


def poly_center(a: list[int]) -> list[int]:
    """Map all coefficients to centered representation."""
    return [center_coeff(x) for x in a]


def inf_norm(a: list[int]) -> int:
    """Infinity norm: max |coeff| where coefficients are centered mod Q."""
    return max(abs(center_coeff(x)) for x in a)


def vec_inf_norm(v: list[list[int]]) -> int:
    """Infinity norm of a vector of polynomials."""
    return max(inf_norm(p) for p in v)


# ─── Power2Round and Decompose ────────────────────────────────────────────────

def power2round(r: int, d: int) -> tuple[int, int]:
    """
    Decompose r ∈ Z_q into (r1, r0) such that:
        r ≡ r1 * 2^d + r0  (mod q)
    with |r0| <= 2^(d-1).

    Used for splitting t into t1 (high bits, public) and t0 (low bits, secret).

    Args:
        r: coefficient in [0, q-1]
        d: number of low bits to drop (d=13 in Dilithium)

    Returns:
        (r1, r0) where r1 in [0, (q-1)/2^d] and r0 in (-2^(d-1), 2^(d-1)]
    """
    r = r % Q
    r0 = r % (1 << d)
    if r0 > (1 << (d - 1)):
        r0 -= (1 << d)
    r1 = (r - r0) >> d
    return r1, r0


def decompose(r: int, alpha: int) -> tuple[int, int]:
    """
    Decompose r ∈ Z_q into (r1, r0) such that:
        r ≡ r1 * alpha + r0  (mod q)
    with |r0| <= alpha/2.

    Special case: if r+ - r0 = q-1 (would give r1 = (q-1)/alpha),
    set r1=0 and adjust r0.

    Used for HighBits/LowBits decomposition.

    Args:
        r: coefficient in [0, q-1]
        alpha: decomposition parameter (= 2*gamma2)

    Returns:
        (r1, r0)
    """
    r = r % Q
    r0 = r % alpha
    if r0 > alpha // 2:
        r0 -= alpha
    # Special case: r - r0 = q-1 (top residue class)
    if r - r0 == Q - 1:
        return 0, r0 - 1
    r1 = (r - r0) // alpha
    return r1, r0


def high_bits(r: int, alpha: int) -> int:
    """Return the high bits r1 from decompose(r, alpha)."""
    r1, _ = decompose(r, alpha)
    return r1


def low_bits(r: int, alpha: int) -> int:
    """Return the low bits r0 from decompose(r, alpha)."""
    _, r0 = decompose(r, alpha)
    return r0


def make_hint(z: int, r: int, alpha: int) -> int:
    """
    Compute hint bit h indicating whether adding z to r changes the high bits.

    h = 0 if HighBits(r, alpha) == HighBits(r+z, alpha)
    h = 1 otherwise

    In signing: z = -c*t0, r = w - c*s2 (= w - c*s2)
    The hint allows the verifier to recover the correct high bits.
    """
    r1 = high_bits(r % Q, alpha)
    v1 = high_bits((r + z) % Q, alpha)
    return 0 if r1 == v1 else 1


def use_hint(h: int, r: int, alpha: int) -> int:
    """
    Use hint bit h to recover the correct high bits of r + z.

    Given:
        h = MakeHint(-ct0, w - cs2, alpha)
        r = (A*z - c*t1*2^d) mod q  (what verifier computes)

    Returns the correct w1 value.
    """
    m = (Q - 1) // alpha    # number of distinct r1 values
    r1, r0 = decompose(r % Q, alpha)
    if h == 1:
        if r0 > 0:
            return (r1 + 1) % m
        else:
            return (r1 - 1) % m
    return r1


# ─── Polynomial-level decomposition ──────────────────────────────────────────

def poly_power2round(poly: list[int], d: int) -> tuple[list[int], list[int]]:
    """Apply Power2Round to each coefficient of polynomial."""
    r1_coeffs, r0_coeffs = [], []
    for c in poly:
        r1, r0 = power2round(c, d)
        r1_coeffs.append(r1)
        r0_coeffs.append(r0)
    return r1_coeffs, r0_coeffs


def poly_decompose(poly: list[int], alpha: int) -> tuple[list[int], list[int]]:
    """Apply Decompose to each coefficient."""
    r1_coeffs, r0_coeffs = [], []
    for c in poly:
        r1, r0 = decompose(c, alpha)
        r1_coeffs.append(r1)
        r0_coeffs.append(r0)
    return r1_coeffs, r0_coeffs


def poly_high_bits(poly: list[int], alpha: int) -> list[int]:
    return [high_bits(x, alpha) for x in poly]


def poly_low_bits(poly: list[int], alpha: int) -> list[int]:
    return [low_bits(x, alpha) for x in poly]


def poly_make_hint(z_poly: list[int], r_poly: list[int], alpha: int) -> list[int]:
    return [make_hint(z, r, alpha) for z, r in zip(z_poly, r_poly)]


def poly_use_hint(h_poly: list[int], r_poly: list[int], alpha: int) -> list[int]:
    return [use_hint(h, r, alpha) for h, r in zip(h_poly, r_poly)]


# ─── Vector/matrix operations ─────────────────────────────────────────────────

def vec_add(u: list[list[int]], v: list[list[int]]) -> list[list[int]]:
    """Add two polynomial vectors."""
    return [poly_add(a, b) for a, b in zip(u, v)]


def vec_sub(u: list[list[int]], v: list[list[int]]) -> list[list[int]]:
    """Subtract polynomial vector v from u."""
    return [poly_sub(a, b) for a, b in zip(u, v)]


def vec_negate(v: list[list[int]]) -> list[list[int]]:
    return [poly_negate(p) for p in v]


def mat_vec_mul(A_ntt: list[list[list[int]]], v: list[list[int]]) -> list[list[int]]:
    """
    Multiply matrix A (in NTT domain) by polynomial vector v.

    A_ntt is k×l matrix of NTT polynomials.
    v is l-vector of polynomials (not in NTT domain).

    Algorithm:
      1. Convert each v[j] to NTT domain
      2. For each row i: accumulate sum_j A[i][j] * v_ntt[j] pointwise
      3. Convert accumulated result back from NTT

    This is the Module-LWE multiplication: result = A * v mod (q, x^n + 1)
    """
    v_ntt = [ntt(p) for p in v]
    result = []
    for row in A_ntt:
        # Accumulate sum in NTT domain
        acc = [0] * N
        for a_poly_ntt, s_poly_ntt in zip(row, v_ntt):
            for i in range(N):
                acc[i] = (acc[i] + a_poly_ntt[i] * s_poly_ntt[i]) % Q
        result.append(inv_ntt(acc))
    return result


def vec_scalar_poly_mul(c: list[int], v: list[list[int]]) -> list[list[int]]:
    """
    Multiply each polynomial in vector v by polynomial c.
    Uses NTT for efficiency.
    """
    c_ntt = ntt(c)
    result = []
    for p in v:
        p_ntt = ntt(p)
        r_ntt = poly_pointwise_mul(c_ntt, p_ntt)
        result.append(inv_ntt(r_ntt))
    return result


def vec_power2round(v: list[list[int]], d: int) -> tuple[list[list[int]], list[list[int]]]:
    """Apply Power2Round to each polynomial in vector."""
    v1, v0 = [], []
    for poly in v:
        r1, r0 = poly_power2round(poly, d)
        v1.append(r1)
        v0.append(r0)
    return v1, v0


def vec_high_bits(v: list[list[int]], alpha: int) -> list[list[int]]:
    return [poly_high_bits(p, alpha) for p in v]


def vec_low_bits(v: list[list[int]], alpha: int) -> list[list[int]]:
    return [poly_low_bits(p, alpha) for p in v]


def vec_make_hint(z_vec: list[list[int]], r_vec: list[list[int]], alpha: int) -> list[list[int]]:
    return [poly_make_hint(z, r, alpha) for z, r in zip(z_vec, r_vec)]


def vec_use_hint(h_vec: list[list[int]], r_vec: list[list[int]], alpha: int) -> list[list[int]]:
    return [poly_use_hint(h, r, alpha) for h, r in zip(h_vec, r_vec)]


def count_hints(h_vec: list[list[int]]) -> int:
    """Count total number of 1 bits in hint vector."""
    return sum(sum(h) for h in h_vec)


if __name__ == "__main__":
    print("[poly.py] Testing polynomial operations...")

    # Test Power2Round roundtrip
    r = 12345
    d = 13
    r1, r0 = power2round(r, d)
    recovered = (r1 * (1 << d) + r0) % Q
    assert recovered == r % Q, f"Power2Round failed: {r1}*2^{d} + {r0} != {r}"
    print(f"  power2round({r}, {d}) = ({r1}, {r0}) -> recovers {recovered}: PASS")

    # Test decompose
    alpha = (Q - 1) // 44   # gamma2 = (q-1)/88, alpha = 2*gamma2 = (q-1)/44
    r = 500000
    r1, r0 = decompose(r, alpha)
    assert abs(r0) <= alpha // 2, f"r0={r0} out of range"
    print(f"  decompose({r}, {alpha}) = ({r1}, {r0}): PASS")

    # Test inf_norm
    p = [0] * N
    p[0] = Q - 1   # coefficient = -1 in centered representation
    assert inf_norm(p) == 1, f"Expected norm 1, got {inf_norm(p)}"
    print("  inf_norm test: PASS")

    # Test matrix-vector multiply dimensions
    from utils import expand_A, DilithiumParams
    print("[poly.py] All basic tests PASS")
