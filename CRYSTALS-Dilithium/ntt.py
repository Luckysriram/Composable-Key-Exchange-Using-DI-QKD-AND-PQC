"""
ntt.py - Number Theoretic Transform for CRYSTALS-Dilithium

The NTT operates in Z_q[x]/(x^256 + 1) where q = 8380417.
This is a "negative wrapped convolution" NTT using the primitive
512th root of unity zeta = 1753.

The transform maps polynomial multiplication to pointwise multiplication,
enabling O(n log n) polynomial multiplication instead of O(n^2).
"""

Q = 8380417       # Dilithium prime: 2^23 - 2^13 + 1
N = 256           # Polynomial degree
ZETA = 1753       # Primitive 512th root of unity mod Q
# (ZETA^256 ≡ -1 mod Q, so ZETA^512 ≡ 1 mod Q)


def _bit_rev8(x: int) -> int:
    """Reverse the 8 least significant bits of x."""
    result = 0
    for _ in range(8):
        result = (result << 1) | (x & 1)
        x >>= 1
    return result


# Precomputed zeta powers in bit-reversed order.
# ZETAS[k] = ZETA^(bit_rev8(k)) mod Q
# ZETAS[0] = 1 (unused), ZETAS[1..255] used in NTT butterflies.
ZETAS: list[int] = [pow(ZETA, _bit_rev8(i), Q) for i in range(256)]

# Precomputed modular inverse of N for use in inverse NTT
N_INV: int = pow(N, Q - 2, Q)   # 8347681


def ntt(poly: list[int]) -> list[int]:
    """
    Forward NTT: polynomial -> NTT representation.

    Uses Cooley-Tukey butterfly with decreasing lengths (128 -> 1).
    Each butterfly: t = zeta * a[j+len]; a[j+len] = a[j]-t; a[j] = a[j]+t

    Args:
        poly: List of N coefficients mod Q

    Returns:
        NTT representation (list of N values mod Q)
    """
    a = list(poly)
    k = 0
    length = 128
    while length >= 1:
        start = 0
        while start < N:
            k += 1
            zeta = ZETAS[k]
            for j in range(start, start + length):
                t = zeta * a[j + length] % Q
                a[j + length] = (a[j] - t) % Q
                a[j] = (a[j] + t) % Q
            start += 2 * length
        length >>= 1
    return a


def inv_ntt(poly: list[int]) -> list[int]:
    """
    Inverse NTT: NTT representation -> polynomial.

    Uses Gentleman-Sande butterfly with increasing lengths (1 -> 128).
    Uses negated zetas (since inverse uses zeta^-1 = -(zeta) for this ring).

    Args:
        poly: NTT representation (list of N values mod Q)

    Returns:
        Polynomial coefficients (list of N values mod Q)
    """
    a = list(poly)
    k = 256
    length = 1
    while length <= 128:
        start = 0
        while start < N:
            k -= 1
            zeta = (Q - ZETAS[k]) % Q   # negated zeta = zeta^(-1) in this ring
            for j in range(start, start + length):
                t = a[j]
                a[j] = (t + a[j + length]) % Q
                a[j + length] = zeta * (t - a[j + length]) % Q
            start += 2 * length
        length <<= 1
    # Final scaling by N^(-1)
    return [(x * N_INV) % Q for x in a]


def poly_pointwise_mul(a: list[int], b: list[int]) -> list[int]:
    """
    Pointwise (coefficient-wise) multiplication of two NTT-domain polynomials.
    Both inputs must already be in NTT form.
    """
    return [(x * y) % Q for x, y in zip(a, b)]


def poly_mul_ntt(a: list[int], b: list[int]) -> list[int]:
    """
    Full polynomial multiplication using NTT:
    1. Forward NTT on both inputs
    2. Pointwise multiply
    3. Inverse NTT
    """
    return inv_ntt(poly_pointwise_mul(ntt(a), ntt(b)))


if __name__ == "__main__":
    # Self-test: verify NTT roundtrip
    import random
    p = [random.randint(0, Q - 1) for _ in range(N)]
    assert p == inv_ntt(ntt(p)), "NTT roundtrip failed!"
    print("[ntt.py] NTT roundtrip: PASS")

    # Verify polynomial multiplication via NTT is correct
    # p(x) = x, q(x) = x => p*q = x^2
    a = [0] * N; a[1] = 1   # polynomial x
    b = [0] * N; b[1] = 1   # polynomial x
    c = poly_mul_ntt(a, b)
    assert c[2] == 1 and c[0] == 0, "x * x should be x^2"
    print("[ntt.py] Poly multiply x*x=x^2: PASS")

    # Verify x^N = -1 in Z_q[x]/(x^N + 1)
    # x^128 * x^128 = x^256 = -1 mod (x^256+1) => coefficient of x^0 is Q-1
    a = [0] * N; a[128] = 1
    b = [0] * N; b[128] = 1
    c = poly_mul_ntt(a, b)
    assert c[0] == Q - 1, f"x^128 * x^128 should give -1 at index 0, got {c[0]}"
    print("[ntt.py] Negacyclic property x^256=-1: PASS")
