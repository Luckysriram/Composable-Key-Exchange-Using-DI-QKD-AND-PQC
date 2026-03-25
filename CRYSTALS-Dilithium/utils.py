"""
utils.py - Utility functions for CRYSTALS-Dilithium

Implements:
- SHAKE128 / SHAKE256 hash wrappers (XOF - extendable output functions)
- Polynomial sampling:
    * expand_A: sample uniform matrix from seed rho
    * expand_S: sample secret vectors from seed rho_prime
    * expand_mask: sample mask vector from seed rho_prime_prime
    * sample_in_ball: generate sparse challenge polynomial
- Bit-packing / bit-unpacking for keys and signatures
- Parameter sets for Dilithium2, Dilithium3, Dilithium5
"""

import hashlib
import os
from dataclasses import dataclass
from ntt import Q, N, ntt


# ─── Dilithium Parameter Sets ─────────────────────────────────────────────────

@dataclass
class DilithiumParams:
    """
    Parameter set for one Dilithium variant.

    Security levels:
      Dilithium2: NIST Level 2 (≈ AES-128)
      Dilithium3: NIST Level 3 (≈ AES-192)
      Dilithium5: NIST Level 5 (≈ AES-256)

    Parameters:
      k, l  : matrix dimensions (A is k×l)
      eta   : secret polynomial coefficient bound (s1, s2 ∈ [-eta, eta])
      tau   : number of ±1 coefficients in challenge c
      beta  : bound for rejection: beta = tau * eta
      gamma1: mask bound
      gamma2: low bits bound (= (q-1)/(2*m) for m = 44 or 16)
      omega  : max number of hint bits
      d     : dropped bits in t (always 13)
    """
    name: str
    k: int
    l: int
    eta: int
    tau: int
    beta: int
    gamma1: int
    gamma2: int
    omega: int
    d: int = 13


DILITHIUM2 = DilithiumParams(
    name="Dilithium2",
    k=4, l=4, eta=2, tau=39, beta=78,
    gamma1=1 << 17,            # 131072
    gamma2=(Q - 1) // 88,     # 95232  = (q-1)/88
    omega=80,
)

DILITHIUM3 = DilithiumParams(
    name="Dilithium3",
    k=6, l=5, eta=4, tau=49, beta=196,
    gamma1=1 << 19,            # 524288
    gamma2=(Q - 1) // 32,     # 261888 = (q-1)/32
    omega=55,
)

DILITHIUM5 = DilithiumParams(
    name="Dilithium5",
    k=8, l=7, eta=2, tau=60, beta=120,
    gamma1=1 << 19,            # 524288
    gamma2=(Q - 1) // 32,     # 261888
    omega=75,
)

ALL_PARAMS = {"Dilithium2": DILITHIUM2, "Dilithium3": DILITHIUM3, "Dilithium5": DILITHIUM5}


# ─── Hash / XOF Wrappers ──────────────────────────────────────────────────────

def shake128(data: bytes, length: int) -> bytes:
    """SHAKE-128 eXtendable Output Function: output `length` bytes."""
    return hashlib.shake_128(data).digest(length)


def shake256(data: bytes, length: int) -> bytes:
    """SHAKE-256 eXtendable Output Function: output `length` bytes."""
    return hashlib.shake_256(data).digest(length)


def H(data: bytes, length: int = 32) -> bytes:
    """H(·) = SHAKE-256 used for challenge hashing and key derivation."""
    return shake256(data, length)


# ─── Polynomial Sampling ──────────────────────────────────────────────────────

def sample_uniform_poly(rho: bytes, i: int, j: int) -> list[int]:
    """
    Sample a uniform polynomial in Z_q from seed rho using SHAKE-128.

    Used in ExpandA to generate the public matrix A[i][j].
    Each coefficient is sampled uniformly from [0, q-1] using rejection sampling
    on 24-bit values (3 bytes per candidate, reject if >= q).

    Args:
        rho: 32-byte seed
        i:   row index
        j:   column index

    Returns:
        Polynomial with N uniformly random coefficients in [0, q-1]
    """
    # Domain separation: seed = rho || j || i
    seed = rho + bytes([j, i])

    # Request enough bytes for ~3x coefficient count (rejection ~6%)
    buf = shake128(seed, 672)  # 672 = 256 * 2.625 bytes
    offset = 0
    coeffs = []

    while len(coeffs) < N:
        if offset + 3 > len(buf):
            # Extend buffer if needed (rare)
            buf = shake128(seed, len(buf) + 168)

        b0 = buf[offset]
        b1 = buf[offset + 1]
        b2 = buf[offset + 2] & 0x7F   # top bit is always 0 for 23-bit q
        offset += 3

        val = b0 | (b1 << 8) | (b2 << 16)
        if val < Q:   # Rejection: accept only if < q
            coeffs.append(val)

    return coeffs[:N]


def sample_eta_poly(rho_prime: bytes, nonce: int, eta: int) -> list[int]:
    """
    Sample small polynomial with coefficients in [-eta, eta].

    Used in ExpandS to generate secret polynomials s1, s2.
    Uses 4-bit rejection sampling: extract 4-bit values from bytes,
    accept if < 2*eta+1, map to eta - value.

    Args:
        rho_prime: 64-byte seed
        nonce:     counter for domain separation
        eta:       coefficient bound (2 or 4)

    Returns:
        Polynomial with coefficients in [-eta, eta] (stored as mod-q values)
    """
    seed = rho_prime + nonce.to_bytes(2, 'little')

    # Enough bytes for 256 coefficients
    if eta == 2:
        buf = shake256(seed, 136)
    else:  # eta == 4
        buf = shake256(seed, 227)

    limit = 2 * eta + 1   # accept if nibble < limit (5 for eta=2, 9 for eta=4)
    coeffs = []
    idx = 0

    while len(coeffs) < N:
        if idx >= len(buf):
            buf = shake256(seed, len(buf) + 64)

        byte = buf[idx]
        idx += 1

        v0 = byte & 0x0F          # lower nibble
        v1 = (byte >> 4) & 0x0F   # upper nibble

        if v0 < limit:
            # Map to [-eta, eta]: eta - v0
            c = (eta - v0) % Q    # store as mod-q value
            coeffs.append(c)

        if len(coeffs) < N and v1 < limit:
            c = (eta - v1) % Q
            coeffs.append(c)

    return coeffs[:N]


def sample_in_ball(c_tilde: bytes, tau: int) -> list[int]:
    """
    Generate challenge polynomial c with exactly tau nonzero coefficients (+/-1).

    Algorithm (from Dilithium spec):
      1. Read 8 bytes from SHAKE-256(c_tilde) as sign bits (64-bit integer)
      2. For i from N-tau to N-1:
           Read byte j from XOF
           while j > i: read next byte j
           c[i] = c[j]
           c[j] = (-1)^(sign_bit)
           shift sign_bits right by 1
      3. Return c

    The result has exactly tau coefficients equal to +1 or -1 (mod q).

    Args:
        c_tilde: 32-byte hash seed
        tau:     number of nonzero coefficients

    Returns:
        Challenge polynomial (coefficients in {0, 1, Q-1})
    """
    buf = shake256(c_tilde, 8 + N)    # 8 for sign bits + N for positions
    signs = int.from_bytes(buf[:8], 'little')   # 64-bit sign mask

    c = [0] * N
    k = 8    # byte index (after sign bytes)

    for i in range(N - tau, N):
        # Rejection loop: find j <= i
        while True:
            if k >= len(buf):
                buf = shake256(c_tilde, len(buf) + 64)
            j = buf[k]
            k += 1
            if j <= i:
                break

        c[i] = c[j]
        # Set c[j] to +1 or -1 based on sign bit (stored as 0/Q-1 mod Q)
        c[j] = (Q - 1) if (signs & 1) else 1
        signs >>= 1

    return c


# ─── Key Expansion Functions ──────────────────────────────────────────────────

def expand_A(rho: bytes, k: int, l: int) -> list[list[list[int]]]:
    """
    Generate k×l matrix A from seed rho (stored in NTT domain).

    Each entry A[i][j] is a uniform polynomial sampled via SHAKE-128.
    Storing in NTT domain saves repeated NTT calls during multiplication.

    Args:
        rho: 32-byte public seed
        k:   number of rows
        l:   number of columns

    Returns:
        k×l matrix of NTT-domain polynomials
    """
    return [[ntt(sample_uniform_poly(rho, i, j)) for j in range(l)] for i in range(k)]


def expand_S(rho_prime: bytes, eta: int, l: int, k: int) -> tuple[list[list[int]], list[list[int]]]:
    """
    Generate secret vectors s1 (l polys) and s2 (k polys) from rho_prime.

    Both have small coefficients in [-eta, eta].

    Domain separation: s1 uses nonces 0..l-1, s2 uses nonces l..l+k-1.

    Returns:
        (s1, s2): vectors of polynomials with coefficients in [-eta, eta] mod q
    """
    s1 = [sample_eta_poly(rho_prime, nonce, eta) for nonce in range(l)]
    s2 = [sample_eta_poly(rho_prime, nonce, eta) for nonce in range(l, l + k)]
    return s1, s2


def expand_mask(rho_pp: bytes, kappa: int, l: int, gamma1: int) -> list[list[int]]:
    """
    Generate mask vector y with l polynomials, coefficients in (-gamma1, gamma1].

    Used in signing to generate the randomness y for the commitment.
    Domain separation: uses kappa, kappa+1, ..., kappa+l-1 as nonces.

    Encoding:
      gamma1 = 2^17: coefficients need 18 bits (gamma1 - coeff encoded as 18-bit)
      gamma1 = 2^19: coefficients need 20 bits

    Returns:
        l-vector of polynomials with coefficients in (-gamma1, gamma1] mod q
    """
    result = []
    for i in range(l):
        result.append(_sample_gamma1_poly(rho_pp, kappa + i, gamma1))
    return result


def _sample_gamma1_poly(rho_pp: bytes, nonce: int, gamma1: int) -> list[int]:
    """Sample one polynomial with coefficients in (-gamma1, gamma1]."""
    seed = rho_pp + nonce.to_bytes(2, 'little')

    if gamma1 == (1 << 17):
        # 18 bits per coefficient: 256 * 18 / 8 = 576 bytes
        buf = shake256(seed, 576)
        coeffs = []
        # Unpack: 9 bytes = 72 bits = 4 * 18 bits
        for base in range(0, 576, 9):
            # Read 72 bits as little-endian integer
            val = int.from_bytes(buf[base:base + 9], 'little')
            for _ in range(4):
                c = val & ((1 << 18) - 1)
                # Map: c = gamma1 - coefficient, so coeff = gamma1 - c
                coeffs.append((gamma1 - c) % Q)
                val >>= 18

    else:  # gamma1 == (1 << 19)
        # 20 bits per coefficient: 256 * 20 / 8 = 640 bytes
        buf = shake256(seed, 640)
        coeffs = []
        # 5 bytes = 40 bits = 2 * 20 bits
        for base in range(0, 640, 5):
            val = int.from_bytes(buf[base:base + 5], 'little')
            for _ in range(2):
                c = val & ((1 << 20) - 1)
                coeffs.append((gamma1 - c) % Q)
                val >>= 20

    return coeffs[:N]


# ─── Bit Packing / Unpacking ──────────────────────────────────────────────────

def pack_bits(values: list[int], bits: int) -> bytes:
    """
    Pack a list of non-negative integers into a byte array using `bits` bits each.

    Example: pack_bits([3, 1, 2], 4) packs three 4-bit values.
    """
    buf = 0
    filled = 0
    result = bytearray()
    mask = (1 << bits) - 1

    for v in values:
        buf |= (v & mask) << filled
        filled += bits
        while filled >= 8:
            result.append(buf & 0xFF)
            buf >>= 8
            filled -= 8

    if filled > 0:
        result.append(buf & 0xFF)

    return bytes(result)


def unpack_bits(data: bytes, count: int, bits: int) -> list[int]:
    """
    Unpack `count` integers of `bits` bits each from a byte array.
    """
    buf = 0
    filled = 0
    result = []
    mask = (1 << bits) - 1
    idx = 0

    for _ in range(count):
        while filled < bits:
            if idx < len(data):
                buf |= data[idx] << filled
                idx += 1
            filled += 8
        result.append(buf & mask)
        buf >>= bits
        filled -= bits

    return result


# ─── Public Key Packing ───────────────────────────────────────────────────────

def pack_pk(rho: bytes, t1: list[list[int]]) -> bytes:
    """
    Pack public key: pk = rho || encode_t1(t1)

    t1 coefficients are in [0, 1023] (10 bits each, since t = t1*2^13 + t0
    and t1 = (t mod q) >> 13, max = (q-1) >> 13 = 1023).

    Size: 32 + k*256*10/8 = 32 + k*320 bytes
    """
    t1_bytes = b''.join(pack_bits(poly, 10) for poly in t1)
    return rho + t1_bytes


def unpack_pk(pk: bytes, k: int) -> tuple[bytes, list[list[int]]]:
    """Unpack public key into (rho, t1)."""
    rho = pk[:32]
    t1_bytes = pk[32:]
    poly_size = N * 10 // 8   # = 320 bytes
    t1 = []
    for i in range(k):
        poly_data = t1_bytes[i * poly_size:(i + 1) * poly_size]
        t1.append(unpack_bits(poly_data, N, 10))
    return rho, t1


# ─── Secret Key Packing ───────────────────────────────────────────────────────

def _eta_bits(eta: int) -> int:
    """Number of bits needed to encode coefficients in [-eta, eta]."""
    return 3 if eta == 2 else 4   # eta=2: values 0-4 (3 bits), eta=4: values 0-8 (4 bits)


def _pack_eta_poly(poly: list[int], eta: int) -> bytes:
    """Pack polynomial with small coefficients: store as (eta - coeff) in [0, 2*eta]."""
    # coeff is stored mod q; convert to signed, then shift to [0, 2*eta]
    values = []
    for c in poly:
        # Center: map mod-q value to [-eta, eta]
        c = c % Q
        if c > Q // 2:
            c -= Q
        values.append(eta - c)   # shift to [0, 2*eta]
    return pack_bits(values, _eta_bits(eta))


def _unpack_eta_poly(data: bytes, eta: int) -> list[int]:
    """Unpack polynomial with small coefficients."""
    bits = _eta_bits(eta)
    values = unpack_bits(data, N, bits)
    # Map from [0, 2*eta] back to [-eta, eta] then to mod-q
    result = []
    for v in values:
        c = eta - v   # in [-eta, eta]
        result.append(c % Q)
    return result


def _pack_t0_poly(poly: list[int]) -> bytes:
    """
    Pack t0 polynomial with coefficients in (-2^12, 2^12].
    Shift by 2^12 to get [0, 2^13-1] (13 bits).
    """
    half = 1 << 12   # = 4096
    values = []
    for c in poly:
        # Center: t0 is already small, but stored mod q
        c = c % Q
        if c > Q // 2:
            c -= Q
        values.append(half - c)   # shift to [0, 2^13-1]
    return pack_bits(values, 13)


def _unpack_t0_poly(data: bytes) -> list[int]:
    """Unpack t0 polynomial."""
    half = 1 << 12
    values = unpack_bits(data, N, 13)
    return [(half - v) % Q for v in values]


def pack_sk(rho: bytes, K: bytes, tr: bytes,
            s1: list[list[int]], s2: list[list[int]],
            t0: list[list[int]], params: 'DilithiumParams') -> bytes:
    """
    Pack secret key: sk = rho || K || tr || s1 || s2 || t0

    Components:
      rho: 32 bytes (public seed)
      K:   32 bytes (signing randomness seed)
      tr:  64 bytes (hash of public key)
      s1:  l * ceil(N * bits(eta) / 8) bytes
      s2:  k * ceil(N * bits(eta) / 8) bytes
      t0:  k * ceil(N * 13 / 8) bytes = k * 416 bytes
    """
    eta = params.eta
    s1_bytes = b''.join(_pack_eta_poly(p, eta) for p in s1)
    s2_bytes = b''.join(_pack_eta_poly(p, eta) for p in s2)
    t0_bytes = b''.join(_pack_t0_poly(p) for p in t0)
    return rho + K + tr + s1_bytes + s2_bytes + t0_bytes


def unpack_sk(sk: bytes, params: 'DilithiumParams') -> tuple:
    """Unpack secret key into (rho, K, tr, s1, s2, t0)."""
    k, l, eta = params.k, params.l, params.eta
    bits = _eta_bits(eta)
    eta_poly_size = N * bits // 8     # bytes per small polynomial
    t0_poly_size = N * 13 // 8       # = 416 bytes

    offset = 0
    rho = sk[offset:offset + 32]; offset += 32
    K   = sk[offset:offset + 32]; offset += 32
    tr  = sk[offset:offset + 64]; offset += 64

    s1 = []
    for _ in range(l):
        s1.append(_unpack_eta_poly(sk[offset:offset + eta_poly_size], eta))
        offset += eta_poly_size

    s2 = []
    for _ in range(k):
        s2.append(_unpack_eta_poly(sk[offset:offset + eta_poly_size], eta))
        offset += eta_poly_size

    t0 = []
    for _ in range(k):
        t0.append(_unpack_t0_poly(sk[offset:offset + t0_poly_size]))
        offset += t0_poly_size

    return rho, K, tr, s1, s2, t0


# ─── Signature Packing ────────────────────────────────────────────────────────

def _gamma1_bits(gamma1: int) -> int:
    return 18 if gamma1 == (1 << 17) else 20


def _pack_z_poly(poly: list[int], gamma1: int) -> bytes:
    """
    Pack z polynomial with coefficients in (-gamma1, gamma1].
    Encode as (gamma1 - coeff) in [0, 2*gamma1-1].
    """
    bits = _gamma1_bits(gamma1)
    values = []
    for c in poly:
        c = c % Q
        if c > Q // 2:
            c -= Q
        values.append(gamma1 - c)   # in [0, 2*gamma1-1]
    return pack_bits(values, bits)


def _unpack_z_poly(data: bytes, gamma1: int) -> list[int]:
    """Unpack z polynomial."""
    bits = _gamma1_bits(gamma1)
    values = unpack_bits(data, N, bits)
    return [(gamma1 - v) % Q for v in values]


def _w1_bits(gamma2: int) -> int:
    """Number of bits for w1 encoding."""
    # gamma2 = (q-1)/88: m = (q-1)/(2*gamma2) = 44, need 6 bits
    # gamma2 = (q-1)/32: m = 16, need 4 bits
    return 6 if gamma2 == (Q - 1) // 88 else 4


def encode_w1(w1: list[list[int]], gamma2: int) -> bytes:
    """Encode w1 (high bits vector) for hashing."""
    bits = _w1_bits(gamma2)
    return b''.join(pack_bits(poly, bits) for poly in w1)


def pack_hint(h: list[list[int]], omega: int, k: int) -> bytes:
    """
    Pack hint vector as position indices + end markers.

    Format: omega bytes for positions + k bytes for end markers.
    h[i][j] = 1 means position j in polynomial i is a hint.
    Positions are stored sequentially, end markers indicate where each poly ends.
    """
    result = bytearray(omega + k)
    idx = 0
    for i, poly in enumerate(h):
        for j, bit in enumerate(poly):
            if bit == 1:
                if idx < omega:
                    result[idx] = j
                    idx += 1
        result[omega + i] = idx   # end marker for polynomial i
    return bytes(result)


def unpack_hint(data: bytes, omega: int, k: int) -> list[list[int]]:
    """Unpack hint vector from packed format."""
    h = [[0] * N for _ in range(k)]
    start = 0
    for i in range(k):
        end = data[omega + i]
        if end < start or end > omega:
            return None   # invalid hint encoding
        for pos in range(start, end):
            if data[pos] >= N:
                return None   # invalid position
            if pos > start and data[pos] <= data[pos - 1]:
                return None   # positions must be strictly increasing within each poly
            h[i][data[pos]] = 1
        start = end
    return h


def pack_sig(c_tilde: bytes, z: list[list[int]], h: list[list[int]],
             params: 'DilithiumParams') -> bytes:
    """
    Pack signature: sig = c_tilde || z || h

    Size:
      c_tilde: 32 bytes
      z: l * N * gamma1_bits / 8 bytes
      h: omega + k bytes
    """
    z_bytes = b''.join(_pack_z_poly(p, params.gamma1) for p in z)
    h_bytes = pack_hint(h, params.omega, params.k)
    return c_tilde + z_bytes + h_bytes


def unpack_sig(sig: bytes, params: 'DilithiumParams') -> tuple:
    """Unpack signature into (c_tilde, z, h). Returns None if malformed."""
    k, l = params.k, params.l
    bits = _gamma1_bits(params.gamma1)
    z_poly_size = N * bits // 8

    c_tilde = sig[:32]
    offset = 32

    z = []
    for _ in range(l):
        z.append(_unpack_z_poly(sig[offset:offset + z_poly_size], params.gamma1))
        offset += z_poly_size

    h = unpack_hint(sig[offset:], params.omega, k)
    return c_tilde, z, h


if __name__ == "__main__":
    print("[utils.py] Running tests...")

    # Test bit packing roundtrip
    values = list(range(44))
    packed = pack_bits(values, 6)
    recovered = unpack_bits(packed, len(values), 6)
    assert recovered == values, f"Bit packing failed: {recovered} != {values}"
    print("  pack/unpack_bits (6-bit): PASS")

    values = list(range(256))
    packed = pack_bits(values, 10)
    recovered = unpack_bits(packed, 256, 10)
    assert recovered == values, "10-bit packing failed"
    print("  pack/unpack_bits (10-bit, 256 values): PASS")

    # Test polynomial sampling
    rho = os.urandom(32)
    poly = sample_uniform_poly(rho, 0, 0)
    assert len(poly) == N
    assert all(0 <= c < Q for c in poly)
    print("  sample_uniform_poly: PASS")

    poly = sample_eta_poly(os.urandom(64), 0, 2)
    assert len(poly) == N
    # Check all coefficients are in [-2, 2] mod q
    for c in poly:
        cc = c % Q
        if cc > Q // 2: cc -= Q
        assert -2 <= cc <= 2, f"eta=2 coeff out of range: {cc}"
    print("  sample_eta_poly (eta=2): PASS")

    poly = sample_eta_poly(os.urandom(64), 0, 4)
    for c in poly:
        cc = c % Q
        if cc > Q // 2: cc -= Q
        assert -4 <= cc <= 4, f"eta=4 coeff out of range: {cc}"
    print("  sample_eta_poly (eta=4): PASS")

    # Test SampleInBall
    tau = 39
    c = sample_in_ball(os.urandom(32), tau)
    nonzero = sum(1 for x in c if x != 0)
    assert nonzero == tau, f"Expected {tau} nonzero, got {nonzero}"
    assert all(x in (0, 1, Q - 1) for x in c), "Invalid challenge coefficients"
    print(f"  sample_in_ball (tau={tau}): PASS ({tau} nonzero coefficients)")

    print("[utils.py] All tests PASS")
