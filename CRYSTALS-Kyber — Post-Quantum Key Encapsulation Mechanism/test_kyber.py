"""
Test Suite for CRYSTALS-Kyber Implementation
==============================================
Covers: NTT roundtrip, polynomial arithmetic, CBD sampling,
compression, and full KEM roundtrip for all parameter sets.

Run with:  python -m pytest test_kyber.py -v
      or:  python test_kyber.py
"""

import sys
import os
import random
import time

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(__file__))

from params import Q, N, KYBER512, KYBER768, KYBER1024
from ntt import ntt, ntt_inv, poly_basemul
from poly import Poly, PolyVec
from sampling import cbd, sample_noise, sample_ntt
from compress import compress_int, decompress_int, compress_poly, decompress_poly
from utils import encode_poly, decode_poly, hash_h, hash_g, prf
from kyber import (
    cpapke_keygen, cpapke_enc, cpapke_dec,
    kem_keygen, kem_encaps, kem_decaps,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Test helpers
# ══════════════════════════════════════════════════════════════════════════════

passed = 0
failed = 0


def test(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✔ {name}")
    else:
        failed += 1
        msg = f"  ✘ {name}"
        if detail:
            msg += f"  — {detail}"
        print(msg)


# ══════════════════════════════════════════════════════════════════════════════
#  NTT Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_ntt_roundtrip():
    """NTT(INTT(p)) == p and INTT(NTT(p)) == p for random polynomials."""
    print("\n─── NTT Roundtrip Tests ───")
    for trial in range(5):
        coeffs = [random.randint(0, Q - 1) for _ in range(N)]
        forward = ntt(coeffs)
        back = ntt_inv(forward)
        match = all((a - b) % Q == 0 for a, b in zip(coeffs, back))
        test(f"NTT roundtrip (trial {trial + 1})", match)


def test_ntt_linearity():
    """NTT(a + b) == NTT(a) + NTT(b)."""
    print("\n─── NTT Linearity Test ───")
    a = [random.randint(0, Q - 1) for _ in range(N)]
    b = [random.randint(0, Q - 1) for _ in range(N)]
    ab = [(x + y) % Q for x, y in zip(a, b)]

    ntt_a = ntt(a)
    ntt_b = ntt(b)
    ntt_ab = ntt(ab)
    ntt_sum = [(x + y) % Q for x, y in zip(ntt_a, ntt_b)]

    match = all((x - y) % Q == 0 for x, y in zip(ntt_ab, ntt_sum))
    test("NTT linearity: NTT(a+b) == NTT(a)+NTT(b)", match)


# ══════════════════════════════════════════════════════════════════════════════
#  Polynomial Arithmetic Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_poly_add_sub():
    """Poly addition and subtraction are inverses."""
    print("\n─── Polynomial Arithmetic Tests ───")
    a = Poly([random.randint(0, Q - 1) for _ in range(N)])
    b = Poly([random.randint(0, Q - 1) for _ in range(N)])
    c = a.add(b).sub(b)
    match = all((x - y) % Q == 0 for x, y in zip(a.coeffs, c.coeffs))
    test("Poly: (a + b) - b == a", match)


def test_poly_ntt_mul():
    """Multiplication in NTT domain: verify via simple schoolbook check for small polys."""
    print("\n─── NTT Multiplication Test ───")
    # Use small coefficients to make schoolbook multiplication feasible
    a_coeffs = [random.randint(0, 10) for _ in range(N)]
    b_coeffs = [random.randint(0, 10) for _ in range(N)]

    a = Poly(a_coeffs)
    b = Poly(b_coeffs)

    # NTT-based multiplication
    a_ntt = a.to_ntt()
    b_ntt = b.to_ntt()
    c_ntt = a_ntt.mul_ntt(b_ntt)
    c = c_ntt.from_ntt()

    # Schoolbook multiplication in Z_q[x]/(x^256 + 1)
    expected = [0] * N
    for i in range(N):
        for j in range(N):
            idx = i + j
            if idx >= N:
                expected[idx - N] = (expected[idx - N] - a_coeffs[i] * b_coeffs[j]) % Q
            else:
                expected[idx] = (expected[idx] + a_coeffs[i] * b_coeffs[j]) % Q

    match = all((c.coeffs[i] - expected[i]) % Q == 0 for i in range(N))
    test("NTT multiplication matches schoolbook", match)


# ══════════════════════════════════════════════════════════════════════════════
#  Sampling Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_cbd_distribution():
    """CBD output should be in {-η, ..., η}."""
    print("\n─── CBD Sampling Tests ───")
    for eta in [2, 3]:
        buf = os.urandom(64 * eta)
        p = cbd(eta, buf)
        # Map back from [0, q) to signed
        signed = [(c if c <= Q // 2 else c - Q) for c in p.coeffs]
        in_range = all(-eta <= s <= eta for s in signed)
        test(f"CBD(η={eta}): all coefficients in [-{eta}, {eta}]", in_range,
             f"min={min(signed)}, max={max(signed)}")


def test_sample_ntt_bounds():
    """Uniform samples should all be in [0, q)."""
    print("\n─── Uniform Sampling Tests ───")
    seed = os.urandom(32)
    p = sample_ntt(seed, 0, 0)
    in_range = all(0 <= c < Q for c in p.coeffs)
    test("Uniform sample: all coefficients in [0, q)", in_range)
    test("Uniform sample: has 256 coefficients", len(p.coeffs) == N)


# ══════════════════════════════════════════════════════════════════════════════
#  Compression Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_compression_roundtrip():
    """decompress(compress(x)) ≈ x within tolerance."""
    print("\n─── Compression Roundtrip Tests ───")
    for d in [4, 5, 10, 11]:
        max_error = 0
        for _ in range(100):
            x = random.randint(0, Q - 1)
            c = compress_int(x, d)
            x_approx = decompress_int(c, d)
            error = min(abs(x - x_approx), Q - abs(x - x_approx))
            max_error = max(max_error, error)
        # Maximum error should be bounded by q / 2^{d+1}
        bound = Q // (1 << (d + 1)) + 1
        test(f"Compression d={d}: max error {max_error} ≤ {bound}", max_error <= bound)


# ══════════════════════════════════════════════════════════════════════════════
#  Byte Encoding Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_encode_decode_roundtrip():
    """encode then decode should recover original coefficients."""
    print("\n─── Byte Encoding Tests ───")
    for bits in [1, 4, 5, 10, 11, 12]:
        coeffs = [random.randint(0, (1 << bits) - 1) for _ in range(N)]
        encoded = encode_poly(coeffs, bits)
        decoded = decode_poly(encoded, bits)
        match = coeffs == decoded[:N]
        test(f"Encode/decode roundtrip (bits={bits})", match)


# ══════════════════════════════════════════════════════════════════════════════
#  IND-CPA PKE Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_cpapke_roundtrip():
    """CPA-PKE encrypt then decrypt recovers the original message."""
    print("\n─── IND-CPA PKE Tests ───")
    for params in [KYBER512, KYBER768, KYBER1024]:
        pk, sk = cpapke_keygen(params)
        msg = os.urandom(32)
        coins = os.urandom(32)
        ct = cpapke_enc(params, pk, msg, coins)
        msg_dec = cpapke_dec(params, sk, ct)
        test(f"CPAPKE roundtrip ({params.name})", msg == msg_dec)


# ══════════════════════════════════════════════════════════════════════════════
#  IND-CCA KEM Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_kem_roundtrip():
    """KEM encaps/decaps produces the same shared secret."""
    print("\n─── IND-CCA KEM Tests ───")
    for params in [KYBER512, KYBER768, KYBER1024]:
        t0 = time.perf_counter()
        pk, sk = kem_keygen(params)
        ct, ss_enc = kem_encaps(params, pk)
        ss_dec = kem_decaps(params, sk, ct)
        elapsed = time.perf_counter() - t0
        test(f"KEM roundtrip ({params.name}) [{elapsed*1000:.0f}ms]",
             ss_enc == ss_dec)


def test_kem_implicit_rejection():
    """Tampered ciphertext should yield a different shared secret."""
    print("\n─── Implicit Rejection Tests ───")
    for params in [KYBER512, KYBER768]:
        pk, sk = kem_keygen(params)
        ct, ss_enc = kem_encaps(params, pk)

        # Tamper with ciphertext
        tampered = bytearray(ct)
        tampered[0] ^= 0xFF
        ss_tampered = kem_decaps(params, sk, bytes(tampered))

        test(f"Implicit rejection ({params.name})", ss_tampered != ss_enc)


def test_kem_key_sizes():
    """Verify key and ciphertext sizes match expected values."""
    print("\n─── Key/CT Size Tests ───")
    for params in [KYBER512, KYBER768, KYBER1024]:
        pk, sk = kem_keygen(params)
        ct, _ = kem_encaps(params, pk)

        test(f"{params.name} pk size = {params.pk_size}", len(pk) == params.pk_size)
        test(f"{params.name} sk size = {params.sk_size}", len(sk) == params.sk_size)
        test(f"{params.name} ct size = {params.ct_size}", len(ct) == params.ct_size)


# ══════════════════════════════════════════════════════════════════════════════
#  Hash Function Tests
# ══════════════════════════════════════════════════════════════════════════════

def test_hash_functions():
    """Basic sanity checks for hash wrappers."""
    print("\n─── Hash Function Tests ───")
    data = b"CRYSTALS-Kyber test"

    h = hash_h(data)
    test("H (SHA3-256) output length = 32", len(h) == 32)

    g1, g2 = hash_g(data)
    test("G (SHA3-512) output lengths = 32, 32", len(g1) == 32 and len(g2) == 32)

    p = prf(b"seed" + b"\x00" * 28, 0, 128)
    test("PRF (SHAKE-256) output length = 128", len(p) == 128)

    # Determinism
    p2 = prf(b"seed" + b"\x00" * 28, 0, 128)
    test("PRF is deterministic", p == p2)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def run_all_tests():
    global passed, failed
    passed = 0
    failed = 0

    print("=" * 60)
    print("  CRYSTALS-Kyber Test Suite")
    print("=" * 60)

    test_hash_functions()
    test_ntt_roundtrip()
    test_ntt_linearity()
    test_poly_add_sub()
    test_poly_ntt_mul()
    test_cbd_distribution()
    test_sample_ntt_bounds()
    test_compression_roundtrip()
    test_encode_decode_roundtrip()
    test_cpapke_roundtrip()
    test_kem_roundtrip()
    test_kem_implicit_rejection()
    test_kem_key_sizes()

    print("\n" + "=" * 60)
    total = passed + failed
    if failed == 0:
        print(f"  ALL {total} TESTS PASSED ✔")
    else:
        print(f"  {passed}/{total} passed, {failed} FAILED ✘")
    print("=" * 60 + "\n")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
