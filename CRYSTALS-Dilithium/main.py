"""
main.py - CRYSTALS-Dilithium Interactive Demo & CLI

Features:
  - Interactive menu (CLI) for key generation, signing, verification
  - Step-by-step signing/verification visualization
  - Test suite for all three variants
  - Comparison with RSA/ECDSA
  - Security concept explanations

Run modes:
  python main.py         → interactive CLI menu
  python main.py demo    → full demo with all variants
  python main.py test    → run test suite
  python main.py info    → show security concepts and comparisons
"""

import sys
import os
import time
import textwrap

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dilithium import Dilithium, keygen, sign, verify
from utils import DILITHIUM2, DILITHIUM3, DILITHIUM5


# ─── Display Helpers ──────────────────────────────────────────────────────────

def banner(text: str, width: int = 60) -> None:
    print("═" * width)
    print(f"  {text}")
    print("═" * width)


def section(text: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {text}")
    print("─" * 50)


def hex_preview(data: bytes, max_bytes: int = 24) -> str:
    """Show hex preview of bytes data."""
    if len(data) <= max_bytes:
        return data.hex()
    return data[:max_bytes].hex() + f"... ({len(data)} bytes total)"


def print_step(n: int, desc: str, detail: str = "") -> None:
    print(f"\n  [{n}] {desc}")
    if detail:
        for line in textwrap.wrap(detail, 56):
            print(f"      {line}")


# ─── Security Concepts ────────────────────────────────────────────────────────

SECURITY_CONCEPTS = """
╔══════════════════════════════════════════════════════════╗
║      CRYSTALS-Dilithium: Security Foundations            ║
╚══════════════════════════════════════════════════════════╝

1. LATTICE-BASED CRYPTOGRAPHY
─────────────────────────────
A lattice is a regular grid of points in n-dimensional space.
Hard problems on lattices (finding short vectors, approximate
closest vector) are believed to resist quantum attacks.

Unlike RSA (factoring) and ECDSA (discrete log), no efficient
quantum algorithm is known for lattice problems. Shor's algorithm
breaks RSA/ECDSA but does NOT apply to lattices.

2. MODULE-LWE (Learning With Errors)
──────────────────────────────────────
Given: A (random matrix), t = A·s + e
Hard: Recover secret s from (A, t)

Where s, e are small ("noisy") polynomials. This is the basis of
Dilithium's key generation hardness. In Dilithium:
  t = A·s₁ + s₂  (s₁, s₂ are small, A is random)

The "Module" prefix means we work over polynomial rings
Z_q[x]/(x^n + 1) instead of plain integers, giving smaller
key sizes with the same security.

3. MODULE-SIS (Short Integer Solution)
───────────────────────────────────────
Given: A (random matrix)
Hard: Find short z such that A·z = 0 (mod q)

This underpins signature unforgeability. A forger would need to
find a short response z satisfying the verification equation,
which requires solving Module-SIS.

4. WHY QUANTUM-RESISTANT?
──────────────────────────
• RSA/ECDSA: broken by Shor's algorithm on quantum computers
  (factoring and discrete log in O(log²n) quantum time)
• Lattice problems: best known quantum algorithm (BKZ + quantum
  speedup) gives only √2 speedup over classical — not enough to
  break 2^128 security at NIST Level 2

5. DILITHIUM vs KYBER (signatures vs encryption)
──────────────────────────────────────────────────
Both use the same mathematical structures but for different goals:

  CRYSTALS-Dilithium  → Digital Signatures (sign/verify)
  CRYSTALS-Kyber      → Key Encapsulation (encrypt/decrypt)

Dilithium proves the signer knows a short secret (Fiat-Shamir
paradigm). Kyber creates a shared secret using LWE encryption.

Both were standardized by NIST in 2024 (FIPS 203, 204).
"""

COMPARISON_TABLE = """
╔═══════════════╦═══════════╦════════════╦══════════════════╗
║  Algorithm    ║ Key Size  ║  Sig Size  ║  Quantum-Safe?   ║
╠═══════════════╬═══════════╬════════════╬══════════════════╣
║  RSA-2048     ║ 256 B     ║  256 B     ║  NO (Shor's alg) ║
║  ECDSA P-256  ║  64 B     ║   64 B     ║  NO (Shor's alg) ║
║  Ed25519      ║  32 B     ║   64 B     ║  NO (Shor's alg) ║
╠═══════════════╬═══════════╬════════════╬══════════════════╣
║  Dilithium2   ║ 1312 B    ║ 2420 B     ║  YES (NIST L2)   ║
║  Dilithium3   ║ 1952 B    ║ 3293 B     ║  YES (NIST L3)   ║
║  Dilithium5   ║ 2592 B    ║ 4595 B     ║  YES (NIST L5)   ║
╚═══════════════╩═══════════╩════════════╩══════════════════╝

Dilithium uses larger keys/signatures due to the mathematical
structure of lattice-based cryptography — but provides security
against quantum adversaries, unlike classical algorithms.

NIST PQC Standardization:
  2016: NIST PQC competition launched
  2022: CRYSTALS-Dilithium selected as primary signature standard
  2024: Standardized as FIPS 204 (ML-DSA)
"""

SIGNING_FLOW_DIAGRAM = """
CRYSTALS-Dilithium Signing Flow
════════════════════════════════

Secret Key (ρ, K, tr, s₁, s₂, t₀)
         │
         │ 1. Hash message with public key binding
         ▼
    μ = H(tr ‖ message)
         │
         │ 2. Derive per-signature randomness seed
         ▼
    ρ″ = H(K ‖ rnd ‖ μ)
         │
         ▼
    ┌────────────────────────────────────────┐
    │           REJECTION LOOP               │
    │                                        │
    │  y = ExpandMask(ρ″, κ)  ← sample mask │
    │  w = A·y                ← commitment  │
    │  w₁ = HighBits(w, 2γ₂)               │
    │                                        │
    │  c̃ = H(μ ‖ Encode(w₁))  ← hash       │
    │  c = SampleInBall(c̃, τ) ← challenge  │
    │                                        │
    │  z = y + c·s₁           ← response   │
    │  r₀ = LowBits(w - c·s₂)              │
    │                                        │
    │  REJECT if ‖z‖∞ ≥ γ₁ - β            │
    │  REJECT if ‖r₀‖∞ ≥ γ₂ - β           │
    │                                        │
    │  h = MakeHint(-c·t₀, w-c·s₂+c·t₀)   │
    │  REJECT if ‖c·t₀‖∞ ≥ γ₂             │
    │  REJECT if Σh > ω                    │
    │                                        │
    │  κ += l (increment for next try)       │
    └────────────────────────────────────────┘
         │
         ▼
    σ = (c̃, z, h)  ← signature


CRYSTALS-Dilithium Verification Flow
═════════════════════════════════════

Public Key (ρ, t₁)    Signature (c̃, z, h)    Message
       │                      │                  │
       └──────────────┬───────┘                  │
                      │                          │
       1. μ = H(H(pk) ‖ message) ←──────────────┘
                      │
       2. c = SampleInBall(c̃, τ)
                      │
       3. A = ExpandA(ρ)
                      │
       4. Compute: A·z - c·t₁·2ᵈ
                      │
       5. w₁' = UseHint(h, A·z - c·t₁·2ᵈ, 2γ₂)
                      │
       ┌──────────────▼─────────────────────────┐
       │  CHECK 1: ‖z‖∞ < γ₁ - β ?            │
       │  CHECK 2: Σh ≤ ω ?                    │
       │  CHECK 3: c̃ = H(μ ‖ Encode(w₁')) ?  │
       └──────────────┬─────────────────────────┘
                      │
                 VALID / INVALID
"""


# ─── Interactive Demo ─────────────────────────────────────────────────────────

def demo_variant(variant: str, message: bytes = None) -> dict:
    """Run a complete demo for one Dilithium variant."""
    if message is None:
        message = b"Hello from the post-quantum future!"

    banner(f"  CRYSTALS-{variant} Demo")

    dil = Dilithium(variant)
    params = dil.params

    print(f"\n  Parameters:")
    print(f"    k={params.k}, l={params.l}, η={params.eta}, τ={params.tau}")
    print(f"    γ₁={params.gamma1}, γ₂={params.gamma2}, ω={params.omega}")
    print(f"    Public key size:  {dil.pk_size} bytes")
    print(f"    Secret key size:  {dil.sk_size} bytes")
    print(f"    Signature size:   {dil.sig_size} bytes")

    # Key Generation
    section("Step 1: Key Generation")
    t0 = time.time()
    pk, sk = dil.keygen()
    t_keygen = time.time() - t0

    print(f"\n  Public Key ({len(pk)} bytes):")
    print(f"    {hex_preview(pk)}")
    print(f"\n  Secret Key ({len(sk)} bytes):")
    print(f"    {hex_preview(sk)}")
    print(f"\n  Time: {t_keygen * 1000:.1f} ms")

    # Signing
    section("Step 2: Signing")
    print(f"\n  Message: {message!r}")

    t0 = time.time()
    sig = dil.sign(sk, message)
    t_sign = time.time() - t0

    print(f"\n  Signature ({len(sig)} bytes):")
    print(f"    c̃:  {sig[:32].hex()}")
    print(f"    z:  {hex_preview(sig[32:])}")
    print(f"\n  Time: {t_sign * 1000:.1f} ms")

    # Verification
    section("Step 3: Verification")
    t0 = time.time()
    valid = dil.verify(pk, message, sig)
    t_verify = time.time() - t0

    result = "✓ VALID" if valid else "✗ INVALID"
    print(f"\n  Result: {result}")
    print(f"  Time: {t_verify * 1000:.1f} ms")

    # Negative tests
    section("Security Tests")

    # Modified message
    bad_msg = dil.verify(pk, message + b"!", sig)
    print(f"\n  Modified message rejected: {'✓ PASS' if not bad_msg else '✗ FAIL'}")

    # Modified signature
    bad_sig = bytearray(sig)
    bad_sig[40] ^= 0x01
    bad_sig_result = dil.verify(pk, message, bytes(bad_sig))
    print(f"  Tampered signature rejected: {'✓ PASS' if not bad_sig_result else '✗ FAIL'}")

    # Wrong key
    pk2, _ = dil.keygen()
    wrong_key = dil.verify(pk2, message, sig)
    print(f"  Wrong public key rejected: {'✓ PASS' if not wrong_key else '✗ FAIL'}")

    return {
        "variant": variant,
        "pk_size": len(pk),
        "sig_size": len(sig),
        "t_keygen": t_keygen,
        "t_sign": t_sign,
        "t_verify": t_verify,
        "valid": valid,
    }


def demo_all() -> None:
    """Full demo with all three variants."""
    banner("CRYSTALS-Dilithium - Complete Demo", 60)
    print("""
  CRYSTALS-Dilithium is a post-quantum digital signature scheme
  standardized by NIST as FIPS 204 (ML-DSA) in August 2024.

  It provides security against quantum computers using the
  hardness of Module-LWE and Module-SIS lattice problems.
    """)

    results = []
    for variant in ["Dilithium2", "Dilithium3", "Dilithium5"]:
        r = demo_variant(variant)
        results.append(r)
        print()

    # Summary table
    section("Performance Summary")
    print(f"\n  {'Variant':<14} {'PK':>8} {'Sig':>8} {'KeyGen':>10} {'Sign':>10} {'Verify':>10}")
    print("  " + "─" * 64)
    for r in results:
        print(f"  {r['variant']:<14} "
              f"{r['pk_size']:>7}B "
              f"{r['sig_size']:>7}B "
              f"{r['t_keygen']*1000:>9.1f}ms "
              f"{r['t_sign']*1000:>9.1f}ms "
              f"{r['t_verify']*1000:>9.1f}ms")

    print(COMPARISON_TABLE)


# ─── Visualized Signing ───────────────────────────────────────────────────────

def demo_signing_steps(variant: str = "Dilithium2") -> None:
    """Show detailed step-by-step signing and verification."""
    banner(f"Step-by-Step {variant} Signing", 60)

    print(SIGNING_FLOW_DIAGRAM)

    dil = Dilithium(variant)
    params = dil.params

    section(f"Live Demo: {variant}")
    message = b"Post-quantum signature demo"
    print(f"\n  Message: {message!r}")

    print_step(1, "Key Generation", "Generating keypair from random seed...")
    pk, sk = dil.keygen()
    print(f"      pk[0:32] (ρ) = {pk[:32].hex()}")
    print(f"      pk size = {len(pk)} bytes")

    print_step(2, "Signing", "Applying Fiat-Shamir with Aborts...")
    sig = dil.sign(sk, message)
    c_tilde = sig[:32]
    print(f"      c̃ (challenge hash) = {c_tilde.hex()}")
    print(f"      sig size = {len(sig)} bytes")

    print_step(3, "Verification", "Checking signature validity...")
    valid = dil.verify(pk, message, sig)
    print(f"      Result: {'✓ SIGNATURE VALID' if valid else '✗ SIGNATURE INVALID'}")

    print_step(4, "Tampering Test", "Flipping 1 bit in the signature...")
    tampered = bytearray(sig)
    tampered[100] ^= 0x80
    result = dil.verify(pk, message, bytes(tampered))
    print(f"      Result: {'✗ TAMPERED SIG ACCEPTED (BUG!)' if result else '✓ TAMPERING DETECTED'}")


# ─── Test Suite ───────────────────────────────────────────────────────────────

def run_tests() -> None:
    """Comprehensive test suite."""
    banner("Dilithium Test Suite", 60)
    passed = 0
    failed = 0

    def test(name: str, cond: bool) -> None:
        nonlocal passed, failed
        status = "PASS" if cond else "FAIL"
        print(f"  [{status}] {name}")
        if cond:
            passed += 1
        else:
            failed += 1

    for variant in ["Dilithium2", "Dilithium3", "Dilithium5"]:
        section(f"Testing {variant}")
        dil = Dilithium(variant)

        # Basic keygen
        pk, sk = dil.keygen()
        test("Keypair generation succeeds", pk is not None and sk is not None)
        test(f"Public key size = {dil.pk_size}", len(pk) == dil.pk_size)

        # Basic sign/verify
        msg = b"Test message 12345"
        sig = dil.sign(sk, msg)
        test(f"Signature size ≈ {dil.sig_size}", abs(len(sig) - dil.sig_size) <= 10)
        test("Valid signature accepted", dil.verify(pk, msg, sig))

        # Security tests
        test("Wrong message rejected",
             not dil.verify(pk, b"wrong message", sig))
        test("Bit-flipped signature rejected",
             not dil.verify(pk, msg, bytes([sig[0] ^ 0xFF]) + sig[1:]))

        wrong_pk, _ = dil.keygen()
        test("Wrong public key rejected",
             not dil.verify(wrong_pk, msg, sig))

        # Multiple messages
        msgs = [b"msg1", b"msg2" * 100, b"\x00" * 1000, bytes(range(256))]
        for m in msgs:
            s = dil.sign(sk, m)
            ok = dil.verify(pk, m, s)
            test(f"Sign/verify msg of {len(m)} bytes", ok)

        # Signature uniqueness (each call should produce different sig in randomized mode)
        sig1 = dil.sign(sk, msg, randomized=True)
        sig2 = dil.sign(sk, msg, randomized=True)
        # They verify but are different (with high probability)
        test("Randomized signing: sig1 valid", dil.verify(pk, msg, sig1))
        test("Randomized signing: sig2 valid", dil.verify(pk, msg, sig2))
        test("Randomized signing: sigs differ", sig1 != sig2)

        # Deterministic signing
        sig_det1 = dil.sign(sk, msg, randomized=False)
        sig_det2 = dil.sign(sk, msg, randomized=False)
        test("Deterministic signing: sigs identical", sig_det1 == sig_det2)

    section("Summary")
    total = passed + failed
    print(f"\n  Results: {passed}/{total} passed")
    if failed == 0:
        print("  ✓ All tests passed!")
    else:
        print(f"  ✗ {failed} test(s) failed!")


# ─── Interactive CLI Menu ─────────────────────────────────────────────────────

def cli_menu() -> None:
    """Interactive CLI menu for Dilithium operations."""
    banner("CRYSTALS-Dilithium Post-Quantum Signatures", 60)

    print("""
  A quantum-resistant digital signature scheme based on
  Module Lattice problems. Standardized by NIST as FIPS 204.
    """)

    # State: current keypair and variant
    state = {"pk": None, "sk": None, "variant": "Dilithium3", "dil": Dilithium("Dilithium3")}

    while True:
        print(f"\n  Current variant: {state['variant']}")
        print(f"  Keys loaded: {'Yes' if state['pk'] else 'No'}")
        print("""
  ┌─────────────────────────────────────┐
  │  MENU                               │
  │  1. Generate Keys                   │
  │  2. Sign Message                    │
  │  3. Verify Signature                │
  │  4. Change Variant                  │
  │  5. Step-by-Step Visualization      │
  │  6. Run Test Suite                  │
  │  7. Security Info & Comparison      │
  │  8. Full Demo (all variants)        │
  │  0. Exit                            │
  └─────────────────────────────────────┘""")

        choice = input("\n  Enter choice: ").strip()

        if choice == "0":
            print("\n  Goodbye!\n")
            break

        elif choice == "1":
            print(f"\n  Generating {state['variant']} keypair...")
            t0 = time.time()
            pk, sk = state["dil"].keygen()
            elapsed = time.time() - t0
            state["pk"], state["sk"] = pk, sk
            print(f"\n  ✓ Keypair generated in {elapsed*1000:.1f} ms")
            print(f"\n  Public Key ({len(pk)} bytes):")
            print(f"    {hex_preview(pk, 32)}")
            print(f"\n  Secret Key ({len(sk)} bytes):")
            print(f"    {hex_preview(sk, 32)}")

        elif choice == "2":
            if not state["sk"]:
                print("\n  ✗ No keys loaded. Please generate keys first (option 1).")
                continue
            msg_input = input("\n  Enter message to sign: ").strip()
            if not msg_input:
                msg_input = "Default test message"
            message = msg_input.encode()

            print(f"\n  Signing with {state['variant']}...")
            t0 = time.time()
            sig = state["dil"].sign(state["sk"], message)
            elapsed = time.time() - t0

            state["last_sig"] = sig
            state["last_msg"] = message
            print(f"\n  ✓ Signature generated in {elapsed*1000:.1f} ms")
            print(f"\n  Signature ({len(sig)} bytes):")
            print(f"    c̃ (challenge): {sig[:32].hex()}")
            print(f"    z (response):  {hex_preview(sig[32:], 24)}")

        elif choice == "3":
            if not state.get("pk"):
                print("\n  ✗ No keys loaded.")
                continue
            if not state.get("last_sig"):
                print("\n  ✗ No signature available. Sign a message first (option 2).")
                continue

            print(f"\n  Verifying signature...")
            t0 = time.time()
            valid = state["dil"].verify(state["pk"], state["last_msg"], state["last_sig"])
            elapsed = time.time() - t0

            print(f"\n  Message: {state['last_msg']!r}")
            print(f"  Result: {'✓ VALID SIGNATURE' if valid else '✗ INVALID SIGNATURE'}")
            print(f"  Time: {elapsed*1000:.1f} ms")

        elif choice == "4":
            print("\n  Available variants:")
            print("    1. Dilithium2 (NIST Level 2, fastest)")
            print("    2. Dilithium3 (NIST Level 3, balanced)")
            print("    3. Dilithium5 (NIST Level 5, highest security)")
            v = input("  Choose variant (1-3): ").strip()
            variants = {"1": "Dilithium2", "2": "Dilithium3", "3": "Dilithium5"}
            if v in variants:
                state["variant"] = variants[v]
                state["dil"] = Dilithium(variants[v])
                state["pk"] = state["sk"] = None   # reset keys for new variant
                print(f"\n  ✓ Switched to {state['variant']}")
                print(f"  {state['dil']}")
            else:
                print("  Invalid choice.")

        elif choice == "5":
            demo_signing_steps(state["variant"])

        elif choice == "6":
            run_tests()

        elif choice == "7":
            print(SECURITY_CONCEPTS)
            print(COMPARISON_TABLE)

        elif choice == "8":
            demo_all()

        else:
            print("  Invalid choice. Please try again.")


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main() -> None:
    args = sys.argv[1:]

    if not args:
        cli_menu()

    elif args[0] == "demo":
        demo_all()

    elif args[0] == "test":
        run_tests()

    elif args[0] == "info":
        print(SECURITY_CONCEPTS)
        print(COMPARISON_TABLE)

    elif args[0] == "steps":
        variant = args[1] if len(args) > 1 else "Dilithium3"
        demo_signing_steps(variant)

    elif args[0] == "quick":
        # Quick single variant demo
        variant = args[1] if len(args) > 1 else "Dilithium3"
        demo_variant(variant)

    else:
        print("Usage:")
        print("  python main.py          → interactive menu")
        print("  python main.py demo     → demo all variants")
        print("  python main.py test     → run test suite")
        print("  python main.py info     → security concepts")
        print("  python main.py steps    → step-by-step visualization")
        print("  python main.py quick    → quick single-variant demo")


if __name__ == "__main__":
    main()
