"""
CRYSTALS-Kyber — Interactive CLI Demo
=======================================
Provides a menu-driven interface to explore the Kyber KEM:
  1. Generate Keys (Kyber512 / Kyber768 / Kyber1024)
  2. Encapsulate (produce ciphertext + shared secret)
  3. Decapsulate (recover shared secret from ciphertext)
  4. Run Full Demo (all steps with intermediate output)
  5. Run Performance Benchmark
  0. Exit

Also prints step-by-step flow diagrams and intermediate values.
"""

import sys
import time
from params import KYBER512, KYBER768, KYBER1024, PARAM_SETS, KyberParams
from kyber import kem_keygen, kem_encaps, kem_decaps


# ── Styling helpers ───────────────────────────────────────────────────────────

BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RED = "\033[91m"
DIM = "\033[2m"
RESET = "\033[0m"
SEPARATOR = f"{DIM}{'─' * 72}{RESET}"


def header(text: str):
    print(f"\n{BOLD}{CYAN}{'═' * 72}")
    print(f"  {text}")
    print(f"{'═' * 72}{RESET}\n")


def step(num: int, text: str):
    print(f"  {YELLOW}[Step {num}]{RESET} {text}")


def success(text: str):
    print(f"  {GREEN}✔ {text}{RESET}")


def error(text: str):
    print(f"  {RED}✘ {text}{RESET}")


def hex_preview(data: bytes, label: str, n: int = 32):
    """Print a hex preview of bytes data."""
    preview = data[:n].hex()
    if len(data) > n:
        preview += "..."
    print(f"    {DIM}{label}:{RESET} {preview}")
    print(f"    {DIM}({len(data)} bytes){RESET}")


# ── Parameter selection ───────────────────────────────────────────────────────

def select_params() -> KyberParams:
    """Let user choose a Kyber parameter set."""
    print(f"\n  {BOLD}Select parameter set:{RESET}")
    print(f"    1. Kyber512  (NIST Level 1 — ~AES-128)")
    print(f"    2. Kyber768  (NIST Level 3 — ~AES-192) {GREEN}[recommended]{RESET}")
    print(f"    3. Kyber1024 (NIST Level 5 — ~AES-256)")
    choice = input(f"\n  Choice [1-3, default=2]: ").strip()
    if choice == "1":
        return KYBER512
    elif choice == "3":
        return KYBER1024
    else:
        return KYBER768


# ── Menu actions ──────────────────────────────────────────────────────────────

# Store state between menu calls
state = {
    "params": None,
    "pk": None,
    "sk": None,
    "ct": None,
    "ss_enc": None,
}


def action_keygen():
    """Generate a Kyber key pair."""
    params = select_params()
    state["params"] = params

    header(f"Key Generation — {params.name}")
    print_keygen_flow()

    step(1, "Generating keypair...")
    t0 = time.perf_counter()
    pk, sk = kem_keygen(params)
    elapsed = time.perf_counter() - t0

    state["pk"] = pk
    state["sk"] = sk

    step(2, "Key pair generated!")
    hex_preview(pk, "Public key (pk)")
    hex_preview(sk, "Secret key (sk)")

    print(f"\n  {SEPARATOR}")
    print(f"  {BOLD}Key Sizes:{RESET}")
    print(f"    Public key:  {len(pk):>5} bytes  (expected: {params.pk_size})")
    print(f"    Secret key:  {len(sk):>5} bytes  (expected: {params.sk_size})")
    print(f"    Time:        {elapsed*1000:.2f} ms")
    success("Keys stored in memory. Ready for encapsulation.\n")


def action_encaps():
    """Encapsulate a shared secret."""
    if state["pk"] is None:
        error("No keys generated yet. Run KeyGen first.")
        return

    params = state["params"]
    header(f"Encapsulation — {params.name}")
    print_encaps_flow()

    step(1, "Encapsulating shared secret...")
    t0 = time.perf_counter()
    ct, ss = kem_encaps(params, state["pk"])
    elapsed = time.perf_counter() - t0

    state["ct"] = ct
    state["ss_enc"] = ss

    step(2, "Encapsulation complete!")
    hex_preview(ct, "Ciphertext (ct)")
    hex_preview(ss, "Shared secret (K)")

    print(f"\n  {SEPARATOR}")
    print(f"  {BOLD}Sizes:{RESET}")
    print(f"    Ciphertext:    {len(ct):>5} bytes  (expected: {params.ct_size})")
    print(f"    Shared secret: {len(ss):>5} bytes")
    print(f"    Time:          {elapsed*1000:.2f} ms")
    success("Ciphertext and shared secret stored. Ready for decapsulation.\n")


def action_decaps():
    """Decapsulate and verify the shared secret."""
    if state["ct"] is None:
        error("No ciphertext yet. Run Encapsulate first.")
        return

    params = state["params"]
    header(f"Decapsulation — {params.name}")
    print_decaps_flow()

    step(1, "Decapsulating shared secret...")
    t0 = time.perf_counter()
    ss_dec = kem_decaps(params, state["sk"], state["ct"])
    elapsed = time.perf_counter() - t0

    step(2, "Decapsulation complete!")
    hex_preview(state["ss_enc"], "Shared secret (encaps)")
    hex_preview(ss_dec, "Shared secret (decaps)")

    print(f"\n  {SEPARATOR}")
    if state["ss_enc"] == ss_dec:
        success(f"Shared secrets MATCH! ✓  ({elapsed*1000:.2f} ms)")
    else:
        error(f"Shared secrets DO NOT MATCH! ✗  ({elapsed*1000:.2f} ms)")

    # Test tampered ciphertext
    print(f"\n  {BOLD}Implicit Rejection Test:{RESET}")
    step(3, "Tampering with ciphertext (flip one byte)...")
    tampered_ct = bytearray(state["ct"])
    tampered_ct[0] ^= 0xFF
    tampered_ct = bytes(tampered_ct)

    ss_tampered = kem_decaps(params, state["sk"], tampered_ct)
    hex_preview(ss_tampered, "Shared secret (tampered)")

    if ss_tampered != state["ss_enc"]:
        success("Tampered ciphertext correctly rejected (different shared secret).\n")
    else:
        error("WARNING: Tampered ciphertext was not rejected!\n")


def action_full_demo():
    """Run a complete demo with all steps and verbose output."""
    params = select_params()
    state["params"] = params

    header(f"FULL DEMO — CRYSTALS-Kyber {params.name}")

    # ── Print overview ──
    print(f"  {BOLD}Parameters:{RESET}")
    print(f"    n = {params.n}    q = {params.q}    k = {params.k}")
    print(f"    η₁ = {params.eta1}    η₂ = {params.eta2}")
    print(f"    dᵤ = {params.du}    dᵥ = {params.dv}")
    print(f"    Public key:    {params.pk_size} bytes")
    print(f"    Secret key:    {params.sk_size} bytes")
    print(f"    Ciphertext:    {params.ct_size} bytes")
    print(f"    Shared secret: {params.ss_size} bytes")
    print()

    # ── Flow diagram ──
    print_full_flow()

    # ── KeyGen ──
    print(f"\n{SEPARATOR}")
    step(1, f"KEY GENERATION ({params.name})")
    t0 = time.perf_counter()
    pk, sk = kem_keygen(params)
    t_kg = time.perf_counter() - t0
    state["pk"], state["sk"] = pk, sk

    hex_preview(pk, "pk")
    hex_preview(sk, "sk")
    print(f"    {DIM}KeyGen time: {t_kg*1000:.2f} ms{RESET}")

    # ── Encapsulation ──
    print(f"\n{SEPARATOR}")
    step(2, "ENCAPSULATION")
    t0 = time.perf_counter()
    ct, ss_enc = kem_encaps(params, pk)
    t_enc = time.perf_counter() - t0
    state["ct"], state["ss_enc"] = ct, ss_enc

    hex_preview(ct, "ct")
    hex_preview(ss_enc, "K (encaps)")
    print(f"    {DIM}Encaps time: {t_enc*1000:.2f} ms{RESET}")

    # ── Decapsulation ──
    print(f"\n{SEPARATOR}")
    step(3, "DECAPSULATION")
    t0 = time.perf_counter()
    ss_dec = kem_decaps(params, sk, ct)
    t_dec = time.perf_counter() - t0

    hex_preview(ss_dec, "K (decaps)")
    print(f"    {DIM}Decaps time: {t_dec*1000:.2f} ms{RESET}")

    # ── Verification ──
    print(f"\n{SEPARATOR}")
    step(4, "VERIFICATION")
    if ss_enc == ss_dec:
        success("Shared secrets MATCH — KEM roundtrip successful!")
    else:
        error("Shared secrets DO NOT MATCH — something is wrong!")

    # ── Implicit rejection test ──
    print(f"\n{SEPARATOR}")
    step(5, "IMPLICIT REJECTION TEST")
    tampered_ct = bytearray(ct)
    tampered_ct[len(tampered_ct) // 2] ^= 0x42
    ss_reject = kem_decaps(params, sk, bytes(tampered_ct))

    hex_preview(ss_reject, "K (tampered)")
    if ss_reject != ss_enc:
        success("Tampered ciphertext correctly yields a DIFFERENT shared secret.")
    else:
        error("WARNING: Implicit rejection may have failed!")

    # ── Summary ──
    print(f"\n{SEPARATOR}")
    print(f"  {BOLD}Performance Summary ({params.name}):{RESET}")
    print(f"    KeyGen:  {t_kg*1000:8.2f} ms")
    print(f"    Encaps:  {t_enc*1000:8.2f} ms")
    print(f"    Decaps:  {t_dec*1000:8.2f} ms")
    print(f"    Total:   {(t_kg+t_enc+t_dec)*1000:8.2f} ms\n")


def action_benchmark():
    """Benchmark all three parameter sets."""
    header("Performance Benchmark")
    iterations = 5

    for params in [KYBER512, KYBER768, KYBER1024]:
        print(f"  {BOLD}{params.name}{RESET} ({iterations} iterations):")

        times_kg, times_enc, times_dec = [], [], []

        for _ in range(iterations):
            t0 = time.perf_counter()
            pk, sk = kem_keygen(params)
            times_kg.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            ct, ss = kem_encaps(params, pk)
            times_enc.append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            ss2 = kem_decaps(params, sk, ct)
            times_dec.append(time.perf_counter() - t0)

            assert ss == ss2, "KEM roundtrip failure!"

        avg = lambda lst: sum(lst) / len(lst) * 1000
        print(f"    KeyGen:  {avg(times_kg):8.2f} ms avg")
        print(f"    Encaps:  {avg(times_enc):8.2f} ms avg")
        print(f"    Decaps:  {avg(times_dec):8.2f} ms avg")
        print(f"    {DIM}pk={params.pk_size}B  sk={params.sk_size}B  ct={params.ct_size}B{RESET}")
        print()

    success("All roundtrips verified.\n")


# ── ASCII flow diagrams ──────────────────────────────────────────────────────

def print_keygen_flow():
    print(f"""
  {DIM}┌──────────────────────────────────────────────────────┐
  │                  KEY GENERATION                      │
  │                                                      │
  │  seed ──→ G(seed) ──→ (ρ, σ)                       │
  │                         │                            │
  │                    ┌────┴────┐                       │
  │                    ▼         ▼                       │
  │              Sample Â    Sample s,e                  │
  │             (from ρ)    (from σ, CBD)                │
  │                    │         │                       │
  │                    └────┬────┘                       │
  │                         ▼                            │
  │                 t̂ = Â·NTT(s) + NTT(e)               │
  │                         │                            │
  │                    ┌────┴────┐                       │
  │                    ▼         ▼                       │
  │               pk = (t̂,ρ)   sk = ŝ                   │
  └──────────────────────────────────────────────────────┘{RESET}
""")


def print_encaps_flow():
    print(f"""
  {DIM}┌──────────────────────────────────────────────────────┐
  │                  ENCAPSULATION                       │
  │                                                      │
  │  m ←$ random ──→ H(m) ──→ G(m||H(pk))              │
  │                              │                       │
  │                         ┌────┴────┐                  │
  │                         ▼         ▼                  │
  │                    K_bar (key)   r (coins)           │
  │                                   │                  │
  │                                   ▼                  │
  │                           CPAPKE.Enc(pk, m, r)       │
  │                                   │                  │
  │                                   ▼                  │
  │                          ct (ciphertext)             │
  │                                   │                  │
  │                     K = KDF(K_bar || H(ct))          │
  │                                   │                  │
  │                         ┌─────────┴─────────┐       │
  │                         ▼                   ▼       │
  │                    ciphertext         shared secret  │
  └──────────────────────────────────────────────────────┘{RESET}
""")


def print_decaps_flow():
    print(f"""
  {DIM}┌──────────────────────────────────────────────────────┐
  │                  DECAPSULATION                       │
  │                                                      │
  │  ct, sk=(sk',pk,h,z)                                │
  │         │                                            │
  │         ▼                                            │
  │  m' = CPAPKE.Dec(sk', ct)                           │
  │         │                                            │
  │         ▼                                            │
  │  (K_bar', r') = G(m' || h)                          │
  │         │                                            │
  │         ▼                                            │
  │  ct' = CPAPKE.Enc(pk, m', r')    ← re-encrypt!     │
  │         │                                            │
  │         ▼                                            │
  │  ┌────────────────┐                                  │
  │  │  ct == ct' ?   │                                  │
  │  └───┬────────┬───┘                                  │
  │   YES│        │NO                                    │
  │      ▼        ▼                                      │
  │  KDF(K̄'||H(ct))  KDF(z||H(ct))  ← implicit reject  │
  └──────────────────────────────────────────────────────┘{RESET}
""")


def print_full_flow():
    print(f"""
  {DIM}┌────────────────────────────────────────────────────────────────┐
  │              CRYSTALS-Kyber KEM — Full Flow                    │
  │                                                                │
  │   ALICE (sender)                          BOB (receiver)       │
  │   ──────────────                          ──────────────       │
  │                                                                │
  │                                    1. (pk, sk) ← KeyGen()      │
  │                         ┌── pk ──────────────┘                 │
  │                         ▼                                      │
  │   2. (ct, K) ← Encaps(pk)                                     │
  │          │                                                     │
  │          └── ct ──────────────→  3. K ← Decaps(sk, ct)        │
  │                                                                │
  │   Alice has K ←──── K must match ────→ Bob has K               │
  │                                                                │
  │   Both parties now share secret K for symmetric encryption.    │
  └────────────────────────────────────────────────────────────────┘{RESET}
""")


# ── Main menu ─────────────────────────────────────────────────────────────────

def main():
    print(f"""
{BOLD}{CYAN}╔════════════════════════════════════════════════════════════╗
║           CRYSTALS-Kyber — Post-Quantum KEM              ║
║         Implementation for Study and Research            ║
╚════════════════════════════════════════════════════════════╝{RESET}

  This tool implements the CRYSTALS-Kyber Key Encapsulation
  Mechanism, a post-quantum cryptographic algorithm selected
  by NIST for standardization (FIPS 203 / ML-KEM).

  Security is based on the Module Learning With Errors (MLWE)
  problem, believed to be resistant to both classical and
  quantum attacks.
""")

    while True:
        print(f"\n  {BOLD}Menu:{RESET}")
        print(f"    1. Generate Key Pair")
        print(f"    2. Encapsulate (create shared secret)")
        print(f"    3. Decapsulate (recover shared secret)")
        print(f"    4. Run Full Demo")
        print(f"    5. Performance Benchmark")
        print(f"    0. Exit")
        print()

        choice = input(f"  {BOLD}Select [0-5]: {RESET}").strip()

        if choice == "1":
            action_keygen()
        elif choice == "2":
            action_encaps()
        elif choice == "3":
            action_decaps()
        elif choice == "4":
            action_full_demo()
        elif choice == "5":
            action_benchmark()
        elif choice == "0":
            print(f"\n  {DIM}Goodbye!{RESET}\n")
            break
        else:
            print(f"  {RED}Invalid choice.{RESET}")


if __name__ == "__main__":
    main()
