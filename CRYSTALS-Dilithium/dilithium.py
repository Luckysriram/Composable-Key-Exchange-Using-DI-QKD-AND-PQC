"""
dilithium.py - CRYSTALS-Dilithium Digital Signature Scheme

Implements KeyGen, Sign, and Verify for all three variants:
  - Dilithium2 (NIST Security Level 2)
  - Dilithium3 (NIST Security Level 3)
  - Dilithium5 (NIST Security Level 5)

Security is based on the hardness of:
  - Module-LWE (Module Learning With Errors) - used for key generation hardness
  - Module-SIS (Module Short Integer Solution) - used for signature unforgeability

NIST standardized as FIPS 204 (ML-DSA) in 2024.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNING FLOW:
  1. Hash message: μ = H(tr || msg)
  2. Generate nonce seed: ρ'' = H(K || rnd || μ)
  3. Loop:
       a. Sample mask: y from expand_mask(ρ'', κ)
       b. Commit: w = A*y, w1 = HighBits(w, 2γ₂)
       c. Hash: c̃ = H(μ || encode(w1))
       d. Sample challenge: c = SampleInBall(c̃)
       e. Compute: z = y + c*s1
       f. Rejection tests on z and LowBits(w - c*s2)
       g. Compute hints h from c*t0
       h. Rejection tests on hints
       i. Return (c̃, z, h)

VERIFICATION FLOW:
  1. μ = H(tr || msg)
  2. c = SampleInBall(c̃, τ)
  3. w1' = UseHint(h, A*z - c*t1*2^d, 2γ₂)
  4. Checks: ||z||∞ < γ₁-β, sum(h) ≤ ω, c̃ = H(μ || encode(w1'))
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from ntt import Q, N, ntt, inv_ntt, poly_pointwise_mul
from poly import (
    poly_add, poly_sub, poly_negate, poly_scalar_mul, poly_zero,
    center_coeff, inf_norm, vec_inf_norm,
    poly_power2round, vec_power2round,
    poly_high_bits, poly_low_bits, vec_high_bits, vec_low_bits,
    poly_make_hint, poly_use_hint, vec_make_hint, vec_use_hint,
    count_hints, mat_vec_mul, vec_add, vec_sub, vec_scalar_poly_mul,
)
from utils import (
    Q, N, shake256, H,
    expand_A, expand_S, expand_mask, sample_in_ball, encode_w1,
    pack_pk, unpack_pk, pack_sk, unpack_sk, pack_sig, unpack_sig,
    DilithiumParams, DILITHIUM2, DILITHIUM3, DILITHIUM5,
)


# ─── Key Generation ───────────────────────────────────────────────────────────

def keygen(params: DilithiumParams, seed: bytes = None) -> tuple[bytes, bytes]:
    """
    Dilithium Key Generation.

    Algorithm:
      1. ξ  ←  random 32 bytes  (master seed)
      2. (ρ, ρ', K) = H(ξ, 96) split into 32+32+32
      3. A  = ExpandA(ρ)        (k×l matrix, stored in NTT domain)
      4. (s1, s2) = ExpandS(ρ', η, l, k)   (small secret vectors)
      5. t  = A·s1 + s2         (module-LWE instance)
      6. (t1, t0) = Power2Round(t, d)
      7. pk = (ρ, t1)
      8. tr = H(pk, 64)
      9. sk = (ρ, K, tr, s1, s2, t0)

    Security: Breaking the scheme requires solving Module-LWE to find s1, s2
              from t = A·s1 + s2 (computationally hard, quantum-resistant).

    Args:
        params: Dilithium parameter set (Dilithium2/3/5)
        seed:   Optional 32-byte master seed (random if None)

    Returns:
        (pk, sk): packed public key and secret key as bytes
    """
    k, l, eta, d = params.k, params.l, params.eta, params.d

    # Step 1: Generate random seed
    xi = seed if seed is not None else os.urandom(32)

    # Step 2: Derive sub-seeds
    expanded = H(xi, 96)
    rho      = expanded[:32]       # public matrix seed
    rho_p    = expanded[32:64]     # secret polynomial seed
    K        = expanded[64:96]     # signing key material

    # Step 3: Generate public matrix A in NTT domain
    A_ntt = expand_A(rho, k, l)

    # Step 4: Generate secret vectors with small coefficients
    s1, s2 = expand_S(rho_p, eta, l, k)

    # Step 5: t = A·s1 + s2 (module-LWE "encryption")
    t = _mat_vec_add(A_ntt, s1, s2)

    # Step 6: Power2Round — split t into public t1 and secret t0
    t1_list, t0_list = [], []
    for poly in t:
        t1_poly, t0_poly = poly_power2round(poly, d)
        t1_list.append(t1_poly)
        t0_list.append(t0_poly)

    # Step 7-8: Pack public key and compute hash of it
    pk = pack_pk(rho, t1_list)
    tr = H(pk, 64)

    # Step 9: Pack secret key
    sk = pack_sk(rho, K, tr, s1, s2, t0_list, params)

    return pk, sk


# ─── Signing ──────────────────────────────────────────────────────────────────

def sign(sk: bytes, message: bytes, params: DilithiumParams,
         randomized: bool = False) -> bytes:
    """
    Dilithium Signing.

    Algorithm:
      1. Unpack sk: (ρ, K, tr, s1, s2, t0)
      2. μ = H(tr || message, 64)         [message hash with public key binding]
      3. rnd = random 32 bytes if randomized, else 0^32
      4. ρ'' = H(K || rnd || μ, 64)       [per-signature randomness seed]
      5. κ = 0, retry loop:
           y = ExpandMask(ρ'', κ, l, γ₁)  [sample fresh randomness]
           w = A·y                         [commitment]
           w1 = HighBits(w, 2γ₂)
           c̃ = H(μ || encode(w1), 32)     [challenge hash]
           c = SampleInBall(c̃, τ)         [sparse challenge polynomial]
           cs1 = c·s1                      [apply challenge to secret]
           cs2 = c·s2
           z = y + cs1                     [response]
           r0 = LowBits(w - cs2, 2γ₂)
           REJECT if ||z||∞ ≥ γ₁ - β     [check response norm]
           REJECT if ||r0||∞ ≥ γ₂ - β    [check low bits]
           ct0 = c·t0
           h = MakeHint(-ct0, w - cs2 + ct0, 2γ₂)
           REJECT if ||ct0||∞ ≥ γ₂        [check hint validity]
           REJECT if sum(h) > ω           [check hint count]
           κ += l, continue
      6. Return σ = (c̃, z, h)

    Args:
        sk:          packed secret key bytes
        message:     message to sign (arbitrary bytes)
        params:      Dilithium parameter set
        randomized:  if True, use randomized signing (default: deterministic)

    Returns:
        Packed signature bytes
    """
    k, l, eta, d = params.k, params.l, params.eta, params.d
    tau, beta     = params.tau, params.beta
    gamma1, gamma2 = params.gamma1, params.gamma2
    omega         = params.omega
    alpha         = 2 * gamma2

    # Step 1: Unpack secret key
    rho, K, tr, s1, s2, t0 = unpack_sk(sk, params)

    # Step 2: Generate matrix A (needed for commitment w = A·y)
    A_ntt = expand_A(rho, k, l)

    # Step 3-4: Compute message representative and per-signature seed
    mu      = H(tr + message, 64)
    rnd     = os.urandom(32) if randomized else bytes(32)
    rho_pp  = H(K + rnd + mu, 64)

    # Step 5: Rejection sampling loop
    kappa = 0
    while True:
        # 5a. Sample mask vector y with coefficients in (-γ₁, γ₁]
        y = expand_mask(rho_pp, kappa, l, gamma1)
        kappa += l

        # 5b. Compute commitment w = A·y, then w1 = HighBits(w)
        w = mat_vec_mul(A_ntt, y)
        w1 = [poly_high_bits(p, alpha) for p in w]

        # 5c. Hash to get challenge seed
        c_tilde = H(mu + encode_w1(w1, gamma2), 32)

        # 5d. Sample sparse challenge polynomial
        c = sample_in_ball(c_tilde, tau)

        # 5e. Compute c·s1 and c·s2
        cs1 = vec_scalar_poly_mul(c, s1)
        cs2 = vec_scalar_poly_mul(c, s2)

        # 5f. Response z = y + c·s1
        z = vec_add(y, cs1)

        # Check 1: ||z||∞ < γ₁ - β
        if vec_inf_norm(z) >= gamma1 - beta:
            continue   # reject and retry

        # Compute w - c·s2
        w_minus_cs2 = vec_sub(w, cs2)

        # Check 2: ||LowBits(w - cs2)||∞ < γ₂ - β
        r0 = [poly_low_bits(p, alpha) for p in w_minus_cs2]
        if vec_inf_norm(r0) >= gamma2 - beta:
            continue   # reject and retry

        # 5g. Compute c·t0 (for hint generation)
        ct0 = vec_scalar_poly_mul(c, t0)

        # Check 3: ||c·t0||∞ < γ₂
        if vec_inf_norm(ct0) >= gamma2:
            continue   # reject and retry

        # 5h. Compute hints h
        # h[i] = MakeHint(-ct0[i], w[i] - cs2[i] + ct0[i], 2γ₂)
        neg_ct0 = [poly_negate(p) for p in ct0]
        w_minus_cs2_plus_ct0 = vec_add(w_minus_cs2, ct0)
        h = vec_make_hint(neg_ct0, w_minus_cs2_plus_ct0, alpha)

        # Check 4: sum of hints ≤ ω
        if count_hints(h) > omega:
            continue   # reject and retry

        # All checks passed — pack and return signature
        return pack_sig(c_tilde, z, h, params)


# ─── Verification ─────────────────────────────────────────────────────────────

def verify(pk: bytes, message: bytes, signature: bytes,
           params: DilithiumParams) -> bool:
    """
    Dilithium Signature Verification.

    Algorithm:
      1. Unpack pk: (ρ, t1)
      2. Unpack sig: (c̃, z, h)
      3. Compute μ = H(H(pk, 64) || message, 64)
      4. Recover challenge: c = SampleInBall(c̃, τ)
      5. Recompute A from ρ
      6. Compute: w1' = UseHint(h, A·z - c·t1·2^d, 2γ₂)
      7. Check: ||z||∞ < γ₁ - β
      8. Check: sum(h) ≤ ω  (and h not malformed)
      9. Check: c̃ = H(μ || encode(w1'), 32)

    Args:
        pk:        packed public key
        message:   message that was signed
        signature: packed signature

    Returns:
        True if signature is valid, False otherwise
    """
    k, l, d   = params.k, params.l, params.d
    tau, beta = params.tau, params.beta
    gamma1, gamma2 = params.gamma1, params.gamma2
    omega     = params.omega
    alpha     = 2 * gamma2

    # Step 2: Unpack public key and signature
    rho, t1 = unpack_pk(pk, k)
    unpacked = unpack_sig(signature, params)
    if unpacked is None:
        return False
    c_tilde, z, h = unpacked

    if h is None:
        return False

    # Step 7: Early norm check on z (before expensive operations)
    if vec_inf_norm(z) >= gamma1 - beta:
        return False

    # Step 8: Check hint count
    if count_hints(h) > omega:
        return False

    # Step 3: Compute message hash
    tr = H(pk, 64)
    mu = H(tr + message, 64)

    # Step 4: Recover challenge polynomial
    c = sample_in_ball(c_tilde, tau)

    # Step 5: Rebuild public matrix A
    A_ntt = expand_A(rho, k, l)

    # Step 6: Compute w1' = UseHint(h, A·z - c·t1·2^d, 2γ₂)
    # First: A·z
    Az = mat_vec_mul(A_ntt, z)

    # Then: c·t1·2^d
    # t1 coefficients are stored as t1; we need c·t1·2^d
    # t1*2^d means left-shift coefficients (scale by 2^d)
    t1_scaled = [[x * (1 << d) % Q for x in poly] for poly in t1]
    ct1_2d = vec_scalar_poly_mul(c, t1_scaled)

    # w_approx = A·z - c·t1·2^d
    w_approx = vec_sub(Az, ct1_2d)

    # Recover high bits using hints
    w1_prime = [poly_use_hint(h[i], w_approx[i], alpha) for i in range(k)]

    # Step 9: Verify challenge hash
    c_tilde_prime = H(mu + encode_w1(w1_prime, gamma2), 32)

    return c_tilde == c_tilde_prime


# ─── Helper Functions ─────────────────────────────────────────────────────────

def _mat_vec_add(A_ntt: list, s1: list, s2: list) -> list:
    """Compute A·s1 + s2 (used in KeyGen)."""
    t = mat_vec_mul(A_ntt, s1)
    return vec_add(t, s2)


# ─── Convenience API ──────────────────────────────────────────────────────────

class Dilithium:
    """
    High-level API for CRYSTALS-Dilithium.

    Usage:
        dil = Dilithium("Dilithium3")
        pk, sk = dil.keygen()
        sig = dil.sign(sk, b"Hello, post-quantum world!")
        valid = dil.verify(pk, b"Hello, post-quantum world!", sig)

    Variants: "Dilithium2", "Dilithium3", "Dilithium5"
    """

    PARAMS = {"Dilithium2": DILITHIUM2, "Dilithium3": DILITHIUM3, "Dilithium5": DILITHIUM5}

    def __init__(self, variant: str = "Dilithium3"):
        if variant not in self.PARAMS:
            raise ValueError(f"Unknown variant '{variant}'. Choose from: {list(self.PARAMS)}")
        self.params = self.PARAMS[variant]
        self.variant = variant

    def keygen(self, seed: bytes = None) -> tuple[bytes, bytes]:
        """Generate a keypair. Returns (public_key, secret_key)."""
        return keygen(self.params, seed)

    def sign(self, sk: bytes, message: bytes, randomized: bool = False) -> bytes:
        """Sign a message. Returns signature bytes."""
        return sign(sk, message, self.params, randomized)

    def verify(self, pk: bytes, message: bytes, signature: bytes) -> bool:
        """Verify a signature. Returns True if valid."""
        return verify(pk, message, signature, self.params)

    @property
    def pk_size(self) -> int:
        """Public key size in bytes."""
        return 32 + self.params.k * 320

    @property
    def sk_size(self) -> int:
        """Secret key size in bytes."""
        bits = 3 if self.params.eta == 2 else 4
        eta_poly = N * bits // 8
        return (32 + 32 + 64
                + self.params.l * eta_poly
                + self.params.k * eta_poly
                + self.params.k * 416)

    @property
    def sig_size(self) -> int:
        """Signature size in bytes."""
        bits = 18 if self.params.gamma1 == (1 << 17) else 20
        return 32 + self.params.l * N * bits // 8 + self.params.omega + self.params.k

    def __repr__(self) -> str:
        return (f"Dilithium({self.variant}: "
                f"pk={self.pk_size}B, sk={self.sk_size}B, sig={self.sig_size}B)")


if __name__ == "__main__":
    print("Testing Dilithium implementations...\n")

    for variant in ["Dilithium2", "Dilithium3", "Dilithium5"]:
        dil = Dilithium(variant)
        print(f"=== {dil} ===")

        pk, sk = dil.keygen()
        msg = b"Post-quantum cryptography test message"
        sig = dil.sign(sk, msg)

        valid = dil.verify(pk, msg, sig)
        print(f"  Sign + Verify: {'PASS' if valid else 'FAIL'}")

        # Test invalid signature detection
        bad_sig = bytearray(sig)
        bad_sig[50] ^= 0xFF   # flip bits
        invalid = dil.verify(pk, msg, bytes(bad_sig))
        print(f"  Invalid sig rejected: {'PASS' if not invalid else 'FAIL'}")

        # Test wrong message detection
        wrong_msg = dil.verify(pk, b"wrong message", sig)
        print(f"  Wrong message rejected: {'PASS' if not wrong_msg else 'FAIL'}")
        print()
