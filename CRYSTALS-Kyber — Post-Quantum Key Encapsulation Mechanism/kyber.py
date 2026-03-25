"""
CRYSTALS-Kyber: IND-CPA PKE + IND-CCA KEM
============================================
Implements the full Kyber Key Encapsulation Mechanism:

  ┌──────────────── IND-CPA PKE ────────────────┐
  │  cpapke_keygen() → (pk, sk)                 │
  │  cpapke_enc(pk, msg, coins) → ct            │
  │  cpapke_dec(sk, ct) → msg                   │
  └──────────────────────────────────────────────┘

  ┌──────────────── IND-CCA KEM ────────────────┐
  │  kem_keygen() → (pk, sk)                    │  Fujisaki-Okamoto
  │  kem_encaps(pk) → (ct, shared_secret)       │  transform applied
  │  kem_decaps(sk, ct) → shared_secret         │  on top of IND-CPA
  └──────────────────────────────────────────────┘

The Fujisaki-Okamoto (FO) transform converts the IND-CPA-secure PKE
into an IND-CCA-secure KEM by:
  1. Re-encrypting during decapsulation and checking consistency
  2. Using implicit rejection (return a pseudorandom value on failure)

Security relies on the Module Learning With Errors (MLWE) problem,
which is believed to be hard even for quantum computers.
"""

from params import Q, N, KyberParams, KYBER512, KYBER768, KYBER1024
from poly import Poly, PolyVec
from ntt import ntt, ntt_inv
from sampling import sample_noise, generate_matrix
from compress import (
    compress_poly, decompress_poly,
    compress_polyvec, decompress_polyvec,
)
from utils import (
    hash_g, hash_h, prf, kdf, random_bytes,
    encode_poly, decode_poly,
)


# ══════════════════════════════════════════════════════════════════════════════
#  IND-CPA Public Key Encryption (CPAPKE)
# ══════════════════════════════════════════════════════════════════════════════

def cpapke_keygen(params: KyberParams) -> tuple[bytes, bytes]:
    """
    IND-CPA Key Generation.

    ┌─────────────────────────────────────────────────────┐
    │  1. d ← random 32 bytes                            │
    │  2. (ρ, σ) ← G(d)          — expand seed           │
    │  3. Â ← Sam(ρ)             — sample matrix in NTT  │
    │  4. s ← CBD_η1(σ, 0..k-1)  — secret vector         │
    │  5. e ← CBD_η1(σ, k..2k-1) — error vector          │
    │  6. ŝ ← NTT(s)                                     │
    │  7. ê ← NTT(e)                                     │
    │  8. t̂ ← Â·ŝ + ê            — public key poly       │
    │  9. pk ← Encode(t̂) || ρ                            │
    │ 10. sk ← Encode(ŝ)                                 │
    └─────────────────────────────────────────────────────┘

    Returns: (pk_bytes, sk_bytes)
    """
    k = params.k
    eta1 = params.eta1

    # Step 1-2: Generate seed and expand
    d = random_bytes(32)
    rho, sigma = hash_g(d)

    # Step 3: Sample public matrix  (in NTT domain)
    A_hat = generate_matrix(rho, k, transpose=False)

    # Step 4-5: Sample secret and error vectors
    s = sample_noise(sigma, eta1, nonce=0, k=k)
    e = sample_noise(sigma, eta1, nonce=k, k=k)

    s_vec = PolyVec(s)
    e_vec = PolyVec(e)

    # Step 6-7: Transform to NTT domain
    s_hat = s_vec.to_ntt()
    e_hat = e_vec.to_ntt()

    # Step 8: Compute public key polynomial t̂ = Â·ŝ + ê
    t_hat = PolyVec.matrix_vec_ntt(A_hat, s_hat).add(e_hat).reduce()

    # Step 9-10: Encode public and secret keys
    pk_bytes = _encode_polyvec(t_hat) + rho
    sk_bytes = _encode_polyvec(s_hat)

    return pk_bytes, sk_bytes


def cpapke_enc(params: KyberParams, pk: bytes, msg: bytes, coins: bytes) -> bytes:
    """
    IND-CPA Encryption.

    ┌─────────────────────────────────────────────────────┐
    │  1. t̂, ρ ← Decode(pk)                              │
    │  2. Â^T ← Sam(ρ)           — transposed matrix      │
    │  3. r ← CBD_η1(coins, 0..k-1) — random vector       │
    │  4. e1 ← CBD_η2(coins, k..2k-1) — error vector 1    │
    │  5. e2 ← CBD_η2(coins, 2k) — error polynomial       │
    │  6. r̂ ← NTT(r)                                      │
    │  7. u ← NTT⁻¹(Â^T · r̂) + e1  — ciphertext part 1  │
    │  8. v ← NTT⁻¹(t̂ᵀ · r̂) + e2 + ⌈q/2⌋·m             │
    │  9. ct ← Compress(u, du) || Compress(v, dv)         │
    └─────────────────────────────────────────────────────┘

    Returns: ciphertext bytes
    """
    k = params.k
    eta1, eta2 = params.eta1, params.eta2
    du, dv = params.du, params.dv

    # Step 1: Decode public key
    t_hat = _decode_polyvec(pk[:12 * k * N // 8], k)
    rho = pk[12 * k * N // 8:]

    # Step 2: Re-generate matrix (transposed for encryption)
    A_hat_T = generate_matrix(rho, k, transpose=True)

    # Step 3-5: Sample randomness and errors
    r = sample_noise(coins, eta1, nonce=0, k=k)
    e1 = sample_noise(coins, eta2, nonce=k, k=k)
    e2 = sample_noise(coins, eta2, nonce=2 * k, k=1)

    r_vec = PolyVec(r)
    e1_vec = PolyVec(e1)

    # Step 6: Transform r to NTT domain
    r_hat = r_vec.to_ntt()

    # Step 7: u = NTT⁻¹(Â^T · r̂) + e1
    u = PolyVec.matrix_vec_ntt(A_hat_T, r_hat).from_ntt().add(e1_vec).reduce()

    # Step 8: v = NTT⁻¹(t̂ᵀ · r̂) + e2 + Decode(msg)
    v = t_hat.inner_product_ntt(r_hat)
    v = Poly(ntt_inv(v.coeffs))
    v = v.add(e2[0])

    # Encode message: each bit → ⌈q/2⌋ or 0
    msg_poly = _msg_to_poly(msg)
    v = v.add(msg_poly).reduce()

    # Step 9: Compress and encode ciphertext
    u_compressed = compress_polyvec(u, du)
    v_compressed = compress_poly(v, dv)

    ct = b''
    for uc in u_compressed:
        ct += encode_poly(uc, du)
    ct += encode_poly(v_compressed, dv)

    return ct


def cpapke_dec(params: KyberParams, sk: bytes, ct: bytes) -> bytes:
    """
    IND-CPA Decryption.

    ┌─────────────────────────────────────────────────────┐
    │  1. u, v ← Decompress(Decode(ct))                  │
    │  2. ŝ ← Decode(sk)                                 │
    │  3. m ← Compress₁(v − NTT⁻¹(ŝᵀ · NTT(u)))        │
    └─────────────────────────────────────────────────────┘

    Returns: 32-byte message
    """
    k = params.k
    du, dv = params.du, params.dv

    # Step 1: Decode and decompress ciphertext
    u_bytes_len = du * k * N // 8
    u_polys = []
    offset = 0
    for i in range(k):
        chunk_len = du * N // 8
        coeffs = decode_poly(ct[offset:offset + chunk_len], du)
        u_polys.append(coeffs)
        offset += chunk_len

    v_coeffs = decode_poly(ct[offset:offset + dv * N // 8], dv)

    u_decompressed = decompress_polyvec(u_polys, du)
    v_decompressed = decompress_poly(v_coeffs, dv)

    # Step 2: Decode secret key
    s_hat = _decode_polyvec(sk, k)

    # Step 3: Compute v - s^T · u
    u_hat = u_decompressed.to_ntt()
    su = s_hat.inner_product_ntt(u_hat)
    su = Poly(ntt_inv(su.coeffs))

    mp = v_decompressed.sub(su).reduce()

    # Recover message: each coefficient → closest bit
    return _poly_to_msg(mp)


# ══════════════════════════════════════════════════════════════════════════════
#  IND-CCA KEM (Fujisaki-Okamoto Transform)
# ══════════════════════════════════════════════════════════════════════════════

def kem_keygen(params: KyberParams = KYBER768) -> tuple[bytes, bytes]:
    """
    KEM Key Generation.

    ┌─────────────────────────────────────────────────────┐
    │  1. (pk, sk') ← CPAPKE.KeyGen()                    │
    │  2. sk ← sk' || pk || H(pk) || z                   │
    │     where z ← random 32 bytes (implicit reject key) │
    └─────────────────────────────────────────────────────┘

    The secret key bundles everything needed for decapsulation:
      - sk': CPA secret key for decryption
      - pk:  public key for re-encryption check
      - H(pk): cached hash of public key
      - z:  random bytes for implicit rejection

    Returns: (public_key, secret_key)
    """
    pk, sk_cpa = cpapke_keygen(params)
    z = random_bytes(32)

    # Bundle: sk_cpa || pk || H(pk) || z
    sk = sk_cpa + pk + hash_h(pk) + z

    return pk, sk


def kem_encaps(params: KyberParams, pk: bytes) -> tuple[bytes, bytes]:
    """
    KEM Encapsulation.

    ┌─────────────────────────────────────────────────────┐
    │  1. m ← random 32 bytes                             │
    │  2. m ← H(m)               — domain separation      │
    │  3. (K̄, r) ← G(m || H(pk)) — derive key + coins    │
    │  4. ct ← CPAPKE.Enc(pk, m, r)                       │
    │  5. K ← KDF(K̄ || H(ct))   — final shared secret    │
    └─────────────────────────────────────────────────────┘

    Returns: (ciphertext, shared_secret)
    """
    m = random_bytes(32)
    m = hash_h(m)  # domain separation

    pk_hash = hash_h(pk)
    K_bar, r = hash_g(m + pk_hash)

    ct = cpapke_enc(params, pk, m, r)
    shared_secret = kdf(K_bar + hash_h(ct))

    return ct, shared_secret


def kem_decaps(params: KyberParams, sk: bytes, ct: bytes) -> bytes:
    """
    KEM Decapsulation (with Fujisaki-Okamoto implicit rejection).

    ┌─────────────────────────────────────────────────────┐
    │  1. Parse sk = sk' || pk || h || z                   │
    │  2. m' ← CPAPKE.Dec(sk', ct)                        │
    │  3. (K̄', r') ← G(m' || h)                          │
    │  4. ct' ← CPAPKE.Enc(pk, m', r')  — re-encrypt      │
    │  5. if ct == ct':                                    │
    │       K ← KDF(K̄' || H(ct))       — accept           │
    │     else:                                            │
    │       K ← KDF(z  || H(ct))        — implicit reject  │
    └─────────────────────────────────────────────────────┘

    The re-encryption check (step 4-5) is the core of the FO transform:
    it ensures CCA security by verifying that the decrypted message
    would produce the same ciphertext. On failure, a pseudorandom
    value derived from the secret z is returned (implicit rejection),
    preventing chosen-ciphertext attacks.

    Returns: 32-byte shared secret
    """
    k = params.k

    # Step 1: Parse secret key components
    sk_cpa_len = 12 * k * N // 8
    pk_len = 12 * k * N // 8 + 32

    sk_cpa = sk[:sk_cpa_len]
    pk = sk[sk_cpa_len:sk_cpa_len + pk_len]
    h = sk[sk_cpa_len + pk_len:sk_cpa_len + pk_len + 32]
    z = sk[sk_cpa_len + pk_len + 32:sk_cpa_len + pk_len + 64]

    # Step 2: Decrypt ciphertext
    m_prime = cpapke_dec(params, sk_cpa, ct)

    # Step 3: Re-derive randomness
    K_bar_prime, r_prime = hash_g(m_prime + h)

    # Step 4: Re-encrypt
    ct_prime = cpapke_enc(params, pk, m_prime, r_prime)

    # Step 5: Compare (constant-time comparison for side-channel resistance)
    ct_hash = hash_h(ct)
    if _ct_compare(ct, ct_prime):
        # Accept: ciphertext matches
        return kdf(K_bar_prime + ct_hash)
    else:
        # Implicit rejection: return pseudorandom value from z
        return kdf(z + ct_hash)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper functions
# ══════════════════════════════════════════════════════════════════════════════

def _encode_polyvec(pv: PolyVec) -> bytes:
    """Encode a PolyVec using 12 bits per coefficient."""
    result = b''
    for p in pv.polys:
        result += encode_poly(p.coeffs, 12)
    return result


def _decode_polyvec(data: bytes, k: int) -> PolyVec:
    """Decode bytes into a PolyVec of k polynomials (12 bits per coeff)."""
    polys = []
    chunk = 12 * N // 8  # 384 bytes per polynomial
    for i in range(k):
        coeffs = decode_poly(data[i * chunk:(i + 1) * chunk], 12)
        polys.append(Poly(coeffs))
    return PolyVec(polys)


def _msg_to_poly(msg: bytes) -> Poly:
    """Convert a 32-byte message to a polynomial.
    Each bit of the message maps to ⌈q/2⌋ (≈ 1665) or 0."""
    coeffs = [0] * N
    for i in range(32):
        for j in range(8):
            bit = (msg[i] >> j) & 1
            coeffs[8 * i + j] = bit * ((Q + 1) // 2)  # bit * 1665
    return Poly(coeffs)


def _poly_to_msg(p: Poly) -> bytes:
    """Convert a polynomial back to a 32-byte message.
    Each coefficient is rounded to the nearest bit:
      closer to 0 → 0,  closer to ⌈q/2⌋ → 1."""
    msg = bytearray(32)
    for i in range(N):
        # Coefficient is closer to q/2 than to 0?
        t = ((p.coeffs[i] << 1) + Q // 2) // Q & 1
        msg[i // 8] |= t << (i % 8)
    return bytes(msg)


def _ct_compare(a: bytes, b: bytes) -> bool:
    """Compare two byte strings in constant time (best-effort in Python).
    Returns True if they are equal."""
    if len(a) != len(b):
        return False
    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0
