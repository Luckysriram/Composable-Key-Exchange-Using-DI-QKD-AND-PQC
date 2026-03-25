# ML-KEM-768 Implementation in Python
# Based on FIPS 203 (Module-Lattice-based Key-Encapsulation Mechanism)
# Security Level 3: 768-bit security

import os
import hashlib
import secrets
from typing import List, Tuple

# Parameters for ML-KEM-768
N = 256
K = 3
Q = 3329
ETA1 = 2
ETA2 = 2
DU = 10
DV = 4
DT = 12  # For key generation

# Primitive root for NTT
ZETA = 17

# Modular inverse of 2^8 mod Q for decompression
INV_256 = pow(256, Q-2, Q)

# Bit reversal table for NTT
BIT_REV = [0] * N
for i in range(N):
    BIT_REV[i] = int(''.join(reversed(format(i, '08b'))), 2)

# NTT zetas
zetas = [pow(ZETA, BIT_REV[i], Q) for i in range(128)]

def mod_reduce(x: int) -> int:
    """Reduce x modulo Q"""
    return ((x % Q) + Q) % Q

def poly_add(a: List[int], b: List[int]) -> List[int]:
    """Add two polynomials"""
    return [(x + y) % Q for x, y in zip(a, b)]

def poly_sub(a: List[int], b: List[int]) -> List[int]:
    """Subtract two polynomials"""
    return [(x - y) % Q for x, y in zip(a, b)]

def poly_mul(a: List[int], b: List[int]) -> List[int]:
    """Multiply two polynomials using NTT"""
    a_ntt = ntt(a)
    b_ntt = ntt(b)
    c_ntt = [(x * y) % Q for x, y in zip(a_ntt, b_ntt)]
    return intt(c_ntt)

def poly_mul_ntt(a_ntt: List[int], b_ntt: List[int]) -> List[int]:
    """Multiply two polynomials in NTT domain"""
    c_ntt = [(x * y) % Q for x, y in zip(a_ntt, b_ntt)]
    return intt(c_ntt)

def ntt(f: List[int]) -> List[int]:
    """Number Theoretic Transform"""
    f_hat = f.copy()
    k = 1
    for len in [128,64,32,16,8,4,2]:
        for start in range(0, N, 2*len):
            zeta = zetas[k]
            k += 1
            for j in range(len):
                t = (zeta * f_hat[start + j + len]) % Q
                f_hat[start + j + len] = (f_hat[start + j] - t) % Q
                f_hat[start + j] = (f_hat[start + j] + t) % Q
    return f_hat

def intt(f_hat: List[int]) -> List[int]:
    """Inverse Number Theoretic Transform"""
    f = f_hat.copy()
    k = 127
    for len in [2,4,8,16,32,64,128]:
        for start in range(0, N, 2*len):
            zeta = pow(zetas[k], Q-2, Q)
            k -= 1
            for j in range(len):
                t = f[start + j]
                f[start + j] = (t + f[start + j + len]) % Q
                f[start + j + len] = ((t - f[start + j + len]) * zeta) % Q
    inv_n = pow(N, Q-2, Q)
    return [(x * inv_n) % Q for x in f]

def cbd(eta: int, bytes_in: bytes) -> List[int]:
    """Centered Binomial Distribution sampling"""
    f = [0] * N
    if eta == 2:
        for i in range(N//8):  # 32
            t = int.from_bytes(bytes_in[4*i:4*i+4], 'little')
            for j in range(8):
                bits = (t >> (4*j)) & 0xF
                a = (bits & 1) + ((bits >> 1) & 1)
                b = ((bits >> 2) & 1) + ((bits >> 3) & 1)
                f[8*i + j] = a - b
                f[8*i + j] = (f[8*i + j] % Q + Q) % Q
    return f

def prf(eta: int, key: bytes, nonce: int) -> bytes:
    """Pseudorandom function for sampling"""
    input_bytes = key + bytes([nonce])
    shake = hashlib.shake_256()
    shake.update(input_bytes)
    return shake.digest(64 * eta)  # For eta=2, 128 bytes

def sample_matrix(key: bytes, transpose: bool) -> List[List[List[int]]]:
    """Sample the matrix A"""
    A = [[[0] * N for _ in range(K)] for _ in range(K)]
    for i in range(K):
        for j in range(K):
            if transpose:
                shake = hashlib.shake_128()
                shake.update(key + bytes([i, j]))
            else:
                shake = hashlib.shake_128()
                shake.update(key + bytes([j, i]))
            buf = shake.digest(672)
            coeffs = []
            pos = 0
            while len(coeffs) < N and pos < 670:
                val = (buf[pos] | (buf[pos+1] << 8)) & 0xFFF
                if val < Q:
                    coeffs.append(val)
                pos += 2
            A[i][j] = coeffs[:N]
    return A

def poly_from_bytes(bytes_in: bytes) -> List[int]:
    """Decode polynomial from bytes (for decompression)"""
    f = [0] * N
    if len(bytes_in) == 320:  # du=10, 320 bytes
        for i in range(N//4):
            t0 = bytes_in[5*i] + 256 * (bytes_in[5*i + 1] % 4)
            t1 = (bytes_in[5*i + 1] >> 2) + 64 * (bytes_in[5*i + 2] % 16)
            t2 = (bytes_in[5*i + 2] >> 4) + 16 * bytes_in[5*i + 3]
            t3 = bytes_in[5*i + 4]
            f[4*i] = ((t0 * Q + (1 << 9)) >> 10) % Q
            f[4*i + 1] = ((t1 * Q + (1 << 9)) >> 10) % Q
            f[4*i + 2] = ((t2 * Q + (1 << 9)) >> 10) % Q
            f[4*i + 3] = ((t3 * Q + (1 << 9)) >> 10) % Q
    elif len(bytes_in) == 128:  # dv=4, 128 bytes
        for i in range(N//2):
            byte = bytes_in[i]
            f[2*i] = ((byte % 16) * Q + (1 << 3)) >> 4
            f[2*i + 1] = ((byte >> 4) * Q + (1 << 3)) >> 4
    elif len(bytes_in) == 384:  # dt=12, 384 bytes
        for i in range(N//2):
            t0 = bytes_in[3*i] + 256 * (bytes_in[3*i + 1] % 16)
            t1 = (bytes_in[3*i + 1] >> 4) + 16 * bytes_in[3*i + 2]
            f[2*i] = ((t0 * Q + (1 << 11)) >> 12) % Q
            f[2*i + 1] = ((t1 * Q + (1 << 11)) >> 12) % Q
    return f

def poly_to_bytes(f: List[int], d: int) -> bytes:
    """Encode polynomial to bytes with compression d bits"""
    f = [x % Q for x in f]  # Ensure coefficients are in 0 to Q-1
    bytes_out = bytearray()
    if d == 10:
        for i in range(N//4):
            t0 = ((f[4*i] * (1 << d)) + (Q >> 1)) // Q
            t1 = ((f[4*i + 1] * (1 << d)) + (Q >> 1)) // Q
            t2 = ((f[4*i + 2] * (1 << d)) + (Q >> 1)) // Q
            t3 = ((f[4*i + 3] * (1 << d)) + (Q >> 1)) // Q
            t0 %= (1 << d)
            t1 %= (1 << d)
            t2 %= (1 << d)
            t3 %= (1 << d)
            bytes_out.append(t0 & 0xFF)
            bytes_out.append(((t0 >> 8) & 0x03) | ((t1 & 0x3F) << 2))
            bytes_out.append((t1 >> 6) | ((t2 & 0x0F) << 4))
            bytes_out.append((t2 >> 4) | ((t3 & 0x03) << 6))
            bytes_out.append(t3 >> 2)
    elif d == 4:
        for i in range(N//2):
            t0 = ((f[2*i] * 16) + (Q >> 1)) // Q
            t1 = ((f[2*i + 1] * 16) + (Q >> 1)) // Q
            t0 %= 16
            t1 %= 16
            bytes_out.append(t0 | (t1 << 4))
    elif d == 12:
        for i in range(N//2):
            t0 = ((f[2*i] * 4096) + (Q >> 1)) // Q
            t1 = ((f[2*i + 1] * 4096) + (Q >> 1)) // Q
            t0 %= 4096
            t1 %= 4096
            bytes_out.append(t0 & 0xFF)
            bytes_out.append(((t0 >> 8) & 0x0F) | ((t1 & 0x0F) << 4))
            bytes_out.append(t1 >> 4)
    return bytes(bytes_out)

def poly_from_msg(msg: bytes) -> List[int]:
    """Convert message to polynomial"""
    f = [0] * N
    for i in range(32):  # 256 bits
        for j in range(8):
            f[8*i + j] = (msg[i] >> j) & 1
    return f

def poly_to_msg(f: List[int]) -> bytes:
    """Convert polynomial to message"""
    msg = bytearray(32)
    for i in range(32):
        for j in range(8):
            msg[i] |= ((f[8*i + j] & 1) << j)
    return bytes(msg)

# K-PKE

def k_pke_keygen() -> Tuple[bytes, bytes]:
    """K-PKE Key Generation"""
    d = secrets.token_bytes(32)
    rho, sigma = d[:32], d[32:]
    A = sample_matrix(rho, False)
    A_hat = [[ntt(A[i][j]) for j in range(K)] for i in range(K)]
    s = []
    e = []
    for i in range(K):
        prf_out = prf(ETA1, sigma, i)
        s.append(cbd(ETA1, prf_out))
        prf_out = prf(ETA1, sigma, i + K)
        e.append(cbd(ETA1, prf_out))
    s_hat = [ntt(s[i]) for i in range(K)]
    t = []
    for i in range(K):
        sum_poly = [0] * N
        for j in range(K):
            prod = poly_mul_ntt(A_hat[i][j], s_hat[j])
            sum_poly = poly_add(sum_poly, prod)
        t.append(poly_add(sum_poly, e[i]))
    # Encode pk
    pk = rho
    for i in range(K):
        pk += poly_to_bytes(t[i], DT)
    # Encode sk
    sk = b''
    for i in range(K):
        sk += poly_to_bytes(s[i], DT)
    return pk, sk

def k_pke_encrypt(pk: bytes, m: bytes, r: bytes) -> bytes:
    """K-PKE Encryption"""
    rho = pk[:32]
    t_start = 32
    t = []
    for i in range(K):
        t_bytes = pk[t_start : t_start + 384]
        t.append(poly_from_bytes(t_bytes))
        t_start += 384
    A = sample_matrix(rho, True)
    A_hat = [[ntt(A[i][j]) for j in range(K)] for i in range(K)]
    y = []
    for i in range(K):
        prf_out = prf(ETA1, r, i)
        y.append(cbd(ETA1, prf_out))
    y_hat = [ntt(y[i]) for i in range(K)]
    e1 = []
    for i in range(K):
        prf_out = prf(ETA1, r, i + K)
        e1.append(cbd(ETA1, prf_out))
    prf_out = prf(ETA1, r, 2 * K)
    e2 = cbd(ETA1, prf_out)
    u = []
    for i in range(K):
        sum_poly = [0] * N
        for j in range(K):
            prod = poly_mul_ntt(A_hat[i][j], y_hat[j])
            sum_poly = poly_add(sum_poly, prod)
        u.append(poly_add(sum_poly, e1[i]))
    mu = poly_from_msg(m)
    v = [0] * N
    t_hat = [ntt(t[i]) for i in range(K)]
    for i in range(K):
        prod = poly_mul_ntt(t_hat[i], y_hat[i])
        v = poly_add(v, prod)
    v = poly_add(v, e2)
    v = poly_add(v, mu)
    # Encode ct
    ct = b''
    for i in range(K):
        ct += poly_to_bytes(u[i], DU)
    ct += poly_to_bytes(v, DV)
    return ct

def k_pke_decrypt(sk: bytes, ct: bytes) -> bytes:
    """K-PKE Decryption"""
    s = []
    sk_start = 0
    for i in range(K):
        s_bytes = sk[sk_start : sk_start + 384]
        s.append(poly_from_bytes(s_bytes))
        sk_start += 384
    u = []
    ct_start = 0
    for i in range(K):
        u_bytes = ct[ct_start : ct_start + 320]
        u.append(poly_from_bytes(u_bytes))
        ct_start += 320
    v_bytes = ct[ct_start : ct_start + 128]
    v = poly_from_bytes(v_bytes)
    sum_poly = [0] * N
    for i in range(K):
        prod = poly_mul(s[i], u[i])
        sum_poly = poly_add(sum_poly, prod)
    w = poly_sub(v, sum_poly)
    m = poly_to_msg(w)
    return m

# ML-KEM

def ml_kem_keygen() -> Tuple[bytes, bytes]:
    """ML-KEM Key Generation"""
    z = secrets.token_bytes(32)
    pk, sk_pke = k_pke_keygen()
    sk = sk_pke + pk + hashlib.sha3_256(pk).digest() + z
    return pk, sk

def ml_kem_encapsulate(pk: bytes) -> Tuple[bytes, bytes]:
    """ML-KEM Encapsulation"""
    m = secrets.token_bytes(32)
    h = hashlib.sha3_256(pk).digest()
    kr = hashlib.shake_256(m + h).digest(64)
    k, r = kr[:32], kr[32:]
    c = k_pke_encrypt(pk, m, r)
    return c, k

def ml_kem_decapsulate(sk: bytes, c: bytes) -> bytes:
    """ML-KEM Decapsulation"""
    sk_pke_len = K * 384  # 3*384=1152
    pk_len = 32 + K * 384  # rho + t's = 32 + 3*384 = 1184
    h_len = 32
    z_len = 32
    sk_pke = sk[:sk_pke_len]
    pk = sk[sk_pke_len:sk_pke_len + pk_len]
    h = sk[sk_pke_len + pk_len:sk_pke_len + pk_len + h_len]
    z = sk[sk_pke_len + pk_len + h_len:]
    m_prime = k_pke_decrypt(sk_pke, c)
    h_prime = hashlib.sha3_256(pk).digest()
    if h != h_prime:
        raise ValueError("Hash mismatch")
    kr_prime = hashlib.shake_256(m_prime + h).digest(64)
    k_prime, r_prime = kr_prime[:32], kr_prime[32:]
    c_prime = k_pke_encrypt(pk, m_prime, r_prime)
    if c != c_prime:
        k_prime = hashlib.shake_256(z + c).digest(32)
    return k_prime

# Main demonstration

if __name__ == "__main__":
    print("ML-KEM-768 Key Encapsulation Demonstration")
    print("Generating keys...")
    pk, sk = ml_kem_keygen()
    print(f"Public key length: {len(pk)} bytes")
    print(f"Secret key length: {len(sk)} bytes")
    print("Encapsulating...")
    c, k = ml_kem_encapsulate(pk)
    print(f"Ciphertext length: {len(c)} bytes")
    print(f"Shared secret length: {len(k)} bytes")
    print("Decapsulating...")
    k_recovered = ml_kem_decapsulate(sk, c)
    print(f"Shared secret matches: {k == k_recovered}")
    print("Success!")