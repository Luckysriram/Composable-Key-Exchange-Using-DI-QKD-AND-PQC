# ML-KEM-768 Implementation
# Post-Quantum Key Encapsulation Mechanism
# Based on FIPS 203

import os
import hashlib
import math

# Parameters for ML-KEM-768
n = 256
k = 3
q = 3329
eta1 = 2
eta2 = 2
du = 10
dv = 4

# Primitive root for NTT
zeta = 17

# Modular inverse
def mod_inverse(a, m):
    m0 = m
    y = 0
    x = 1
    if m == 1:
        return 0
    while a > 1:
        q = a // m
        t = m
        m = a % m
        a = t
        t = y
        y = x - q * y
        x = t
    if x < 0:
        x += m0
    return x

# Bit reversal
def bitrev(i, bits=8):
    return int(''.join(reversed(format(i, f'0{bits}b'))), 2)

# NTT constants
ntt_zetas = [pow(zeta, i, q) for i in range(255)]
ntt_zetas_inv = [pow(zeta, -i, q) for i in range(255)]

# NTT
def ntt(f):
    f_hat = f[:]
    zeta_idx = 0
    len_ = 2
    while len_ <= n:
        for start in range(0, n, len_):
            zeta = ntt_zetas[zeta_idx]
            zeta_idx += 1
            for j in range(len_ // 2):
                t = (zeta * f_hat[start + j + len_ // 2]) % q
                f_hat[start + j + len_ // 2] = (f_hat[start + j] - t) % q
                f_hat[start + j] = (f_hat[start + j] + t) % q
        len_ *= 2
    return f_hat

# Inverse NTT
def intt(f_hat):
    f = f_hat[:]
    zeta_idx = 0
    len_ = n
    while len_ >= 2:
        for start in range(0, n, len_):
            zeta = ntt_zetas_inv[zeta_idx]
            zeta_idx += 1
            for j in range(len_ // 2):
                t = f[start + j]
                f[start + j] = (t + f[start + j + len_ // 2]) % q
                f[start + j + len_ // 2] = (t - f[start + j + len_ // 2]) % q
                f[start + j + len_ // 2] = (f[start + j + len_ // 2] * zeta) % q
        len_ //= 2
    inv_n = mod_inverse(n, q)
    for i in range(n):
        f[i] = (f[i] * inv_n) % q
    return f

# Polynomial class
class Poly:
    def __init__(self, coeffs=None):
        if coeffs is None:
            self.coeffs = [0] * n
        else:
            self.coeffs = coeffs[:n] if len(coeffs) >= n else coeffs + [0] * (n - len(coeffs))

    def __add__(self, other):
        return Poly([(a + b) % q for a, b in zip(self.coeffs, other.coeffs)])

    def __sub__(self, other):
        return Poly([(a - b) % q for a, b in zip(self.coeffs, other.coeffs)])

    def __mul__(self, other):
        if isinstance(other, int):
            return Poly([(a * other) % q for a in self.coeffs])
        # Polynomial multiplication using NTT
        a_hat = ntt(self.coeffs)
        b_hat = ntt(other.coeffs)
        c_hat = [(a * b) % q for a, b in zip(a_hat, b_hat)]
        c = intt(c_hat)
        return Poly(c)

# Sampling functions
def prf(s, nonce, eta):
    input = s + bytes([nonce])
    shake = hashlib.shake_128()
    shake.update(input)
    return shake.digest(256 * 2 * eta)  # Adjusted for n=256, eta=2

def sample_eta(eta, seed, nonce):
    buf = prf(seed, nonce, eta)
    r = [0] * n
    for i in range(n):
        d = 0
        t = 0
        for j in range(eta):
            byte_idx = 2 * i * eta + 2 * j
            a = buf[byte_idx]
            d += (a >> 0) & 1
            d += (a >> 1) & 1
            t += (a >> 4) & 1
            t += (a >> 5) & 1
        r[i] = d - t
    return Poly(r)

def sample_uniform(seed, i, j, l):
    input = seed + bytes([i, j, l])
    shake = hashlib.shake_128()
    shake.update(input)
    buf = shake.digest(2)  # 2 bytes
    val = int.from_bytes(buf, 'little') % q
    return val

# Matrix A
def gen_matrix(d):
    A = []
    for i in range(k):
        row = []
        for j in range(k):
            input_ = d + bytes([i, j])
            shake = hashlib.shake_128()
            shake.update(input_)
            buf = shake.digest(768)  # 3 bytes per coeff
            coeffs = []
            for l in range(n):
                val = int.from_bytes(buf[3*l:3*l+3], 'little') % q
                coeffs.append(val)
            row.append(Poly(coeffs))
        A.append(row)
    return A

# Encoding/Decoding
def encode_du(f):
    bytes_out = bytearray(320)
    for i in range(n):
        val = ((f.coeffs[i] * (1 << du)) + (q // 2)) // q
        for b in range(du):
            if (val >> b) & 1:
                byte_idx = (i * du + b) // 8
                bit_idx = (i * du + b) % 8
                bytes_out[byte_idx] |= (1 << bit_idx)
    return bytes(bytes_out)

def decode_du(bytes_in):
    f = [0] * n
    for i in range(n):
        val = 0
        for b in range(du):
            byte_idx = (i * du + b) // 8
            bit_idx = (i * du + b) % 8
            if bytes_in[byte_idx] & (1 << bit_idx):
                val |= (1 << b)
        f[i] = ((val * q) + (1 << (du - 1))) >> du
    return Poly(f)

def encode_dv(f):
    bytes_out = bytearray(128)
    for i in range(n):
        val = ((f.coeffs[i] * (1 << dv)) + (q // 2)) // q
        for b in range(dv):
            if (val >> b) & 1:
                byte_idx = (i * dv + b) // 8
                bit_idx = (i * dv + b) % 8
                bytes_out[byte_idx] |= (1 << bit_idx)
    return bytes(bytes_out)

def decode_dv(bytes_in):
    f = [0] * n
    for i in range(n):
        val = 0
        for b in range(dv):
            byte_idx = (i * dv + b) // 8
            bit_idx = (i * dv + b) % 8
            if bytes_in[byte_idx] & (1 << bit_idx):
                val |= (1 << b)
        f[i] = ((val * q) + (1 << (dv - 1))) >> dv
    return Poly(f)

# K-PKE
def kpke_keygen(d):
    A = gen_matrix(d)
    s = [sample_eta(eta1, d, nonce) for nonce in range(k)]
    e = [sample_eta(eta2, d, nonce) for nonce in range(k, 2*k)]
    t = []
    for i in range(k):
        ti = e[i]
        for j in range(k):
            ti = ti + A[i][j] * s[j]
        t.append(ti)
    pk = encode_du(t[0]) + encode_du(t[1]) + encode_du(t[2]) + d
    return pk, s

def kpke_encrypt(pk, m, r):
    t_bytes = pk[:320*3]
    d = pk[320*3:]
    t = [decode_du(t_bytes[i*320:(i+1)*320]) for i in range(k)]
    A = gen_matrix(d)
    r_poly = [sample_eta(eta1, r, nonce) for nonce in range(k)]
    e1 = sample_eta(eta2, r, k)
    e2 = sample_eta(eta2, r, k+1)
    u = []
    for i in range(k):
        ui = e1
        for j in range(k):
            ui = ui + A[j][i] * r_poly[j]
        u.append(ui)
    v = e2
    for i in range(k):
        v = v + t[i] * r_poly[i]
    m_poly = Poly([((m[i//8] >> (i%8)) & 1) * (q//2) for i in range(256)])
    v = v + m_poly
    c = b''
    for ui in u:
        c += encode_du(ui)
    c += encode_dv(v)
    return c

def kpke_decrypt(sk, c):
    u_bytes = c[:320*k]
    v_bytes = c[320*k:]
    u = [decode_du(u_bytes[i*320:(i+1)*320]) for i in range(k)]
    v = decode_dv(v_bytes)
    v_minus = v
    for i in range(k):
        v_minus = v_minus - sk[i] * u[i]
    m_bits = [1 if coeff > q//4 else 0 for coeff in v_minus.coeffs]
    m = bytearray(32)
    for i in range(256):
        if m_bits[i]:
            m[i//8] |= (1 << (i%8))
    return bytes(m)

# ML-KEM
def ml_kem_keygen():
    d = os.urandom(32)
    z = os.urandom(32)
    pk, sk = kpke_keygen(d)
    h = hashlib.sha3_256(pk).digest()
    ek = pk
    dk = (sk, pk, h, z)
    return ek, dk

def ml_kem_encapsulate(ek):
    m = os.urandom(32)
    r = os.urandom(32)
    c = kpke_encrypt(ek, m, r)
    h_c = hashlib.sha3_256(c).digest()
    K = hashlib.sha3_256(m + h_c).digest()
    return K, c

def ml_kem_decapsulate(dk, c):
    sk, pk, h, z = dk
    h_pk = hashlib.sha3_256(pk).digest()
    if h_pk != h:
        h_c = hashlib.sha3_256(c).digest()
        K = hashlib.sha3_256(z + h_c).digest()
        return K
    m = kpke_decrypt(sk, c)
    h_c = hashlib.sha3_256(c).digest()
    K = hashlib.sha3_256(m + h_c).digest()
    return K

if __name__ == "__main__":
    ek, dk = ml_kem_keygen()
    K1, c = ml_kem_encapsulate(ek)
    K2 = ml_kem_decapsulate(dk, c)
    print("Keys match:", K1 == K2)