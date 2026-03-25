# CRYSTALS-Kyber — Post-Quantum Key Encapsulation Mechanism

A complete, from-scratch Python implementation of **CRYSTALS-Kyber** (ML-KEM, FIPS 203),
the post-quantum cryptographic algorithm selected by NIST for standardization.

Supports all three parameter sets: **Kyber512**, **Kyber768**, **Kyber1024**.

---

## 📁 Project Structure

```
├── params.py       # Parameter sets & NTT constants
├── poly.py         # Polynomial & PolyVec arithmetic
├── ntt.py          # Number Theoretic Transform
├── sampling.py     # CBD noise & uniform rejection sampling
├── compress.py     # Lossy compression / decompression
├── utils.py        # SHA3/SHAKE wrappers, byte encoding
├── kyber.py        # IND-CPA PKE + IND-CCA KEM (core scheme)
├── main.py         # CLI demo with menus & ASCII flow diagrams
├── gui.py          # Tkinter GUI
├── test_kyber.py   # Comprehensive test suite
└── README.md       # This file
```

---

## 🚀 Quick Start

```bash
# Run the interactive CLI demo
python main.py

# Run the Tkinter GUI
python gui.py

# Run tests
python test_kyber.py
```

**No external dependencies** — only Python 3.10+ standard library (`hashlib`, `os`, `tkinter`).

---

## 🔐 What is CRYSTALS-Kyber?

CRYSTALS-Kyber is a **Key Encapsulation Mechanism (KEM)** — it lets two parties establish
a shared secret key over an insecure channel. Unlike RSA or ECDH, Kyber's security is
based on **lattice problems** that are believed to resist attacks from quantum computers.

### How It Works (Simplified)

```
Alice                                          Bob
──────                                         ────
                                         1. (pk, sk) ← KeyGen()
                         ← pk ──────────────┘
2. (ct, K) ← Encaps(pk)
        └── ct ──────────────→  3. K ← Decaps(sk, ct)

Both now share secret K → use for AES, ChaCha20, etc.
```

### The Three Operations

| Operation      | What it does                                    |
|----------------|-------------------------------------------------|
| **KeyGen**     | Generate a public/secret key pair               |
| **Encaps**     | Create a ciphertext + shared secret from the PK |
| **Decaps**     | Recover the shared secret from CT + SK          |

---

## 🧮 Core Components Explained

### 1. Polynomial Ring  ℤ_q[x]/(x^n + 1)

All arithmetic happens in the ring of polynomials modulo x²⁵⁶ + 1, with coefficients
modulo q = 3329. This quotient ring has special algebraic properties that enable the NTT.

### 2. Number Theoretic Transform (NTT)

The NTT is the "finite field FFT" — it converts polynomial multiplication from O(n²) to
O(n log n). Since q = 3329 is prime and q ≡ 1 (mod 256), we can find a primitive 256-th
root of unity (ζ = 17) and use Cooley-Tukey / Gentleman-Sande butterflies.

### 3. Centered Binomial Distribution (CBD)

Noise polynomials are sampled from CBD_η, where each coefficient is the difference of
two sums of η random bits. This produces small coefficients in {-η, ..., η} and is
much faster than discrete Gaussian sampling.

### 4. Compression / Decompression

To reduce ciphertext size, coefficients are rounded from q bits to d bits:
- `Compress_d(x) = ⌈(2^d / q) · x⌋ mod 2^d`
- `Decompress_d(x) = ⌈(q / 2^d) · x⌋`

This is lossy but the error is bounded and absorbed by the scheme's error tolerance.

### 5. Hash Functions

| Kyber Name | Underlying Primitive | Purpose                     |
|------------|---------------------|-----------------------------|
| H          | SHA3-256            | Hashing public keys         |
| G          | SHA3-512            | Deriving (key, coins) pairs |
| PRF        | SHAKE-256           | Deterministic noise generation |
| XOF        | SHAKE-128           | Matrix sampling stream      |
| KDF        | SHAKE-256           | Final shared secret derivation |

---

## 🛡️ Security

### Lattice-Based Cryptography

Kyber's security relies on the **Module Learning With Errors (MLWE)** problem:

> Given a random matrix **A** and a vector **t = A·s + e** (where s and e have
> small coefficients), find **s**.

This is a generalization of the LWE problem to modules over polynomial rings.
The best known algorithms (both classical and quantum) require exponential time.

### Why Quantum-Resistant?

- **Shor's algorithm** breaks RSA/ECC by efficiently solving factoring/discrete-log
- **No known quantum algorithm** efficiently solves lattice problems (LWE, MLWE)
- Grover's algorithm gives only a quadratic speedup for searching, which Kyber's
  parameter sizes already account for

### Fujisaki-Okamoto Transform (FO)

Kyber internally uses an IND-CPA-secure public-key encryption scheme. The FO
transform upgrades it to **IND-CCA security** (resistance to chosen-ciphertext
attacks) by:

1. **Re-encrypting** the decrypted message and comparing to the received ciphertext
2. **Implicit rejection** — returning a pseudorandom value (not an error) on failure,
   preventing timing side-channel attacks

---

## 📊 Parameter Sets

| Parameter  | k | η₁ | η₂ | dᵤ | dᵥ | PK (B) | SK (B) | CT (B) | Security |
|------------|---|----|----|----|----|--------|--------|--------|----------|
| Kyber512   | 2 | 3  | 2  | 10 | 4  | 800    | 1632   | 768    | AES-128  |
| Kyber768   | 3 | 2  | 2  | 10 | 4  | 1184   | 2400   | 1088   | AES-192  |
| Kyber1024  | 4 | 2  | 2  | 11 | 5  | 1568   | 3168   | 1568   | AES-256  |

---

## 🔄 Kyber vs RSA

| Feature               | RSA-2048         | Kyber768           |
|-----------------------|------------------|--------------------|
| **Quantum-safe**      | ❌ No            | ✅ Yes              |
| **Key size (PK)**     | 256 bytes        | 1,184 bytes        |
| **Ciphertext size**   | 256 bytes        | 1,088 bytes        |
| **Security basis**    | Integer factoring | Module-LWE         |
| **Key generation**    | Slow (primes)    | Fast (sampling)    |
| **Enc/Dec speed**     | Moderate         | Fast               |
| **NIST status**       | Legacy           | FIPS 203 standard  |

RSA keys are smaller, but Kyber is dramatically faster and resistant to quantum attacks.
For new systems, Kyber (ML-KEM) is the recommended choice.

---

## 🌍 Real-World Usage

- **NIST FIPS 203** — Standardized as ML-KEM (August 2024)
- **Google Chrome** — Hybrid X25519+Kyber768 key exchange since 2023
- **Signal Protocol** — PQXDH uses Kyber768 for post-quantum key agreement
- **Cloudflare** — Post-quantum TLS support using Kyber
- **AWS KMS** — Hybrid post-quantum TLS support

---

## ⚠️ Disclaimer

This is an **educational implementation** for study and research. It has **not** been
audited for production use. For real-world PQC, use:
- [liboqs](https://github.com/open-quantum-safe/liboqs) (C)
- [pqcrypto](https://pypi.org/project/pqcrypto/) (Python bindings)
- [CIRCL](https://github.com/cloudflare/circl) (Go)

---

## 📜 License

MIT — Free for educational and research use.
