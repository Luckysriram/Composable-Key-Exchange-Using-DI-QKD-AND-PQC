"""
CRYSTALS-Kyber — Tkinter GUI
==============================
A graphical interface for exploring the Kyber KEM.
Provides buttons for key generation, encapsulation, and decapsulation
with a scrollable output pane showing intermediate results.
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import threading
import time
import sys
import io

from params import KYBER512, KYBER768, KYBER1024, KyberParams
from kyber import kem_keygen, kem_encaps, kem_decaps


class KyberGUI:
    """Tkinter-based GUI for CRYSTALS-Kyber KEM."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CRYSTALS-Kyber — Post-Quantum KEM")
        self.root.geometry("900x700")
        self.root.configure(bg="#1a1a2e")

        # State
        self.params: KyberParams | None = None
        self.pk: bytes | None = None
        self.sk: bytes | None = None
        self.ct: bytes | None = None
        self.ss_enc: bytes | None = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Consolas", 16, "bold"),
                         foreground="#e94560", background="#1a1a2e")
        style.configure("Info.TLabel", font=("Consolas", 10),
                         foreground="#a0a0c0", background="#1a1a2e")
        style.configure("Action.TButton", font=("Consolas", 11, "bold"),
                         padding=10)

        # ── Title ──
        ttk.Label(self.root, text="🔐 CRYSTALS-Kyber KEM",
                  style="Title.TLabel").pack(pady=(15, 5))
        ttk.Label(self.root, text="Post-Quantum Key Encapsulation Mechanism",
                  style="Info.TLabel").pack(pady=(0, 10))

        # ── Parameter selection ──
        param_frame = tk.Frame(self.root, bg="#1a1a2e")
        param_frame.pack(pady=5)
        ttk.Label(param_frame, text="Parameter Set:",
                  style="Info.TLabel").pack(side=tk.LEFT, padx=5)
        self.param_var = tk.StringVar(value="Kyber768")
        for name in ["Kyber512", "Kyber768", "Kyber1024"]:
            ttk.Radiobutton(param_frame, text=name, variable=self.param_var,
                            value=name).pack(side=tk.LEFT, padx=5)

        # ── Buttons ──
        btn_frame = tk.Frame(self.root, bg="#1a1a2e")
        btn_frame.pack(pady=10)

        self.btn_keygen = ttk.Button(btn_frame, text="🔑 Generate Keys",
                                      command=self._on_keygen, style="Action.TButton")
        self.btn_keygen.pack(side=tk.LEFT, padx=8)

        self.btn_encaps = ttk.Button(btn_frame, text="📦 Encapsulate",
                                      command=self._on_encaps, style="Action.TButton",
                                      state=tk.DISABLED)
        self.btn_encaps.pack(side=tk.LEFT, padx=8)

        self.btn_decaps = ttk.Button(btn_frame, text="🔓 Decapsulate",
                                      command=self._on_decaps, style="Action.TButton",
                                      state=tk.DISABLED)
        self.btn_decaps.pack(side=tk.LEFT, padx=8)

        self.btn_demo = ttk.Button(btn_frame, text="🚀 Full Demo",
                                    command=self._on_full_demo, style="Action.TButton")
        self.btn_demo.pack(side=tk.LEFT, padx=8)

        # ── Output area ──
        self.output = scrolledtext.ScrolledText(
            self.root, wrap=tk.WORD, font=("Consolas", 10),
            bg="#0f0f23", fg="#c0c0e0", insertbackground="#e94560",
            width=100, height=30
        )
        self.output.pack(padx=15, pady=10, fill=tk.BOTH, expand=True)

        # ── Status bar ──
        self.status_var = tk.StringVar(value="Ready. Select a parameter set and generate keys.")
        ttk.Label(self.root, textvariable=self.status_var,
                  style="Info.TLabel").pack(pady=(0, 10))

    def _log(self, text: str):
        """Append text to the output area."""
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def _clear(self):
        self.output.delete("1.0", tk.END)

    def _get_params(self) -> KyberParams:
        name = self.param_var.get()
        return {"Kyber512": KYBER512, "Kyber768": KYBER768,
                "Kyber1024": KYBER1024}[name]

    def _run_async(self, func):
        """Run a function in a background thread to keep UI responsive."""
        thread = threading.Thread(target=func, daemon=True)
        thread.start()

    # ── Actions ──

    def _on_keygen(self):
        self._run_async(self._do_keygen)

    def _do_keygen(self):
        self.params = self._get_params()
        self._log(f"{'═' * 60}")
        self._log(f"  KEY GENERATION — {self.params.name}")
        self._log(f"{'═' * 60}")
        self.status_var.set("Generating keys...")

        t0 = time.perf_counter()
        self.pk, self.sk = kem_keygen(self.params)
        elapsed = time.perf_counter() - t0

        self._log(f"  Public key:  {self.pk[:32].hex()}...")
        self._log(f"               ({len(self.pk)} bytes)")
        self._log(f"  Secret key:  {self.sk[:32].hex()}...")
        self._log(f"               ({len(self.sk)} bytes)")
        self._log(f"  Time:        {elapsed*1000:.2f} ms")
        self._log(f"  ✔ Keys generated successfully.\n")

        self.status_var.set("Keys generated. Ready to encapsulate.")
        self.btn_encaps.configure(state=tk.NORMAL)

    def _on_encaps(self):
        self._run_async(self._do_encaps)

    def _do_encaps(self):
        self._log(f"{'═' * 60}")
        self._log(f"  ENCAPSULATION — {self.params.name}")
        self._log(f"{'═' * 60}")
        self.status_var.set("Encapsulating...")

        t0 = time.perf_counter()
        self.ct, self.ss_enc = kem_encaps(self.params, self.pk)
        elapsed = time.perf_counter() - t0

        self._log(f"  Ciphertext:     {self.ct[:32].hex()}...")
        self._log(f"                  ({len(self.ct)} bytes)")
        self._log(f"  Shared secret:  {self.ss_enc.hex()}")
        self._log(f"                  ({len(self.ss_enc)} bytes)")
        self._log(f"  Time:           {elapsed*1000:.2f} ms")
        self._log(f"  ✔ Encapsulation complete.\n")

        self.status_var.set("Ciphertext created. Ready to decapsulate.")
        self.btn_decaps.configure(state=tk.NORMAL)

    def _on_decaps(self):
        self._run_async(self._do_decaps)

    def _do_decaps(self):
        self._log(f"{'═' * 60}")
        self._log(f"  DECAPSULATION — {self.params.name}")
        self._log(f"{'═' * 60}")
        self.status_var.set("Decapsulating...")

        t0 = time.perf_counter()
        ss_dec = kem_decaps(self.params, self.sk, self.ct)
        elapsed = time.perf_counter() - t0

        self._log(f"  Shared secret (encaps): {self.ss_enc.hex()}")
        self._log(f"  Shared secret (decaps): {ss_dec.hex()}")
        self._log(f"  Time: {elapsed*1000:.2f} ms")

        if ss_dec == self.ss_enc:
            self._log(f"  ✔ MATCH — Shared secrets are identical!")
        else:
            self._log(f"  ✘ MISMATCH — Something went wrong!")

        self._log("")
        self.status_var.set("Done. Secrets match!" if ss_dec == self.ss_enc else "Error: mismatch!")

    def _on_full_demo(self):
        self._run_async(self._do_full_demo)

    def _do_full_demo(self):
        self._clear()
        self._do_keygen()
        self._do_encaps()
        self._do_decaps()

        # Implicit rejection test
        self._log(f"{'═' * 60}")
        self._log(f"  IMPLICIT REJECTION TEST")
        self._log(f"{'═' * 60}")
        tampered = bytearray(self.ct)
        tampered[0] ^= 0xFF
        ss_bad = kem_decaps(self.params, self.sk, bytes(tampered))
        self._log(f"  Tampered secret:  {ss_bad.hex()}")
        self._log(f"  Original secret:  {self.ss_enc.hex()}")
        if ss_bad != self.ss_enc:
            self._log(f"  ✔ Tampered ciphertext correctly rejected.\n")
        else:
            self._log(f"  ✘ WARNING: rejection may have failed!\n")
        self.status_var.set("Full demo complete.")


def main():
    root = tk.Tk()
    app = KyberGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
