#!/usr/bin/env python3
"""
CuKEM Visual Demo Dashboard

Runs all CuKEM demos, captures step-by-step results, and generates
a beautiful interactive HTML dashboard that opens in the browser.
"""

import sys
import time
import json
import webbrowser
import tempfile
import os
from datetime import datetime
import logging
import traceback

# Suppress logging to console during visual demo
logging.disable(logging.CRITICAL)

# ============================================================
# Demo Execution Engine
# ============================================================

def run_demo_1_pqc():
    """Post-Quantum Key Exchange (ML-KEM-768)"""
    from pqc_kem import MLKEM768
    steps = []

    kem = MLKEM768()
    steps.append({"text": "Initialized ML-KEM-768 (Kyber768) engine", "status": "ok", "detail": "NIST FIPS 203 compliant"})

    keypair = kem.generate_keypair()
    steps.append({
        "text": "Generated responder keypair",
        "status": "ok",
        "detail": f"Public key: {len(keypair.public_key)} bytes | Secret key: {len(keypair.secret_key)} bytes"
    })

    encap = kem.encapsulate(keypair.public_key)
    if not encap.success:
        steps.append({"text": f"Encapsulation failed: {encap.error}", "status": "fail"})
        return steps, False
    steps.append({
        "text": "Initiator encapsulated shared secret",
        "status": "ok",
        "detail": f"Ciphertext: {len(encap.ciphertext)} bytes | Secret: {len(encap.shared_secret)} bytes | Fingerprint: {encap.get_key_fingerprint()}"
    })

    decap = kem.decapsulate(encap.ciphertext, keypair.secret_key)
    if not decap.success:
        steps.append({"text": f"Decapsulation failed: {decap.error}", "status": "fail"})
        return steps, False
    steps.append({"text": "Responder decapsulated shared secret", "status": "ok"})

    match = encap.shared_secret == decap.shared_secret
    steps.append({
        "text": "Shared secrets match" if match else "Shared secrets DO NOT match",
        "status": "ok" if match else "fail",
        "detail": "Key exchange verified - both parties hold identical 32-byte secret"
    })

    return steps, match


def run_demo_2_bb84():
    """BB84 Quantum Key Distribution"""
    from bb84_simulator import BB84Simulator
    steps = []

    bb84 = BB84Simulator(n_qubits=256, error_correction_threshold=0.11)
    steps.append({"text": "Initialized BB84 simulator", "status": "ok", "detail": "256 qubits, QBER threshold: 11%"})

    steps.append({"text": "Alice generates random bits and bases", "status": "info"})
    steps.append({"text": "Bob generates random measurement bases", "status": "info"})
    steps.append({"text": "Simulating quantum channel transmission...", "status": "info"})

    result = bb84.execute_protocol(noise_level=0.0)

    if result.success:
        steps.append({
            "text": "BB84 protocol completed successfully",
            "status": "ok",
            "detail": f"Generated key: {result.key_length} bytes | Sifted: {result.sifted_key_length} qubits | QBER: {result.qber:.4f} | Efficiency: {result.metadata['sifting_efficiency']:.2%}"
        })
    else:
        steps.append({"text": f"BB84 protocol failed: {result.error}", "status": "fail"})

    return steps, result.success


def run_demo_3_entropy():
    """Entropy Estimation (NIST 800-90B)"""
    from bb84_simulator import BB84Simulator
    from entropy_estimator import EntropyEstimator
    steps = []

    estimator = EntropyEstimator(min_entropy_per_bit=0.8)
    steps.append({"text": "Initialized NIST 800-90B entropy estimator", "status": "ok", "detail": "Min entropy threshold: 0.8 bits/symbol"})

    bb84 = BB84Simulator(n_qubits=2048)
    steps.append({"text": "Generating quantum key material (2048 qubits)...", "status": "info"})

    bb84_result = bb84.execute_protocol(noise_level=0.0)
    if not bb84_result.success or not bb84_result.raw_key:
        steps.append({"text": "Failed to generate quantum key material", "status": "fail"})
        return steps, False

    steps.append({"text": f"Generated {len(bb84_result.raw_key)} bytes of key material", "status": "ok"})

    entropy_result = estimator.estimate_entropy(bb84_result.raw_key)
    if entropy_result.success:
        steps.append({
            "text": "Entropy analysis complete",
            "status": "ok",
            "detail": (
                f"Min-entropy: {entropy_result.min_entropy:.4f} | "
                f"Shannon: {entropy_result.estimations.get('shannon_entropy', 0):.4f} | "
                f"MCV: {entropy_result.estimations.get('mcv_estimate', 0):.4f} | "
                f"Collision: {entropy_result.estimations.get('collision_estimate', 0):.4f}"
            )
        })
        steps.append({
            "text": f"Entropy sufficient: {'Yes' if entropy_result.sufficient else 'No'}",
            "status": "ok" if entropy_result.sufficient else "warn"
        })
        return steps, entropy_result.sufficient
    else:
        steps.append({"text": f"Entropy estimation failed: {entropy_result.error}", "status": "fail"})
        return steps, False


def run_demo_4_cukem():
    """CuKEM Hybrid Mode (PQC + QKD)"""
    from cukem import CuKEM, CuKEMConfig, CuKEMMode
    steps = []

    config = CuKEMConfig(
        mode=CuKEMMode.HYBRID,
        n_qubits=256,
        min_entropy=0.8,
        qber_threshold=0.11,
        chsh_verification=False,
        output_key_length=32
    )

    cukem = CuKEM(config)
    steps.append({"text": "Initialized CuKEM in HYBRID mode", "status": "ok", "detail": "PQC (ML-KEM-768) + QKD (BB84) combined"})

    keypair = cukem.generate_keypair()
    steps.append({"text": "Responder generated keypair", "status": "ok"})

    steps.append({"text": "Initiator performing hybrid key exchange...", "status": "info"})

    result = cukem.initiate_exchange(
        responder_public_key=keypair.public_key,
        noise_level=0.0
    )

    if result.success:
        steps.append({
            "text": "Hybrid key exchange successful",
            "status": "ok",
            "detail": f"Mode: {result.mode.value} | Key: {result.key_length} bytes | Fallback: {result.fallback_used}"
        })
        if result.warnings:
            for w in result.warnings:
                steps.append({"text": f"Warning: {w}", "status": "warn"})
    else:
        steps.append({"text": f"Key exchange failed: {result.error}", "status": "fail"})

    return steps, result.success


def run_demo_5_adaptive():
    """Adaptive Controller & Health Monitoring"""
    from adaptive_controller import AdaptiveController
    steps = []

    controller = AdaptiveController()
    steps.append({"text": "Initialized adaptive state machine controller", "status": "ok", "detail": "States: idle, initializing, hybrid_active, pqc_only, degraded, failed, recovering"})

    success = controller.initialize_system()
    if success:
        steps.append({"text": "System initialized -> hybrid_active state", "status": "ok"})
    else:
        steps.append({"text": "System initialization failed", "status": "fail"})
        return steps, False

    steps.append({"text": "Performing monitored key exchange...", "status": "info"})
    result = controller.perform_exchange(role="initiator", noise_level=0.0)

    if result.success:
        steps.append({"text": "Monitored key exchange successful", "status": "ok"})
    else:
        steps.append({"text": f"Exchange failed: {result.error}", "status": "fail"})

    health = controller.get_health_status()
    stats = controller.get_statistics()
    steps.append({
        "text": f"Health: {health.value} | State: {stats['state']}",
        "status": "ok" if health.value == "healthy" else "warn",
        "detail": f"Total exchanges: {stats['metrics']['total_exchanges']} | PQC success: {stats['metrics']['pqc_success_rate']:.0%} | Avg QBER: {stats['metrics']['avg_qber']:.4f}"
    })

    return steps, result.success


def run_demo_6_breaker():
    """Circuit Breaker Pattern"""
    from circuit_breaker import CircuitBreaker, CircuitBreakerConfig
    steps = []

    config = CircuitBreakerConfig(failure_threshold=3, success_threshold=2, timeout_seconds=5)
    breaker = CircuitBreaker(config, name="demo_breaker")
    steps.append({"text": "Initialized circuit breaker", "status": "ok", "detail": "Failure threshold: 3 | Recovery threshold: 2 | Timeout: 5s"})

    call_count = 0

    def unreliable_function():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise Exception("Simulated failure")
        return f"Success on call {call_count}"

    for attempt in range(1, 6):
        try:
            result = breaker.call(unreliable_function)
            steps.append({"text": f"Attempt {attempt}: {result}", "status": "ok"})
        except Exception as e:
            steps.append({"text": f"Attempt {attempt}: {str(e)}", "status": "fail"})

    stats = breaker.get_statistics()
    steps.append({
        "text": f"Circuit breaker state: {stats['state']}",
        "status": "ok",
        "detail": f"Total calls: {stats['stats']['total_calls']}"
    })

    return steps, True


# ============================================================
# HTML Dashboard Generator
# ============================================================

DEMO_LIST = [
    ("Layer 1: Post-Quantum KEM", "ML-KEM-768 (Kyber) key encapsulation for quantum-resistant key exchange", run_demo_1_pqc, "pqc"),
    ("Layer 2: Quantum Key Distribution", "BB84 protocol simulation for quantum-secured key generation", run_demo_2_bb84, "qkd"),
    ("Layer 3: Entropy Estimation", "NIST 800-90B entropy validation of quantum-derived key material", run_demo_3_entropy, "entropy"),
    ("Layer 5: CuKEM Hybrid Mode", "Combined PQC + QKD key exchange with automatic fallback", run_demo_4_cukem, "cukem"),
    ("Layer 7: Adaptive Controller", "State machine with health monitoring and automatic mode switching", run_demo_5_adaptive, "adaptive"),
    ("Layer 8: Circuit Breaker", "Fault tolerance pattern for resilient key exchange operations", run_demo_6_breaker, "breaker"),
]

LAYER_COLORS = {
    "pqc": "#6C63FF",
    "qkd": "#00D2FF",
    "entropy": "#FF6B9D",
    "cukem": "#FFD93D",
    "adaptive": "#4ECDC4",
    "breaker": "#FF8A5C",
}


def generate_html(results, total_time):
    """Generate the full HTML dashboard."""
    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    # Build demo cards HTML
    cards_html = ""
    for i, r in enumerate(results):
        color = LAYER_COLORS.get(r["tag"], "#6C63FF")
        status_class = "pass" if r["passed"] else "fail"
        status_label = "PASS" if r["passed"] else "FAIL"

        steps_html = ""
        for step in r["steps"]:
            s = step["status"]
            icon = {"ok": "&#10003;", "fail": "&#10007;", "warn": "&#9888;", "info": "&#8226;"}.get(s, "&#8226;")
            cls = {"ok": "step-ok", "fail": "step-fail", "warn": "step-warn", "info": "step-info"}.get(s, "step-info")
            detail_html = f'<div class="step-detail">{step["detail"]}</div>' if step.get("detail") else ""
            steps_html += f'<div class="step {cls}" style="animation-delay: {0.06 * (i + 1)}s"><span class="step-icon">{icon}</span><div class="step-content"><span class="step-text">{step["text"]}</span>{detail_html}</div></div>\n'

        cards_html += f'''
        <div class="demo-card" style="animation-delay: {0.1 * i}s">
            <div class="card-header" style="border-left: 4px solid {color}">
                <div class="card-title-row">
                    <span class="card-number" style="background: {color}">{i + 1}</span>
                    <div>
                        <h3 class="card-title">{r["name"]}</h3>
                        <p class="card-desc">{r["desc"]}</p>
                    </div>
                </div>
                <div class="card-meta">
                    <span class="badge badge-{status_class}">{status_label}</span>
                    <span class="card-time">{r["time"]:.3f}s</span>
                </div>
            </div>
            <div class="card-steps">
                {steps_html}
            </div>
        </div>
        '''

    # Summary stats
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CuKEM System Dashboard</title>
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {{
        --bg-primary: #0B0E17;
        --bg-secondary: #111827;
        --bg-card: #1A1F35;
        --bg-card-hover: #1E2440;
        --text-primary: #E8ECF4;
        --text-secondary: #8892A8;
        --text-muted: #5A6478;
        --border: #2A3050;
        --accent-blue: #6C63FF;
        --accent-cyan: #00D2FF;
        --accent-green: #34D399;
        --accent-red: #F87171;
        --accent-yellow: #FBBF24;
        --accent-orange: #FF8A5C;
        --glow-blue: rgba(108, 99, 255, 0.15);
    }}

    * {{ margin: 0; padding: 0; box-sizing: border-box; }}

    body {{
        font-family: 'Inter', -apple-system, sans-serif;
        background: var(--bg-primary);
        color: var(--text-primary);
        min-height: 100vh;
        overflow-x: hidden;
    }}

    /* Animated background */
    body::before {{
        content: '';
        position: fixed;
        top: 0; left: 0;
        width: 100%; height: 100%;
        background:
            radial-gradient(ellipse at 20% 20%, rgba(108, 99, 255, 0.08) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 80%, rgba(0, 210, 255, 0.06) 0%, transparent 50%),
            radial-gradient(ellipse at 50% 50%, rgba(255, 107, 157, 0.04) 0%, transparent 50%);
        z-index: 0;
        pointer-events: none;
    }}

    .container {{
        max-width: 1100px;
        margin: 0 auto;
        padding: 24px 20px 60px;
        position: relative;
        z-index: 1;
    }}

    /* Header */
    .header {{
        text-align: center;
        padding: 40px 20px 30px;
        margin-bottom: 30px;
    }}

    .header-badge {{
        display: inline-block;
        padding: 4px 14px;
        background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        color: #fff;
        margin-bottom: 16px;
    }}

    .header h1 {{
        font-size: 2.2em;
        font-weight: 700;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #E8ECF4 30%, var(--accent-cyan));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }}

    .header p {{
        color: var(--text-secondary);
        font-size: 1em;
        font-weight: 300;
    }}

    /* Stats bar */
    .stats-bar {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-bottom: 36px;
    }}

    .stat-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        animation: fadeInUp 0.5s ease forwards;
        opacity: 0;
    }}

    .stat-card:nth-child(1) {{ animation-delay: 0.05s; }}
    .stat-card:nth-child(2) {{ animation-delay: 0.1s; }}
    .stat-card:nth-child(3) {{ animation-delay: 0.15s; }}
    .stat-card:nth-child(4) {{ animation-delay: 0.2s; }}

    .stat-value {{
        font-size: 2em;
        font-weight: 700;
        font-family: 'JetBrains Mono', monospace;
    }}

    .stat-value.green {{ color: var(--accent-green); }}
    .stat-value.red {{ color: var(--accent-red); }}
    .stat-value.cyan {{ color: var(--accent-cyan); }}
    .stat-value.blue {{ color: var(--accent-blue); }}

    .stat-label {{
        font-size: 0.78em;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}

    /* Architecture Section */
    .arch-section {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 36px;
        animation: fadeInUp 0.5s ease 0.25s forwards;
        opacity: 0;
    }}

    .arch-section h2 {{
        font-size: 1.1em;
        font-weight: 600;
        margin-bottom: 20px;
        color: var(--text-secondary);
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.85em;
    }}

    .arch-flow {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0;
        flex-wrap: wrap;
        padding: 10px 0;
    }}

    .arch-node {{
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 6px;
        padding: 14px 16px;
        border-radius: 10px;
        border: 1px solid var(--border);
        background: var(--bg-secondary);
        min-width: 100px;
        transition: transform 0.2s, box-shadow 0.2s;
    }}

    .arch-node:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
    }}

    .arch-node-label {{
        font-size: 0.72em;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}

    .arch-node-sub {{
        font-size: 0.65em;
        color: var(--text-muted);
    }}

    .arch-arrow {{
        color: var(--text-muted);
        font-size: 1.2em;
        padding: 0 6px;
        animation: pulseArrow 2s ease-in-out infinite;
    }}

    @keyframes pulseArrow {{
        0%, 100% {{ opacity: 0.4; }}
        50% {{ opacity: 1; }}
    }}

    /* Demo cards */
    .demo-card {{
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 14px;
        margin-bottom: 20px;
        overflow: hidden;
        animation: fadeInUp 0.5s ease forwards;
        opacity: 0;
        transition: border-color 0.3s, box-shadow 0.3s;
    }}

    .demo-card:hover {{
        border-color: rgba(108, 99, 255, 0.3);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }}

    .card-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 20px 24px;
        border-bottom: 1px solid var(--border);
    }}

    .card-title-row {{
        display: flex;
        align-items: center;
        gap: 14px;
    }}

    .card-number {{
        display: flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 0.85em;
        color: #fff;
        flex-shrink: 0;
    }}

    .card-title {{
        font-size: 1.05em;
        font-weight: 600;
    }}

    .card-desc {{
        font-size: 0.8em;
        color: var(--text-secondary);
        margin-top: 2px;
    }}

    .card-meta {{
        display: flex;
        align-items: center;
        gap: 12px;
        flex-shrink: 0;
    }}

    .card-time {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8em;
        color: var(--text-muted);
    }}

    .badge {{
        display: inline-block;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.72em;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }}

    .badge-pass {{
        background: rgba(52, 211, 153, 0.15);
        color: var(--accent-green);
        border: 1px solid rgba(52, 211, 153, 0.3);
    }}

    .badge-fail {{
        background: rgba(248, 113, 113, 0.15);
        color: var(--accent-red);
        border: 1px solid rgba(248, 113, 113, 0.3);
    }}

    /* Steps */
    .card-steps {{
        padding: 16px 24px 20px;
    }}

    .step {{
        display: flex;
        align-items: flex-start;
        gap: 10px;
        padding: 8px 0;
        animation: fadeInLeft 0.3s ease forwards;
        opacity: 0;
    }}

    .step + .step {{
        border-top: 1px solid rgba(42, 48, 80, 0.5);
    }}

    .step-icon {{
        flex-shrink: 0;
        width: 22px;
        height: 22px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        font-size: 0.7em;
        font-weight: 700;
        margin-top: 1px;
    }}

    .step-ok .step-icon {{
        background: rgba(52, 211, 153, 0.15);
        color: var(--accent-green);
    }}

    .step-fail .step-icon {{
        background: rgba(248, 113, 113, 0.15);
        color: var(--accent-red);
    }}

    .step-warn .step-icon {{
        background: rgba(251, 191, 36, 0.15);
        color: var(--accent-yellow);
    }}

    .step-info .step-icon {{
        background: rgba(108, 99, 255, 0.15);
        color: var(--accent-blue);
    }}

    .step-content {{
        flex: 1;
        min-width: 0;
    }}

    .step-text {{
        font-size: 0.88em;
        color: var(--text-primary);
    }}

    .step-detail {{
        font-size: 0.78em;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
        margin-top: 3px;
        word-break: break-all;
    }}

    /* Footer */
    .footer {{
        text-align: center;
        padding: 30px;
        color: var(--text-muted);
        font-size: 0.8em;
    }}

    /* Animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(20px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    @keyframes fadeInLeft {{
        from {{
            opacity: 0;
            transform: translateX(-10px);
        }}
        to {{
            opacity: 1;
            transform: translateX(0);
        }}
    }}

    /* Responsive */
    @media (max-width: 700px) {{
        .stats-bar {{
            grid-template-columns: repeat(2, 1fr);
        }}
        .card-header {{
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }}
        .card-meta {{
            align-self: flex-end;
        }}
        .arch-flow {{
            gap: 4px;
        }}
        .arch-node {{
            min-width: 70px;
            padding: 10px 8px;
        }}
        .arch-node-label {{
            font-size: 0.6em;
        }}
    }}
</style>
</head>
<body>

<div class="container">

    <!-- Header -->
    <header class="header">
        <div class="header-badge">Quantum-Safe Communication</div>
        <h1>CuKEM System Dashboard</h1>
        <p>8-Layer Hybrid PQC-QKD Architecture &mdash; Demo Execution Report</p>
    </header>

    <!-- Summary Stats -->
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-value green">{passed}/{total}</div>
            <div class="stat-label">Tests Passed</div>
        </div>
        <div class="stat-card">
            <div class="stat-value cyan">{total_time:.2f}s</div>
            <div class="stat-label">Total Time</div>
        </div>
        <div class="stat-card">
            <div class="stat-value blue">{total}</div>
            <div class="stat-label">Demos Run</div>
        </div>
        <div class="stat-card">
            <div class="stat-value {'green' if passed == total else 'red'}">{passed / total * 100:.0f}%</div>
            <div class="stat-label">Success Rate</div>
        </div>
    </div>

    <!-- Architecture Flow -->
    <div class="arch-section">
        <h2>Architecture Pipeline</h2>
        <div class="arch-flow">
            <div class="arch-node" style="border-color: {LAYER_COLORS['pqc']}">
                <span class="arch-node-label" style="color: {LAYER_COLORS['pqc']}">PQC KEM</span>
                <span class="arch-node-sub">ML-KEM-768</span>
            </div>
            <span class="arch-arrow">&#10132;</span>
            <div class="arch-node" style="border-color: {LAYER_COLORS['qkd']}">
                <span class="arch-node-label" style="color: {LAYER_COLORS['qkd']}">QKD</span>
                <span class="arch-node-sub">BB84</span>
            </div>
            <span class="arch-arrow">&#10132;</span>
            <div class="arch-node" style="border-color: {LAYER_COLORS['entropy']}">
                <span class="arch-node-label" style="color: {LAYER_COLORS['entropy']}">Entropy</span>
                <span class="arch-node-sub">NIST 800-90B</span>
            </div>
            <span class="arch-arrow">&#10132;</span>
            <div class="arch-node" style="border-color: #A78BFA">
                <span class="arch-node-label" style="color: #A78BFA">HKDF</span>
                <span class="arch-node-sub">Key Combine</span>
            </div>
            <span class="arch-arrow">&#10132;</span>
            <div class="arch-node" style="border-color: {LAYER_COLORS['cukem']}">
                <span class="arch-node-label" style="color: {LAYER_COLORS['cukem']}">CuKEM</span>
                <span class="arch-node-sub">Hybrid Core</span>
            </div>
            <span class="arch-arrow">&#10132;</span>
            <div class="arch-node" style="border-color: #F472B6">
                <span class="arch-node-label" style="color: #F472B6">TLS</span>
                <span class="arch-node-sub">Transport</span>
            </div>
            <span class="arch-arrow">&#10132;</span>
            <div class="arch-node" style="border-color: {LAYER_COLORS['adaptive']}">
                <span class="arch-node-label" style="color: {LAYER_COLORS['adaptive']}">Controller</span>
                <span class="arch-node-sub">Adaptive</span>
            </div>
            <span class="arch-arrow">&#10132;</span>
            <div class="arch-node" style="border-color: {LAYER_COLORS['breaker']}">
                <span class="arch-node-label" style="color: {LAYER_COLORS['breaker']}">Breaker</span>
                <span class="arch-node-sub">Resilience</span>
            </div>
        </div>
    </div>

    <!-- Demo Cards -->
    {cards_html}

    <!-- Footer -->
    <div class="footer">
        CuKEM Hybrid PQC-QKD System &mdash; Generated {now}
    </div>

</div>

</body>
</html>'''

    return html


# ============================================================
# Main Entry Point
# ============================================================

def main():
    print()
    print("=" * 60)
    print("  CuKEM Visual Dashboard - Running Demos...")
    print("=" * 60)
    print()

    results = []
    total_start = time.time()

    for name, desc, func, tag in DEMO_LIST:
        print(f"  Running: {name}...", end=" ", flush=True)
        start = time.time()
        try:
            steps, passed = func()
            elapsed = time.time() - start
            results.append({
                "name": name,
                "desc": desc,
                "tag": tag,
                "steps": steps,
                "passed": passed,
                "time": elapsed
            })
            status = "PASS" if passed else "FAIL"
            print(f"{status} ({elapsed:.2f}s)")
        except Exception as e:
            elapsed = time.time() - start
            results.append({
                "name": name,
                "desc": desc,
                "tag": tag,
                "steps": [{"text": f"Error: {str(e)}", "status": "fail"}],
                "passed": False,
                "time": elapsed
            })
            print(f"ERROR ({elapsed:.2f}s)")

    total_time = time.time() - total_start

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print()
    print(f"  Results: {passed}/{total} passed in {total_time:.2f}s")
    print()

    # Generate HTML dashboard
    html = generate_html(results, total_time)

    # Write to temp file and open
    output_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "cukem_dashboard.html"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"  Dashboard saved to: {output_path}")
    print("  Opening in browser...")
    print()

    webbrowser.open(f"file:///{output_path.replace(os.sep, '/')}")


if __name__ == "__main__":
    main()
