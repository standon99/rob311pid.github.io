"""
=============================================================================
Root Locus PID Tuning Lab
=============================================================================

PLANT MODEL
-----------
The ballbot near the upright equilibrium is an inverted pendulum on a
rolling ball.  We extract the SISO transfer function from motor torque T
to body tilt angle theta:

    G(s) = b / (s^2 - a)

where:
    a = gravitational instability term  (positive — gives the RHP pole)
    b = torque-to-tilt coupling         (negative — non-minimum-phase
        reaction: positive torque initially tips the body backward)

Because b < 0, the negative feedback loop 1 + C(s)*G(s) = 0 needs the
controller to command torque *opposing* the tilt.  We absorb the sign
so students see G_eff(s) = |b| / (s^2 - a) and use standard positive-
gain PID tuning with negative feedback.

USAGE
-----
    1.  Edit the CONFIGURATION section below with your robot's values.
    2.  python app.py
    3.  Open http://localhost:5000
    4.  Move the sliders — the plots update live.

Author : Siddhant Tandon
=============================================================================
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend for server use
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, render_template, jsonify, request


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  CONFIGURATION — EDIT THIS SECTION FOR YOUR ROBOT                       ║
# ╚═══════════════════════════════════════════════════════════════════════════╝
#
# You have TWO options:
#
#   OPTION A (recommended): Provide the two key linearised coefficients
#       directly.  These come from your own linearisation (MATLAB,
#       Mathematica, or by hand).  The defaults are from the ETH Rezero
#       thesis, Section 2.4.1 (yz-plane model).
#
#   OPTION B: Provide physical parameters and let the code compute an
#       approximate linearisation (simplified inverted pendulum on ball).
#
# Set USE_DIRECT = True  for Option A,
#     USE_DIRECT = False for Option B.
# ─────────────────────────────────────────────────────────────────────────

USE_DIRECT = True

# --- OPTION A: Direct linearisation coefficients ---
# From the linearised equation  theta_ddot = a42 * theta + b4 * T :
#   a42 > 0  →  gravitational instability (gives the RHP pole)
#   b4  < 0  →  reaction torque (positive T tips body backward initially)
DIRECT_A42 =  16.3333    # [1/s^2]  ETH thesis value
DIRECT_B4  =  -2.8567    # [1/s^2]  ETH thesis value

# --- OPTION B: Physical parameters ---
PHYS = {
    "m_ball":  2.29,     # [kg]
    "m_body":  9.2,      # [kg]
    "r_ball":  0.125,    # [m]
    "l_cog":   0.339,    # [m]   Height of body COG above ball centre
    "J_ball":  0.0239,   # [kg*m^2]
    "J_wheel": 0.00236,  # [kg*m^2]  Virtual actuating wheel inertia
    "J_body":  4.76,     # [kg*m^2]
    "g":       9.81,     # [m/s^2]
}

# --- Derivative filter time constant ---
TAU_F = 0.01   # [s]  For filtered derivative Kd*s / (tau_f*s + 1)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1 — PLANT TRANSFER FUNCTION                                    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def build_plant_tf():
    """
    Build G_eff(s) = |b4| / (s^2 - a42).

    Returns (num, den) as polynomial coefficient arrays in descending
    powers of s.
    """
    if USE_DIRECT:
        a42 = DIRECT_A42
        b4  = DIRECT_B4
    else:
        p = PHYS
        m_b = p["m_body"]; m_k = p["m_ball"]
        r_k = p["r_ball"]; l = p["l_cog"]; g = p["g"]
        J_b = p["J_body"]; J_k = p["J_ball"]; J_w = p["J_wheel"]

        alpha = J_k + (m_b + m_k) * r_k**2 + J_w
        beta  = m_b * r_k * l
        gamma = J_b + m_b * l**2
        det_M = alpha * gamma - beta**2
        grav  = m_b * g * l

        a42 =  (alpha * grav) / det_M
        b4  = -(beta + alpha) / det_M

    # Absorb sign: use |b4| so positive PID gains stabilise
    b_eff = abs(b4)

    num = np.array([b_eff])               # constant numerator
    den = np.array([1.0, 0.0, -a42])      # s^2 + 0*s - a42

    return num, den


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2 — PID CONTROLLER                                             ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def pid_tf(Kp, Ki, Kd, tau_f=TAU_F):
    """
    PID controller with filtered derivative:

        C(s) = Kp + Ki/s + Kd*s / (tau_f*s + 1)

    When Ki > 0 (full PID), the common denominator is s*(tau_f*s + 1):
        num = (Kp*tau_f + Kd)*s^2 + (Kp + Ki*tau_f)*s + Ki
        den = tau_f*s^2 + s

    When Ki = 0 (PD only), the 1/s term vanishes and the common factor
    of s in both num and den cancels, leaving:
        num = (Kp*tau_f + Kd)*s + Kp
        den = tau_f*s + 1

    When Ki = 0 and Kd = 0 (P only), the controller is just:
        num = [Kp]
        den = [1]

    We handle these cases explicitly to avoid spurious poles at s=0.
    """
    if abs(Ki) < 1e-15 and abs(Kd) < 1e-15:
        # P-only: C(s) = Kp
        return np.array([Kp]), np.array([1.0])
    elif abs(Ki) < 1e-15:
        # PD only: C(s) = Kp + Kd*s/(tau_f*s + 1) = [(Kp*tau_f+Kd)*s + Kp] / (tau_f*s + 1)
        num = np.array([Kp*tau_f + Kd,  Kp])
        den = np.array([tau_f,          1.0])
        return num, den
    else:
        # Full PID
        num = np.array([Kp*tau_f + Kd,  Kp + Ki*tau_f,  Ki])
        den = np.array([tau_f,          1.0,             0.0])
        return num, den


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3 — ROOT LOCUS & CLOSED-LOOP ANALYSIS                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def open_loop_tf(num_g, den_g, Kp, Ki, Kd):
    """L(s) = C(s) * G(s).  Returns (num_L, den_L)."""
    nc, dc = pid_tf(Kp, Ki, Kd)
    num_L = np.convolve(nc, num_g)
    den_L = np.convolve(dc, den_g)
    return num_L, den_L


def cl_poles(num_L, den_L):
    """Closed-loop poles: roots of  den_L + num_L = 0."""
    n = max(len(num_L), len(den_L))
    np_ = np.zeros(n); dp_ = np.zeros(n)
    np_[n - len(num_L):] = num_L
    dp_[n - len(den_L):] = den_L
    return np.roots(dp_ + np_)


def root_locus(num_L, den_L, k_max=500.0, n_pts=600):
    """
    Sweep gain K in  1 + K*L(s) = 0  from 0 to k_max.
    Returns (k_vals, roots_array).
    """
    k_vals = np.concatenate([
        np.linspace(0,    5,    n_pts // 3),
        np.linspace(5,    50,   n_pts // 3),
        np.linspace(50,   k_max, n_pts // 3),
    ])
    n = len(den_L)
    num_pad = np.zeros(n)
    num_pad[n - len(num_L):] = num_L

    roots_list = []
    for K in k_vals:
        r = np.roots(den_L + K * num_pad)
        r = r[np.argsort(r.real)]
        roots_list.append(r)

    return k_vals, np.array(roots_list)


def step_response(num_g, den_g, Kp, Ki, Kd, t_end=5.0, n_pts=1000):
    """
    Compute the closed-loop step response T(s) = L / (1+L).
    Returns (t, y, poles, stable).
    """
    nL, dL = open_loop_tf(num_g, den_g, Kp, Ki, Kd)
    poles = cl_poles(nL, dL)
    stable = bool(np.all(poles.real < -1e-10))

    n = max(len(nL), len(dL))
    np_ = np.zeros(n); dp_ = np.zeros(n)
    np_[n - len(nL):] = nL
    dp_[n - len(dL):] = dL
    cl_num = np_
    cl_den = dp_ + np_

    # Clean leading zeros
    while len(cl_num) > 1 and abs(cl_num[0]) < 1e-14:
        cl_num = cl_num[1:]
    while len(cl_den) > 1 and abs(cl_den[0]) < 1e-14:
        cl_den = cl_den[1:]

    t = np.linspace(0, t_end if stable else min(t_end, 2.0), n_pts)
    try:
        sys_cl = signal.TransferFunction(cl_num, cl_den)
        t, y = signal.step(sys_cl, T=t)
        if not stable:
            y = np.clip(y, -20, 20)
    except Exception:
        y = np.zeros_like(t)

    return t, y, poles, stable


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4 — MATPLOTLIB FIGURE GENERATION                               ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

def make_plots(num_g, den_g, Kp, Ki, Kd):
    """
    Generate root locus + step response as a base64 PNG image.
    Returns dict with 'img', 'stable', 'poles_text'.
    """
    # ── Compute everything ──
    nL, dL = open_loop_tf(num_g, den_g, Kp, Ki, Kd)
    ol_p = np.roots(dL)
    ol_z = np.roots(nL) if len(nL) > 1 else np.array([])
    kv, rl = root_locus(nL, dL)
    poles = cl_poles(nL, dL)
    stable = bool(np.all(poles.real < -1e-10))
    t, y, _, _ = step_response(num_g, den_g, Kp, Ki, Kd)

    # ── Colours ──
    BG      = "#0a0e17"
    GRID    = "#1a2234"
    TEXT    = "#8899b4"
    LOCUS   = "#3b5998"
    OLP_C   = "#34d399"    # open-loop pole colour
    OLZ_C   = "#22d3ee"    # open-loop zero colour
    CLP_OK  = "#FFCB05"    # closed-loop pole (stable)
    CLP_BAD = "#f87171"    # closed-loop pole (unstable)
    RESP    = "#FFCB05"
    REF     = "#556680"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG, dpi=110)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    # ──────────── ROOT LOCUS ────────────
    ax1.set_title("Root Locus — s-Plane", color="#e8edf5", fontsize=13,
                fontweight="bold", pad=10)
    ax1.set_xlabel("Real Axis (σ)", color=TEXT, fontsize=10)
    ax1.set_ylabel("Imaginary Axis (jω)", color=TEXT, fontsize=10)

    # Determine axis limits from data
    pts_re = []; pts_im = []
    for b in range(rl.shape[1]):
        re = rl[:, b].real; im = rl[:, b].imag
        mask = (np.abs(re) < 200) & (np.abs(im) < 200)
        pts_re.extend(re[mask]); pts_im.extend(im[mask])
    for p in np.concatenate([ol_p, poles, ol_z]):
        pts_re.append(p.real); pts_im.append(p.imag)

    if pts_re:
        xlo = min(pts_re) - 2;  xhi = max(pts_re) + 2
        yabs = max(abs(min(pts_im)), abs(max(pts_im))) + 2
    else:
        xlo, xhi, yabs = -10, 10, 10
    xlo = min(xlo, -3); xhi = max(xhi, 3); yabs = max(yabs, 3)
    ax1.set_xlim(xlo, xhi); ax1.set_ylim(-yabs, yabs)

    # RHP shading + jω axis
    ax1.axvspan(0, xhi, alpha=0.04, color=CLP_BAD)
    ax1.axvline(0, color=CLP_BAD, lw=1, alpha=0.3, ls="--")

    # Branches
    for b in range(rl.shape[1]):
        ax1.plot(rl[:, b].real, rl[:, b].imag,
                color=LOCUS, lw=1.2, alpha=0.55)

    # OL poles (×)
    ax1.scatter(ol_p.real, ol_p.imag, marker="x", s=110,
                color=OLP_C, linewidths=2.5, zorder=5, label="OL poles")

    # OL zeros (○)
    if len(ol_z) > 0 and not np.all(np.abs(ol_z) < 1e-10):
        ax1.scatter(ol_z.real, ol_z.imag, marker="o", s=80,
                    facecolors="none", edgecolors=OLZ_C, linewidths=2,
                    zorder=5, label="OL zeros")

    # CL poles (◆)
    for p in poles:
        c = CLP_BAD if p.real > 1e-6 else CLP_OK
        ax1.scatter(p.real, p.imag, marker="D", s=90, color=c,
                    edgecolors="white", linewidths=0.5, zorder=6)

    # Legend entries
    ax1.scatter([], [], marker="D", s=50, color=CLP_OK,  label="CL poles (stable)")
    ax1.scatter([], [], marker="D", s=50, color=CLP_BAD, label="CL poles (unstable)")
    ax1.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID,
            labelcolor=TEXT, framealpha=0.9)

    # ──────────── STEP RESPONSE ────────────
    ax2.set_title("Closed-Loop Step Response", color="#e8edf5",
                fontsize=13, fontweight="bold", pad=10)
    ax2.set_xlabel("Time (s)", color=TEXT, fontsize=10)
    ax2.set_ylabel("θ (rad)", color=TEXT, fontsize=10)
    ax2.axhline(1.0, color=REF, lw=1, ls=":", alpha=0.6, label="Reference")
    ax2.axhline(0.0, color=REF, lw=0.5, alpha=0.3)

    rc = RESP if stable else CLP_BAD
    ax2.plot(t, y, color=rc, lw=2, label="θ(t)", zorder=3)

    if not stable:
        ax2.text(0.5, 0.5, "UNSTABLE", transform=ax2.transAxes,
                fontsize=28, fontweight="bold", color=CLP_BAD,
                ha="center", va="center", alpha=0.3)

    ax2.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID,labelcolor=TEXT, framealpha=0.9)

    plt.tight_layout(pad=2)

    # ── Encode to base64 ──
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

    # ── Format poles for display ──
    pole_strs = []
    for p in poles:
        unstable = bool(p.real > 1e-6)
        if abs(p.imag) < 1e-6:
            txt = f"{p.real:.4f}"
        else:
            sign = "+" if p.imag >= 0 else "−"
            txt = f"{p.real:.4f} {sign} {abs(p.imag):.4f}j"
        pole_strs.append({"text": txt, "unstable": unstable})

    return {"img": img_b64, "stable": stable, "poles": pole_strs}


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5 — FLASK APPLICATION                                          ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

app = Flask(__name__, template_folder='docs')

# Build plant once at startup
G_NUM, G_DEN = build_plant_tf()

ol = np.roots(G_DEN)
print("\n" + "=" * 62)
print("  ROB 311 — Ballbot Root Locus Lab")
print("=" * 62)
print(f"  G(s) = {G_NUM[0]:.4f} / (s^2 - {-G_DEN[2]:.4f})")
print(f"  Open-loop poles: s = {ol}")
rp = ol[ol.real > 0]
if len(rp) > 0:
    print(f"  Unstable pole at s = +{rp[0].real:.4f}")
print(f"\n  Open http://localhost:5000")
print("=" * 62 + "\n")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/update", methods=["POST"])
def api_update():
    d = request.get_json()
    Kp = float(d.get("Kp", 0))
    Ki = float(d.get("Ki", 0))
    Kd = float(d.get("Kd", 0))
    result = make_plots(G_NUM, G_DEN, Kp, Ki, Kd)
    return jsonify(result)


@app.route("/api/plant", methods=["GET"])
def api_plant():
    return jsonify({
        "num": G_NUM.tolist(),
        "den": G_DEN.tolist(),
        "poles": [{"re": float(p.real), "im": float(p.imag)}
                for p in np.roots(G_DEN)],
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8100)
