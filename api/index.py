"""
Vercel serverless function — Root Locus PID Tuning Lab API.

Exposes the Flask WSGI `app` object that Vercel's Python runtime picks up
automatically.  Static frontend is served from ../public/ by Vercel CDN.
"""

import numpy as np
from scipy import signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import base64
from flask import Flask, jsonify, request


# ── Configuration (mirrors app.py) ──────────────────────────────────────────

USE_DIRECT = True

DIRECT_A42 = 16.3333
DIRECT_B4  = -2.8567

PHYS = {
    "m_ball":  2.29,
    "m_body":  9.2,
    "r_ball":  0.125,
    "l_cog":   0.339,
    "J_ball":  0.0239,
    "J_wheel": 0.00236,
    "J_body":  4.76,
    "g":       9.81,
}

TAU_F = 0.01


# ── Plant transfer function ─────────────────────────────────────────────────

def build_plant_tf():
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

    b_eff = abs(b4)
    num = np.array([b_eff])
    den = np.array([1.0, 0.0, -a42])
    return num, den


# ── PID controller ──────────────────────────────────────────────────────────

def pid_tf(Kp, Ki, Kd, tau_f=TAU_F):
    if abs(Ki) < 1e-15 and abs(Kd) < 1e-15:
        return np.array([Kp]), np.array([1.0])
    elif abs(Ki) < 1e-15:
        num = np.array([Kp*tau_f + Kd,  Kp])
        den = np.array([tau_f,          1.0])
        return num, den
    else:
        num = np.array([Kp*tau_f + Kd,  Kp + Ki*tau_f,  Ki])
        den = np.array([tau_f,          1.0,             0.0])
        return num, den


# ── Root locus & closed-loop analysis ───────────────────────────────────────

def open_loop_tf(num_g, den_g, Kp, Ki, Kd):
    nc, dc = pid_tf(Kp, Ki, Kd)
    num_L = np.convolve(nc, num_g)
    den_L = np.convolve(dc, den_g)
    return num_L, den_L


def cl_poles(num_L, den_L):
    n = max(len(num_L), len(den_L))
    np_ = np.zeros(n); dp_ = np.zeros(n)
    np_[n - len(num_L):] = num_L
    dp_[n - len(den_L):] = den_L
    return np.roots(dp_ + np_)


def root_locus(num_L, den_L, k_max=500.0, n_pts=600):
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
    nL, dL = open_loop_tf(num_g, den_g, Kp, Ki, Kd)
    poles = cl_poles(nL, dL)
    stable = bool(np.all(poles.real < -1e-10))

    n = max(len(nL), len(dL))
    np_ = np.zeros(n); dp_ = np.zeros(n)
    np_[n - len(nL):] = nL
    dp_[n - len(dL):] = dL
    cl_num = np_
    cl_den = dp_ + np_

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


# ── Matplotlib figure generation ────────────────────────────────────────────

def make_plots(num_g, den_g, Kp, Ki, Kd):
    nL, dL = open_loop_tf(num_g, den_g, Kp, Ki, Kd)
    ol_p = np.roots(dL)
    ol_z = np.roots(nL) if len(nL) > 1 else np.array([])
    kv, rl = root_locus(nL, dL)
    poles = cl_poles(nL, dL)
    stable = bool(np.all(poles.real < -1e-10))
    t, y, _, _ = step_response(num_g, den_g, Kp, Ki, Kd)

    BG      = "#0a0e17"
    GRID    = "#1a2234"
    TEXT    = "#8899b4"
    LOCUS   = "#3b5998"
    OLP_C   = "#34d399"
    OLZ_C   = "#22d3ee"
    CLP_OK  = "#FFCB05"
    CLP_BAD = "#f87171"
    RESP    = "#FFCB05"
    REF     = "#556680"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG, dpi=110)

    for ax in (ax1, ax2):
        ax.set_facecolor(BG)
        ax.tick_params(colors=TEXT, labelsize=9)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)

    ax1.set_title("Root Locus — s-Plane", color="#e8edf5", fontsize=13,
                  fontweight="bold", pad=10)
    ax1.set_xlabel("Real Axis (σ)", color=TEXT, fontsize=10)
    ax1.set_ylabel("Imaginary Axis (jω)", color=TEXT, fontsize=10)

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

    ax1.axvspan(0, xhi, alpha=0.04, color=CLP_BAD)
    ax1.axvline(0, color=CLP_BAD, lw=1, alpha=0.3, ls="--")

    for b in range(rl.shape[1]):
        ax1.plot(rl[:, b].real, rl[:, b].imag,
                 color=LOCUS, lw=1.2, alpha=0.55)

    ax1.scatter(ol_p.real, ol_p.imag, marker="x", s=110,
                color=OLP_C, linewidths=2.5, zorder=5, label="OL poles")

    if len(ol_z) > 0 and not np.all(np.abs(ol_z) < 1e-10):
        ax1.scatter(ol_z.real, ol_z.imag, marker="o", s=80,
                    facecolors="none", edgecolors=OLZ_C, linewidths=2,
                    zorder=5, label="OL zeros")

    for p in poles:
        c = CLP_BAD if p.real > 1e-6 else CLP_OK
        ax1.scatter(p.real, p.imag, marker="D", s=90, color=c,
                    edgecolors="white", linewidths=0.5, zorder=6)

    ax1.scatter([], [], marker="D", s=50, color=CLP_OK,  label="CL poles (stable)")
    ax1.scatter([], [], marker="D", s=50, color=CLP_BAD, label="CL poles (unstable)")
    ax1.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID,
               labelcolor=TEXT, framealpha=0.9)

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

    ax2.legend(loc="upper right", fontsize=8, facecolor=BG, edgecolor=GRID,
               labelcolor=TEXT, framealpha=0.9)

    plt.tight_layout(pad=2)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")

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


# ── Flask application ───────────────────────────────────────────────────────

app = Flask(__name__)

G_NUM, G_DEN = build_plant_tf()


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
