# ROB 311 — Root Locus PID Tuning Lab

Interactive Flask web app for tuning a PID controller on a ballbot using root locus analysis.

## Quick Start

```bash
pip install flask numpy scipy matplotlib
python app.py
# Open http://localhost:5000
```

## Controls

1. **Three gain sliders** (Kp, Ki, Kd) at the top of the page
2. **Root locus plot** (left) showing how closed-loop poles move in the s-plane
3. **Step response plot** (right) showing the corresponding time-domain behavior
4. **Stability badge** and **closed-loop pole readout** that update in real time

## Plant Model

The ballbot tilt dynamics are modeled as an inverted pendulum:

```
G(s) = b / (s² − a)
```

## Configuring for Your Robot

Edit the top of `app.py`:

### option A — Direct linearisation values (recommended)

If you have linearised your ballbot's dynamics (in MATLAB, Mathematica, etc.),
set `USE_DIRECT = True` and provide:

```python
DIRECT_A42 = 16.3333   # θ̈ due to θ  (positive = unstable)
DIRECT_B4  = -2.8567   # θ̈ due to T  (negative = reaction torque)
```

### Option B — Physical parameters

Set `USE_DIRECT = False` and fill in `PHYS` with your robot's masses, radii,
inertias, and COG height. The code computes an approximate linearisation.

## Deps

- Python 3.8+
- Flask
- NumPy
- SciPy  
- Matplotlib
