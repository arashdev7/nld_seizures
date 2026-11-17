import numpy as np
import matplotlib.pyplot as plt

fs = 256
t = np.arange(0, 10, 1/fs)
freqs = [0.3, 4.0]
max_order = 4

# Differencing scaled by fs each time (derivative-like), so 4 Hz visibly grows.
# Set SCALE = False for raw np.diff (both signals then shrink).
SCALE = True

stages = {}
for f in freqs:
    x = np.sin(2*np.pi*f*t)
    s = [x]
    for k in range(max_order):
        x = np.diff(x) * (fs if SCALE else 1)
        s.append(x)
    stages[f] = s

fig, axes = plt.subplots(max_order+1, 2, figsize=(12, 11), sharex=True)
for k in range(max_order+1):
    # shared y-limits within a row, so both signals are on the SAME scale
    lim = max(np.max(np.abs(stages[f][k][50:-50])) if k else 1.0 for f in freqs) * 1.1
    for col, f in enumerate(freqs):
        ax = axes[k, col]
        y = stages[f][k]
        ax.plot(t[:len(y)], y, lw=1, color=["tab:blue", "tab:red"][col])
        ax.set_ylim(-lim, lim)
        ax.grid(alpha=0.3)
        amp = np.max(np.abs(y[50:-50])) if k else 1.0
        ax.set_title(f"{f} Hz, order {k}  (peak ≈ {amp:.3g})", fontsize=9)
axes[-1, 0].set_xlabel("time (s)"); axes[-1, 1].set_xlabel("time (s)")
fig.suptitle("Repeated differencing. Each row shares one y-scale across both signals", y=0.995)
fig.tight_layout()
fig.savefig("differencing_side_by_side.png", dpi=150)