"""
Which device is fastest for this (tiny) model?  python bench_device.py [n_windows]
Times a training epoch on every available device (cpu / mps / cuda). The model is small
(18-step sequences, hidden size 32), so an Apple GPU is not guaranteed to beat the CPU;
measure before launching a long sweep, then pin the winner with  SEIZURE_DEVICE=cpu|mps.
"""
import os
import sys
import time

import numpy as np
import torch

from common import train_torch

n = int(sys.argv[1]) if len(sys.argv) > 1 else 40000
rng = np.random.default_rng(0)
X = rng.normal(size=(n, 432)).astype(np.float32)
y = (rng.random(n) < 0.3).astype(int)
X[y == 1, :24] += 0.5
Xv, yv = X[:5000], y[:5000]

devs = ["cpu"]
if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
    devs.append("mps")
if torch.cuda.is_available():
    devs.append("cuda")
quiet = lambda *a, **k: None
for dev in devs:
    os.environ["SEIZURE_DEVICE"] = dev
    train_torch("full", X, y, Xv, yv, epochs=1, log=quiet)            # warm-up / kernel compile
    t = time.time()
    train_torch("full", X, y, Xv, yv, epochs=3, patience=99, log=quiet)
    per = (time.time() - t) / 3
    print(f"{dev:5s}: {per:6.2f} s/epoch on {n} windows  -> ~{per * 100 / 60:.1f} min for 100 epochs")
