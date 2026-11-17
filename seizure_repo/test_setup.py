"""
Run this first:  python test_setup.py
Checks the parts that depend on PyWavelets and PyTorch, on random data, in < 1 min.
"""
import numpy as np

print("1. PyWavelets: db4 bands")
import pywt
from features import CHANNELS, FS, preprocess, subbands, window_features

x = np.random.default_rng(0).normal(0, 30, (18, 256))
sb = subbands(x)
assert sb.shape == (6, 18, 256), sb.shape
assert np.allclose(sb.sum(axis=0), x, atol=1e-6), "bands don't add back up to the window"
print("   6 bands x 18 channels x 256 samples, and they add back up to the window: OK")

print("2. One window of features")
f = window_features(x)
assert f.shape == (18, 24) and np.isfinite(f).all()
print(f"   (18, 24), all finite. Higuchi FD range {f[:, 2::4].min():.2f}-{f[:, 2::4].max():.2f} "
      f"(expect ~1-2), fuzzy entropy {f[:, 3::4].min():.2f}-{f[:, 3::4].max():.2f}: OK")

print("3. Filtering a 1-minute recording")
y = preprocess(np.random.default_rng(1).normal(0, 30, (18, 60 * FS)))
assert y.shape == (18, 60 * FS) and np.isfinite(y).all()
print("   OK")

print("4. PyTorch: every model variant")
import torch
from model import ABLATIONS, DCSAEBiLSTM

for name, kw in ABLATIONS.items():
    m = DCSAEBiLSTM(**kw)
    out = m(torch.randn(8, 432))
    assert out.shape == (8, 2), (name, out.shape)
    print(f"   {name:20s} {sum(p.numel() for p in m.parameters()):>8,} parameters  OK")

print("5. Training loop: 3 epochs on random data with a planted signal")
from common import fit_predict, window_metrics

rng = np.random.default_rng(2)
X = rng.normal(size=(1200, 432)).astype(np.float32)
yy = (rng.random(1200) < 0.3).astype(int)
X[yy == 1, :24] += 1.0                         # make class 1 learnable
p = fit_predict("full", X[:800], yy[:800], X[800:1000], yy[800:1000], X[1000:],
                torch_args=dict(epochs=3))
print(f"   test AUC {window_metrics(yy[1000:], p)['AUC']:.3f} (should be well above 0.5)")
print(f"\nAll good. Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
