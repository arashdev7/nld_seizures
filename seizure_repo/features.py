"""
Feature extraction for Wang et al. (2026), DC-SA-EBiLSTM seizure prediction.

Pipeline per 1-s window (18 channels x 256 samples):
  filter (done once on the continuous recording) -> db4 5-level DWT ->
  6 sub-bands [D1, D2, D3, D4, D5, A5] -> 4 features each [STD, RMS, HFD, FuEn]
  -> (18, 24) matrix, flattened to 432 like the paper.

Interpretation choices (the paper is ambiguous) are marked with NOTE.
"""
import numpy as np
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

FS = 256
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
BANDS = ["D1", "D2", "D3", "D4", "D5", "A5"]   # order from Figure 5
FEATS = ["STD", "RMS", "HFD", "FuEn"]          # order from Figure 5


# ---------------------------------------------------------------- filtering
def preprocess(x, fs=FS):
    """x: (C, T) continuous recording. 4th-order 0.5 Hz high-pass + 60 Hz notch.
    Apply to the whole recording BEFORE cutting 1-s windows (avoids edge effects)."""
    pad = min(10 * fs, x.shape[-1] - 1)    # 10 s of mirrored padding: the default
                                           # (15 samples) is far shorter than the
                                           # filter's ~1 s memory -> edge errors
    sos = butter(4, 0.5, btype="highpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=-1, padlen=pad)
    b, a = iirnotch(60.0, Q=30.0, fs=fs)   # NOTE: Q not given in paper
    return filtfilt(b, a, x, axis=-1, padlen=pad)


# ---------------------------------------------------------------- DWT
def subbands(win, wavelet="db4", level=5):
    """win: (C, N). Returns (6, C, N): each sub-band reconstructed to full length.

    NOTE: The paper sets Higuchi kmax = 128 'because each segment has 256 samples'.
    That only makes sense if every sub-band has 256 samples, i.e. each band is
    reconstructed back to the time domain (raw coefficient arrays are only ~14-131
    long). We therefore reconstruct each band with the other coefficients zeroed."""
    import pywt
    N = win.shape[-1]
    coeffs = pywt.wavedec(win, wavelet, level=level, axis=-1)  # [A5, D5, D4, D3, D2, D1]
    recon = []
    for i in range(len(coeffs)):
        c = [np.zeros_like(cc) for cc in coeffs]
        c[i] = coeffs[i]
        recon.append(pywt.waverec(c, wavelet, axis=-1)[..., :N])
    a5, d5, d4, d3, d2, d1 = recon
    return np.stack([d1, d2, d3, d4, d5, a5])


# ---------------------------------------------------------------- features
def std_feat(X):
    return X.std(axis=-1)


def rms_feat(X):
    return np.sqrt(np.mean(X ** 2, axis=-1))


def higuchi_fd(X, kmax=128):
    """Vectorised Higuchi FD. X: (B, N) -> (B,).
    Standard Higuchi (1988): uses |differences| and averages over m. (The paper's
    Eq. 1-2 drop the absolute value and sum instead of average, likely typos.)"""
    X = np.asarray(X, dtype=float)
    B, N = X.shape
    ks = np.arange(1, kmax + 1)
    L = np.empty((B, kmax))
    for j, k in enumerate(ks):
        d = np.abs(X[:, k:] - X[:, :-k])                 # (B, N-k); element i belongs to m = i % k
        pad = (-d.shape[1]) % k
        d = np.pad(d, ((0, 0), (0, pad)))
        sums = d.reshape(B, -1, k).sum(axis=1)           # (B, k): sum per starting offset m
        m = np.arange(k)
        n_m = (N - 1 - m) // k                           # number of increments per m
        Lm = sums * (N - 1) / (n_m * k) / k
        L[:, j] = Lm.mean(axis=1)
    L = np.maximum(L, 1e-12)
    # FD = slope of log L(k) vs log(1/k)
    x = np.log(1.0 / ks)
    x = x - x.mean()
    y = np.log(L)
    return (y - y.mean(axis=1, keepdims=True)) @ x / (x @ x)


def fuzzy_entropy(X, m=2, n=1, r_factor=0.2):
    """Vectorised fuzzy entropy (Chen et al. 2007 definition). X: (B, N) -> (B,).
    Paper: m=2, fuzzy power n=1, r=0.2*SD."""
    X = np.asarray(X, dtype=np.float32)                  # float32: ~4x faster, same
    B, N = X.shape                                       # result to ~1e-7
    r = (r_factor * X.std(axis=1) + 1e-12).astype(np.float32)
    K = N - m                                            # same template count for m and m+1

    def phi(dim):
        idx = np.arange(K)[:, None] + np.arange(dim)
        V = X[:, idx]                                     # (B, K, dim)
        V = V - V.mean(axis=2, keepdims=True)             # baseline removal
        d = np.abs(V[:, :, None, 0] - V[:, None, :, 0])   # Chebyshev distance
        for t in range(1, dim):
            np.maximum(d, np.abs(V[:, :, None, t] - V[:, None, :, t]), out=d)
        if n != 1:
            d **= n
        d /= r[:, None, None]                             # similarity = exp(-d^n / r),
        np.negative(d, out=d)                             # computed in place
        np.exp(d, out=d)
        return (d.sum(axis=(1, 2), dtype=np.float64) - K) / (K * (K - 1))  # drop i == j

    return np.log(phi(m) + 1e-12) - np.log(phi(m + 1) + 1e-12)


def window_features(win, kmax=128):
    """win: (18, 256) filtered window -> (18, 24) features,
    columns ordered band-major: D1[STD,RMS,HFD,FuEn], D2[...], ..., A5[...]."""
    C, N = win.shape
    sb = subbands(win)                                   # (6, C, N)
    flat = sb.reshape(-1, N)                             # (6*C, N)
    f = np.stack([std_feat(flat), rms_feat(flat),
                  higuchi_fd(flat, kmax), fuzzy_entropy(flat)], axis=-1)  # (6*C, 4)
    f = f.reshape(6, C, 4).transpose(1, 0, 2).reshape(C, 24)
    return f


def windows_to_features(windows, n_jobs=-1):
    """windows: (W, 18, 256) -> (W, 432). Parallelised with joblib."""
    from joblib import Parallel, delayed
    feats = Parallel(n_jobs=n_jobs, batch_size=64)(
        delayed(window_features)(w) for w in windows)
    return np.stack(feats).reshape(len(windows), -1)


if __name__ == "__main__":
    # Sanity checks: FD of white noise ~2, of a smooth sine ~1; FuEn noise > sine
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((4, 256))
    t = np.arange(256) / FS
    sine = np.tile(np.sin(2 * np.pi * 3 * t), (4, 1))
    print("HFD noise", higuchi_fd(noise, kmax=10).round(2), " sine", higuchi_fd(sine, kmax=10).round(2))
    print("FuEn noise", fuzzy_entropy(noise).round(2), " sine", fuzzy_entropy(sine).round(2))
