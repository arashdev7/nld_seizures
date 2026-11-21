import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, iirnotch, sosfiltfilt, filtfilt

FS = 256
CHANNELS = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]
BANDS = ["D1", "D2", "D3", "D4", "D5", "A5"]
FEATS = ["STD", "RMS", "HFD", "FuEn"]


def _feat_device():
    req = os.environ.get("SEIZURE_DEVICE", "").strip().lower()
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def preprocess(x, fs=FS):
    pad = min(10 * fs, x.shape[-1] - 1)
    sos = butter(4, 0.5, btype="highpass", fs=fs, output="sos")
    x = sosfiltfilt(sos, x, axis=-1, padlen=pad)
    b, a = iirnotch(60.0, Q=30.0, fs=fs)
    return filtfilt(b, a, x, axis=-1, padlen=pad)


def subbands_batch(wins, wavelet="db4", level=5):
    import pywt
    N = wins.shape[-1]
    coeffs = pywt.wavedec(wins, wavelet, level=level, axis=-1)
    recon = []
    for i in range(len(coeffs)):
        c = [np.zeros_like(cc) for cc in coeffs]
        c[i] = coeffs[i]
        recon.append(pywt.waverec(c, wavelet, axis=-1)[..., :N])
    a5, d5, d4, d3, d2, d1 = recon
    return np.stack([d1, d2, d3, d4, d5, a5], axis=1)


def _higuchi_torch(X, kmax, x_log, x_denom):
    B, N = X.shape
    dev = X.device
    L = torch.empty((B, kmax), dtype=torch.float32, device=dev)
    for j, k in enumerate(range(1, kmax + 1)):
        d = torch.abs(X[:, k:] - X[:, :-k])
        pad = (-d.shape[1]) % k
        if pad:
            d = F.pad(d, (0, pad))
        sums = d.reshape(B, -1, k).sum(dim=1)
        m = torch.arange(k, device=dev)
        n_m = (N - 1 - m) // k
        Lm = sums * ((N - 1) / (n_m.to(torch.float32) * (k * k)))
        L[:, j] = Lm.mean(dim=1)
    L = torch.clamp(L, min=1e-12)
    y = torch.log(L)
    return ((y - y.mean(dim=1, keepdim=True)) @ x_log) / x_denom


def _fuzzy_entropy_torch(X, idx_m, idx_m1, m=2, n=1, r_factor=0.2):
    K = X.shape[1] - m
    r = r_factor * X.std(dim=1, correction=0) + 1e-12

    def phi(V):
        V = V - V.mean(dim=2, keepdim=True)
        d = torch.abs(V[:, :, None, 0] - V[:, None, :, 0])
        for t in range(1, V.shape[2]):
            d = torch.maximum(d, torch.abs(V[:, :, None, t] - V[:, None, :, t]))
        if n != 1:
            d = d ** n
        d = torch.exp(-d / r[:, None, None])
        return (d.sum(dim=(1, 2)) - K) / (K * (K - 1))

    pm = phi(X[:, idx_m])
    pm1 = phi(X[:, idx_m1])
    return torch.log(pm + 1e-12) - torch.log(pm1 + 1e-12)


def windows_to_features(windows, n_jobs=-1, kmax=128, chunk=2048):
    W, C, N = windows.shape
    if W == 0:
        return np.zeros((0, C * 24), dtype=np.float32)

    sb = subbands_batch(np.asarray(windows, dtype=np.float32))
    flat = sb.reshape(-1, N)
    B_total = flat.shape[0]

    dev = _feat_device()
    m = 2
    K = N - m
    ks = torch.arange(1, kmax + 1, dtype=torch.float32, device=dev)
    x_log = torch.log(1.0 / ks)
    x_log = x_log - x_log.mean()
    x_denom = x_log @ x_log
    idx_m = torch.arange(K, device=dev)[:, None] + torch.arange(m, device=dev)
    idx_m1 = torch.arange(K, device=dev)[:, None] + torch.arange(m + 1, device=dev)

    out_flat = np.empty((B_total, 4), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, B_total, chunk):
            e = min(s + chunk, B_total)
            xb = torch.from_numpy(flat[s:e]).to(dev)
            f_std = xb.std(dim=-1, correction=0)
            f_rms = torch.sqrt(torch.mean(xb ** 2, dim=-1))
            f_hfd = _higuchi_torch(xb, kmax, x_log, x_denom)
            f_fuen = _fuzzy_entropy_torch(xb, idx_m, idx_m1, m=m)
            out_flat[s:e] = torch.stack([f_std, f_rms, f_hfd, f_fuen], dim=-1).cpu().numpy()

    if dev.type == "mps":
        torch.mps.empty_cache()

    return out_flat.reshape(W, 6, C, 4).transpose(0, 2, 1, 3).reshape(W, C * 24)


def window_features(win, kmax=128):
    return windows_to_features(win[None], kmax=kmax)[0]
