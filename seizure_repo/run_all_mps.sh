#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -x "../.venv/bin/python" ]; then
  PY="../.venv/bin/python"
  PIP="../.venv/bin/pip"
else
  PY="python"
  PIP="pip"
fi

export SEIZURE_DEVICE=mps
DATA_ROOT="."
PATIENT="chb01"

echo "==> [0/4] Verifying Python, MPS, and plotting dependencies..."
"$PY" -c "import torch; assert torch.backends.mps.is_available(), 'MPS not available!'; print('MPS ready on', torch.__version__)"
"$PY" -c "import matplotlib" 2>/dev/null || "$PIP" install -q matplotlib

if [ ! -f "${DATA_ROOT}/${PATIENT}/${PATIENT}-summary.txt" ]; then
  echo "ERROR: ${DATA_ROOT}/${PATIENT}/${PATIENT}-summary.txt not found."
  exit 1
fi

echo "==> [1/4] Updating index_windows.py, features.py, common.py, and sweep_preictal.py..."

cat << 'EOF_PY' > index_windows.py
import argparse
import os
import re
import sys
import pandas as pd

FS = 256
PREICTAL_START = 33 * 60
PREICTAL_END = 3 * 60
INTERICTAL_GAP = 4 * 3600

REQUIRED = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


def _clock(s):
    h, m, sec = (int(v) for v in s.strip().split(":"))
    return h * 3600 + m * 60 + sec


def parse_summary(path):
    text = open(path).read()
    channels, recs, cur = [], [], None
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Channels changed") or line.startswith("Channels in EDF"):
            channels = []
            continue
        m = re.match(r"Channel\s+\d+:\s*(\S+)", line)
        if m:
            channels.append(m.group(1).upper())
            continue
        m = re.match(r"File Name:\s*(\S+)", line)
        if m:
            cur = dict(file=m.group(1), start=None, end=None, seizures=[], channels=list(channels))
            recs.append(cur)
            continue
        if cur is None:
            continue
        m = re.match(r"File Start Time:\s*(\S+)", line)
        if m:
            cur["start"] = _clock(m.group(1))
            continue
        m = re.match(r"File End Time:\s*(\S+)", line)
        if m:
            cur["end"] = _clock(m.group(1))
            continue
        m = re.match(r"Seizure\s*\d*\s*Start Time:\s*(\d+)", line)
        if m:
            cur["seizures"].append([int(m.group(1)), None])
            continue
        m = re.match(r"Seizure\s*\d*\s*End Time:\s*(\d+)", line)
        if m:
            cur["seizures"][-1][1] = int(m.group(1))
    return recs


def to_absolute(recs):
    day, prev_end = 0, None
    for r in recs:
        if r["start"] is None:
            raise ValueError(f"{r['file']}: no start time in summary")
        start = r["start"] + day * 86400
        if prev_end is not None:
            while start < prev_end - 60:
                day += 1
                start += 86400
        end = r["end"] + day * 86400
        while end <= start:
            day += 1
            end += 86400
        r["abs_start"], r["abs_end"] = start, end
        r["abs_seizures"] = [(start + a, start + b) for a, b in r["seizures"]]
        prev_end = end
    return recs


def has_channels(r):
    chans = set(r["channels"])
    return all(c in chans for c in REQUIRED)


def effective_pre_starts(seizures, pre_start, pre_end):
    eff = []
    for i, (on, _) in enumerate(seizures):
        ms = pre_start
        if i > 0:
            prev_off = seizures[i - 1][1]
            ms = min(pre_start, (on - prev_off) - pre_end)
        eff.append(max(pre_end, ms))
    return eff


def build_index(recs, pre_start=PREICTAL_START, pre_end=PREICTAL_END, gap=INTERICTAL_GAP):
    seizures = sorted(s for r in recs for s in r["abs_seizures"])
    max_starts = effective_pre_starts(seizures, pre_start, pre_end)
    rows = []
    for r in recs:
        if not has_channels(r):
            print(f"  skip {r['file']}: missing some of the 18 channels", file=sys.stderr)
            continue
        n_win = r["abs_end"] - r["abs_start"]
        for w in range(n_win):
            t0 = r["abs_start"] + w
            t1 = t0 + 1
            label, sid = None, -1
            for i, (on, off) in enumerate(seizures):
                ms = max_starts[i]
                if ms > pre_end and on - ms <= t0 and t1 <= on - pre_end:
                    label, sid = 1, i
                    break
            if label == 1:
                if any(t0 < off and on < t1 for on, off in seizures):
                    label = None
            else:
                if all(t1 <= on - gap or t0 >= off + gap for on, off in seizures):
                    label = 0
            if label is not None:
                rows.append((r["file"], w * FS, t0, label, sid))
    return pd.DataFrame(rows, columns=["file", "start_sample", "abs_time_s", "label", "seizure_id"])


def report(recs, idx, pre_start, pre_end):
    seizures = sorted(s for r in recs for s in r["abs_seizures"])
    max_starts = effective_pre_starts(seizures, pre_start, pre_end)
    hours = sum(r["abs_end"] - r["abs_start"] for r in recs) / 3600
    print(f"  {len(recs)} recordings, {hours:.1f} h, {len(seizures)} seizures")
    pre = idx[idx.label == 1]
    for i, (on, off) in enumerate(seizures):
        n = (pre.seizure_id == i).sum()
        nom = max_starts[i] - pre_end
        gap = (on - seizures[i - 1][1]) / 60 if i else float("nan")
        cap = f" [capped at -{max_starts[i] / 60:.0f}m]" if max_starts[i] < pre_start else ""
        print(f"  seizure {i}: onset t={on}s, {n:4d}/{nom} preictal windows (gap: {gap:.0f}m){cap}")
    print(f"  preictal windows:   {len(pre)}")
    print(f"  interictal windows: {(idx.label == 0).sum()} ({(idx.label == 0).sum() / 3600:.1f} h)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("patients", nargs="+")
    ap.add_argument("--out", default="index")
    ap.add_argument("--preictal-start-min", type=float, default=33)
    ap.add_argument("--preictal-end-min", type=float, default=3)
    ap.add_argument("--interictal-gap-h", type=float, default=4)
    args = ap.parse_args()
    pre_start = int(args.preictal_start_min * 60)
    pre_end = int(args.preictal_end_min * 60)
    gap = int(args.interictal_gap_h * 3600)
    os.makedirs(args.out, exist_ok=True)
    for p in args.patients:
        print(p)
        recs = to_absolute(parse_summary(os.path.join(args.root, p, f"{p}-summary.txt")))
        idx = build_index(recs, pre_start, pre_end, gap)
        report(recs, idx, pre_start, pre_end)
        idx.to_csv(os.path.join(args.out, f"{p}_index.csv"), index=False)
        sz = sorted(s for r in recs for s in r["abs_seizures"])
        max_starts = effective_pre_starts(sz, pre_start, pre_end)
        pd.DataFrame({
            "seizure_id": range(len(sz)),
            "onset_s": [a for a, _ in sz],
            "offset_s": [b for _, b in sz],
            "preictal_start_s": [a - ms for (a, _), ms in zip(sz, max_starts)],
            "preictal_end_s": [a - pre_end for a, _ in sz],
        }).to_csv(os.path.join(args.out, f"{p}_seizures.csv"), index=False)


if __name__ == "__main__":
    main()
EOF_PY

cat << 'EOF_PY' > features.py
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
EOF_PY

cat << 'EOF_PY' > common.py
import copy
import os
import time
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SKLEARN_MODELS = ("svm", "knn")


def load_features(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    order = np.argsort(out["abs_time_s"], kind="stable")
    out = {k: v[order] for k, v in out.items()}
    out["X"] = out["X"].astype(np.float32)
    bad = ~np.isfinite(out["X"]).all(axis=1)
    if bad.any():
        print(f"  dropping {bad.sum()} windows with NaN/inf features")
        out = {k: v[~bad] for k, v in out.items()}
    return out


def scale(X, tr):
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X).astype(np.float32)


def undersample(idx, y, ratio, rng):
    pre = idx[y[idx] == 1]
    inter = idx[y[idx] == 0]
    k = min(len(inter), int(ratio * len(pre)))
    return np.sort(np.concatenate([pre, rng.choice(inter, k, replace=False)]))


def window_metrics(y, prob, thr=0.5):
    pred = prob >= thr
    tp = np.sum(pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    return dict(ACC=(tp + tn) / len(y),
                SEN=tp / (tp + fn) if tp + fn else np.nan,
                SPE=tn / (tn + fp) if tn + fp else np.nan,
                AUC=auc, n_pre=int(np.sum(y == 1)), n_inter=int(np.sum(y == 0)))


def alarms(times, pred, win=30, k=10, refractory_s=0):
    if len(times) == 0:
        return np.array([], dtype=int)
    t0 = int(times.min())
    line = np.zeros(int(times.max()) - t0 + 1)
    line[(times - t0).astype(int)] = pred
    votes = np.convolve(line, np.ones(win), mode="full")[:len(line)]
    on = votes >= k
    on[:win - 1] = False
    starts = np.flatnonzero(on & ~np.r_[False, on[:-1]]) + t0
    if refractory_s and len(starts):
        kept = [starts[0]]
        for s in starts[1:]:
            if s - kept[-1] >= refractory_s:
                kept.append(s)
        starts = np.array(kept)
    return starts


def split_segments(times, max_gap=1):
    cuts = np.flatnonzero(np.diff(times) > max_gap) + 1
    return np.split(np.arange(len(times)), cuts)


def fit_predict(name, Xtr, ytr, Xva, yva, Xte, class_weight=False, seed=0,
                torch_args=None, log=print):
    if name == "svm":
        from sklearn.svm import SVC
        m = SVC(kernel="rbf", C=10, gamma="scale",
                class_weight="balanced" if class_weight else None).fit(Xtr, ytr)
        return 1 / (1 + np.exp(-m.decision_function(Xte)))
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        m = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    model = train_torch(name, Xtr, ytr, Xva, yva, class_weight=class_weight,
                        seed=seed, log=log, **(torch_args or {}))
    return predict_torch(model, Xte)


def _device():
    import torch
    req = os.environ.get("SEIZURE_DEVICE", "").strip().lower()
    if req:
        return torch.device(req)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_torch(variant, Xtr, ytr, Xva, yva, epochs=100, patience=30, batch=128,
                lr=1e-3, weight_decay=1e-4, step=80, gamma=0.1,
                class_weight=False, seed=0, log=print):
    import torch
    import torch.nn as nn
    from model import ABLATIONS, DCSAEBiLSTM

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = _device()
    model = DCSAEBiLSTM(**ABLATIONS[variant]).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=step, gamma=gamma)
    w = None
    if class_weight:
        counts = np.bincount(ytr, minlength=2).astype(np.float32)
        w = torch.tensor(len(ytr) / (2 * counts), dtype=torch.float32, device=dev)
    lossf = nn.CrossEntropyLoss(weight=w)

    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr).long())
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True,
                                     drop_last=len(ds) > batch)
    best, best_state, wait = -np.inf, None, 0
    t = time.time()
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for xb, yb in dl:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        sched.step()
        pva = predict_torch(model, Xva)
        score = roc_auc_score(yva, pva) if len(np.unique(yva)) == 2 else -tot
        if score > best:
            best, best_state, wait = score, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
        if ep % 10 == 0 or wait == 0:
            log(f"    epoch {ep:3d}  train loss {tot / len(ds):.4f}  val AUC {score:.4f}"
                f"{'  *' if wait == 0 else ''}")
        if wait >= patience:
            log(f"    early stop at epoch {ep}")
            break
    model.load_state_dict(best_state)
    log(f"    trained in {time.time() - t:.0f}s, best val AUC {best:.4f}")
    return model


def predict_torch(model, X, batch=4096):
    import torch
    dev = next(model.parameters()).device
    model.eval()
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(dev)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
EOF_PY

"$PY" - << 'EOF_PATCH'
with open("sweep_preictal.py", "r") as f:
    src = f.read()

old_relabel = '''def relabel(d, onset, start_min, end_min):
    """Boolean mask of windows to keep for preictal = (onset - start, onset - end].
    Interictal windows (y == 0) are always kept."""
    lead = onset[d["seizure_id"]] - d["abs_time_s"]            # seconds before onset
    pre = (d["y"] == 1) & (lead <= start_min * 60) & (lead >= end_min * 60 + 1)
    return (d["y"] == 0) | pre'''

new_relabel = '''def relabel(d, onset, offset, start_min, end_min):
    """Boolean mask of windows to keep for preictal = (onset - max_lead, onset - end],
    capping lookback at (onset_i - offset_{i-1}) - end_min for close seizures."""
    max_lead = np.full(len(onset), start_min * 60.0)
    if len(onset) > 1:
        max_lead[1:] = np.minimum(max_lead[1:], (onset[1:] - offset[:-1]) - end_min * 60.0)
    lead = onset[d["seizure_id"]] - d["abs_time_s"]
    pre = (d["y"] == 1) & (lead <= max_lead[d["seizure_id"]]) & (lead >= end_min * 60 + 1)
    return (d["y"] == 0) | pre'''

if old_relabel in src:
    src = src.replace(old_relabel, new_relabel)
    src = src.replace(
        'onset = sz.set_index("seizure_id").onset_s.sort_index().to_numpy()',
        'onset = sz.set_index("seizure_id").onset_s.sort_index().to_numpy()\n    offset = sz.set_index("seizure_id").offset_s.sort_index().to_numpy()'
    )
    src = src.replace("relabel(d, onset, S, args.end)", "relabel(d, onset, offset, S, args.end)")
    with open("sweep_preictal.py", "w") as f:
        f.write(src)
    print("  patched sweep_preictal.py")
else:
    print("  sweep_preictal.py already patched")
EOF_PATCH

echo "==> [2/4] Indexing windows up to 60 min preictal (63m -> 3m) with close-seizure cap..."
"$PY" index_windows.py "$DATA_ROOT" "$PATIENT" \
  --preictal-start-min 63 \
  --preictal-end-min 3 \
  --out index

echo "==> [3/4] Clearing stale seizure caches and extracting missing windows on MPS..."
"$PY" -c "import os, pandas as pd; df=pd.read_csv('index/${PATIENT}_index.csv'); [os.remove(f'features/${PATIENT}/{f[:-4]}.npz') for f in df[df.seizure_id>=0].file.unique() if os.path.exists(f'features/${PATIENT}/{f[:-4]}.npz')]"
"$PY" extract_features.py "$DATA_ROOT" "$PATIENT" --index index --out features

echo "==> [4/4] Running preictal sweep (10m to 60m) on MPS..."
"$PY" sweep_preictal.py "features/${PATIENT}.npz" \
  --index index \
  --model dcsa \
  --protocols loso \
  --seeds 0 \
  --out sweep_dcsa

"$PY" sweep_preictal.py "features/${PATIENT}.npz" \
  --index index \
  --model dcsa \
  --protocols loso grouped \
  --seeds 0 1 2 \
  --out sweep_dcsa

echo "==> ALL DONE! Results saved in sweep_dcsa/summary.csv and sweep_dcsa/sweep.png"
