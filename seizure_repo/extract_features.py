"""
Step 2 of data prep: load the EEG for every indexed window and compute the
432 features.

For each recording that has indexed windows:
    read EDF -> pick the paper's 18 channels in order -> microvolts ->
    filter the WHOLE recording (0.5 Hz high-pass + 60 Hz notch) ->
    cut the indexed 1-s windows -> features (18 x 24 -> 432)
Results are cached per recording, so an interrupted run resumes where it stopped.

Usage:
    python extract_features.py /mnt/windows/eeg/chbmit chb01 --index index --out features
    # first pass, faster: keep 3 interictal windows per preictal window
    python extract_features.py /mnt/windows/eeg/chbmit chb01 --interictal-ratio 3

Output: features/chb01.npz with
    X (n, 432) float32, y (n,), seizure_id (n,), abs_time_s (n,), recording (n,)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from features import CHANNELS, FS, preprocess, windows_to_features


# ---------------------------------------------------------------- EDF reading
def read_edf(path):
    """Minimal EDF reader (CHB-MIT: 16-bit, all channels at 256 Hz).
    Returns (labels, data) with data in microvolts, shape (n_channels, n_samples)."""
    with open(path, "rb") as f:
        head = f.read(256)
        n_bytes = int(head[184:192])
        n_rec = int(head[236:244])
        ns = int(head[252:256])
        sh = f.read(n_bytes - 256)

    def field(offset, width):
        return [sh[offset + i * width: offset + (i + 1) * width].decode("ascii").strip()
                for i in range(ns)]

    o = 0
    labels = field(o, 16); o += 16 * ns
    o += 80 * ns                                   # transducer
    units = field(o, 8); o += 8 * ns
    pmin = np.array(field(o, 8), float); o += 8 * ns
    pmax = np.array(field(o, 8), float); o += 8 * ns
    dmin = np.array(field(o, 8), float); o += 8 * ns
    dmax = np.array(field(o, 8), float); o += 8 * ns
    o += 80 * ns                                   # prefiltering
    spr = np.array(field(o, 8), int)
    if not np.all(spr == spr[0]):
        raise ValueError(f"{path}: channels have different sampling rates")
    spr = spr[0]

    raw = np.memmap(path, dtype="<i2", mode="r", offset=n_bytes,
                    shape=(n_rec, ns, spr))
    gain = (pmax - pmin) / (dmax - dmin)
    scale = np.array([1e3 if u.lower() == "mv" else 1e-3 if u.lower() == "nv" else 1.0
                      for u in units])            # convert to microvolts
    data = np.asarray(raw).transpose(1, 0, 2).reshape(ns, -1).astype(np.float64)
    data = ((data - dmin[:, None]) * gain[:, None] + pmin[:, None]) * scale[:, None]
    return labels, data


def pick_channels(labels, data):
    """Return the paper's 18 channels in the paper's order.
    CHB-MIT stores T8-P8 twice (identical); the first copy is used."""
    pos = {}
    for i, lab in enumerate(labels):
        pos.setdefault(lab.upper(), i)             # keeps first occurrence
    missing = [c for c in CHANNELS if c not in pos]
    if missing:
        raise KeyError(f"missing channels {missing}")
    return data[[pos[c] for c in CHANNELS]]


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("patient")
    ap.add_argument("--index", default="index")
    ap.add_argument("--out", default="features")
    ap.add_argument("--interictal-ratio", type=float, default=None,
                    help="keep this many interictal windows per preictal window "
                         "(random, seeded). Default: keep all.")
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    p = args.patient
    idx = pd.read_csv(os.path.join(args.index, f"{p}_index.csv"))
    if args.interictal_ratio is not None:
        pre = idx[idx.label == 1]
        inter = idx[idx.label == 0]
        k = min(len(inter), int(args.interictal_ratio * len(pre)))
        inter = inter.sample(n=k, random_state=args.seed)
        idx = pd.concat([pre, inter]).sort_values(["abs_time_s"])
    print(f"{p}: {(idx.label == 1).sum()} preictal + {(idx.label == 0).sum()} "
          f"interictal windows from {idx.file.nunique()} recordings")

    cache = os.path.join(args.out, p)
    os.makedirs(cache, exist_ok=True)
    parts = []
    for fname, rows in idx.groupby("file", sort=False):
        cfile = os.path.join(cache, fname.replace(".edf", ".npz"))
        if os.path.exists(cfile):
            parts.append(cfile)
            continue
        t = time.time()
        labels, data = read_edf(os.path.join(args.root, p, fname))
        x = preprocess(pick_channels(labels, data))
        starts = rows.start_sample.to_numpy()
        ok = starts + FS <= x.shape[1]
        rows, starts = rows[ok], starts[ok]
        wins = np.stack([x[:, s:s + FS] for s in starts])          # (n, 18, 256)
        X = windows_to_features(wins, n_jobs=args.jobs).astype(np.float32)
        np.savez(cfile, X=X, y=rows.label.to_numpy(),
                 seizure_id=rows.seizure_id.to_numpy(),
                 abs_time_s=rows.abs_time_s.to_numpy(),
                 recording=np.array([fname] * len(rows)))
        parts.append(cfile)
        print(f"  {fname}: {len(rows)} windows in {time.time() - t:.0f}s")

    out = {k: np.concatenate([np.load(c)[k] for c in parts])
           for k in ["X", "y", "seizure_id", "abs_time_s", "recording"]}
    np.savez(os.path.join(args.out, f"{p}.npz"), **out)
    print(f"saved {args.out}/{p}.npz  X={out['X'].shape}")


if __name__ == "__main__":
    main()
