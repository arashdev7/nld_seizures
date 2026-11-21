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
    Xs, ys, sids, ts, recs = [], [], [], [], []
    for fname, rows in idx.groupby("file", sort=False):
        cfile = os.path.join(cache, fname.replace(".edf", ".npz"))
        # The per-recording cache holds ONLY features, keyed by window time. Labels always
        # come from the current index, so re-running with a different preictal definition
        # can never pick up stale labels. (The old cache stored y / seizure_id too and
        # silently reused them.) Old caches still work: only abs_time_s and X are read.
        c_t, c_X = np.zeros(0, dtype=np.int64), np.zeros((0, 432), dtype=np.float32)
        if os.path.exists(cfile):
            c = np.load(cfile)
            c_t, c_X = c["abs_time_s"].astype(np.int64), c["X"]
        need = rows[~rows.abs_time_s.isin(c_t)]
        if len(need):
            t = time.time()
            labels, data = read_edf(os.path.join(args.root, p, fname))
            x = preprocess(pick_channels(labels, data))
            starts = need.start_sample.to_numpy()
            ok = starts + FS <= x.shape[1]
            need, starts = need[ok], starts[ok]
            if len(need):
                wins = np.stack([x[:, s:s + FS] for s in starts])      # (n, 18, 256)
                Xn = windows_to_features(wins, n_jobs=args.jobs).astype(np.float32)
                c_t = np.concatenate([c_t, need.abs_time_s.to_numpy().astype(np.int64)])
                c_X = np.concatenate([c_X, Xn])
                np.savez(cfile, X=c_X, abs_time_s=c_t)
            print(f"  {fname}: {len(need)} new windows in {time.time() - t:.0f}s "
                  f"({len(rows) - len(need)} from cache)")
        pos = pd.Series(np.arange(len(c_t)), index=c_t)
        keep = rows.abs_time_s.isin(c_t).to_numpy()
        rows = rows[keep]
        Xs.append(c_X[pos.loc[rows.abs_time_s.to_numpy()].to_numpy()])
        ys.append(rows.label.to_numpy()); sids.append(rows.seizure_id.to_numpy())
        ts.append(rows.abs_time_s.to_numpy()); recs.append(np.array([fname] * len(rows)))

    out = dict(X=np.concatenate(Xs), y=np.concatenate(ys), seizure_id=np.concatenate(sids),
               abs_time_s=np.concatenate(ts), recording=np.concatenate(recs))
    np.savez(os.path.join(args.out, f"{p}.npz"), **out)
    print(f"saved {args.out}/{p}.npz  X={out['X'].shape}  "
          f"({(out['y'] == 1).sum()} preictal, {(out['y'] == 0).sum()} interictal)")


if __name__ == "__main__":
    main()