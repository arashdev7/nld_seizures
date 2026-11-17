"""
Step 1 of data prep: decide which 1-second windows are preictal / interictal.

Reads only the chbXX-summary.txt file (no EEG is loaded), puts every recording
on one continuous clock, and writes a table with one row per usable window:

    file, start_sample, abs_time_s, label (1 = preictal, 0 = interictal), seizure_id

Paper's definitions (Wang et al. 2026):
    preictal    = 33 min to 3 min before a seizure onset (30 min = 1800 windows)
    interictal  = at least 4 h away from any seizure
Everything else (ictal, the 3 min just before onset, the zone between the two
classes) is discarded.

Usage:
    python index_windows.py /mnt/windows/eeg/chbmit chb01 --out index
"""
import argparse
import os
import re
import sys

import pandas as pd

FS = 256
PREICTAL_START = 33 * 60      # seconds before onset
PREICTAL_END = 3 * 60         # seconds before onset (seizure prediction horizon)
INTERICTAL_GAP = 4 * 3600     # seconds from any seizure

REQUIRED = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ",
]


def _clock(s):
    """'13:43:04' -> seconds. Hours can be >= 24 in some CHB-MIT summaries."""
    h, m, sec = (int(v) for v in s.strip().split(":"))
    return h * 3600 + m * 60 + sec


def parse_summary(path):
    """Return a list of recordings, each with clock times, seizures and channels."""
    text = open(path).read()
    channels = []
    recs = []
    cur = None
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
            cur = dict(file=m.group(1), start=None, end=None, seizures=[],
                       channels=list(channels))
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
    """Put all recordings on one clock (seconds since the first recording's
    midnight), handling midnight roll-overs."""
    day = 0
    prev_end = None
    for r in recs:
        if r["start"] is None:
            raise ValueError(f"{r['file']}: no start time in summary "
                             "(chb24's summary has none; the paper doesn't use chb24)")
        start = r["start"] + day * 86400
        if prev_end is not None:
            while start < prev_end - 60:          # clock went past midnight
                day += 1
                start += 86400
        end = r["end"] + day * 86400
        while end <= start:                        # recording itself crossed midnight
            day += 1
            end += 86400
        r["abs_start"], r["abs_end"] = start, end
        r["abs_seizures"] = [(start + a, start + b) for a, b in r["seizures"]]
        prev_end = end
    return recs


def has_channels(r):
    chans = set(r["channels"])
    return all(c in chans for c in REQUIRED)


def build_index(recs, pre_start=PREICTAL_START, pre_end=PREICTAL_END,
                gap=INTERICTAL_GAP):
    seizures = sorted(s for r in recs for s in r["abs_seizures"])
    rows = []
    for r in recs:
        if not has_channels(r):
            print(f"  skip {r['file']}: missing some of the 18 channels", file=sys.stderr)
            continue
        n_win = (r["abs_end"] - r["abs_start"])           # 1-s windows in this file
        for w in range(n_win):
            t0 = r["abs_start"] + w                       # window covers [t0, t0+1)
            t1 = t0 + 1
            label, sid = None, -1
            # preictal: inside [onset-33min, onset-3min) of some seizure,
            # and not overlapping any seizure itself
            for i, (on, off) in enumerate(seizures):
                if on - pre_start <= t0 and t1 <= on - pre_end:
                    label, sid = 1, i
                    break
            if label == 1:
                if any(t0 < off and on < t1 for on, off in seizures):
                    label = None
            else:
                # interictal: at least 4 h from every seizure (onset and offset)
                if all(t1 <= on - gap or t0 >= off + gap
                       for on, off in seizures):
                    label = 0
            if label is not None:
                rows.append((r["file"], w * FS, t0, label, sid))
    return pd.DataFrame(rows, columns=["file", "start_sample", "abs_time_s",
                                       "label", "seizure_id"])


def report(recs, idx, n_pre=PREICTAL_START - PREICTAL_END):
    seizures = sorted(s for r in recs for s in r["abs_seizures"])
    hours = sum(r["abs_end"] - r["abs_start"] for r in recs) / 3600
    print(f"  {len(recs)} recordings, {hours:.1f} h, {len(seizures)} seizures")
    pre = idx[idx.label == 1]
    for i, (on, off) in enumerate(seizures):
        n = (pre.seizure_id == i).sum()
        gap = (on - seizures[i - 1][1]) / 60 if i else float("nan")
        print(f"  seizure {i}: onset t={on}s, {n:4d}/{n_pre} preictal windows"
              f"  (minutes since previous seizure: {gap:.0f})")
    print(f"  preictal windows:   {len(pre)}")
    print(f"  interictal windows: {(idx.label == 0).sum()} "
          f"({(idx.label == 0).sum() / 3600:.1f} h)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="folder containing chb01/, chb02/, ...")
    ap.add_argument("patients", nargs="+", help="e.g. chb01 chb05")
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
        report(recs, idx, pre_start - pre_end)
        idx.to_csv(os.path.join(args.out, f"{p}_index.csv"), index=False)
        # seizure table (absolute seconds), used for warning times in run_loso.py
        sz = sorted(s for r in recs for s in r["abs_seizures"])
        pd.DataFrame({"seizure_id": range(len(sz)),
                      "onset_s": [a for a, _ in sz], "offset_s": [b for _, b in sz],
                      "preictal_start_s": [a - pre_start for a, _ in sz],
                      "preictal_end_s": [a - pre_end for a, _ in sz]}
                     ).to_csv(os.path.join(args.out, f"{p}_seizures.csv"), index=False)


if __name__ == "__main__":
    main()
