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
