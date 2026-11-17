"""
Experiment 2 of the paper: event-level leave-one-seizure-out with alarms.

For each usable seizure S:
    test       = S's preictal windows  + one block of interictal data
    validation = the next seizure's preictal windows + the next interictal block
    training   = all other seizures' preictal windows + the other interictal blocks
Interictal data is cut into as many contiguous time blocks as there are
seizures (the paper only says "clean interictal blocks"). Training uses a
class-weighted loss instead of undersampling.

Alarm: >= 10 of the last 30 one-second windows predicted preictal.
    event sensitivity = share of test seizures with an alarm in their preictal period
    FPR/h             = alarms during interictal test data / interictal test hours
    warning time      = seizure onset - first alarm, in minutes

Needs ALL interictal windows (run extract_features.py without
--interictal-ratio), otherwise FPR/h is meaningless.

Usage:
    python run_loso.py features/chb01.npz --index index --model dcsa
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from common import alarms, fit_predict, load_features, scale, split_segments, window_metrics

MODEL_ALIASES = {"dcsa": "full"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--index", default="index", help="folder with <patient>_seizures.csv")
    ap.add_argument("--model", default="dcsa")
    ap.add_argument("--min-preictal", type=int, default=900,
                    help="a seizure is usable if it has at least this many preictal windows")
    ap.add_argument("--vote-window", type=int, default=30)
    ap.add_argument("--vote-k", type=int, default=10)
    ap.add_argument("--refractory-min", type=float, default=0,
                    help="ignore new alarms this long after one (paper: not stated)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    name = MODEL_ALIASES.get(args.model, args.model)

    patient = os.path.basename(args.features).replace(".npz", "")
    d = load_features(args.features)
    X, y, sid, t = d["X"], d["y"].astype(int), d["seizure_id"], d["abs_time_s"]
    sz = pd.read_csv(os.path.join(args.index, f"{patient}_seizures.csv")).set_index("seizure_id")

    counts = pd.Series(sid[y == 1]).value_counts()
    usable = sorted(int(s) for s in counts.index if counts[s] >= args.min_preictal)
    print(f"{patient}: {len(counts)} seizures with preictal data, "
          f"{len(usable)} usable (>= {args.min_preictal} windows): {usable}")
    if len(usable) < 3:
        raise SystemExit("need at least 3 usable seizures (the paper's inclusion rule)")

    inter = np.flatnonzero(y == 0)                       # already in time order
    blocks = np.array_split(inter, len(usable))
    if len(inter) and np.median(np.diff(t[inter])) > 1.5:
        print("  WARNING: interictal windows look subsampled; FPR/h will be wrong. "
              "Re-run extract_features.py without --interictal-ratio.")
    pre_of = {s: np.flatnonzero((y == 1) & (sid == s)) for s in usable}

    rows, false_alarms, inter_hours, all_y, all_p = [], 0, 0.0, [], []
    for i, s in enumerate(usable):
        v = usable[(i + 1) % len(usable)]
        te = np.concatenate([pre_of[s], blocks[i]])
        va = np.concatenate([pre_of[v], blocks[(i + 1) % len(usable)]])
        others = [pre_of[u] for u in usable if u not in (s, v)]
        others += [b for j, b in enumerate(blocks) if j not in (i, (i + 1) % len(usable))]
        tr = np.sort(np.concatenate(others))
        Xs = scale(X, tr)
        print(f"\nround {i + 1}/{len(usable)}: test seizure {s}, val seizure {v}; "
              f"train {len(tr)} (pre {y[tr].sum()}), test {len(te)}")
        prob = fit_predict(name, Xs[tr], y[tr], Xs[va], y[va], Xs[te], class_weight=True,
                           seed=args.seed + i, torch_args=dict(epochs=args.epochs))
        pred = (prob >= 0.5).astype(float)
        all_y.append(y[te]); all_p.append(prob)

        # --- the held-out seizure: was there an alarm in its preictal period?
        n_pre = len(pre_of[s])
        a = alarms(t[te][:n_pre], pred[:n_pre], args.vote_window, args.vote_k)
        onset = sz.loc[s, "onset_s"]
        detected = len(a) > 0
        warning = (onset - a[0]) / 60 if detected else np.nan

        # --- interictal test block: count false alarms per continuous segment
        ti, pi = t[te][n_pre:], pred[n_pre:]
        fa = 0
        for seg in split_segments(ti, max_gap=1):
            fa += len(alarms(ti[seg], pi[seg], args.vote_window, args.vote_k,
                             int(args.refractory_min * 60)))
        hours = len(ti) / 3600
        false_alarms += fa
        inter_hours += hours
        m = window_metrics(y[te], prob)
        rows.append(dict(test_seizure=s, detected=detected, warning_min=warning,
                         false_alarms=fa, interictal_h=hours, **m))
        print(f"  detected={detected}  warning={warning:.2f} min  "
              f"false alarms={fa} in {hours:.2f} h  window AUC={m['AUC']:.4f}")

    res = pd.DataFrame(rows)
    summary = dict(patient=patient, model=args.model, n_events=len(res),
                   event_sensitivity=float(res.detected.mean()),
                   fpr_per_h=false_alarms / inter_hours if inter_hours else float("nan"),
                   mean_warning_min=float(res.warning_min.mean()),
                   pooled_window_metrics=window_metrics(np.concatenate(all_y),
                                                        np.concatenate(all_p)))
    print(f"\n{patient} {args.model} leave-one-seizure-out:")
    print(f"  events {summary['n_events']}, event sensitivity "
          f"{100 * summary['event_sensitivity']:.2f}%, FPR/h {summary['fpr_per_h']:.3f}, "
          f"mean warning {summary['mean_warning_min']:.2f} min")
    os.makedirs(args.out, exist_ok=True)
    tag = f"{patient}_{args.model}_loso"
    res.to_csv(os.path.join(args.out, tag + ".csv"), index=False)
    json.dump(summary, open(os.path.join(args.out, tag + "_summary.json"), "w"),
              indent=1, default=float)


if __name__ == "__main__":
    main()
