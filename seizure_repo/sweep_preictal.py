"""
Preictal-length sweep: how do the results change if the preictal period is
[onset - S min, onset - END min) for S = 13, 18, ..., 63 (preictal LENGTH 10 ... 60 min, ending 3 min before onset) instead of only 33?

Features do not depend on the label, so they are extracted ONCE for the widest
period and every S is just a relabelling of that file:

    # 1. index + features for the widest period (windows 63 -> 3 min before onset)
    python index_windows.py /mnt/windows/eeg/chbmit chb01 --preictal-start-min 63 --out index
    python extract_features.py /mnt/windows/eeg/chbmit chb01 --index index --out features
    #    (existing per-recording caches in features/chb01/ are reused; only the
    #     extra windows are computed)

    # 2. sweep (each S runs run_loso.py and a grouped 10-fold via run_kfold.py)
    python sweep_preictal.py features/chb01.npz --index index --model dcsa --seeds 0 1 2

For every S this script
  * keeps all interictal windows (they do not depend on S),
  * keeps preictal windows with  END*60 < onset - t <= S*60,
  * writes a temporary features file + seizure table for that S,
  * calls run_loso.py on the SAME set of seizures for every S (so the interictal test
    blocks are identical and the S values are comparable) and, if asked, run_kfold.py --split grouped,
  * collects everything in sweep/summary.csv and sweep/sweep.png.

Alarm settings for the sweep: 10-of-30 vote, refractory 30 min with re-arming, so a
classifier that stays "preictal" for hours is charged one false alarm per 30 min
(the original code counted it once). Use the same value for every S so FPR/h is comparable.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def relabel(d, onset, offset, start_min, end_min):
    """Boolean mask of windows to keep for preictal = (onset - max_lead, onset - end],
    capping lookback at (onset_i - offset_{i-1}) - end_min for close seizures."""
    max_lead = np.full(len(onset), start_min * 60.0)
    if len(onset) > 1:
        max_lead[1:] = np.minimum(max_lead[1:], (onset[1:] - offset[:-1]) - end_min * 60.0)
    lead = onset[d["seizure_id"]] - d["abs_time_s"]
    pre = (d["y"] == 1) & (lead <= max_lead[d["seizure_id"]]) & (lead >= end_min * 60 + 1)
    return (d["y"] == 0) | pre


def run(cmd):
    print("  $", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=HERE)
    if r.returncode != 0:
        print(f"  !! command failed ({r.returncode}); continuing with the next one")
    return r.returncode == 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features", help="features/chb01.npz built from the WIDEST preictal period")
    ap.add_argument("--index", default="index")
    ap.add_argument("--starts", type=float, nargs="+",
                    default=[13 + 5 * i for i in range(11)],     # 13 ... 63  = lengths 10 ... 60 min
                    help="minutes before onset where the preictal period starts")
    ap.add_argument("--end", type=float, default=3,
                    help="minutes before onset where it ends (prediction horizon)")
    ap.add_argument("--model", default="dcsa")
    ap.add_argument("--protocols", nargs="+", default=["loso"],
                    choices=["loso", "grouped"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--refractory-min", type=float, default=30)
    ap.add_argument("--min-frac", type=float, default=0.5,
                    help="a seizure is usable if it has this fraction of the SHORTEST period's "
                         "nominal window count (0.5 x 600 = 300 windows for a 10-min period); "
                         "the same absolute threshold is used for every S")
    ap.add_argument("--per-s-seizures", action="store_true",
                    help="let the usable seizures differ between S (default: use only the "
                         "seizures usable at every S)")
    ap.add_argument("--out", default="sweep")
    ap.add_argument("--keep-files", action="store_true", help="keep the per-S feature files")
    ap.add_argument("--force", action="store_true", help="recompute finished runs")
    args = ap.parse_args()

    patient = os.path.basename(args.features).replace(".npz", "")
    raw = np.load(args.features, allow_pickle=False)
    d = {k: raw[k] for k in raw.files}
    sz = pd.read_csv(os.path.join(args.index, f"{patient}_seizures.csv"))
    onset = sz.set_index("seizure_id").onset_s.sort_index().to_numpy()
    offset = sz.set_index("seizure_id").offset_s.sort_index().to_numpy()

    pre = d["y"] == 1
    lead = (onset[d["seizure_id"]] - d["abs_time_s"])[pre]
    print(f"{patient}: features cover preictal windows from {lead.max() / 60:.1f} to "
          f"{lead.min() / 60:.1f} min before onset")
    if max(args.starts) * 60 > lead.max() + 120:
        sys.exit(f"--starts goes up to {max(args.starts)} min but the features only reach "
                 f"{lead.max() / 60:.1f} min. Re-run index_windows.py with "
                 f"--preictal-start-min {max(args.starts):g} and extract_features.py.")
    if args.end * 60 + 1 < lead.min():
        sys.exit(f"--end {args.end} is closer to onset than the features go "
                 f"({lead.min() / 60:.1f} min); re-run index_windows.py with a smaller "
                 "--preictal-end-min.")

    tmp = os.path.join(args.out, "tmp")
    os.makedirs(tmp, exist_ok=True)

    # ---- pass 1: windows per seizure for every S -> which seizures are usable everywhere
    min_pre = int(args.min_frac * (min(args.starts) - args.end) * 60)
    counts = {}
    for S in args.starts:
        k = relabel(d, onset, offset, S, args.end)
        sid_k = d["seizure_id"][k & (d["y"] == 1)]
        counts[S] = {int(a): int(n) for a, n in zip(*np.unique(sid_k, return_counts=True))}
    usable_by_S = {S: sorted(a for a, n in c.items() if n >= min_pre) for S, c in counts.items()}
    common = sorted(set.intersection(*[set(u) for u in usable_by_S.values()]))
    print(f"\nusable seizures (>= {min_pre} preictal windows) per S:")
    for S in args.starts:
        print(f"  S={S:g}: {usable_by_S[S]}   windows {counts[S]}")
    if not args.per_s_seizures:
        print(f"-> using the seizures usable at every S: {common}")
        if len(common) < 3:
            sys.exit("fewer than 3 seizures usable at every S; lower --min-frac or use --per-s-seizures")

    rows = []
    for S in args.starts:
        keep = relabel(d, onset, offset, S, args.end)
        sub = {k: v[keep] for k, v in d.items()}
        y, sid = sub["y"], sub["seizure_id"]
        per_sz = counts[S]
        usable = usable_by_S[S] if args.per_s_seizures else common
        tag = f"{patient}_pre{S:g}"
        print(f"\n=== preictal {S:g} -> {args.end:g} min before onset ({S - args.end:g} min long): "
              f"{(y == 1).sum()} preictal / {(y == 0).sum()} interictal windows; "
              f"per seizure {per_sz}; using {usable}")
        fpath = os.path.join(tmp, tag + ".npz")
        np.savez(fpath, **sub)
        s2 = sz.copy()
        s2["preictal_start_s"] = s2.onset_s - int(S * 60)
        s2["preictal_end_s"] = s2.onset_s - int(args.end * 60)
        s2.to_csv(os.path.join(tmp, f"{tag}_seizures.csv"), index=False)

        for seed in args.seeds:
            odir = os.path.join(args.out, f"seed{seed}")
            os.makedirs(odir, exist_ok=True)
            row = dict(start_min=S, end_min=args.end, seed=seed,
                       n_pre=int((y == 1).sum()), n_inter=int((y == 0).sum()),
                       n_usable_seizures=len(usable), windows_per_seizure=json.dumps(per_sz))
            if "loso" in args.protocols:
                f = os.path.join(odir, f"{tag}_{args.model}_loso_summary.json")
                if args.force or not os.path.exists(f):
                    run([sys.executable, "run_loso.py", fpath, "--index", tmp,
                         "--model", args.model, "--min-preictal", str(min_pre),
                         "--only-seizures", *[str(u) for u in usable],
                         "--refractory-min", str(args.refractory_min), "--rearm",
                         "--epochs", str(args.epochs), "--seed", str(seed), "--out", odir])
                if os.path.exists(f):
                    j = json.load(open(f))
                    pw = j["pooled_window_metrics"]
                    row.update(loso_events=j["n_events"], loso_event_sens=j["event_sensitivity"],
                               loso_fpr_per_h=j["fpr_per_h"], loso_warning_min=j["mean_warning_min"],
                               loso_AUC=pw["AUC"], loso_SEN=pw["SEN"], loso_SPE=pw["SPE"])
            if "grouped" in args.protocols:
                f = os.path.join(odir, f"{tag}_{args.model}_grouped_kfold_mean.json")
                if args.force or not os.path.exists(f):
                    run([sys.executable, "run_kfold.py", fpath, "--model", args.model,
                         "--split", "grouped", "--epochs", str(args.epochs),
                         "--seed", str(seed), "--out", odir])
                if os.path.exists(f):
                    j = json.load(open(f))
                    row.update({f"grouped_{k}": v for k, v in j.items()})
            rows.append(row)
        if not args.keep_files:
            os.remove(fpath)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(args.out, "summary_all_seeds.csv"), index=False)
    num = [c for c in res.columns if c.startswith(("loso_", "grouped_")) or c in ("n_pre",)]
    num = [c for c in num if pd.api.types.is_numeric_dtype(res[c])]
    g = res.groupby("start_min")
    mean, sd = g[num].mean(), g[num].std()
    summ = mean.copy()
    summ.columns = [c + "_mean" for c in num]
    if len(args.seeds) > 1:
        for c in num:
            summ[c + "_sd"] = sd[c]
    summ["n_usable_seizures"] = g.n_usable_seizures.first()
    summ.reset_index().to_csv(os.path.join(args.out, "summary.csv"), index=False)
    pd.set_option("display.width", 200, "display.max_columns", 30)
    print("\n=== summary (mean over seeds) ===")
    print(mean.round(3).to_string())

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for col, lab in [("loso_AUC", "LOSO pooled AUC"), ("grouped_AUC", "grouped 10-fold AUC")]:
            if col in mean:
                ax[0].errorbar(mean.index, mean[col], yerr=sd[col] if len(args.seeds) > 1 else None,
                               marker="o", label=lab, capsize=3)
        ax[0].set_ylabel("AUC"); ax[0].legend()
        for col, lab in [("loso_event_sens", "event sensitivity"), ("loso_SPE", "window specificity")]:
            if col in mean:
                ax[1].errorbar(mean.index, mean[col], yerr=sd[col] if len(args.seeds) > 1 else None,
                               marker="o", label=lab, capsize=3)
        ax[1].set_ylabel("LOSO"); ax[1].legend()
        if "loso_fpr_per_h" in mean:
            ax[2].errorbar(mean.index, mean["loso_fpr_per_h"],
                           yerr=sd["loso_fpr_per_h"] if len(args.seeds) > 1 else None,
                           marker="o", color="tab:red", capsize=3)
        ax[2].set_ylabel("LOSO false alarms / h")
        for a in ax:
            a.set_xlabel(f"preictal start (min before onset); end = {args.end:g} min"); a.grid(alpha=.3)
        fig.suptitle(f"{patient}, {args.model}: effect of preictal length")
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "sweep.png"), dpi=140)
    except Exception as e:                                    # plotting is optional
        print("  (no plot:", e, ")")
    if not args.keep_files:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
