"""
Experiment 1 of the paper: per-patient 10-fold cross-validation.

For each fold: 1 fold = test, the next fold = validation (the paper only says
"one part of the remaining data"), the other 8 = training.
Training is undersampled to 1 preictal : 2 interictal; validation and test keep
the real ratio. Features are standardized with training statistics only.

--split stratified   the paper's protocol (random windows; neighbouring seconds
                     of the same seizure end up in both train and test)
--split grouped      stricter check: whole seizures / whole interictal hours
                     stay together, so no near-duplicate leakage

Usage:
    python run_kfold.py features/chb01.npz --model dcsa
    python run_kfold.py features/chb01.npz --model svm --split grouped
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from common import fit_predict, load_features, scale, undersample, window_metrics

MODEL_ALIASES = {"dcsa": "full"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("features")
    ap.add_argument("--model", default="dcsa",
                    help="dcsa | svm | knn | any ABLATIONS key in model.py")
    ap.add_argument("--folds", type=int, default=10)
    ap.add_argument("--split", choices=["stratified", "grouped"], default="stratified")
    ap.add_argument("--ratio", type=float, default=2.0, help="interictal:preictal in training")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()
    name = MODEL_ALIASES.get(args.model, args.model)

    d = load_features(args.features)
    X, y = d["X"], d["y"].astype(int)
    patient = os.path.basename(args.features).replace(".npz", "")
    print(f"{patient}: {np.sum(y == 1)} preictal, {np.sum(y == 0)} interictal windows")

    rng = np.random.default_rng(args.seed)
    if args.split == "stratified":
        folds = [te for _, te in StratifiedKFold(args.folds, shuffle=True,
                                                 random_state=args.seed).split(X, y)]
    else:
        # group = seizure id for preictal, recording hour for interictal
        groups = np.where(y == 1, d["seizure_id"], 10_000 + d["abs_time_s"] // 3600)
        n = min(args.folds, len(np.unique(d["seizure_id"][y == 1])))
        folds = [te for _, te in StratifiedGroupKFold(n, shuffle=True,
                                                      random_state=args.seed).split(X, y, groups)]
        print(f"  grouped split: {n} folds (one per seizure at most)")

    rows = []
    for i, te in enumerate(folds):
        va = folds[(i + 1) % len(folds)]
        tr = np.setdiff1d(np.arange(len(y)), np.concatenate([te, va]))
        tr = undersample(tr, y, args.ratio, rng)
        Xs = scale(X, tr)
        print(f"fold {i + 1}/{len(folds)}: train {len(tr)} (pre {np.sum(y[tr])}), "
              f"val {len(va)}, test {len(te)} (pre {np.sum(y[te])})")
        prob = fit_predict(name, Xs[tr], y[tr], Xs[va], y[va], Xs[te], seed=args.seed + i,
                           torch_args=dict(epochs=args.epochs))
        m = window_metrics(y[te], prob)
        m["fold"] = i + 1
        rows.append(m)
        print("  " + "  ".join(f"{k} {m[k]:.4f}" for k in ["ACC", "SEN", "SPE", "AUC"]))

    res = pd.DataFrame(rows)
    mean = res[["ACC", "SEN", "SPE", "AUC"]].mean()
    print(f"\n{patient} {args.model} ({args.split}) mean over folds:")
    print("  " + "  ".join(f"{k} {100 * v:.2f}%" for k, v in mean.items()))
    os.makedirs(args.out, exist_ok=True)
    tag = f"{patient}_{args.model}_{args.split}_kfold"
    res.to_csv(os.path.join(args.out, tag + ".csv"), index=False)
    json.dump({k: float(v) for k, v in mean.items()},
              open(os.path.join(args.out, tag + "_mean.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
