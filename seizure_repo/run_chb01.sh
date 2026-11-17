#!/usr/bin/env bash
# Full pipeline for one patient. Usage:  bash run_chb01.sh [patient] [data_root]
# Choose the Python with:  PYTHON=python3.12 bash run_chb01.sh
set -euo pipefail
PY=${PYTHON:-python}
P=${1:-chb01}
ROOT=${2:-/mnt/windows/eeg/chbmit}

echo "== 1. label windows (seconds)"
"$PY" index_windows.py "$ROOT" "$P" --out index

echo "== 2. features for every labelled window (the slow step; resumable)"
"$PY" extract_features.py "$ROOT" "$P" --index index --out features

echo "== 3. paper's 10-fold experiment with DC-SA-EBiLSTM"
"$PY" run_kfold.py "features/$P.npz" --model dcsa

echo "== 4. paper's leave-one-seizure-out alarm experiment"
"$PY" run_loso.py "features/$P.npz" --index index --model dcsa

echo "== 5. stricter 10-fold check: whole seizures kept together"
"$PY" run_kfold.py "features/$P.npz" --model dcsa --split grouped

echo "== 6. baselines from the paper's Table 3 (SVM can take a while)"
"$PY" run_kfold.py "features/$P.npz" --model knn
"$PY" run_kfold.py "features/$P.npz" --model svm

echo "== done: see results/"