# Reproducing DC-SA-EBiLSTM (Wang et al., 2026) on CHB-MIT

## Setup (once)

```bash
pip install numpy scipy pandas scikit-learn joblib pywavelets torch
python test_setup.py        # < 1 minute; must end with "All good"
```

## Run everything for chb01

```bash
bash run_chb01.sh chb01 /mnt/windows/eeg/chbmit
```

Or step by step:

| Step | Command | Output | Time (8-core laptop, rough) |
|---|---|---|---|
| 1. Label windows | `python index_windows.py /mnt/windows/eeg/chbmit chb01` | `index/chb01_index.csv`, `index/chb01_seizures.csv` | seconds |
| 2. Features | `python extract_features.py /mnt/windows/eeg/chbmit chb01` | `features/chb01.npz` (n × 432) | ~0.17 s per window per core; tens of minutes for chb01. Resumable |
| 3. 10-fold (paper's Table 5) | `python run_kfold.py features/chb01.npz --model dcsa` | `results/chb01_dcsa_stratified_kfold*.{csv,json}` | minutes on GPU, longer on CPU |
| 4. Leave-one-seizure-out (Table 6) | `python run_loso.py features/chb01.npz --model dcsa` | `results/chb01_dcsa_loso*.{csv,json}` | longer than step 3 (no undersampling) |
| 5. Stricter check | `python run_kfold.py features/chb01.npz --model dcsa --split grouped` | same pattern | like step 3 |
| 6. Baselines (Table 3) | `--model svm`, `--model knn`, `--model CNN`, `--model LSTM`, `--model BiLSTM`, `--model SA-EBiLSTM`, `--model CNN-EBiLSTM` | | SVM is slow on large sets |
| Ablations (Table 4) | `--model DC-SA-BiLSTM`, `no_gate`, `no_residual_proj`, `no_layernorm`, `no_self_attention`, `standard_conv1d`, `no_posenc` | | |

Quick first pass: `python extract_features.py ... --interictal-ratio 3` keeps 3 interictal
windows per preictal one. Fine for step 3, **not** for step 4 (false alarms per hour need
all interictal data).

## What to compare with the paper (chb01)

- Table 1: chb01 has 7 seizures. Step 1 prints how many of each seizure's 1,800 preictal
  windows exist in the recordings.
- Table 5 (10-fold): the paper's per-patient row for chb01.
- Table 6 (leave-one-seizure-out): chb01 had 6 test events, 100% event sensitivity,
  0.000 FPR/h, 29.16 min mean warning.

## Choices the paper doesn't specify (all marked NOTE / in --help)

| Detail | What this code does | Flag to change |
|---|---|---|
| Filtering | zero-phase (forward-backward), 10 s edge padding; notch Q = 30 | `features.py` |
| Wavelet sub-bands | each band rebuilt to 256 samples (only reading where kmax = 128 works) | `features.py` |
| 10-fold validation set | the fold after the test fold | `run_kfold.py` |
| Early stopping metric | validation AUC | `common.py` |
| Leave-one-seizure-out interictal | split into contiguous time blocks, one per seizure | `run_loso.py` |
| "Usable" seizure | at least 900 of 1,800 preictal windows | `--min-preictal` |
| Alarm vote | needs a full 30-s history (reproduces the paper's 32.52-min maximum warning) | `--vote-window`, `--vote-k` |
| Refractory period after an alarm | none | `--refractory-min` |
| Seizures close together | preictal windows overlapping a seizure are dropped; nothing else | `index_windows.py` |
| Labelling windows | 33-3 min preictal, 4 h interictal gap | `--preictal-start-min` etc. |
