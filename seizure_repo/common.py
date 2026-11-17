"""
Shared pieces for run_kfold.py and run_loso.py:
loading features, scaling, the models (DC-SA-EBiLSTM via PyTorch, SVM/KNN via
scikit-learn), window-level metrics, and the 10-of-30 alarm rule.
"""
import copy
import time

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

SKLEARN_MODELS = ("svm", "knn")


# ---------------------------------------------------------------- data
def load_features(path):
    d = np.load(path, allow_pickle=False)
    out = {k: d[k] for k in d.files}
    order = np.argsort(out["abs_time_s"], kind="stable")      # time order
    out = {k: v[order] for k, v in out.items()}
    out["X"] = out["X"].astype(np.float32)
    bad = ~np.isfinite(out["X"]).all(axis=1)
    if bad.any():
        print(f"  dropping {bad.sum()} windows with NaN/inf features")
        out = {k: v[~bad] for k, v in out.items()}
    return out


def scale(X, tr):
    """Standardize with mean/SD from the training windows only (no leakage)."""
    sc = StandardScaler().fit(X[tr])
    return sc.transform(X).astype(np.float32)


def undersample(idx, y, ratio, rng):
    """Keep all preictal windows in idx and `ratio` x as many interictal."""
    pre = idx[y[idx] == 1]
    inter = idx[y[idx] == 0]
    k = min(len(inter), int(ratio * len(pre)))
    return np.sort(np.concatenate([pre, rng.choice(inter, k, replace=False)]))


# ---------------------------------------------------------------- metrics
def window_metrics(y, prob, thr=0.5):
    pred = prob >= thr
    tp = np.sum(pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    fp = np.sum(pred & (y == 0)); fn = np.sum(~pred & (y == 1))
    auc = roc_auc_score(y, prob) if len(np.unique(y)) == 2 else np.nan
    return dict(ACC=(tp + tn) / len(y),
                SEN=tp / (tp + fn) if tp + fn else np.nan,
                SPE=tn / (tn + fp) if tn + fp else np.nan,
                AUC=auc, n_pre=int(np.sum(y == 1)), n_inter=int(np.sum(y == 0)))


# ---------------------------------------------------------------- alarm rule
def alarms(times, pred, win=30, k=10, refractory_s=0):
    """10-of-30 voting. times: window start times (s); pred: 0/1 per window.
    The vote at second t uses windows in (t-win, t]; it is only evaluated once
    a full `win`-second history exists (this reproduces the paper's earliest
    possible warning of 32.52 min). Missing seconds count as 'not preictal'.
    Returns the times at which a new alarm starts."""
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
    """Split sorted times into runs with no gap bigger than max_gap seconds."""
    cuts = np.flatnonzero(np.diff(times) > max_gap) + 1
    return np.split(np.arange(len(times)), cuts)


# ---------------------------------------------------------------- models
def fit_predict(name, Xtr, ytr, Xva, yva, Xte, class_weight=False, seed=0,
                torch_args=None, log=print):
    """Train on (Xtr, ytr), use (Xva, yva) for early stopping (PyTorch models),
    return P(preictal) for Xte."""
    if name == "svm":
        from sklearn.svm import SVC
        m = SVC(kernel="rbf", C=10, gamma="scale",
                class_weight="balanced" if class_weight else None).fit(Xtr, ytr)
        return 1 / (1 + np.exp(-m.decision_function(Xte)))   # >0.5 <=> class 1
    if name == "knn":
        from sklearn.neighbors import KNeighborsClassifier
        m = KNeighborsClassifier(n_neighbors=5, weights="distance").fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1]
    model = train_torch(name, Xtr, ytr, Xva, yva, class_weight=class_weight,
                        seed=seed, log=log, **(torch_args or {}))
    return predict_torch(model, Xte)


def _device():
    import torch
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_torch(variant, Xtr, ytr, Xva, yva, epochs=100, patience=30, batch=128,
                lr=1e-3, weight_decay=1e-4, step=80, gamma=0.1,
                class_weight=False, seed=0, log=print):
    """Paper settings: Adam(lr 1e-3, wd 1e-4), batch 128, <=100 epochs,
    early stopping (patience 30) on validation, StepLR(80, 0.1).
    Early stopping monitors validation AUC (the paper doesn't say which metric)."""
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
                                     drop_last=len(ds) > batch)   # BatchNorm needs >1
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
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch]).to(dev)
            out.append(torch.softmax(model(xb), dim=1)[:, 1].cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)
