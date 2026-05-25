"""Threshold 재조정 — 여러 선택 기준 비교.

현재: val MCC-max → 0.31 → test MCC 0.223, TPR 0.36, TNR 0.888 (비대칭)

시도:
  - MCC max (현재)
  - bAcc max (TPR+TNR 균형)
  - Youden J max (TPR - FPR)
  - F1 max
  - bAcc 와 TPR>=0.5 제약

각 기준 → test 에서 MCC + TPR + TNR 측정.
"""

from __future__ import annotations

import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, f1_score, matthews_corrcoef,
                              roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


def load_ensemble_probs(domain: str, df: pd.DataFrame):
    """모델 + FP 캐시로 prediction 계산. ensemble probability 반환."""
    from src.train_domain_models import fp_xy
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    meta = json.load(open(os.path.join(MODELS_DIR, domain, "ensemble_meta.json")))
    weights = np.array(meta["weights"])
    members = meta["members"]
    probs = []
    y_ref = None
    for name in members:
        kind, fp_name = name.split("_", 1)
        X, y, _, _ = fp_xy(fp_name, df, domain, db=db)
        sub_dir = os.path.join(MODELS_DIR, domain, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sub_dir, "model.pkl"))
        else:
            m = CatBoostClassifier(); m.load_model(os.path.join(sub_dir, "model.cbm"))
        p = m.predict_proba(X)[:, 1]
        probs.append(p)
        if y_ref is None: y_ref = y
    probs = np.array(probs).T
    score = probs @ weights
    return score, y_ref


def metrics_at(thr, y, score):
    pred = (score >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    return {
        "thr": float(thr),
        "tpr": tp/max(tp+fn,1), "tnr": tn/max(fp+tn,1),
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "bacc": float(balanced_accuracy_score(y, pred)),
        "acc": float(accuracy_score(y, pred)),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def best_thr_by(y, score, criterion="mcc"):
    grid = np.linspace(0.05, 0.95, 181)
    best, bv = grid[0], -1e9
    for t in grid:
        pred = (score >= t).astype(int)
        if criterion == "mcc":   v = matthews_corrcoef(y, pred)
        elif criterion == "bacc": v = balanced_accuracy_score(y, pred)
        elif criterion == "f1":   v = f1_score(y, pred, zero_division=0)
        elif criterion == "youden":
            cm = confusion_matrix(y, pred, labels=[1, 0])
            tp, fn = cm[0]; fp, tn = cm[1]
            tpr = tp/max(tp+fn,1); fpr = fp/max(fp+tn,1)
            v = tpr - fpr
        if v > bv: bv, best = v, t
    return float(best)


def run(domain: str):
    print(f"\n{'='*70}\n  Threshold sweep — {domain}\n{'='*70}")
    val = pd.read_csv(os.path.join(DATA_DIR, "val", f"{domain}.csv"))
    test = pd.read_csv(os.path.join(DATA_DIR, "test", f"{domain}.csv"))

    s_val, y_val = load_ensemble_probs(domain, val)
    s_test, y_test = load_ensemble_probs(domain, test)
    auc_v = roc_auc_score(y_val, s_val)
    auc_t = roc_auc_score(y_test, s_test)
    print(f"  val AUC {auc_v:.3f}  test AUC {auc_t:.3f}")
    print(f"  val pos {y_val.sum()}/{len(y_val)} ({100*y_val.mean():.1f}%) | test pos {y_test.sum()}/{len(y_test)} ({100*y_test.mean():.1f}%)")

    print(f"\n  {'기준':>10s} {'val_thr':>8s} {'test_thr':>9s}  ── val에서 선택한 thr로 test 평가 ──")
    print(f"  {'criterion':>10s} {'thr':>8s} {'mcc':>6s} {'tpr':>6s} {'tnr':>6s} {'bacc':>6s} {'f1':>6s}")

    rows = {}
    for c in ("mcc", "bacc", "youden", "f1"):
        thr = best_thr_by(y_val, s_val, c)
        m = metrics_at(thr, y_test, s_test)
        rows[c] = m
        print(f"  {c:>10s} {thr:>8.3f} {m['mcc']:>6.3f} {m['tpr']:>6.3f} {m['tnr']:>6.3f} {m['bacc']:>6.3f} {m['f1']:>6.3f}")

    # peek: test 자체에서 MCC max — 천장
    thr_peek = best_thr_by(y_test, s_test, "mcc")
    m_peek = metrics_at(thr_peek, y_test, s_test)
    print(f"  {'PEEK MCC max':>10s} {thr_peek:>8.3f} {m_peek['mcc']:>6.3f} {m_peek['tpr']:>6.3f} {m_peek['tnr']:>6.3f} {m_peek['bacc']:>6.3f} {m_peek['f1']:>6.3f}  (cheating — ceiling)")
    rows["peek_mcc"] = m_peek

    return {"auc_val": float(auc_v), "auc_test": float(auc_t),
            "n_test": int(len(y_test)), "pos_test": int(y_test.sum()),
            "criteria": rows}


def main():
    out = {}
    for d in ("vivo", "vitro"):
        out[d] = run(d)
    with open(os.path.join(PROJECT_ROOT, "results", "threshold_sweep.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/threshold_sweep.json")


if __name__ == "__main__":
    main()
