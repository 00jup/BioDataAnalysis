"""Chemprop v9 — vivo 모델 결과 평가 + vitro 학습.

이미 vivo 학습 끝남. KeyError 만 수정 → 빠른 재평가.
그 후 vitro 학습.
"""

from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_chemprop_v9 import train_domain

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v9")
DATA_DIR   = os.path.join(PROJECT_ROOT, "data", "chemprop_v9")
RESULTS    = os.path.join(PROJECT_ROOT, "results")


def evaluate_only(domain: str):
    """이미 예측 결과 있으면 그것만 평가."""
    pred_path = os.path.join(MODELS_DIR, domain, "test_pred.csv")
    te_path   = os.path.join(DATA_DIR,   domain, "test.csv")
    if not os.path.exists(pred_path):
        return None
    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score)
    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    te = pd.read_csv(te_path)
    merged = te.merge(pred_df, on="canonical_smiles", how="left")
    y = merged["label"].to_numpy(int)
    p = merged["pred"].to_numpy(float)
    auc = roc_auc_score(y, p)
    best_t, best_m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best_m: best_m, best_t = m, t
    pred = (p >= best_t).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    print(f"  [{domain}] N={len(y)} (양성 {y.sum()})")
    print(f"  AUC {auc:.3f}  MCC {best_m:.3f}  threshold {best_t:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    return {"auc": float(auc), "mcc": float(best_m), "threshold": float(best_t),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
            "n_test": int(len(y))}


def main():
    out = {}
    print("=== vivo (이미 학습된 모델) 평가 ===")
    r = evaluate_only("vivo")
    if r: out["vivo"] = r

    print("\n=== vitro 학습 + 평가 ===")
    r = train_domain("vitro")
    if r: out["vitro"] = r

    with open(os.path.join(RESULTS, "chemprop_v9.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/chemprop_v9.json")

    # v1 vs chemprop_v9 비교
    print(f"\n{'='*70}\n  v1 vs chemprop_v9 비교\n{'='*70}")
    for d in ("vivo", "vitro"):
        v1 = json.load(open(os.path.join(PROJECT_ROOT, "models", d, "ensemble_meta.json")))["test_metrics"]
        v9 = out.get(d, {})
        print(f"\n[{d}]")
        print(f"  {'metric':<8s} {'v1 RF/CB':>10s} {'chemprop_v9':>12s} {'Δ':>8s}")
        for k in ("auc", "mcc", "tpr", "tnr"):
            v1v = v1.get(k, 0); v9v = v9.get(k, 0)
            print(f"  {k:<8s} {v1v:>10.3f} {v9v:>12.3f} {v9v-v1v:>+8.4f}")


if __name__ == "__main__":
    main()
