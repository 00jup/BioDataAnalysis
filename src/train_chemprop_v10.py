"""Chemprop v10 — 강화 학습.

차이점 vs v9:
  - ensemble-size 3 → 5
  - epochs 30 → 40
  - patience 5 → 8
  - rdkit_2d_normalized featurizer 추가 (분자 디스크립터 booster)
  - 메트릭 binary-mcc + roc
"""

from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHEMPROP_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v9")  # 데이터 재활용
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v10")
RESULTS = os.path.join(PROJECT_ROOT, "results")

PY = sys.executable
CHEMPROP_BIN = os.path.join(os.path.dirname(PY), "chemprop")


def train_domain(domain: str):
    print(f"\n{'='*70}\n  Chemprop v10 — {domain}\n{'='*70}")
    save_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(save_dir, exist_ok=True)
    combined_path = os.path.join(CHEMPROP_DATA, domain, "train_val.csv")
    te_path = os.path.join(CHEMPROP_DATA, domain, "test.csv")

    cmd = [
        CHEMPROP_BIN, "train",
        "-i", combined_path,
        "-s", "canonical_smiles",
        "--target-columns", "label",
        "-t", "classification",
        "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--splits-column", "split",
        "--ensemble-size", "5",
        "--epochs", "40",
        "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "-o", save_dir,
    ]
    log_path = os.path.join(save_dir, "train.log")
    print(f"  command: {' '.join(cmd[:8])} ... (rdkit_2d_normalized + ensemble 5 + epochs 40)")
    t0 = time.time()
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    # predict
    pred_path = os.path.join(save_dir, "test_pred.csv")
    cmd_pred = [
        CHEMPROP_BIN, "predict",
        "--test-path", te_path,
        "-s", "canonical_smiles",
        "--model-paths", save_dir,
        "--preds-path", pred_path,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ]
    print(f"  test 예측 중...")
    t0 = time.time()
    log_p = os.path.join(save_dir, "predict.log")
    with open(log_p, "w") as logf:
        r = subprocess.run(cmd_pred, stdout=logf, stderr=subprocess.STDOUT)
    print(f"  예측 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_p) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    # 평가
    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score)
    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    te = pd.read_csv(te_path)
    m = te.merge(pred_df, on="canonical_smiles", how="left")
    y = m["label"].to_numpy(int); p = m["pred"].to_numpy(float)
    auc = roc_auc_score(y, p)
    bt, bm = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        mcc = matthews_corrcoef(y, (p >= t).astype(int))
        if mcc > bm: bm, bt = mcc, t
    pred = (p >= bt).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    print(f"\n  [{domain} v10 test] N={len(y)} (양성 {y.sum()})")
    print(f"  AUC {auc:.3f}  MCC {bm:.3f}  thr {bt:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}  F1 {f1_score(y, pred):.3f}")
    return {"auc": float(auc), "mcc": float(bm), "threshold": float(bt),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1))}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for d in ("vivo", "vitro"):
        r = train_domain(d)
        if r: out[d] = r
    with open(os.path.join(RESULTS, "chemprop_v10.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/chemprop_v10.json")

    # v9 vs v10 비교
    if os.path.exists(os.path.join(RESULTS, "chemprop_v9.json")):
        v9 = json.load(open(os.path.join(RESULTS, "chemprop_v9.json")))
        print(f"\n{'='*70}\n  Chemprop v9 vs v10 비교\n{'='*70}")
        for d in ("vivo", "vitro"):
            if d not in v9 or d not in out: continue
            print(f"\n[{d}]")
            print(f"  {'metric':<8s} {'v9':>8s} {'v10':>8s} {'Δ':>8s}")
            for k in ("auc", "mcc", "tpr", "tnr"):
                v9v = v9[d].get(k, 0); v10v = out[d].get(k, 0)
                print(f"  {k:<8s} {v9v:>8.3f} {v10v:>8.3f} {v10v-v9v:>+8.4f}")


if __name__ == "__main__":
    main()
