"""Chemprop v31 — class expansion 데이터로 재학습.

v27 hyperparameter 그대로:
  ensemble 15, hidden 600, BCE, scaffold split, seed 1111
Data: data/chemprop_scaffold_v3/
Models: models/chemprop_v31_class_expanded/
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v3")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v31_class_expanded")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def train(domain):
    print(f"\n=== Chemprop v31 — {domain} (class expanded) ===")
    save_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(save_dir, exist_ok=True)
    splits_file = os.path.join(DATA_DIR, domain, "splits.json")
    csv_path = os.path.join(DATA_DIR, domain, "all.csv")

    if not os.path.exists(csv_path):
        print(f"  ERROR: {csv_path} 없음.")
        return None

    cmd = [
        CHEMPROP_BIN,
        "train",
        "-i",
        csv_path,
        "-s",
        "canonical_smiles",
        "--target-columns",
        "label",
        "-t",
        "classification",
        "-l",
        "bce",
        "--metrics",
        "binary-mcc",
        "roc",
        "--splits-file",
        splits_file,
        "--ensemble-size",
        "15",
        "--message-hidden-dim",
        "600",
        "--epochs",
        "40",
        "--patience",
        "8",
        "--molecule-featurizers",
        "v1_rdkit_2d_normalized",
        "--accelerator",
        "cpu",
        "-o",
        save_dir,
    ]
    log_path = os.path.join(save_dir, "train.log")
    t0 = time.time()
    print(f"  학습 시작 ({time.strftime('%H:%M:%S')})")
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time() - t0) / 60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_path) as f:
            print(f.read()[-1500:])
        return None

    # test 평가
    # split 의 test idx 로 직접 평가
    all_df = pd.read_csv(csv_path)
    splits = json.load(open(splits_file))
    s = splits[0] if isinstance(splits, list) else splits
    test_df = all_df.iloc[s["test"]].reset_index(drop=True)
    test_csv = os.path.join(save_dir, "test_data.csv")
    test_df[["canonical_smiles"]].to_csv(test_csv, index=False)

    pred_path = os.path.join(save_dir, "test_pred.csv")
    cmd_p = [
        CHEMPROP_BIN,
        "predict",
        "--test-path",
        test_csv,
        "-s",
        "canonical_smiles",
        "--model-paths",
        save_dir,
        "--preds-path",
        pred_path,
        "--molecule-featurizers",
        "v1_rdkit_2d_normalized",
        "--accelerator",
        "cpu",
    ]
    log_p = os.path.join(save_dir, "predict.log")
    with open(log_p, "w") as f:
        subprocess.run(cmd_p, stdout=f, stderr=subprocess.STDOUT)

    from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    m = test_df.merge(pred_df, on="canonical_smiles", how="left")
    y = m["label"].to_numpy(int)
    p = m["pred"].to_numpy(float)
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    auc = roc_auc_score(y, p)
    bt, bm = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        mc = matthews_corrcoef(y, (p >= t).astype(int))
        if mc > bm:
            bm, bt = mc, t
    pred = (p >= bt).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]
    fp, tn = cm[1]
    out = {
        "auc": float(auc),
        "mcc": float(bm),
        "threshold": float(bt),
        "tpr": float(tp / max(tp + fn, 1)),
        "tnr": float(tn / max(fp + tn, 1)),
        "n_test": int(valid.sum()),
    }
    print(
        f"  [{domain} v31] AUC {auc:.3f}  MCC {bm:.3f}  "
        f"TPR {out['tpr']:.3f}  TNR {out['tnr']:.3f}  N={out['n_test']}"
    )
    return out


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    out["vivo"] = train("vivo")
    # vitro 는 데이터 없을 수 있음
    vitro_csv = os.path.join(DATA_DIR, "vitro", "all.csv")
    if os.path.exists(vitro_csv):
        out["vitro"] = train("vitro")
    with open(os.path.join(RESULTS, "chemprop_v31.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n저장: results/chemprop_v31.json")


if __name__ == "__main__":
    main()
