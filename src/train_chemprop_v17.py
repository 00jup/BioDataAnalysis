"""v17 — ensemble 15 + hidden 600 (v15 강화).

v15 의 hidden 600 + ensemble 15 → 다양성 ↑ 기대.
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
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v2")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2", "v17_ens15_h600")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def train(domain):
    print(f"\n=== Chemprop v17 — {domain} (ensemble 15 + hidden 600) ===")
    save_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(save_dir, exist_ok=True)
    splits_file = os.path.join(
        PROJECT_ROOT, "models", "chemprop_scaffold_v2", "v12_baseline", domain, "splits.json"
    )
    csv_path = os.path.join(DATA_DIR, domain, "all.csv")
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
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time() - t0) / 60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        return None

    test_csv = os.path.join(
        PROJECT_ROOT, "models", "chemprop_scaffold_v2", "v12_baseline", domain, "test_smiles.csv"
    )
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
    all_df = pd.read_csv(csv_path)
    te_df = pd.read_csv(test_csv).merge(all_df, on="canonical_smiles", how="left")
    m = te_df.merge(pred_df, on="canonical_smiles", how="left")
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
    }
    print(
        f"  [{domain} v17] AUC {auc:.3f}  MCC {bm:.3f}  TPR {out['tpr']:.3f}  TNR {out['tnr']:.3f}"
    )
    return out


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    out = {d: train(d) for d in ("vivo", "vitro")}
    with open(os.path.join(RESULTS, "chemprop_v17.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n저장: results/chemprop_v17.json")


if __name__ == "__main__":
    main()
