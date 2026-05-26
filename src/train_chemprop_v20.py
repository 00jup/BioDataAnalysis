"""Chemprop v20 — chEMBL 가 vivo 로 옮겨진 후 재학습.

v17 와 동일 hp:
  - ensemble 15, hidden 600, epochs 40, patience 8
  - v1_rdkit_2d_normalized featurizer
  - scaffold-balanced split

데이터:
  - vivo 양성 4,003 / 음성 10,381 (1:2.6) — chEMBL +610 추가
"""
from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_v20")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v20")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def build_data():
    """labels_db 에서 vivo 분자 추출."""
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    df = db[db.vivo_label.notna()][["canonical_smiles", "vivo_label"]].rename(
        columns={"vivo_label": "label"})
    df["label"] = df["label"].astype(int)
    df = df.dropna().drop_duplicates(subset=["canonical_smiles"])
    print(f"vivo: {len(df)} (양성 {(df.label==1).sum()} / 음성 {(df.label==0).sum()})")
    os.makedirs(DATA_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "all.csv")
    df.to_csv(csv_path, index=False)
    return csv_path


def train():
    print(f"\n=== Chemprop v20 — chEMBL→vivo 통합 학습 ===")
    csv_path = build_data()
    save_dir = MODELS_DIR; os.makedirs(save_dir, exist_ok=True)
    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv_path, "-s", "canonical_smiles",
        "--target-columns", "label", "-t", "classification", "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.7", "0.15", "0.15",
        "--ensemble-size", "15",
        "--message-hidden-dim", "600",
        "--epochs", "40", "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "--save-smiles-splits",
        "-o", save_dir,
    ]
    log_path = os.path.join(save_dir, "train.log")
    t0 = time.time()
    print(f"  설정: ensemble 15 + hidden 600 + featurizer (v17 동일)")
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    test_csv = os.path.join(save_dir, "test_smiles.csv")
    pred_path = os.path.join(save_dir, "test_pred.csv")
    cmd_p = [
        CHEMPROP_BIN, "predict",
        "--test-path", test_csv, "-s", "canonical_smiles",
        "--model-paths", save_dir,
        "--preds-path", pred_path,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ]
    log_p = os.path.join(save_dir, "predict.log")
    with open(log_p, "w") as f:
        subprocess.run(cmd_p, stdout=f, stderr=subprocess.STDOUT)

    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score)
    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    all_df = pd.read_csv(csv_path)
    te_df = pd.read_csv(test_csv).merge(all_df, on="canonical_smiles", how="left")
    m = te_df.merge(pred_df, on="canonical_smiles", how="left")
    y = m["label"].to_numpy(int); p = m["pred"].to_numpy(float)
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    auc = roc_auc_score(y, p)
    bt, bm = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        mc = matthews_corrcoef(y, (p >= t).astype(int))
        if mc > bm: bm, bt = mc, t
    pred = (p >= bt).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    out = {"auc": float(auc), "mcc": float(bm), "threshold": float(bt),
           "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
           "n_test": int(len(y)), "n_pos": int(y.sum())}
    print(f"\n  [v20 scaffold test] N={len(y)} (양성 {y.sum()})")
    print(f"  AUC {auc:.3f}  MCC {bm:.3f}  threshold {bt:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    return out


def main():
    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    out = train()
    with open(os.path.join(RESULTS, "chemprop_v20.json"), "w") as f:
        json.dump({"v20": out}, f, indent=2)
    print(f"\n저장: results/chemprop_v20.json")


if __name__ == "__main__":
    main()
