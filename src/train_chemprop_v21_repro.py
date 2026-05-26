"""v17 재현 실험 — 같은 데이터 + 같은 hp + 같은 split.

목적: v17 MCC 0.694 가 robust 한지 확인.
ensemble 15 의 stochasticity (각 모델 weight 초기화 random) 가 있어서
값이 약간 다를 수 있음. 재실험으로 분산 측정.
"""
from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v2")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v21_repro")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def train():
    print(f"\n=== Chemprop v21 — v17 재현 (같은 hp, 같은 split) ===")
    save_dir = MODELS_DIR; os.makedirs(save_dir, exist_ok=True)
    splits_file = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2",
                                "v12_baseline", "vivo", "splits.json")
    csv_path = os.path.join(DATA_DIR, "vivo", "all.csv")

    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv_path, "-s", "canonical_smiles",
        "--target-columns", "label", "-t", "classification", "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--splits-file", splits_file,
        "--ensemble-size", "15",
        "--message-hidden-dim", "600",
        "--epochs", "40", "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "-o", save_dir,
    ]
    log_path = os.path.join(save_dir, "train.log")
    t0 = time.time()
    print(f"  v17 와 동일 설정: ensemble 15 + hidden 600 + featurizer + same split")
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    # 같은 test set (v17 의 test_smiles.csv) 로 평가
    test_csv = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2",
                             "v12_baseline", "vivo", "test_smiles.csv")
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
    print(f"\n  [v21 재현] N={len(y)} (양성 {y.sum()})")
    print(f"  AUC {auc:.3f}  MCC {bm:.3f}  threshold {bt:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    return {"auc": float(auc), "mcc": float(bm), "threshold": float(bt),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1))}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    out = train()
    # v17 과 직접 비교
    v17 = json.load(open(os.path.join(RESULTS, "chemprop_v17.json")))
    print(f"\n{'='*60}")
    print(f"  v17 vs v21 (재현)")
    print(f"{'='*60}")
    print(f"  {'metric':<8s} {'v17':>8s} {'v21':>8s} {'Δ':>8s}")
    for k in ("auc", "mcc", "tpr", "tnr"):
        v17v = v17["vivo"].get(k, 0); v21v = out.get(k, 0)
        print(f"  {k:<8s} {v17v:>8.3f} {v21v:>8.3f} {v21v-v17v:>+8.4f}")
    with open(os.path.join(RESULTS, "chemprop_v21_repro.json"), "w") as f:
        json.dump({"v17": v17["vivo"], "v21": out}, f, indent=2)


if __name__ == "__main__":
    main()
