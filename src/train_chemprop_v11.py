"""Chemprop v11 — scaffold + class_balance (양성 균형 학습).

차이 vs scaffold:
  - --class-balance 옵션 추가
    → 배치마다 양성/음성 같은 비율 → 양성 학습 더 강화
    → 양성 비율 10.7% (vivo) 의 imbalance 해소
  - 같은 split 사용 (재현성)
"""
from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v11")
RESULTS = os.path.join(PROJECT_ROOT, "results")

PY = sys.executable
CHEMPROP_BIN = os.path.join(os.path.dirname(PY), "chemprop")


def train_vivo_vitro(domain: str):
    print(f"\n{'='*70}\n  Chemprop v11 — {domain} (scaffold + class_balance)\n{'='*70}")
    csv_path = os.path.join(DATA_DIR, domain, "all.csv")
    save_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(save_dir, exist_ok=True)

    # 이전 scaffold 의 splits.json 재활용 → 같은 train/val/test
    splits_file = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold", domain, "splits.json")

    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv_path,
        "-s", "canonical_smiles",
        "--target-columns", "label",
        "-t", "classification",
        "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--splits-file", splits_file,
        "--class-balance",                    # ⭐ 추가
        "--ensemble-size", "5",
        "--epochs", "40",
        "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "-o", save_dir,
    ]
    print("  명령: class_balance + scaffold + ensemble 5 + featurizer")
    t0 = time.time()
    log_path = os.path.join(save_dir, "train.log")
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    # 같은 test split 으로 평가 — scaffold 의 test_smiles.csv 사용
    test_csv = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold", domain, "test_smiles.csv")
    pred_path = os.path.join(save_dir, "test_pred.csv")
    cmd_p = [
        CHEMPROP_BIN, "predict",
        "--test-path", test_csv,
        "-s", "canonical_smiles",
        "--model-paths", save_dir,
        "--preds-path", pred_path,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ]
    log_p = os.path.join(save_dir, "predict.log")
    with open(log_p, "w") as logf:
        subprocess.run(cmd_p, stdout=logf, stderr=subprocess.STDOUT)

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
        mcc = matthews_corrcoef(y, (p >= t).astype(int))
        if mcc > bm: bm, bt = mcc, t
    pred = (p >= bt).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    print(f"\n  [{domain} v11 scaffold] N={len(y)} (양성 {y.sum()})")
    print(f"  AUC {auc:.3f}  MCC {bm:.3f}  thr {bt:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}  F1 {f1_score(y, pred):.3f}")
    return {"auc": float(auc), "mcc": float(bm), "threshold": float(bt),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
            "n_test": int(len(y)), "n_pos": int(y.sum())}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for d in ("vivo", "vitro"):
        r = train_vivo_vitro(d)
        if r: out[d] = r

    with open(os.path.join(RESULTS, "chemprop_v11.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/chemprop_v11.json")

    # scaffold 대비 비교
    if os.path.exists(os.path.join(RESULTS, "chemprop_scaffold.json")):
        scf = json.load(open(os.path.join(RESULTS, "chemprop_scaffold.json")))
        print(f"\n{'='*70}\n  scaffold (v9 변형) vs v11 (class_balance)\n{'='*70}")
        for d in ("vivo", "vitro"):
            if d not in scf or d not in out: continue
            print(f"\n[{d}]")
            print(f"  {'metric':<8s} {'scaffold':>10s} {'v11':>10s} {'Δ':>8s}")
            for k in ("auc", "mcc", "tpr", "tnr"):
                sv = scf[d].get(k, 0); v11v = out[d].get(k, 0)
                print(f"  {k:<8s} {sv:>10.3f} {v11v:>10.3f} {v11v-sv:>+8.4f}")


if __name__ == "__main__":
    main()
