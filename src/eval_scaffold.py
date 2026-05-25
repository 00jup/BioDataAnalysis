"""scaffold 학습된 vivo 평가 + vitro 학습+평가."""
from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_chemprop_scaffold import train_scaffold

MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def predict_and_eval(domain: str):
    save_dir = os.path.join(MODELS_DIR, domain)
    test_csv = os.path.join(save_dir, "test_smiles.csv")
    all_csv = os.path.join(DATA_DIR, domain, "all.csv")
    if not os.path.exists(test_csv): return None

    pred_path = os.path.join(save_dir, "test_pred.csv")
    if not os.path.exists(pred_path):
        cmd = [
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
            subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)

    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score)
    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    te_df = pd.read_csv(test_csv)
    all_df = pd.read_csv(all_csv)
    te_df = te_df.merge(all_df, on="canonical_smiles", how="left")
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
    print(f"\n  [{domain} scaffold test] N={len(y)} (양성 {y.sum()})")
    print(f"  AUC {auc:.3f}  MCC {bm:.3f}  thr {bt:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}  F1 {f1_score(y, pred):.3f}")
    return {"auc": float(auc), "mcc": float(bm), "threshold": float(bt),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
            "n_test": int(len(y)), "n_pos": int(y.sum())}


def main():
    print("=== scaffold vivo 평가 (이미 학습) ===")
    out = {"vivo": predict_and_eval("vivo")}

    print("\n=== scaffold vitro 학습+평가 ===")
    out["vitro"] = train_scaffold("vitro")

    with open(os.path.join(RESULTS, "chemprop_scaffold.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/chemprop_scaffold.json")

    # inchi_key (v10) vs scaffold 비교
    v10 = json.load(open(os.path.join(RESULTS, "chemprop_v10.json")))
    print(f"\n{'='*70}\n  inchi_key (v10) vs scaffold split — 진짜 OOD\n{'='*70}")
    for d in ("vivo", "vitro"):
        if d not in v10 or d not in out or out[d] is None: continue
        print(f"\n[{d}]")
        print(f"  {'metric':<8s} {'v10 (inchi)':>12s} {'scaffold':>10s} {'Δ':>8s}")
        for k in ("auc", "mcc", "tpr", "tnr"):
            v10v = v10[d].get(k, 0); sv = out[d].get(k, 0)
            print(f"  {k:<8s} {v10v:>12.3f} {sv:>10.3f} {sv-v10v:>+8.4f}")


if __name__ == "__main__":
    main()
