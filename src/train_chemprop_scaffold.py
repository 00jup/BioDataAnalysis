"""Chemprop scaffold-balanced split — 진짜 OOD generalization 측정.

기존 (random/inchi_key split): test analog 가 train 에 있을 수 있음 (cheating 잠재)
Scaffold split: 같은 Murcko 골격은 train OR test 중 한 쪽에만 → 진짜 OOD 평가

train/val/test = 70/15/15  scaffold-balanced
ensemble-size 5, epochs 40, v1_rdkit_2d_normalized featurizer.

기대: test MCC 가 inchi_key split (0.425) 보다 낮아질 것 — 진짜 일반화 천장.
"""
from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUT_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold")
RESULTS = os.path.join(PROJECT_ROOT, "results")

PY = sys.executable
CHEMPROP_BIN = os.path.join(os.path.dirname(PY), "chemprop")


def build_data(domain: str):
    """전체 라벨된 분자를 chemprop scaffold split 에게 넘긴다."""
    os.makedirs(os.path.join(OUT_DATA, domain), exist_ok=True)
    db = pd.read_parquet(os.path.join(DATA_DIR, "labels_db", "full.parquet"))
    if domain == "vivo":
        df = db[db.vivo_label.notna()][["canonical_smiles", "vivo_label"]].rename(
            columns={"vivo_label": "label"})
    else:
        df = db[db.vitro_label.notna()][["canonical_smiles", "vitro_label"]].rename(
            columns={"vitro_label": "label"})
    df["label"] = df["label"].astype(int)
    df = df.dropna().drop_duplicates(subset=["canonical_smiles"])
    print(f"  [{domain}] 전체: {len(df)} (양성 {(df.label==1).sum()} / 음성 {(df.label==0).sum()})")

    csv_path = os.path.join(OUT_DATA, domain, "all.csv")
    df.to_csv(csv_path, index=False)
    return csv_path, len(df)


def train_scaffold(domain: str):
    print(f"\n{'='*70}\n  Chemprop scaffold-balanced — {domain}\n{'='*70}")
    csv_path, n = build_data(domain)
    save_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv_path,
        "-s", "canonical_smiles",
        "--target-columns", "label",
        "-t", "classification",
        "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.7", "0.15", "0.15",
        "--ensemble-size", "5",
        "--epochs", "40",
        "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "--save-smiles-splits",
        "-o", save_dir,
    ]
    print(f"  명령: ensemble=5, epochs=40, featurizer=v1_rdkit_2d_normalized, split=SCAFFOLD_BALANCED")
    t0 = time.time()
    log_path = os.path.join(save_dir, "train.log")
    with open(log_path, "w") as logf:
        r = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    # test 평가 — chemprop 가 직접 test_smiles.csv 저장
    test_csv = os.path.join(save_dir, "test_smiles.csv")
    if not os.path.exists(test_csv):
        print(f"  test_smiles.csv 없음")
        return None
    print(f"  scaffold test csv: {test_csv}")

    # predict
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

    # 평가 — test_smiles.csv 에는 SMILES 만, all.csv 와 join 해서 라벨 확보
    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score)
    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    te_df = pd.read_csv(test_csv)
    all_df = pd.read_csv(csv_path)
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
    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for d in ("vivo", "vitro"):
        r = train_scaffold(d)
        if r: out[d] = r
    with open(os.path.join(RESULTS, "chemprop_scaffold.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/chemprop_scaffold.json")

    # inchi_key split (v10) 과 비교
    if os.path.exists(os.path.join(RESULTS, "chemprop_v10.json")):
        v10 = json.load(open(os.path.join(RESULTS, "chemprop_v10.json")))
        print(f"\n{'='*70}\n  inchi_key split (v10) vs scaffold split\n{'='*70}")
        for d in ("vivo", "vitro"):
            if d not in v10 or d not in out: continue
            print(f"\n[{d}]")
            print(f"  {'metric':<8s} {'v10 (inchi)':>12s} {'scaffold':>10s} {'Δ':>8s}")
            for k in ("auc", "mcc", "tpr", "tnr"):
                v10v = v10[d].get(k, 0); sv = out[d].get(k, 0)
                print(f"  {k:<8s} {v10v:>12.3f} {sv:>10.3f} {sv-v10v:>+8.4f}")


if __name__ == "__main__":
    main()
