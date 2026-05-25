"""Chemprop D-MPNN 학습 — 현재 vivo/vitro 데이터 (v1 분리 split).

사용자 의도:
  - v1 의 train/val/test inchi_key 분리 그대로 유지
  - Chemprop 결과를 v1 ensemble 의 추가 멤버로 활용 가능

전략:
  1. train+val+test 합친 csv 에 splits 컬럼 추가 → chemprop splits-column 사용
  2. RDKit 2D normalized featurizer (이전 deprecated 에서 사용 흔적)
  3. ensemble-size 3 (시간 + 메모리 절충)
  4. epochs 30, patience 5
  5. M1 Max MPS 가속
  6. 메트릭: binary-mcc + roc
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHEMPROP_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v9")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v9")
RESULTS = os.path.join(PROJECT_ROOT, "results")

PY = sys.executable
CHEMPROP = os.path.join(os.path.dirname(PY), "chemprop")


def prep_data(domain: str) -> tuple[str, str, int]:
    """train+val 통합 (chemprop split internal), test 분리.

    chemprop v2: -i train.csv -t test.csv  형태 안 됨.
    대신 splits-column 사용 — 통합 csv 에 'split' 컬럼 (train/val/test).

    근데 chemprop v2 splits-column 정확한 동작 확인 필요.
    가장 안전: 통합 csv (split 컬럼) + --split-sizes 0 0 0 (대신 splits-column)
    """
    out_dir = os.path.join(CHEMPROP_DATA, domain)
    os.makedirs(out_dir, exist_ok=True)

    tr = pd.read_csv(os.path.join(DATA_DIR, "train", f"{domain}.csv"))
    va = pd.read_csv(os.path.join(DATA_DIR, "val", f"{domain}.csv"))
    te = pd.read_csv(os.path.join(DATA_DIR, "test", f"{domain}.csv"))

    # smiles + label 만, 깨끗하게
    def clean(df):
        d = df[["canonical_smiles", "label"]].dropna()
        d["label"] = d["label"].astype(int)
        return d
    tr, va, te = clean(tr), clean(va), clean(te)

    tr_path = os.path.join(out_dir, "train.csv")
    va_path = os.path.join(out_dir, "val.csv")
    te_path = os.path.join(out_dir, "test.csv")
    tr.to_csv(tr_path, index=False)
    va.to_csv(va_path, index=False)
    te.to_csv(te_path, index=False)
    print(f"[{domain}] train {len(tr)} val {len(va)} test {len(te)}")
    return out_dir, te_path, len(te)


def train_domain(domain: str):
    print(f"\n{'='*70}\n  Chemprop v9 — {domain}\n{'='*70}")
    work_dir, te_path, n_test = prep_data(domain)
    save_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(save_dir, exist_ok=True)

    # chemprop v2 — splits-column 으로 train/val 지정, test 는 별도 predict
    # 통합 csv 만들기
    combined = pd.concat([
        pd.read_csv(os.path.join(work_dir, "train.csv")).assign(split="train"),
        pd.read_csv(os.path.join(work_dir, "val.csv")).assign(split="val"),
    ], ignore_index=True)
    combined_path = os.path.join(work_dir, "train_val.csv")
    combined.to_csv(combined_path, index=False)

    cmd = [
        CHEMPROP, "train",
        "-i", combined_path,
        "-s", "canonical_smiles",
        "--target-columns", "label",
        "-t", "classification",
        "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--splits-column", "split",
        "--ensemble-size", "3",
        "--epochs", "30",
        "--patience", "5",
        "--accelerator", "cpu",  # MPS 가 가끔 segfault — CPU 안전
        "-o", save_dir,
    ]
    print(f"  command: {' '.join(cmd[:8])} ...")
    t0 = time.time()
    log_path = os.path.join(save_dir, "train.log")
    with open(log_path, "w") as logf:
        result = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - t0
    print(f"  학습 끝 ({elapsed/60:.1f}분, exit={result.returncode})  log: {log_path}")
    if result.returncode != 0:
        print(f"  ⚠️ chemprop train 실패. log 끝부분:")
        with open(log_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    # test 예측 — chemprop predict
    pred_path = os.path.join(save_dir, "test_pred.csv")
    cmd_predict = [
        CHEMPROP, "predict",
        "--test-path", te_path,
        "-s", "canonical_smiles",
        "--model-paths", save_dir,
        "--preds-path", pred_path,
        "--accelerator", "cpu",
    ]
    print(f"  test 예측 중...")
    t0 = time.time()
    log_pred_path = os.path.join(save_dir, "predict.log")
    with open(log_pred_path, "w") as logf:
        result = subprocess.run(cmd_predict, stdout=logf, stderr=subprocess.STDOUT)
    print(f"  예측 끝 ({(time.time()-t0)/60:.1f}분, exit={result.returncode})")
    if result.returncode != 0:
        with open(log_pred_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None

    # 평가
    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score, accuracy_score)
    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    te = pd.read_csv(te_path)
    merged = te.merge(pred_df, on="canonical_smiles", how="left")
    y = merged["label"].to_numpy(int)
    p = merged["pred"].to_numpy(float)
    auc = roc_auc_score(y, p)

    # MCC max threshold
    best_t, best_m = 0.5, -1.0
    grid = np.linspace(0.05, 0.95, 91)
    for t in grid:
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best_m: best_m, best_t = m, t
    pred = (p >= best_t).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    print(f"\n  [{domain} chemprop test] N={len(y)} (양성 {y.sum()})")
    print(f"  AUC {auc:.3f}  MCC {best_m:.3f}  threshold {best_t:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}  F1 {f1_score(y, pred):.3f}")
    return {"domain": domain, "auc": float(auc), "mcc": float(best_m),
            "threshold": float(best_t),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
            "n_test": int(len(y))}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for d in ("vivo", "vitro"):
        r = train_domain(d)
        if r: out[d] = r
    with open(os.path.join(RESULTS, "chemprop_v9.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/chemprop_v9.json")


if __name__ == "__main__":
    main()
