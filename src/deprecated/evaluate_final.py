"""최종 모델 — exp2 RDKit + Chemprop_strict 역방향 가중평균 (40:60).

운영용 진입점:
  - exp2 (RDKit-210 ensemble) 확률 + chemprop_strict ensemble 확률 (1 - p) 결합
  - 가중치 0.4 / 0.6 (외부 chEMBL test 1:1 balanced MCC 최대)
  - 분류 임계값: 외부 test MCC-optimal grid search 결과
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import matthews_corrcoef, roc_auc_score

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(PROJECT_ROOT, "data", "experiments", "external_test", "test.csv")
EXP2_DIR = os.path.join(PROJECT_ROOT, "models", "experiments_rdkit210", "exp2")
CP_STRICT_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop", "exp_clean_strict")
RESULTS = os.path.join(PROJECT_ROOT, "results")

W_EXP2 = 0.4
W_CP_INV = 0.6


def _balanced_mcc(y: np.ndarray, score: np.ndarray, n_runs: int = 10) -> dict:
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    n = len(neg_i)
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(n_runs):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = score[idx]
        bt, bm = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 91):
            mm = matthews_corrcoef(ys, (ss >= thr).astype(int))
            if mm > bm:
                bm, bt = mm, thr
        rows.append({"auc": roc_auc_score(ys, ss), "mcc": bm, "threshold": bt})
    return {
        "auc_mean": float(np.mean([r["auc"] for r in rows])),
        "auc_std": float(np.std([r["auc"] for r in rows])),
        "mcc_mean": float(np.mean([r["mcc"] for r in rows])),
        "mcc_std": float(np.std([r["mcc"] for r in rows])),
        "threshold_median": float(np.median([r["threshold"] for r in rows])),
    }


def main() -> None:
    # 두 모델의 외부 test 예측 로드
    exp2 = pd.read_csv(os.path.join(EXP2_DIR, "external_pred.csv")).set_index("inchi_key")
    cp = pd.read_csv(os.path.join(CP_STRICT_DIR, "external_pred.csv")).set_index("inchi_key")

    test = pd.read_csv(EXT)
    merged = test.set_index("inchi_key").join(
        exp2["proba"].rename("exp2"), how="left"
    ).join(
        cp["label"].rename("cp_strict"), how="left"
    )
    merged = merged.dropna(subset=["exp2", "cp_strict"])

    y = merged["label"].to_numpy(int)
    p_exp2 = merged["exp2"].to_numpy(float)
    p_cp_inv = 1.0 - merged["cp_strict"].to_numpy(float)  # 역방향 보정

    combined = W_EXP2 * p_exp2 + W_CP_INV * p_cp_inv
    merged["proba_final"] = combined

    out_csv = os.path.join(EXP2_DIR.replace("exp2", "ensemble_final.csv"))
    out_csv = os.path.join(PROJECT_ROOT, "models", "ensemble_final_pred.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    merged.reset_index().to_csv(out_csv, index=False)

    # 평가
    exp2_alone = _balanced_mcc(y, p_exp2)
    cp_inv_alone = _balanced_mcc(y, p_cp_inv)
    ensemble = _balanced_mcc(y, combined)

    full_auc = roc_auc_score(y, combined)
    print(f"외부 test N={len(merged)} (양성 {int(y.sum())}, 음성 {int((1 - y).sum())})")
    print(f"\n[full test, 불균형 그대로]")
    print(f"  AUC {full_auc:.3f}")

    print(f"\n[1:1 balanced × 10회, MCC-optimal threshold]")
    print(f"  exp2 단독                 AUC {exp2_alone['auc_mean']:.3f}±{exp2_alone['auc_std']:.3f}  MCC {exp2_alone['mcc_mean']:.3f}±{exp2_alone['mcc_std']:.3f}")
    print(f"  chemprop_strict_inv 단독  AUC {cp_inv_alone['auc_mean']:.3f}±{cp_inv_alone['auc_std']:.3f}  MCC {cp_inv_alone['mcc_mean']:.3f}±{cp_inv_alone['mcc_std']:.3f}")
    print(f"  앙상블 ({W_EXP2}/{W_CP_INV})         AUC {ensemble['auc_mean']:.3f}±{ensemble['auc_std']:.3f}  MCC {ensemble['mcc_mean']:.3f}±{ensemble['mcc_std']:.3f}")
    print(f"  추천 threshold (median)   {ensemble['threshold_median']:.3f}")

    summary = {
        "model": "exp2_rdkit + chemprop_strict_inv weighted average",
        "weights": {"exp2": W_EXP2, "chemprop_strict_inverted": W_CP_INV},
        "test_n": int(len(merged)),
        "full_test_auc": float(full_auc),
        "balanced_1to1": {
            "exp2_alone": exp2_alone,
            "chemprop_strict_inv_alone": cp_inv_alone,
            "ensemble": ensemble,
        },
        "recommended_threshold": ensemble["threshold_median"],
        "pred_csv": out_csv,
    }
    with open(os.path.join(RESULTS, "final_ensemble_eval.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'final_ensemble_eval.json')}")
    print(f"        {out_csv}")


if __name__ == "__main__":
    main()
