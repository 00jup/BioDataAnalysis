"""3개 실험 모델을 공통 chEMBL 외부 test로 평가하고 비교 리포트 생성.

외부 test는 양성 불균형 → AUC는 전체로, 임계값 의존 지표는 균형 부분집합으로 보고.
feature 는 feature_cache.parquet 사용, exp별 selected_features.json 적용.

사용법:
    python src/evaluate_experiments.py
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import joblib
import numpy as np

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from sklearn.metrics import (  # noqa: E402
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.exp_features import load_xy  # noqa: E402

EXP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "experiments")
REPORT = os.path.join(PROJECT_ROOT, "results", "experiments_comparison.md")
EXPS = ["exp1", "exp2", "exp3"]
RANDOM_STATE = 42


def main() -> None:
    X_full, y = load_xy(os.path.join(EXP_DIR, "external_test", "test.csv"))
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    print(f"외부 test 전체: 양성 {n_pos}, 음성 {n_neg}")

    rng = np.random.RandomState(RANDOM_STATE)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    bal_idx = np.concatenate([rng.choice(pos_idx, size=len(neg_idx), replace=False), neg_idx])
    yb = y[bal_idx]

    results = {}
    for exp in EXPS:
        odir = os.path.join(MODELS_DIR, exp)
        model = joblib.load(os.path.join(odir, "best_model.pkl"))
        meta = json.load(open(os.path.join(odir, "meta.json")))
        kept = json.load(open(os.path.join(odir, "selected_features.json")))
        t = float(meta.get("classification_threshold", 0.5))

        X = X_full[kept].to_numpy(dtype=np.float64)
        full_prob = model.predict_proba(X)[:, 1]
        full_pred = (full_prob >= t).astype(int)
        bal_prob = full_prob[bal_idx]
        bal_pred = (bal_prob >= t).astype(int)
        results[exp] = {
            "meta": meta,
            "threshold": t,
            "auc_full": roc_auc_score(y, full_prob),
            "auc_bal": roc_auc_score(yb, bal_prob),
            "bal_acc": balanced_accuracy_score(yb, bal_pred),
            "f1": f1_score(yb, bal_pred, zero_division=0),
            "precision": precision_score(yb, bal_pred, zero_division=0),
            "recall": recall_score(yb, bal_pred, zero_division=0),
            "mcc_bal": matthews_corrcoef(yb, bal_pred),
            "mcc_full": matthews_corrcoef(y, full_pred),
        }

    notes = {
        "exp1": "저장소 양성 only",
        "exp2": "저장소 + DILIst + GoldStandard union",
        "exp3": "union 양성 + marketed_clean 전체 음성",
    }
    lines = [
        "# 실험 변형 비교 — DILI 데이터셋 3종 (Mordred 개선판)",
        "",
        "Feature: Morgan + MACCS + Mordred 2D descriptor + feature selection.",
        "하이퍼파라미터 튜닝, Youden's J 임계값.",
        "",
        "## 데이터셋 구성",
        "",
        "| 변형 | 양성 | 음성 | feature수 | 비고 |",
        "|------|------|------|-----------|------|",
    ]
    for exp in EXPS:
        m = results[exp]["meta"]
        td = m["training_data"]
        lines.append(
            f"| {exp} | {td['positives']} | {td['negatives']} "
            f"| {m['model']['n_features_selected']} | {notes[exp]} |"
        )

    lines += [
        "",
        "## 내부 성능",
        "",
        "| 변형 | 앙상블 CV AUC | Val AUC | 임계값 |",
        "|------|--------------|---------|--------|",
    ]
    for exp in EXPS:
        m = results[exp]["meta"]
        lines.append(
            f"| {exp} | {m['cv_auc']:.4f} | {m['validation']['auc']:.4f} "
            f"| {results[exp]['threshold']:.3f} |"
        )

    lines += [
        "",
        "## 외부 chEMBL test 성능",
        "",
        f"- 전체 test: 양성 {n_pos}, 음성 {n_neg} (불균형 → AUC만 신뢰)",
        f"- 균형 부분집합: 양성 {len(neg_idx)}, 음성 {len(neg_idx)}",
        "",
        "| 변형 | AUC(전체) | AUC(균형) | MCC(전체) | MCC(균형) | Balanced Acc | F1 | Precision | Recall |",
        "|------|-----------|-----------|-----------|-----------|--------------|-----|-----------|--------|",
    ]
    for exp in EXPS:
        r = results[exp]
        lines.append(
            f"| {exp} | {r['auc_full']:.4f} | {r['auc_bal']:.4f} "
            f"| {r['mcc_full']:.4f} | {r['mcc_bal']:.4f} | {r['bal_acc']:.4f} "
            f"| {r['f1']:.4f} | {r['precision']:.4f} | {r['recall']:.4f} |"
        )

    best = max(EXPS, key=lambda e: results[e]["auc_full"])
    lines += ["", f"**외부 test 최고 AUC(전체): {best} ({results[best]['auc_full']:.4f})**", ""]

    os.makedirs(os.path.dirname(REPORT), exist_ok=True)
    with open(REPORT, "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\n리포트 저장: {REPORT}")


if __name__ == "__main__":
    main()
