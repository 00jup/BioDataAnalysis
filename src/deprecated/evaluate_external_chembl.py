"""
외부 검증: 팀원 ChEMBL 간독성 데이터로 v1.3.0 모델 검증.

절차:
  1. 누수 없는 평가 풀 구축
     - 양성: 팀원 chembl_toxic_positive_set.csv (label=1)
     - 음성: data/marketed_drugs/non_hepatotoxic/marketed_clean.csv
     - v1.3.0 학습 분자(InChIKey) 제외 → 누수 차단
     - 양성∩음성 라벨 충돌 분자는 양쪽 모두 제외
  2. 음성 522개를 매 회 랜덤 추출해 1:1 균형 평가셋 구성
  3. 10회 반복 → 지표 평균±표준편차 보고

사용법:
    python src/evaluate_external_chembl.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import extract_features  # noqa: E402
from src.model_versioning import load_versioned_meta, load_versioned_model  # noqa: E402

TEAMMATE_DIR = "/Users/jeje/Downloads/(~260515) 바데분 chEMBL 데이터 다운로드"
POSITIVE_CSV = os.path.join(TEAMMATE_DIR, "chembl_toxic_positive_set.csv")
NEGATIVE_CSV = os.path.join(
    PROJECT_ROOT, "data", "marketed_drugs", "non_hepatotoxic", "marketed_clean.csv"
)
MANIFEST_CSV = os.path.join(PROJECT_ROOT, "models", "v1.3.0", "training_data_manifest.csv")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "external_eval")
RESULTS_JSON = os.path.join(PROJECT_ROOT, "results", "external_chembl_v1.3.0.json")

MODEL_VERSION = "1.3.0"
N_RUNS = 10


def inchikey(smiles: str) -> str | None:
    """SMILES → InChIKey (실패 시 None)."""
    if not smiles or str(smiles).strip().lower() in ("", "nan"):
        return None
    mol = Chem.MolFromSmiles(str(smiles).strip())
    if mol is None:
        return None
    try:
        return Chem.MolToInchiKey(mol)
    except Exception:
        return None


def build_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    """누수·라벨충돌 제거된 양성/음성 풀 DataFrame 반환."""
    # 학습에 쓰인 분자 (v1.3.0)
    seen = set(pd.read_csv(MANIFEST_CSV)["inchi_key"].dropna())
    print(f"v1.3.0 학습 분자: {len(seen)}개")

    # 양성: 팀원 ChEMBL 간독성 positive set
    pos_raw = pd.read_csv(POSITIVE_CSV)
    pos_raw = pos_raw[pos_raw["label"].astype(str).isin(["1", "1.0"])].copy()
    pos_raw["inchi_key"] = pos_raw["canonical_smiles"].map(inchikey)
    pos_raw = pos_raw.dropna(subset=["inchi_key"]).drop_duplicates("inchi_key")
    pos = pd.DataFrame(
        {
            "Name": pos_raw["pref_name"].fillna(pos_raw["molecule_chembl_id"]),
            "SMILES": pos_raw["canonical_smiles"],
            "Label": 1,
            "inchi_key": pos_raw["inchi_key"],
        }
    )

    # 음성: 프로젝트 marketed_clean (시판약물 = 음성 가정)
    neg_raw = pd.read_csv(NEGATIVE_CSV)
    neg_raw = neg_raw.dropna(subset=["inchi_key"]).drop_duplicates("inchi_key")
    neg = pd.DataFrame(
        {
            "Name": neg_raw["name"],
            "SMILES": neg_raw["canonical_smiles"],
            "Label": 0,
            "inchi_key": neg_raw["inchi_key"],
        }
    )

    print(f"원본 고유 분자: 양성 {len(pos)} / 음성 {len(neg)}")

    # 누수 제거: 학습 분자 제외
    pos = pos[~pos["inchi_key"].isin(seen)]
    neg = neg[~neg["inchi_key"].isin(seen)]
    print(f"학습 분자 제외 후: 양성 {len(pos)} / 음성 {len(neg)}")

    # 라벨 충돌 제거: 양쪽에 모두 있는 분자는 신뢰 불가 → 양쪽 제외
    conflict = set(pos["inchi_key"]) & set(neg["inchi_key"])
    pos = pos[~pos["inchi_key"].isin(conflict)]
    neg = neg[~neg["inchi_key"].isin(conflict)]
    print(f"라벨 충돌 {len(conflict)}개 제외 후: 양성 {len(pos)} / 음성 {len(neg)}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pos.to_csv(os.path.join(OUT_DIR, "chembl_external_positives.csv"), index=False)
    neg.to_csv(os.path.join(OUT_DIR, "chembl_external_negatives_pool.csv"), index=False)
    print(f"평가 풀 저장: {OUT_DIR}/")
    return pos.reset_index(drop=True), neg.reset_index(drop=True)


def predict_proba(df: pd.DataFrame, model) -> np.ndarray:
    """DataFrame의 SMILES에 대해 독성 확률 반환 (피처 추출 실패분은 NaN)."""
    feat_df, valid = extract_features(df["SMILES"].tolist(), feature_set="B")
    proba = np.full(len(df), np.nan)
    proba[valid] = model.predict_proba(feat_df.values)[:, 1]
    return proba


def main() -> None:
    pos, neg = build_pools()

    model = load_versioned_model(MODEL_VERSION)
    meta = load_versioned_meta(MODEL_VERSION)
    threshold = float(meta.get("classification_threshold", 0.5))
    print(f"\n모델 v{MODEL_VERSION} 로드 — 분류 임계값 {threshold:.2f}")

    # 피처 추출 + 예측은 전체에 대해 한 번만 (반복 시 재계산 회피)
    print("\n[양성 풀 예측]")
    pos_proba = predict_proba(pos, model)
    print("[음성 풀 예측]")
    neg_proba = predict_proba(neg, model)

    pos_valid = np.where(~np.isnan(pos_proba))[0]
    neg_valid = np.where(~np.isnan(neg_proba))[0]
    n_pos = len(pos_valid)
    print(f"\n피처 추출 성공: 양성 {n_pos} / 음성 {len(neg_valid)}")

    pos_scores = pos_proba[pos_valid]

    # 10회 반복: 음성 n_pos개 랜덤 추출 → 1:1 균형 평가
    metrics = {k: [] for k in ["roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"]}
    per_run = []
    for seed in range(N_RUNS):
        rng = np.random.default_rng(seed)
        sampled = rng.choice(neg_valid, size=n_pos, replace=False)
        neg_scores = neg_proba[sampled]

        y_true = np.array([1] * n_pos + [0] * n_pos)
        y_score = np.concatenate([pos_scores, neg_scores])
        y_pred = (y_score >= threshold).astype(int)

        run = {
            "seed": seed,
            "roc_auc": roc_auc_score(y_true, y_score),
            "pr_auc": average_precision_score(y_true, y_score),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
        per_run.append(run)
        for k in metrics:
            metrics[k].append(run[k])

    # 보고
    print(f"\n{'=' * 72}")
    print(f"  외부 검증 결과 — v{MODEL_VERSION} / ChEMBL 간독성 / 1:1 균형 / {N_RUNS}회 반복")
    print(f"  양성 {n_pos}개 고정 + 음성 {n_pos}개 랜덤추출(매 회) — threshold {threshold:.2f}")
    print(f"{'=' * 72}")
    print(f"{'run':<5}" + "".join(f"{k:>12}" for k in metrics))
    for run in per_run:
        print(f"{run['seed']:<5}" + "".join(f"{run[k]:>12.4f}" for k in metrics))
    print("-" * 72)
    print(f"{'mean':<5}" + "".join(f"{np.mean(metrics[k]):>12.4f}" for k in metrics))
    print(f"{'std':<5}" + "".join(f"{np.std(metrics[k]):>12.4f}" for k in metrics))
    print("=" * 72)

    summary = {
        "model_version": MODEL_VERSION,
        "threshold": threshold,
        "n_runs": N_RUNS,
        "n_positive": int(n_pos),
        "n_negative_pool": int(len(neg_valid)),
        "ratio": "1:1 balanced (negatives randomly resampled each run)",
        "per_run": per_run,
        "summary": {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v))} for k, v in metrics.items()
        },
    }
    os.makedirs(os.path.dirname(RESULTS_JSON), exist_ok=True)
    with open(RESULTS_JSON, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n결과 저장: {RESULTS_JSON}")


if __name__ == "__main__":
    main()
