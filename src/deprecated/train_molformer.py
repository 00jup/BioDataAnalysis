"""IBM MolFormer-XL 사전학습 embedding + 경량 분류기 (DILI 외부 chEMBL 평가).

전략:
  1. ibm-research/MoLFormer-XL-both-10pct 로 분자별 768-dim 임베딩 추출
  2. exp_clean_full 학습 분자 임베딩으로 LogReg + XGBoost 학습
  3. 외부 chEMBL test 임베딩으로 평가 (MCC, AUC, F1)

D-MPNN scratch 학습 대비 강점:
  - PubChem 1억+ 분자로 사전학습된 표현 → OOD 일반화
  - 적은 데이터로도 안정적 (양성 422~1465)

사용:
    python src/train_molformer.py
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
EXT_TEST = os.path.join(EXP_DIR, "external_test", "test.csv")
CACHE_DIR = os.path.join(PROJECT_ROOT, "data", "molformer_cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "molformer")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

MODEL_NAME = "ibm-research/MoLFormer-XL-both-10pct"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
BATCH = 32
RANDOM_STATE = 42


def get_embedder():
    """MolFormer 로드 — 메모리 절약을 위해 eval/no_grad."""
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mdl = AutoModel.from_pretrained(MODEL_NAME, deterministic_eval=True, trust_remote_code=True)
    mdl.eval().to(DEVICE)
    return tok, mdl


@torch.no_grad()
def embed_batch(tok, mdl, smiles_list: list[str]) -> np.ndarray:
    """SMILES 배치 → (N, 768) embedding. CLS 토큰 사용."""
    enc = tok(smiles_list, padding=True, return_tensors="pt").to(DEVICE)
    out = mdl(**enc)
    # MoLFormer는 pooled output 제공
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        emb = out.pooler_output
    else:
        emb = out.last_hidden_state[:, 0, :]  # CLS fallback
    return emb.cpu().float().numpy()


def embed_smiles(smiles: list[str], cache_path: str) -> np.ndarray:
    """SMILES 리스트 임베딩, parquet 캐시."""
    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        if set(df["canonical_smiles"]) >= set(smiles):
            print(f"  cache 적중 {cache_path}")
            d = df.set_index("canonical_smiles").loc[smiles]
            return d.drop(columns=["canonical_smiles"], errors="ignore").to_numpy(dtype=np.float32)
    print(f"  임베딩 추출 {len(smiles)}개...")
    tok, mdl = get_embedder()
    out = np.zeros((len(smiles), 768), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(smiles), BATCH):
        batch = smiles[i : i + BATCH]
        out[i : i + BATCH] = embed_batch(tok, mdl, batch)
        if (i // BATCH) % 10 == 0:
            elapsed = time.time() - t0
            rate = (i + BATCH) / max(1, elapsed)
            eta = (len(smiles) - i - BATCH) / max(1, rate)
            print(f"    {i + BATCH}/{len(smiles)}  {rate:.1f}/s  eta {eta:.0f}s")
    cols = [f"d{j}" for j in range(768)]
    df = pd.DataFrame(out, columns=cols)
    df.insert(0, "canonical_smiles", smiles)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    df.to_parquet(cache_path, index=False)
    print(f"  캐시 저장: {cache_path}")
    return out


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate(name: str, model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """ROC 의 Youden J 임계값 + 1:1 balanced 10회 평가."""
    proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, thr = roc_curve(y_test, proba)
    best_t = float(thr[int(np.argmax(tpr - fpr))])
    best_t = min(max(best_t, 0.05), 0.95)
    overall = _metrics(y_test, proba, best_t)

    pos_i = np.where(y_test == 1)[0]
    neg_i = np.where(y_test == 0)[0]
    n = len(neg_i)
    rng = np.random.default_rng(RANDOM_STATE)
    bal = []
    for _ in range(10):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        bal.append(_metrics(y_test[idx], proba[idx], best_t))
    balanced = {
        k: {"mean": float(np.mean([r[k] for r in bal])), "std": float(np.std([r[k] for r in bal]))}
        for k in ("roc_auc", "pr_auc", "mcc", "accuracy", "f1", "precision", "recall")
    }
    summary = {"name": name, "overall_full_test": overall, "balanced_1to1": balanced}
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    print(f"device={DEVICE}, model={MODEL_NAME}")
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    train_csv = os.path.join(EXP_DIR, "exp_clean_full", "manifest.csv")
    test_csv = EXT_TEST

    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)

    train_smi = train["canonical_smiles"].tolist()
    test_smi = test["canonical_smiles"].tolist()

    all_smi = sorted(set(train_smi) | set(test_smi))
    print(f"임베딩 대상: train {len(train_smi)} + test {len(test_smi)} = unique {len(all_smi)}")

    cache_path = os.path.join(CACHE_DIR, "molformer_emb.parquet")
    embed_smiles(all_smi, cache_path)

    # 캐시에서 다시 정렬 조회
    cache = pd.read_parquet(cache_path).set_index("canonical_smiles")
    Xtr = cache.loc[train_smi].to_numpy(dtype=np.float32)
    Xte = cache.loc[test_smi].to_numpy(dtype=np.float32)
    ytr = train["label"].to_numpy(dtype=int)
    yte = test["label"].to_numpy(dtype=int)

    print(f"\nXtr {Xtr.shape}, Xte {Xte.shape}")

    # 분류기 1: LogReg (class weighted)
    print("\n▶ LogReg")
    lr = LogisticRegression(
        class_weight="balanced", max_iter=2000, C=1.0, random_state=RANDOM_STATE
    )
    lr.fit(Xtr, ytr)
    summ_lr = evaluate("MolFormer + LogReg", lr, Xte, yte)

    # 분류기 2: XGBoost
    print("\n▶ XGBoost")
    spw = float((ytr == 0).sum()) / max(1, int((ytr == 1).sum()))
    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        scale_pos_weight=spw,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="logloss",
    )
    xgb.fit(Xtr, ytr)
    summ_xgb = evaluate("MolFormer + XGBoost", xgb, Xte, yte)

    out = {"device": DEVICE, "model": MODEL_NAME, "logreg": summ_lr, "xgboost": summ_xgb}
    with open(os.path.join(RESULTS_DIR, "molformer_external_eval.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n저장: results/molformer_external_eval.json")


if __name__ == "__main__":
    main()
