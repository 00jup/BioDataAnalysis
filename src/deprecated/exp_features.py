"""실험용 feature 로딩 + selection 유틸.

feature_cache.parquet (Morgan+MACCS+Mordred) 에서 분자별 feature 를 조회하고,
train 기준으로 무분산·고상관 컬럼을 제거하는 feature selection 을 제공한다.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# EXP_FEATURE_CACHE 환경변수로 캐시 교체 가능 (RDKit 210 ↔ Mordred 1613)
CACHE_PATH = os.environ.get(
    "EXP_FEATURE_CACHE",
    os.path.join(PROJECT_ROOT, "data", "experiments", "feature_cache.parquet"),
)

_CACHE: pd.DataFrame | None = None


def _cache() -> pd.DataFrame:
    global _CACHE
    if _CACHE is None:
        _CACHE = pd.read_parquet(CACHE_PATH)
    return _CACHE


def load_xy(csv_path: str) -> tuple[pd.DataFrame, np.ndarray]:
    """csv 의 분자에 대한 feature DataFrame 과 label 배열 반환."""
    df = pd.read_csv(csv_path)
    cache = _cache()
    mask = df["canonical_smiles"].isin(cache.index)
    df = df[mask].reset_index(drop=True)
    X = cache.loc[df["canonical_smiles"]].reset_index(drop=True)
    y = df["label"].to_numpy(dtype=int)
    return X, y


def select_features(X_train: pd.DataFrame, corr_threshold: float = 0.95) -> list[str]:
    """train 기준 feature selection: 무분산 제거 → 고상관 쌍 중 하나 제거."""
    # 1) 분산 0 (상수) 컬럼 제거
    std = X_train.std(axis=0)
    kept = std[std > 0].index.tolist()
    Xk = X_train[kept]

    # 2) 상관계수 |r| > threshold 인 쌍에서 뒤쪽 컬럼 제거
    corr = np.corrcoef(Xk.to_numpy(dtype=np.float64), rowvar=False)
    corr = np.abs(np.nan_to_num(corr))
    upper = np.triu(corr, k=1)
    drop_idx = {j for j in range(upper.shape[1]) if (upper[:, j] > corr_threshold).any()}
    final = [c for i, c in enumerate(kept) if i not in drop_idx]
    return final
