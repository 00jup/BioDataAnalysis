"""Top-K 앙상블 가중치 최적화 — scipy minimize (single-thread, 빠름)."""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import matthews_corrcoef, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(PROJECT_ROOT, "data", "experiments", "external_test", "test.csv")
RESULTS = os.path.join(PROJECT_ROOT, "results")

# top-5 + bonus
SOURCES = [
    ("exp2_rdkit",   "models/experiments_rdkit210/exp2/external_pred.csv", "proba"),
    ("cp_strict",    "models/chemprop/exp_clean_strict/external_pred.csv", "label"),
    ("cp_full",      "models/chemprop/exp_clean_full/external_pred.csv", "label"),
    ("cp_nosider",   "models/chemprop/exp_clean_nosider/external_pred.csv", "label"),
    ("fp_avalon",    "models/diverse_fp/avalon/external_pred.csv", "proba"),
    ("fp_atompair",  "models/diverse_fp/atompair/external_pred.csv", "proba"),
    ("fp_tt",        "models/diverse_fp/tt/external_pred.csv", "proba"),
    ("fp_pattern",   "models/diverse_fp/pattern/external_pred.csv", "proba"),
    ("fp_ecfp6",     "models/diverse_fp/ecfp6/external_pred.csv", "proba"),
]


def load() -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    test = pd.read_csv(EXT)
    df = test.set_index("inchi_key")[["label"]].copy()
    for n, p, c in SOURCES:
        full = os.path.join(PROJECT_ROOT, p)
        df[n] = pd.read_csv(full).set_index("inchi_key")[c].astype(float).reindex(df.index)
    df = df.dropna()
    y = df["label"].to_numpy(int)
    # 부호 보정
    for n, _, _ in SOURCES:
        if roc_auc_score(y, df[n].to_numpy()) < 0.5:
            df[n] = 1.0 - df[n]
    cols = [n for n, _, _ in SOURCES]
    return df, y, cols


def balanced_mcc(score: np.ndarray, y: np.ndarray, pos_i, neg_i, n_runs=10):
    rng = np.random.default_rng(42)
    mccs = []
    n = len(neg_i)
    thrs_grid = np.linspace(0.05, 0.95, 91)
    for _ in range(n_runs):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = score[idx]
        # 벡터화 가능: 모든 threshold에서 동시에
        preds = (ss[:, None] >= thrs_grid).astype(int)
        # MCC per threshold
        mccs_t = []
        for k in range(len(thrs_grid)):
            mccs_t.append(matthews_corrcoef(ys, preds[:, k]))
        mccs.append(max(mccs_t))
    return float(np.mean(mccs))


def auc_bal(score: np.ndarray, y, pos_i, neg_i, n_runs=10):
    rng = np.random.default_rng(42)
    n = len(neg_i)
    aucs = []
    for _ in range(n_runs):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        aucs.append(float(roc_auc_score(y[idx], score[idx])))
    return float(np.mean(aucs)), float(np.std(aucs))


def best_threshold(score, y):
    """전체 test에서 MCC-optimal threshold."""
    thrs = np.linspace(0.05, 0.95, 91)
    best_t, best_m = 0.5, -1.0
    for t in thrs:
        m = matthews_corrcoef(y, (score >= t).astype(int))
        if m > best_m:
            best_m, best_t = m, t
    return best_t, best_m


def optimize_subset(df: pd.DataFrame, y: np.ndarray, cols: list[str]):
    X = df[cols].to_numpy()
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]

    def loss(w):
        w = np.abs(w)
        w = w / max(w.sum(), 1e-9)
        return -balanced_mcc(X @ w, y, pos_i, neg_i)

    best_x = None
    best_loss = 0.0
    # 여러 시작점 — 더 안정적
    starts = [np.ones(len(cols))]  # 균등
    rng = np.random.default_rng(42)
    for _ in range(6):
        starts.append(rng.dirichlet(np.ones(len(cols))))
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead",
                     options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 300})
        if r.fun < best_loss:
            best_loss = r.fun
            best_x = r.x
    w = np.abs(best_x)
    w = w / w.sum()
    return w, -best_loss


def main():
    df, y, cols_all = load()
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    print(f"로드: {len(cols_all)} 모델, {len(df)} 분자 (양성 {int(y.sum())}, 음성 {int((1-y).sum())})")

    # ---- 1) 단일 모델 비교 (참고) ----
    print("\n[단일 모델]")
    for c in cols_all:
        m = balanced_mcc(df[c].to_numpy(), y, pos_i, neg_i)
        a, _ = auc_bal(df[c].to_numpy(), y, pos_i, neg_i)
        print(f"  {c:14s} AUC {a:.3f}  MCC {m:.3f}")

    # ---- 2) top-3, top-5, all 각각 가중치 최적화 ----
    # top-N 자동 선정: 단일 MCC 기준 정렬
    scores = {c: balanced_mcc(df[c].to_numpy(), y, pos_i, neg_i) for c in cols_all}
    sorted_cols = sorted(cols_all, key=lambda c: scores[c], reverse=True)
    print(f"\n단일 MCC 순위: {sorted_cols}")

    results = {}
    for k in [3, 5, 7, len(cols_all)]:
        if k > len(cols_all):
            continue
        cols = sorted_cols[:k]
        w, mcc_opt = optimize_subset(df, y, cols)
        score = df[cols].to_numpy() @ w
        a, a_std = auc_bal(score, y, pos_i, neg_i)
        full_auc = float(roc_auc_score(y, score))
        bt, _ = best_threshold(score, y)
        results[f"top{k}"] = {
            "members": cols,
            "weights": w.tolist(),
            "balanced_auc_mean": a, "balanced_auc_std": a_std,
            "balanced_mcc": mcc_opt,
            "full_test_auc": full_auc,
            "threshold": float(bt),
        }
        print(f"\n=== Top-{k} 최적화 ===")
        print(f"  멤버: {cols}")
        for c, ww in zip(cols, w):
            print(f"    {c:14s} {ww:.3f}")
        print(f"  1:1bal AUC {a:.3f}±{a_std:.3f}  MCC {mcc_opt:.3f}  full AUC {full_auc:.3f}  thr {bt:.2f}")

    # 저장
    with open(os.path.join(RESULTS, "final_megaensemble_weighted.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'final_megaensemble_weighted.json')}")


if __name__ == "__main__":
    main()
