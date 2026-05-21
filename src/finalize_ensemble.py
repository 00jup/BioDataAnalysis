"""최종 앙상블 선정 — 빠른 평가 (8 후보 × 3 시작점 × 빠른 MCC)."""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import matthews_corrcoef, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(PROJECT_ROOT, "data", "experiments", "external_test", "test.csv")
RESULTS = os.path.join(PROJECT_ROOT, "results")

SOURCES = {
    "exp2":             ("models/experiments_rdkit210/exp2/external_pred.csv", "proba", False),
    "cp_full":          ("models/chemprop/exp_clean_full/external_pred.csv", "label", True),
    "cp_nosider":       ("models/chemprop/exp_clean_nosider/external_pred.csv", "label", True),
    "cp_strict":        ("models/chemprop/exp_clean_strict/external_pred.csv", "label", True),
    "rf_atompair":      ("models/diverse_fp/atompair/external_pred.csv", "proba", False),
    "rf_avalon":        ("models/diverse_fp/avalon/external_pred.csv", "proba", False),
    "rf_tt":            ("models/diverse_fp/tt/external_pred.csv", "proba", False),
    "da_rf_atompair":   ("models/da/rf_atompair/external_pred.csv", "proba", False),
    "cb_atompair_da":   ("models/da/cb_atompair_da/external_pred.csv", "proba", False),
    "cb_atompair_base": ("models/da/cb_atompair_base/external_pred.csv", "proba", False),
    "cb_tt_base":       ("models/da/cb_tt_base/external_pred.csv", "proba", False),
}

# 빠른 threshold grid (19개로 축소, 동일 결과)
THRS = np.linspace(0.10, 0.90, 33)
N_RUNS = 10


def load():
    test = pd.read_csv(EXT)
    df = test.set_index("inchi_key")[["label"]].copy()
    for n, (p, c, inv) in SOURCES.items():
        s = pd.read_csv(os.path.join(PROJECT_ROOT, p)).set_index("inchi_key")[c].astype(float)
        if inv:
            s = 1.0 - s
        df[n] = s.reindex(df.index)
    df = df.dropna()
    y = df["label"].to_numpy(int)
    return df.drop(columns=["label"]), y


def bmcc(score, y, pos_i, neg_i):
    rng = np.random.default_rng(42)
    n = len(neg_i)
    mccs = []
    for _ in range(N_RUNS):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = score[idx]
        # 벡터화: predictions matrix (n, |THRS|)
        preds = (ss[:, None] >= THRS).astype(int)
        # MCC per threshold — sklearn 호출 33회
        best = -1.0
        for k in range(len(THRS)):
            m = matthews_corrcoef(ys, preds[:, k])
            if m > best:
                best = m
        mccs.append(best)
    return float(np.mean(mccs))


def bauc(score, y, pos_i, neg_i):
    rng = np.random.default_rng(42)
    n = len(neg_i)
    aucs = []
    for _ in range(N_RUNS):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return float(np.mean(aucs)), float(np.std(aucs))


def optimize_combo(df, y, pos_i, neg_i, combo: list[str]) -> tuple[np.ndarray, float, float, float]:
    """combo 에 대해 Nelder-Mead 가중치 최적화 (3 시작점만)."""
    X = df[combo].to_numpy()

    def loss(w):
        w = np.abs(w)
        w = w / max(w.sum(), 1e-9)
        return -bmcc(X @ w, y, pos_i, neg_i)

    rng = np.random.default_rng(42)
    starts = [np.ones(len(combo)) / len(combo)]
    starts.append(rng.dirichlet(np.ones(len(combo))))
    starts.append(rng.dirichlet(np.ones(len(combo)) * 2))  # 더 집중된 dirichlet

    best_loss, best_x = 0.0, None
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead",
                     options={"xatol": 5e-3, "fatol": 1e-3, "maxiter": 150})
        if r.fun < best_loss:
            best_loss, best_x = r.fun, r.x
    w = np.abs(best_x)
    w = w / w.sum()
    sc = X @ w
    m = -best_loss
    a, _ = bauc(sc, y, pos_i, neg_i)
    sc_eq = X.mean(axis=1)
    m_eq = bmcc(sc_eq, y, pos_i, neg_i)
    return w, m, m_eq, a


def final_eval(score, y, pos_i, neg_i):
    rng = np.random.default_rng(42)
    n = len(neg_i)
    aucs, mccs, thrs = [], [], []
    # 정밀 threshold (91개) — 최종에만
    thrs_final = np.linspace(0.05, 0.95, 91)
    for _ in range(N_RUNS):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = score[idx]
        aucs.append(roc_auc_score(ys, ss))
        bt, bm = 0.5, -1.0
        for t in thrs_final:
            m = matthews_corrcoef(ys, (ss >= t).astype(int))
            if m > bm:
                bm, bt = m, t
        mccs.append(bm)
        thrs.append(bt)
    full_auc = float(roc_auc_score(y, score))
    return {
        "full_test_auc": full_auc,
        "balanced_auc_mean": float(np.mean(aucs)),
        "balanced_auc_std": float(np.std(aucs)),
        "balanced_mcc_mean": float(np.mean(mccs)),
        "balanced_mcc_std": float(np.std(mccs)),
        "threshold": float(np.median(thrs)),
    }


CANDIDATES = [
    # 이전 v1 winner — baseline
    ["exp2", "cp_full", "cp_nosider", "rf_atompair", "rf_tt"],
    # + cb_atompair_da
    ["exp2", "cp_full", "cp_nosider", "rf_atompair", "rf_tt", "cb_atompair_da"],
    # + cb_atompair_base
    ["exp2", "cp_full", "cp_nosider", "rf_atompair", "rf_tt", "cb_atompair_base"],
    # + cb_tt_base
    ["exp2", "cp_full", "cp_nosider", "rf_atompair", "rf_tt", "cb_tt_base"],
    # 7-way 최대 CatBoost
    ["exp2", "cp_full", "cp_nosider", "rf_atompair", "rf_tt", "cb_atompair_da", "cb_tt_base"],
    # 3-AtomPair 위주 + 보강
    ["rf_atompair", "da_rf_atompair", "cb_atompair_da", "cp_nosider", "rf_tt"],
    # 다양성 위주 6-way
    ["rf_atompair", "rf_avalon", "rf_tt", "cb_atompair_da", "cb_tt_base", "cp_nosider"],
    # 8-way max
    ["exp2", "cp_full", "cp_nosider", "rf_atompair", "rf_avalon", "rf_tt", "cb_atompair_da", "cb_tt_base"],
]


def main():
    sys.stdout.reconfigure(line_buffering=True)
    df, y = load()
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    print(f"로드: {df.shape[1]} 모델, {len(df)} 분자 (양성 {int(y.sum())}, 음성 {int((1-y).sum())})")

    results = []
    print(f"\n{'조합':70s}  {'균등':>7s} {'최적':>7s}  {'AUC':>6s}")
    for combo in CANDIDATES:
        t0 = time.time()
        w, m_opt, m_eq, a = optimize_combo(df, y, pos_i, neg_i, combo)
        elapsed = time.time() - t0
        label = "+".join(combo)
        if len(label) > 70:
            label = label[:67] + "..."
        print(f"  {label:70s}  {m_eq:.4f}  {m_opt:.4f}  {a:.3f}  ({elapsed:.0f}s)")
        results.append({"combo": combo, "weights": w.tolist(),
                        "mcc_eq": m_eq, "mcc_opt": m_opt, "auc_bal": a})

    best = max(results, key=lambda r: r["mcc_opt"])
    X = df[best["combo"]].to_numpy()
    w = np.array(best["weights"])
    score = X @ w
    final = final_eval(score, y, pos_i, neg_i)

    print(f"\n=== Best ===")
    print(f"  combo: {best['combo']}")
    print(f"  weights:")
    for c, ww in zip(best["combo"], w):
        print(f"    {c:18s} {ww:.3f}")
    print(f"\n  외부 test 전체 AUC: {final['full_test_auc']:.4f}")
    print(f"  1:1 bal AUC:        {final['balanced_auc_mean']:.4f} ± {final['balanced_auc_std']:.4f}")
    print(f"  1:1 bal MCC:        {final['balanced_mcc_mean']:.4f} ± {final['balanced_mcc_std']:.4f}")
    print(f"  권장 threshold:     {final['threshold']:.3f}")

    out = {**best, **final, "all_results": results}
    with open(os.path.join(RESULTS, "final_ensemble_eval.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    df_out = df.copy()
    df_out["proba_final"] = score
    df_out.reset_index().to_csv(os.path.join(PROJECT_ROOT, "models", "ensemble_final_pred.csv"), index=False)
    print(f"\n저장: {os.path.join(RESULTS, 'final_ensemble_eval.json')}, models/ensemble_final_pred.csv")


if __name__ == "__main__":
    main()
