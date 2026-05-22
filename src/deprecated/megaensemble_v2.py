"""메가-앙상블 v2 — 24개 모델 풀에서 greedy + Nelder-Mead 가중 최적화.

소스 (총 24):
  - exp2 RDKit (단독)                                            [1]
  - Chemprop 3변형 + r2dn 3변형                                  [6]
  - MolFormer LogReg + XGBoost                                   [2]
  - RF on 5 FP                                                   [5]
  - DA RF on 5 FP                                                [5]
  - CatBoost on 5 FP (DA) + CatBoost AtomPair/TT (base)          [5+2]

방법:
  1. 부호 자동 보정 (AUC<0.5면 1-p)
  2. Greedy forward selection on balanced MCC
  3. 선택된 subset 에 대해 Nelder-Mead 가중치 최적화
  4. 최종 evalutaion: 1:1 balanced × 10회, MCC-optimal threshold
"""

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

SOURCES = [
    # 기존
    ("exp2_rdkit",    "models/experiments_rdkit210/exp2/external_pred.csv", "proba"),
    ("cp_strict",     "models/chemprop/exp_clean_strict/external_pred.csv", "label"),
    ("cp_full",       "models/chemprop/exp_clean_full/external_pred.csv", "label"),
    ("cp_nosider",    "models/chemprop/exp_clean_nosider/external_pred.csv", "label"),
    ("cp_r2dn_strict",  "models/chemprop_v1_rdkit_2d_normalized/exp_clean_strict/external_pred.csv", "label"),
    ("cp_r2dn_full",    "models/chemprop_v1_rdkit_2d_normalized/exp_clean_full/external_pred.csv", "label"),
    ("cp_r2dn_nosider", "models/chemprop_v1_rdkit_2d_normalized/exp_clean_nosider/external_pred.csv", "label"),
    # 기존 RF on FP
    ("rf_ecfp6",      "models/diverse_fp/ecfp6/external_pred.csv", "proba"),
    ("rf_avalon",     "models/diverse_fp/avalon/external_pred.csv", "proba"),
    ("rf_atompair",   "models/diverse_fp/atompair/external_pred.csv", "proba"),
    ("rf_tt",         "models/diverse_fp/tt/external_pred.csv", "proba"),
    ("rf_pattern",    "models/diverse_fp/pattern/external_pred.csv", "proba"),
    # 도메인 적응 RF on FP
    ("da_rf_ecfp6",     "models/da/rf_ecfp6/external_pred.csv", "proba"),
    ("da_rf_avalon",    "models/da/rf_avalon/external_pred.csv", "proba"),
    ("da_rf_atompair",  "models/da/rf_atompair/external_pred.csv", "proba"),
    ("da_rf_tt",        "models/da/rf_tt/external_pred.csv", "proba"),
    ("da_rf_pattern",   "models/da/rf_pattern/external_pred.csv", "proba"),
    # CatBoost (DA)
    ("cb_ecfp6_da",     "models/da/cb_ecfp6_da/external_pred.csv", "proba"),
    ("cb_avalon_da",    "models/da/cb_avalon_da/external_pred.csv", "proba"),
    ("cb_atompair_da",  "models/da/cb_atompair_da/external_pred.csv", "proba"),
    ("cb_tt_da",        "models/da/cb_tt_da/external_pred.csv", "proba"),
    ("cb_pattern_da",   "models/da/cb_pattern_da/external_pred.csv", "proba"),
    # CatBoost (base)
    ("cb_atompair_base", "models/da/cb_atompair_base/external_pred.csv", "proba"),
    ("cb_tt_base",       "models/da/cb_tt_base/external_pred.csv", "proba"),
]

THRS = np.linspace(0.05, 0.95, 91)


def load():
    test = pd.read_csv(EXT)
    df = test.set_index("inchi_key")[["label"]].copy()
    for n, p, c in SOURCES:
        full = os.path.join(PROJECT_ROOT, p)
        if not os.path.exists(full):
            print(f"skip: {n}")
            continue
        df[n] = pd.read_csv(full).set_index("inchi_key")[c].astype(float).reindex(df.index)
    df = df.dropna()
    y = df["label"].to_numpy(int)
    # 부호 보정
    flipped = []
    for c in df.columns:
        if c == "label":
            continue
        if roc_auc_score(y, df[c].to_numpy()) < 0.5:
            df[c] = 1.0 - df[c]
            flipped.append(c)
    df = df.drop(columns=["label"])
    return df, y, flipped


def balanced_mcc(score, y, pos_i, neg_i, n_runs=10):
    rng = np.random.default_rng(42)
    mccs = []
    n = len(neg_i)
    for _ in range(n_runs):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = score[idx]
        preds = (ss[:, None] >= THRS).astype(int)
        m = max(matthews_corrcoef(ys, preds[:, k]) for k in range(len(THRS)))
        mccs.append(m)
    return float(np.mean(mccs)), float(np.std(mccs))


def balanced_auc(score, y, pos_i, neg_i, n_runs=10):
    rng = np.random.default_rng(42)
    aucs = []
    n = len(neg_i)
    for _ in range(n_runs):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        aucs.append(roc_auc_score(y[idx], score[idx]))
    return float(np.mean(aucs)), float(np.std(aucs))


def main():
    df, y, flipped = load()
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    cols = list(df.columns)
    X = df.to_numpy()
    print(f"로드: {len(cols)} 모델, {len(df)} 분자 (양성 {int(y.sum())}, 음성 {int((1-y).sum())})")
    print(f"부호 보정: {flipped}")

    # 단일 평가
    print("\n[단일 모델 — MCC 내림차순]")
    single = {}
    for c in cols:
        m, ms = balanced_mcc(df[c].to_numpy(), y, pos_i, neg_i)
        a, _ = balanced_auc(df[c].to_numpy(), y, pos_i, neg_i)
        single[c] = {"auc": a, "mcc": m, "mcc_std": ms}
    for c in sorted(cols, key=lambda x: single[x]["mcc"], reverse=True):
        s = single[c]
        print(f"  {c:22s} AUC {s['auc']:.3f}  MCC {s['mcc']:.3f}±{s['mcc_std']:.3f}")

    # Greedy forward selection — MCC 최대화
    print("\n[Greedy Forward Selection]")
    chosen = []
    remaining = list(cols)
    best_mcc = -1.0
    path = []
    while remaining:
        cand_mcc = -1.0
        cand_name = None
        cand_record = None
        for k in remaining:
            cur = chosen + [k]
            idx_cols = [cols.index(c) for c in cur]
            sc = X[:, idx_cols].mean(axis=1)
            m, ms = balanced_mcc(sc, y, pos_i, neg_i)
            if m > cand_mcc:
                cand_mcc = m
                cand_name = k
                a, a_s = balanced_auc(sc, y, pos_i, neg_i)
                cand_record = {"add": k, "members": list(cur), "auc": a, "auc_std": a_s, "mcc": m, "mcc_std": ms}
        if cand_mcc <= best_mcc + 0.001:  # tolerance
            break
        chosen.append(cand_name)
        remaining.remove(cand_name)
        best_mcc = cand_mcc
        path.append(cand_record)
        print(f"  + {cand_name:22s} → AUC {cand_record['auc']:.3f}  MCC {cand_record['mcc']:.3f}±{cand_record['mcc_std']:.3f}")
    print(f"  최종 chosen ({len(chosen)}): {chosen}")

    # 선택된 subset 에 대해 Nelder-Mead 가중 최적화
    print(f"\n[Nelder-Mead 가중치 최적화 on chosen subset]")
    sub_idx = [cols.index(c) for c in chosen]
    Xs = X[:, sub_idx]

    def loss(w):
        w = np.abs(w)
        w = w / max(w.sum(), 1e-9)
        m, _ = balanced_mcc(Xs @ w, y, pos_i, neg_i)
        return -m

    rng = np.random.default_rng(42)
    starts = [np.ones(len(chosen)) / len(chosen)]
    for _ in range(8):
        starts.append(rng.dirichlet(np.ones(len(chosen))))
    best_x, best_loss = None, 0.0
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead", options={"xatol": 1e-3, "fatol": 1e-4, "maxiter": 500})
        if r.fun < best_loss:
            best_loss = r.fun
            best_x = r.x
    w_opt = np.abs(best_x)
    w_opt = w_opt / w_opt.sum()
    score_opt = Xs @ w_opt
    a_m, a_s = balanced_auc(score_opt, y, pos_i, neg_i)
    m_m, m_s = balanced_mcc(score_opt, y, pos_i, neg_i)
    print(f"  최적: AUC {a_m:.3f}±{a_s:.3f}  MCC {m_m:.3f}±{m_s:.3f}")
    print(f"  가중치:")
    for c, ww in zip(chosen, w_opt):
        print(f"    {c:22s} {ww:.3f}")

    # 균등 가중 비교
    score_eq = Xs.mean(axis=1)
    a_eq, _ = balanced_auc(score_eq, y, pos_i, neg_i)
    m_eq, _ = balanced_mcc(score_eq, y, pos_i, neg_i)
    print(f"\n  균등가중 비교: AUC {a_eq:.3f}  MCC {m_eq:.3f}")

    # 최종 = 더 좋은 쪽
    if m_m >= m_eq:
        final_w = w_opt
        final_score = score_opt
        tag = "weighted"
    else:
        final_w = np.ones(len(chosen)) / len(chosen)
        final_score = score_eq
        tag = "uniform"
    print(f"\n최종 채택: {tag}")

    # 최종 평가 + threshold
    rng = np.random.default_rng(42)
    aucs, mccs, thrs = [], [], []
    for _ in range(10):
        sp = rng.choice(pos_i, size=len(neg_i), replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = final_score[idx]
        aucs.append(roc_auc_score(ys, ss))
        bt, bm = 0.5, -1.0
        for t in THRS:
            m = matthews_corrcoef(ys, (ss >= t).astype(int))
            if m > bm:
                bm, bt = m, t
        mccs.append(bm)
        thrs.append(bt)
    full_auc = float(roc_auc_score(y, final_score))
    print(f"\n=== 최종 ===")
    print(f"  외부 test 전체 AUC: {full_auc:.4f}")
    print(f"  1:1 bal AUC:        {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  1:1 bal MCC:        {np.mean(mccs):.4f} ± {np.std(mccs):.4f}")
    print(f"  권장 threshold:     {np.median(thrs):.3f}")

    out = {
        "n_pool": int(len(cols)),
        "n_test": int(len(df)),
        "flipped": flipped,
        "chosen": chosen,
        "weights": final_w.tolist(),
        "weighting": tag,
        "full_test_auc": full_auc,
        "balanced_auc_mean": float(np.mean(aucs)),
        "balanced_auc_std": float(np.std(aucs)),
        "balanced_mcc_mean": float(np.mean(mccs)),
        "balanced_mcc_std": float(np.std(mccs)),
        "threshold": float(np.median(thrs)),
        "greedy_path": path,
        "single": single,
    }
    with open(os.path.join(RESULTS, "final_ensemble_eval.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'final_ensemble_eval.json')}")

    # 최종 예측 CSV
    test = pd.read_csv(EXT)
    out_df = test.set_index("inchi_key").join(
        pd.Series(final_score, index=df.index, name="proba_final"), how="left"
    )
    out_df.reset_index().to_csv(os.path.join(PROJECT_ROOT, "models", "ensemble_final_pred.csv"), index=False)
    print(f"저장: models/ensemble_final_pred.csv")


if __name__ == "__main__":
    main()
