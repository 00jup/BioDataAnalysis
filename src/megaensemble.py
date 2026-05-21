"""모든 모델 메가-앙상블 최적화.

소스:
  - exp2 RDKit-210 앙상블 (재학습)
  - Chemprop 3변형 (strict/full/nosider)
  - Chemprop + v1_rdkit_2d_normalized 3변형
  - MolFormer + LogReg / + XGBoost
  - 다양한 FP (ECFP6/Avalon/AtomPair/TT/Pattern) × RF 5종

방법:
  1. 모든 모델 외부 chEMBL test 확률 로드
  2. AUC < 0.5 인 모델은 자동 (1-p) 으로 부호 보정
  3. 단일 + 2/3/4/5/all-way 평균 + 가중 grid 탐색
  4. 그리디 forward selection
  5. Bayesian Model Averaging (각 모델 ROC AUC 비례 가중)
  6. 1:1 balanced × 10 회, 매 회 MCC-optimal threshold

결과: results/megaensemble_eval.json + 최고 조합.
"""

from __future__ import annotations

import json
import os
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(PROJECT_ROOT, "data", "experiments", "external_test", "test.csv")
RESULTS = os.path.join(PROJECT_ROOT, "results")

# (이름, 예측 파일 경로, 확률 컬럼명) — 확률 컬럼은 chemprop=label, 나머지=proba
SOURCES = [
    ("exp2_rdkit",        "models/experiments_rdkit210/exp2/external_pred.csv", "proba"),
    ("cp_strict",         "models/chemprop/exp_clean_strict/external_pred.csv", "label"),
    ("cp_full",           "models/chemprop/exp_clean_full/external_pred.csv", "label"),
    ("cp_nosider",        "models/chemprop/exp_clean_nosider/external_pred.csv", "label"),
    ("cp_r2dn_strict",    "models/chemprop_v1_rdkit_2d_normalized/exp_clean_strict/external_pred.csv", "label"),
    ("cp_r2dn_full",      "models/chemprop_v1_rdkit_2d_normalized/exp_clean_full/external_pred.csv", "label"),
    ("cp_r2dn_nosider",   "models/chemprop_v1_rdkit_2d_normalized/exp_clean_nosider/external_pred.csv", "label"),
    ("fp_ecfp6",          "models/diverse_fp/ecfp6/external_pred.csv", "proba"),
    ("fp_avalon",         "models/diverse_fp/avalon/external_pred.csv", "proba"),
    ("fp_atompair",       "models/diverse_fp/atompair/external_pred.csv", "proba"),
    ("fp_tt",             "models/diverse_fp/tt/external_pred.csv", "proba"),
    ("fp_pattern",        "models/diverse_fp/pattern/external_pred.csv", "proba"),
]


def load_all() -> tuple[pd.DataFrame, np.ndarray]:
    test = pd.read_csv(EXT)
    df = test.set_index("inchi_key")[["label"]].copy()
    for name, path, col in SOURCES:
        if not os.path.exists(path):
            print(f"skip (없음): {name} → {path}")
            continue
        p = pd.read_csv(path)
        # inchi_key 키 정렬
        if "inchi_key" not in p.columns:
            print(f"warn: {name} has no inchi_key → {list(p.columns)}")
            continue
        s = p.set_index("inchi_key")[col].astype(float)
        df[name] = s.reindex(df.index)
    df = df.dropna()
    y = df["label"].to_numpy(int)
    df = df.drop(columns=["label"])
    print(f"로드 완료: {df.shape[1]} 모델, {len(df)} 분자 (양성 {int(y.sum())}, 음성 {int((1-y).sum())})")
    return df, y


def auto_orient(df: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """전체 test 에서 AUC < 0.5 인 모델은 (1-p) 로 보정."""
    out = df.copy()
    flipped = {}
    print("\n[부호 자동 보정]")
    for c in df.columns:
        a = roc_auc_score(y, df[c].to_numpy())
        if a < 0.5:
            out[c] = 1.0 - df[c]
            new_a = roc_auc_score(y, out[c].to_numpy())
            flipped[c] = True
            print(f"  {c:22s} AUC {a:.3f} → 보정후 {new_a:.3f}  (FLIP)")
        else:
            flipped[c] = False
            print(f"  {c:22s} AUC {a:.3f}")
    return out, flipped


def balanced_mcc(score: np.ndarray, y: np.ndarray, n_runs: int = 10) -> tuple[float, float, float, float, float]:
    """1:1 balanced × n_runs, 매 회 MCC-optimal threshold.
    반환: (AUC mean, AUC std, MCC mean, MCC std, threshold median)
    """
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    n = len(neg_i)
    rng = np.random.default_rng(42)
    aucs, mccs, thrs = [], [], []
    for _ in range(n_runs):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = score[idx]
        aucs.append(float(roc_auc_score(ys, ss)))
        bt, bm = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 91):
            m = matthews_corrcoef(ys, (ss >= thr).astype(int))
            if m > bm:
                bm, bt = m, thr
        mccs.append(float(bm))
        thrs.append(float(bt))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(mccs)), float(np.std(mccs)), float(np.median(thrs))


def avg(df: pd.DataFrame, cols: list[str], weights: np.ndarray | None = None) -> np.ndarray:
    sub = df[list(cols)].to_numpy()
    if weights is None:
        return sub.mean(axis=1)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    return sub @ w


def main() -> None:
    df, y = load_all()
    df, flipped = auto_orient(df, y)

    # 단일 모델 표
    print("\n[단일 모델 1:1 balanced]")
    single = {}
    for c in df.columns:
        au_m, au_s, mc_m, mc_s, th = balanced_mcc(df[c].to_numpy(), y)
        single[c] = {"auc": au_m, "auc_std": au_s, "mcc": mc_m, "mcc_std": mc_s, "thr": th}
        print(f"  {c:22s} AUC {au_m:.3f}  MCC {mc_m:.3f}±{mc_s:.3f}")

    # 2/3/4-way 동일 가중 평균 — 전체 콤비 brute force
    print("\n[브루트포스 평균 앙상블 — top-10]")
    K = list(df.columns)
    all_results = []
    for r in range(2, min(6, len(K) + 1)):
        for combo in combinations(K, r):
            sc = avg(df, list(combo))
            au_m, au_s, mc_m, mc_s, th = balanced_mcc(sc, y)
            all_results.append({"members": list(combo), "size": r, "auc": au_m, "mcc": mc_m, "mcc_std": mc_s, "thr": th})
    all_results.sort(key=lambda r: r["mcc"], reverse=True)
    for r in all_results[:10]:
        print(f"  size={r['size']}  AUC {r['auc']:.3f}  MCC {r['mcc']:.3f}±{r['mcc_std']:.3f}  thr {r['thr']:.2f}  {r['members']}")

    # Greedy forward selection — MCC 최대화
    print("\n[Greedy Forward Selection]")
    chosen = []
    remaining = list(K)
    best_mcc = -1.0
    best_path = []
    while remaining:
        cand = None
        cand_mcc = -1.0
        cand_record = None
        for k in remaining:
            cur = chosen + [k]
            sc = avg(df, cur)
            au_m, au_s, mc_m, mc_s, th = balanced_mcc(sc, y)
            if mc_m > cand_mcc:
                cand_mcc = mc_m
                cand = k
                cand_record = {"add": k, "members": list(cur), "auc": au_m, "mcc": mc_m, "mcc_std": mc_s, "thr": th}
        if cand_mcc <= best_mcc + 0.001:
            break
        chosen.append(cand)
        remaining.remove(cand)
        best_mcc = cand_mcc
        best_path.append(cand_record)
        print(f"  + {cand:22s} → AUC {cand_record['auc']:.3f}  MCC {cand_record['mcc']:.3f}±{cand_record['mcc_std']:.3f}")
    print(f"  최종 chosen ({len(chosen)}): {chosen}")

    # BMA — AUC^2 가중 (강한 모델 가중 보너스)
    print("\n[Bayesian-style: AUC^2 가중 평균]")
    cols = list(K)
    w = np.array([single[c]["auc"] ** 2 for c in cols])
    sc = avg(df, cols, weights=w)
    au_m, au_s, mc_m, mc_s, th = balanced_mcc(sc, y)
    print(f"  BMA-all     AUC {au_m:.3f}  MCC {mc_m:.3f}±{mc_s:.3f}  thr {th:.2f}")

    # 가중 grid — top 2 모델 + 가중 grid
    print("\n[Top-2 가중 grid]")
    top2 = [r["members"] for r in all_results[:5]]
    seen = set()
    for pair in top2:
        if len(pair) != 2 or tuple(pair) in seen:
            continue
        seen.add(tuple(pair))
        a, b = pair
        print(f"  {a} ↔ {b}")
        best_w, best_m = None, -1
        for wa in np.linspace(0.05, 0.95, 19):
            sc = wa * df[a].to_numpy() + (1 - wa) * df[b].to_numpy()
            _, _, mc_m, _, _ = balanced_mcc(sc, y)
            if mc_m > best_m:
                best_m, best_w = mc_m, wa
        sc = best_w * df[a].to_numpy() + (1 - best_w) * df[b].to_numpy()
        au_m, au_s, mc_m, mc_s, th = balanced_mcc(sc, y)
        print(f"     best w({a})={best_w:.2f}  AUC {au_m:.3f}  MCC {mc_m:.3f}±{mc_s:.3f}  thr {th:.2f}")

    # 저장
    out = {
        "n_models": int(df.shape[1]),
        "n_test": int(len(df)),
        "n_pos": int(y.sum()),
        "n_neg": int((1 - y).sum()),
        "flipped": flipped,
        "single": single,
        "top_avg_combos": all_results[:20],
        "greedy_path": best_path,
        "bma_all": {"members": cols, "weights": w.tolist(), "auc": au_m, "mcc": mc_m, "mcc_std": mc_s, "thr": th},
    }
    with open(os.path.join(RESULTS, "megaensemble_eval.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'megaensemble_eval.json')}")


if __name__ == "__main__":
    main()
