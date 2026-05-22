"""두 가지 강건성 검증.

A) 음성 추가 실험 — chEMBL train 에 시판약 음성 5,000 추가 시 결과 변화
   - chembl_clean/train (485+210) → augmented (485 + 210+5000 = 5695)
   - val/test 와 InChIKey 누수 자동 제거
   - 10-way 재학습 → 같은 chembl_clean/test 에서 보고

B) 5-fold CV variance — 결과 들쭉날쭉한지 확인
   - 전체 fresh chEMBL 풀 994 stratified 5-fold
   - 각 fold 마다 10-way 학습+threshold→test
   - MCC/TPR/TNR mean±std
"""
from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import StratifiedKFold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC = os.path.join(PROJECT_ROOT, "data", "experiments", "chembl_clean")
FP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments", "fp_cache")
RESULTS = os.path.join(PROJECT_ROOT, "results")
RANDOM_STATE = 42
FPS = ["ecfp6", "avalon", "atompair", "tt", "pattern"]


def fp_xy(fp_name: str, df: pd.DataFrame):
    fp = pd.read_parquet(os.path.join(FP_DIR, f"{fp_name}.parquet")).set_index("canonical_smiles")
    cols = list(fp.columns)
    mask = df["canonical_smiles"].isin(fp.index)
    sub = df[mask].reset_index(drop=True)
    X = fp.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
    y = sub["label"].to_numpy(int)
    return X, y, sub


def train_one(fp_name: str, kind: str, train_df, val_df, test_df):
    Xtr, ytr, _ = fp_xy(fp_name, train_df)
    Xv, yv, _ = fp_xy(fp_name, val_df)
    Xte, yte, _ = fp_xy(fp_name, test_df)
    spw = float((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    if kind == "rf":
        clf = RandomForestClassifier(n_estimators=500, max_depth=None, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    else:
        clf = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, scale_pos_weight=spw, verbose=0, random_seed=RANDOM_STATE)
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xv)[:,1], clf.predict_proba(Xte)[:,1], yv, yte


def optimize_weights(val_X, yv, test_X, yte):
    """val에서 가중치+threshold → test 보고."""
    THRS = np.linspace(0.05, 0.95, 91)
    def loss(w):
        w = np.abs(w); w = w / max(w.sum(),1e-9)
        score = val_X @ w
        return -max(matthews_corrcoef(yv, (score>=t).astype(int)) for t in THRS)

    rng = np.random.default_rng(RANDOM_STATE)
    starts = [np.ones(val_X.shape[1])/val_X.shape[1]] + [rng.dirichlet(np.ones(val_X.shape[1])) for _ in range(5)]
    best_loss, best_w = 0.0, None
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead", options={"xatol":5e-3,"fatol":1e-3,"maxiter":200})
        if r.fun < best_loss:
            best_loss, best_w = r.fun, r.x
    w = np.abs(best_w); w = w / w.sum()
    val_score = val_X @ w
    bt, bm = 0.5, -1.0
    for t in THRS:
        m = matthews_corrcoef(yv, (val_score>=t).astype(int))
        if m > bm: bm, bt = m, t
    test_score = test_X @ w
    pred = (test_score >= bt).astype(int)
    cm = confusion_matrix(yte, pred, labels=[1,0])
    tp,fn = cm[0]; fp,tn = cm[1]
    tpr = tp/max(tp+fn,1); tnr = tn/max(fp+tn,1)
    return {
        "weights": w.tolist(), "threshold": float(bt),
        "test_auc": float(roc_auc_score(yte, test_score)),
        "test_mcc": float(matthews_corrcoef(yte, pred)),
        "test_tpr": float(tpr), "test_tnr": float(tnr),
        "test_n": int(len(yte)), "test_pos": int(tp+fn), "test_neg": int(fp+tn),
    }


def run_one(train_df, val_df, test_df, label):
    print(f"\n=== {label} ===")
    print(f"  train: {len(train_df)} (양성 {int((train_df.label==1).sum())} / 음성 {int((train_df.label==0).sum())})")
    print(f"  val:   {len(val_df)} (양성 {int((val_df.label==1).sum())} / 음성 {int((val_df.label==0).sum())})")
    print(f"  test:  {len(test_df)} (양성 {int((test_df.label==1).sum())} / 음성 {int((test_df.label==0).sum())})")
    val_probs, test_probs = [], []
    yv_ref, yte_ref = None, None
    for fp in FPS:
        for kind in ("rf","cb"):
            pv, pte, yv, yte = train_one(fp, kind, train_df, val_df, test_df)
            val_probs.append(pv); test_probs.append(pte)
            if yv_ref is None: yv_ref, yte_ref = yv, yte
    Xv = np.array(val_probs).T; Xte = np.array(test_probs).T
    res = optimize_weights(Xv, yv_ref, Xte, yte_ref)
    print(f"  → test AUC {res['test_auc']:.3f}  MCC {res['test_mcc']:.3f}  TPR {res['test_tpr']:.3f}  TNR {res['test_tnr']:.3f}  (thr {res['threshold']:.3f})")
    return res


def main():
    train = pd.read_csv(os.path.join(CC, "train.csv"))
    val = pd.read_csv(os.path.join(CC, "val.csv"))
    test = pd.read_csv(os.path.join(CC, "test.csv"))
    full_pool = pd.read_csv(os.path.join(CC, "fresh_pool.csv"))

    summary = {}

    # ===== Baseline (재현) =====
    print("\n" + "="*70)
    print("STEP A.0 — Baseline (기존 10-way 재현)")
    print("="*70)
    summary["baseline"] = run_one(train, val, test, "baseline 10-way")

    # ===== A) 음성 추가 — 시판약 음성 N개 추가 =====
    print("\n" + "="*70)
    print("STEP A — 시판약 음성 추가 (chEMBL test/val InChIKey 비누수)")
    print("="*70)
    marketed_neg = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "marketed_drugs", "non_hepatotoxic", "marketed_clean.csv"))
    marketed_neg = marketed_neg.dropna(subset=["canonical_smiles","inchi_key"]).drop_duplicates("inchi_key")
    marketed_neg = marketed_neg[marketed_neg["dilirank_category"] != "Ambiguous-DILI-Concern"].copy()
    # chEMBL val/test/train 분자 제거 (chEMBL 전체 풀 제거)
    chembl_ik = set(full_pool["inchi_key"])
    marketed_neg = marketed_neg[~marketed_neg["inchi_key"].isin(chembl_ik)]
    marketed_neg = marketed_neg[["canonical_smiles","inchi_key"]].assign(label=0, source="marketed_neg_aug")
    print(f"\n시판약 음성 풀 (누수 제거 후): {len(marketed_neg)}")

    for n_add in [500, 2000, 5000]:
        aug_neg = marketed_neg.sample(n=min(n_add, len(marketed_neg)), random_state=RANDOM_STATE)
        aug_train = pd.concat([train, aug_neg[["canonical_smiles","inchi_key","label"]].assign(molecule_chembl_id="")], ignore_index=True)
        summary[f"aug_{n_add}"] = run_one(aug_train, val, test, f"chEMBL train + {n_add} 시판약 음성")

    # ===== B) 5-fold CV variance =====
    print("\n" + "="*70)
    print("STEP B — 5-fold CV (들쭉날쭉 검증)")
    print("="*70)
    y_full = full_pool["label"].to_numpy(int)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_results = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(full_pool, y_full)):
        full_tr = full_pool.iloc[tr_idx].reset_index(drop=True)
        # tr 안에서 다시 val 분리 (85/15)
        from sklearn.model_selection import train_test_split
        tr_inner, val_inner = train_test_split(full_tr, test_size=0.15, random_state=RANDOM_STATE, stratify=full_tr["label"])
        te = full_pool.iloc[te_idx].reset_index(drop=True)
        r = run_one(tr_inner, val_inner, te, f"5-fold CV (fold {fold+1}/5)")
        fold_results.append(r)

    mccs = [r["test_mcc"] for r in fold_results]
    aucs = [r["test_auc"] for r in fold_results]
    tprs = [r["test_tpr"] for r in fold_results]
    tnrs = [r["test_tnr"] for r in fold_results]
    summary["cv_5fold"] = {
        "per_fold": fold_results,
        "mcc_mean": float(np.mean(mccs)), "mcc_std": float(np.std(mccs)),
        "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs)),
        "tpr_mean": float(np.mean(tprs)), "tpr_std": float(np.std(tprs)),
        "tnr_mean": float(np.mean(tnrs)), "tnr_std": float(np.std(tnrs)),
    }
    print(f"\n=== 5-fold CV 요약 ===")
    print(f"  MCC {np.mean(mccs):.3f} ± {np.std(mccs):.3f}    (folds: {[round(m,3) for m in mccs]})")
    print(f"  AUC {np.mean(aucs):.3f} ± {np.std(aucs):.3f}")
    print(f"  TPR {np.mean(tprs):.3f} ± {np.std(tprs):.3f}")
    print(f"  TNR {np.mean(tnrs):.3f} ± {np.std(tnrs):.3f}")

    with open(os.path.join(RESULTS, "chembl_robustness.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n" + "="*70)
    print("=== 종합 비교 ===")
    print("="*70)
    print(f"  {'시나리오':30s} {'train n':>10s} {'AUC':>7s} {'MCC':>7s} {'TPR':>7s} {'TNR':>7s}")
    for k, r in summary.items():
        if k == "cv_5fold":
            print(f"  {'5-fold CV (평균)':30s} {'~795':>10s} {r['auc_mean']:7.3f} {r['mcc_mean']:7.3f} {r['tpr_mean']:7.3f} {r['tnr_mean']:7.3f}  (±{r['mcc_std']:.3f})")
        else:
            n = 695 + (int(k.split('_')[1]) if k.startswith("aug_") else 0)
            print(f"  {k:30s} {n:>10d} {r['test_auc']:7.3f} {r['test_mcc']:7.3f} {r['test_tpr']:7.3f} {r['test_tnr']:7.3f}")
    print(f"\n저장: {os.path.join(RESULTS, 'chembl_robustness.json')}")


if __name__ == "__main__":
    main()
