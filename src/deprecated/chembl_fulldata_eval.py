"""FULL DATA 시나리오 — 양성·음성 모든 출처 합치고 test/val 겹침만 제거.

학습 데이터 = chEMBL pool ∪ 시판약 양성·음성 − chembl_clean(val ∪ test)
평가 = chembl_clean/val (튜닝) → chembl_clean/test (보고)

같은 holdout 비교가 핵심.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem, RDLogger
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC = os.path.join(PROJECT_ROOT, "data", "experiments", "chembl_clean")
MD = os.path.join(PROJECT_ROOT, "data", "marketed_drugs")
FP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments", "fp_cache")
RESULTS = os.path.join(PROJECT_ROOT, "results")
FPS = ["ecfp6", "avalon", "atompair", "tt", "pattern"]
RANDOM_STATE = 42


def canonicalize(s):
    if not isinstance(s, str) or not s.strip(): return None
    m = Chem.MolFromSmiles(s)
    return Chem.MolToSmiles(m) if m else None


def build_full_train(val_ik: set, test_ik: set):
    """모든 양성/음성 합치고 (val ∪ test) 누수만 제거. 라벨 충돌은 제외."""
    holdout = val_ik | test_ik

    # 양성 출처 1: chembl_clean train (이미 정제됨)
    cc_train = pd.read_csv(os.path.join(CC, "train.csv"))
    # 양성 출처 2: 시판약 hepatotoxic_all_lenient
    md_pos = pd.read_csv(os.path.join(MD, "hepatotoxic", "hepatotoxic_all_lenient.csv"))
    md_pos = md_pos.dropna(subset=["canonical_smiles","inchi_key"]).drop_duplicates("inchi_key")
    md_pos["label"] = 1
    # 양성 출처 3: external_positives (DILIst + GoldStandard)
    ext_pos = pd.read_csv(os.path.join(MD, "hepatotoxic", "external", "external_positives.csv"))
    ext_pos = ext_pos.dropna(subset=["canonical_smiles","inchi_key"]).drop_duplicates("inchi_key")
    ext_pos["label"] = 1

    # 음성 출처 1: chembl_clean train (음성)  ← 이미 포함됨
    # 음성 출처 2: 시판약 marketed_clean (Ambiguous 제외)
    md_neg = pd.read_csv(os.path.join(MD, "non_hepatotoxic", "marketed_clean.csv"))
    md_neg = md_neg.dropna(subset=["canonical_smiles","inchi_key"]).drop_duplicates("inchi_key")
    md_neg = md_neg[md_neg["dilirank_category"] != "Ambiguous-DILI-Concern"].copy()
    md_neg["label"] = 0

    pieces = [
        cc_train[["canonical_smiles","inchi_key","label"]],
        md_pos[["canonical_smiles","inchi_key","label"]],
        ext_pos[["canonical_smiles","inchi_key","label"]],
        md_neg[["canonical_smiles","inchi_key","label"]],
    ]
    full = pd.concat(pieces, ignore_index=True)
    full = full.dropna(subset=["canonical_smiles","inchi_key"]).drop_duplicates("inchi_key")
    n_before = len(full)
    # holdout 제거
    full = full[~full["inchi_key"].isin(holdout)]
    # InChIKey 라벨 충돌 제거
    g = full.groupby("inchi_key")["label"].nunique()
    conflict = set(g[g > 1].index)
    if conflict:
        full = full[~full["inchi_key"].isin(conflict)]

    print(f"  raw 합집합: {n_before}, holdout 제거 후: {n_before - (n_before - len(full) - len(conflict))}, 라벨충돌 {len(conflict)} 제거 → {len(full)}")
    return full


def fp_xy(fp_name, df):
    fp = pd.read_parquet(os.path.join(FP_DIR, f"{fp_name}.parquet")).set_index("canonical_smiles")
    cols = list(fp.columns)
    mask = df["canonical_smiles"].isin(fp.index)
    sub = df[mask].reset_index(drop=True)
    if len(sub) < len(df):
        # FP 캐시에 없는 분자가 있으면 계산 후 캐시 보강
        from rdkit.Chem.rdFingerprintGenerator import GetAtomPairGenerator, GetMorganGenerator, GetTopologicalTorsionGenerator
        from rdkit.Avalon import pyAvalonTools
        nBits_map = {"ecfp6":2048,"avalon":512,"atompair":2048,"tt":2048,"pattern":2048}
        missing = sorted(set(df["canonical_smiles"]) - set(fp.index))
        nBits = nBits_map[fp_name]
        arr = np.zeros((len(missing), nBits), dtype=np.uint8)
        for i, smi in enumerate(missing):
            m = Chem.MolFromSmiles(smi)
            if m is None: continue
            try:
                if fp_name == "ecfp6":
                    f = GetMorganGenerator(radius=3, fpSize=nBits).GetFingerprint(m)
                elif fp_name == "avalon":
                    f = pyAvalonTools.GetAvalonFP(m, nBits=nBits)
                elif fp_name == "atompair":
                    f = GetAtomPairGenerator(fpSize=nBits).GetFingerprint(m)
                elif fp_name == "tt":
                    f = GetTopologicalTorsionGenerator(fpSize=nBits).GetFingerprint(m)
                elif fp_name == "pattern":
                    f = Chem.PatternFingerprint(m, fpSize=nBits)
                arr[i, list(f.GetOnBits())] = 1
            except Exception:
                pass
        cols_new = [f"{fp_name}_{j}" for j in range(nBits)]
        new_df = pd.DataFrame(arr, columns=cols_new)
        new_df.insert(0, "canonical_smiles", missing)
        merged = pd.concat([fp.reset_index(), new_df]).drop_duplicates("canonical_smiles").set_index("canonical_smiles")
        merged.reset_index().to_parquet(os.path.join(FP_DIR, f"{fp_name}.parquet"), index=False)
        fp = merged
        sub = df.reset_index(drop=True)
    X = fp.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
    y = sub["label"].to_numpy(int)
    return X, y, sub


def train_one(fp, kind, train_df, val_df, test_df):
    Xtr, ytr, _ = fp_xy(fp, train_df)
    Xv, yv, _ = fp_xy(fp, val_df)
    Xte, yte, _ = fp_xy(fp, test_df)
    spw = float((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    if kind == "rf":
        c = RandomForestClassifier(n_estimators=500, max_depth=None, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    else:
        c = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, scale_pos_weight=spw, verbose=0, random_seed=RANDOM_STATE)
    c.fit(Xtr, ytr)
    return c.predict_proba(Xv)[:,1], c.predict_proba(Xte)[:,1], yv, yte


def optimize_and_eval(Xv, yv, Xte, yte):
    THRS = np.linspace(0.05, 0.95, 91)
    def loss(w):
        w = np.abs(w); w = w/max(w.sum(),1e-9)
        s = Xv @ w
        return -max(matthews_corrcoef(yv, (s>=t).astype(int)) for t in THRS)
    rng = np.random.default_rng(RANDOM_STATE)
    starts = [np.ones(Xv.shape[1])/Xv.shape[1]] + [rng.dirichlet(np.ones(Xv.shape[1])) for _ in range(5)]
    best_loss, best_w = 0.0, None
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead", options={"xatol":5e-3,"fatol":1e-3,"maxiter":200})
        if r.fun < best_loss: best_loss, best_w = r.fun, r.x
    w = np.abs(best_w); w = w/w.sum()
    val_score = Xv @ w
    bt, bm = 0.5, -1.0
    for t in THRS:
        m = matthews_corrcoef(yv, (val_score>=t).astype(int))
        if m > bm: bm, bt = m, t
    test_score = Xte @ w
    pred = (test_score >= bt).astype(int)
    cm = confusion_matrix(yte, pred, labels=[1,0])
    tp,fn = cm[0]; fp,tn = cm[1]
    return {
        "weights": w.tolist(), "threshold": float(bt),
        "test_auc": float(roc_auc_score(yte, test_score)),
        "test_mcc": float(matthews_corrcoef(yte, pred)),
        "test_tpr": float(tp/max(tp+fn,1)), "test_tnr": float(tn/max(fp+tn,1)),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
    }


def main():
    val = pd.read_csv(os.path.join(CC, "val.csv"))
    test = pd.read_csv(os.path.join(CC, "test.csv"))
    val_ik = set(val["inchi_key"]); test_ik = set(test["inchi_key"])

    print("=== FULL DATA 학습 데이터 구축 ===")
    full = build_full_train(val_ik, test_ik)
    print(f"  최종 train: {len(full)} (양성 {int((full.label==1).sum())} / 음성 {int((full.label==0).sum())})")

    # 누수 자체검증
    assert len(set(full["inchi_key"]) & val_ik) == 0, "val 누수"
    assert len(set(full["inchi_key"]) & test_ik) == 0, "test 누수"
    print(f"  누수 검증: val 0, test 0 ✓")

    print(f"\n=== 10-way (RF+CatBoost × 5종 FP) FULL DATA 학습 ===")
    val_probs, test_probs = [], []
    yv_ref, yte_ref = None, None
    for fp in FPS:
        for kind in ("rf","cb"):
            print(f"  {kind}_{fp} ...", flush=True)
            pv, pte, yv, yte = train_one(fp, kind, full, val, test)
            val_probs.append(pv); test_probs.append(pte)
            if yv_ref is None: yv_ref, yte_ref = yv, yte
    Xv = np.array(val_probs).T; Xte = np.array(test_probs).T
    res = optimize_and_eval(Xv, yv_ref, Xte, yte_ref)
    res["train_n"] = int(len(full))
    res["train_pos"] = int((full.label==1).sum())
    res["train_neg"] = int((full.label==0).sum())

    print(f"\n=== FULL DATA 결과 ===")
    print(f"  train n = {res['train_n']} (양성 {res['train_pos']} / 음성 {res['train_neg']})")
    print(f"  threshold = {res['threshold']:.3f} (val 에서 결정)")
    print(f"  test AUC = {res['test_auc']:.4f}")
    print(f"  test MCC = {res['test_mcc']:.4f}")
    print(f"  TPR (양성 잘 찾기) = {res['test_tpr']:.4f}  (TP={res['tp']}/{res['tp']+res['fn']})")
    print(f"  TNR (음성 잘 찾기) = {res['test_tnr']:.4f}  (TN={res['tn']}/{res['fp']+res['tn']})")

    with open(os.path.join(RESULTS, "chembl_fulldata_eval.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'chembl_fulldata_eval.json')}")


if __name__ == "__main__":
    main()
