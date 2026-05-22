"""도메인 적응 — chEMBL 양성/음성 일부를 학습에 추가.

전략:
  - 기존 exp_clean_full 학습 풀
  - + data/external_eval/chembl_external_positives.csv 중 외부 test 와 비누수 분자 (258개)
  - + data/external_eval/chembl_external_negatives_pool.csv 의 일부 (1000개 샘플, test 와 자동 비누수)
  - → exp_clean_da/ 데이터셋 저장
  - → 5종 FP RF + CatBoost(AtomPair) 학습 → 외부 chEMBL test 평가

도메인 갭(시판약 임상 vs chEMBL in vitro)이 학습 분포에 직접 포함되어 해소를 기대.

사용:
    python src/train_domain_adapted.py
"""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(PROJECT_ROOT, "data", "experiments")
EXT_TEST = os.path.join(EXP, "external_test", "test.csv")
DA_DIR = os.path.join(EXP, "exp_clean_da")
FP_DIR = os.path.join(EXP, "fp_cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "da")
RESULTS = os.path.join(PROJECT_ROOT, "results")

RANDOM_STATE = 42
N_NEG_FROM_CHEMBL = 1500   # 추가 음성


def canonicalize(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def build_da_manifest() -> str:
    """exp_clean_full 학습 풀 + chEMBL 양성 258(test 비누수) + chEMBL 음성 일부."""
    base = pd.read_csv(os.path.join(EXP, "exp_clean_full", "manifest.csv"))
    test = pd.read_csv(EXT_TEST)
    test_ik = set(test["inchi_key"].dropna())

    # 1) chEMBL 양성 (테스트 InChIKey 제거)
    cp = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "external_eval", "chembl_external_positives.csv"))
    cp["canonical_smiles"] = cp["SMILES"].map(canonicalize)
    cp = cp.dropna(subset=["canonical_smiles", "inchi_key"])
    cp = cp[~cp["inchi_key"].isin(test_ik)].copy()
    cp = cp[["canonical_smiles", "inchi_key"]].drop_duplicates("inchi_key")
    cp["label"] = 1
    cp["source"] = "chembl_da_pos"

    # 2) chEMBL 음성 (테스트 비누수 — 이미 검증됨, 추가 검증)
    cn = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "external_eval", "chembl_external_negatives_pool.csv"))
    cn["canonical_smiles"] = cn["SMILES"].map(canonicalize)
    cn = cn.dropna(subset=["canonical_smiles", "inchi_key"])
    cn = cn[~cn["inchi_key"].isin(test_ik)].copy()
    cn = cn[["canonical_smiles", "inchi_key"]].drop_duplicates("inchi_key")
    cn = cn.sample(n=min(N_NEG_FROM_CHEMBL, len(cn)), random_state=RANDOM_STATE)
    cn["label"] = 0
    cn["source"] = "chembl_da_neg"

    # 3) 기존 exp_clean_full + 둘 합치기 (InChIKey 충돌 시 chEMBL DA 우선)
    base["source"] = base.get("source", "exp_clean_full")
    chembl = pd.concat([cp, cn], ignore_index=True)
    chembl_ik = set(chembl["inchi_key"])
    base = base[~base["inchi_key"].isin(chembl_ik)].copy()
    merged = pd.concat([base[["canonical_smiles", "inchi_key", "label", "source"]], chembl], ignore_index=True)
    merged = merged.dropna(subset=["canonical_smiles", "inchi_key"]).drop_duplicates("inchi_key")
    merged = merged[~merged["inchi_key"].isin(test_ik)]  # 안전망

    # InChIKey 라벨 충돌 자체점검
    conf = merged.groupby("inchi_key")["label"].nunique()
    assert (conf == 1).all(), f"라벨 충돌 {(conf > 1).sum()} 개"

    os.makedirs(DA_DIR, exist_ok=True)
    path = os.path.join(DA_DIR, "manifest.csv")
    merged.to_csv(path, index=False)

    n_pos = int((merged["label"] == 1).sum())
    n_neg = int((merged["label"] == 0).sum())
    print(f"DA manifest 저장: {path}")
    print(f"  총 {len(merged)} (양성 {n_pos}, 음성 {n_neg})")
    print(f"    - exp_clean_full 잔여: {(merged['source'] == 'positive').sum() + (merged['source'] == 'marketed_clean').sum()}")
    print(f"    - chEMBL DA 양성:    {(merged['source'] == 'chembl_da_pos').sum()}")
    print(f"    - chEMBL DA 음성:    {(merged['source'] == 'chembl_da_neg').sum()}")
    return path


def load_xy(fp_path: str, csv: str):
    df = pd.read_csv(csv)
    fp = pd.read_parquet(fp_path).set_index("canonical_smiles")
    fp_cols = [c for c in fp.columns if c != "canonical_smiles"]
    mask = df["canonical_smiles"].isin(fp.index)
    df = df[mask].reset_index(drop=True)
    X = fp.loc[df["canonical_smiles"], fp_cols].to_numpy(dtype=np.uint8)
    y = df["label"].to_numpy(int)
    return X, y, df


def balanced_mcc_auc(score: np.ndarray, y: np.ndarray, n_runs: int = 10):
    pos_i = np.where(y == 1)[0]
    neg_i = np.where(y == 0)[0]
    n = len(neg_i)
    rng = np.random.default_rng(RANDOM_STATE)
    aucs, mccs = [], []
    thrs = np.linspace(0.05, 0.95, 91)
    for _ in range(n_runs):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = y[idx]
        ss = score[idx]
        aucs.append(float(roc_auc_score(ys, ss)))
        bm = max(matthews_corrcoef(ys, (ss >= t).astype(int)) for t in thrs)
        mccs.append(float(bm))
    return float(np.mean(aucs)), float(np.std(aucs)), float(np.mean(mccs)), float(np.std(mccs))


def train_rf(fp_name: str, manifest_path: str) -> dict:
    """RF on FP — manifest 학습 → 외부 test 평가."""
    fp_path = os.path.join(FP_DIR, f"{fp_name}.parquet")
    Xtr, ytr, _ = load_xy(fp_path, manifest_path)
    Xte, yte, test_df = load_xy(fp_path, EXT_TEST)

    rf = RandomForestClassifier(
        n_estimators=500, max_depth=None, max_features="sqrt",
        min_samples_leaf=2, class_weight="balanced",
        random_state=RANDOM_STATE, n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(Xtr, ytr)
    proba = rf.predict_proba(Xte)[:, 1]
    out_dir = os.path.join(MODELS_DIR, f"rf_{fp_name}")
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(out_dir, "rf.pkl"))
    test_df["proba"] = proba
    test_df.to_csv(os.path.join(out_dir, "external_pred.csv"), index=False)

    a_m, a_s, m_m, m_s = balanced_mcc_auc(proba, yte)
    print(f"  RF on {fp_name:10s}  학습 {time.time()-t0:.1f}s  AUC {a_m:.3f}±{a_s:.3f}  MCC {m_m:.3f}±{m_s:.3f}  (train n={len(ytr)}, pos {int(ytr.sum())})")
    return {"name": f"da_rf_{fp_name}", "auc": a_m, "auc_std": a_s, "mcc": m_m, "mcc_std": m_s}


def train_catboost(fp_name: str, manifest_path: str, *, da: bool = True) -> dict:
    """CatBoost on FP."""
    from catboost import CatBoostClassifier
    fp_path = os.path.join(FP_DIR, f"{fp_name}.parquet")
    Xtr, ytr, _ = load_xy(fp_path, manifest_path)
    Xte, yte, test_df = load_xy(fp_path, EXT_TEST)
    spw = float((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    cb = CatBoostClassifier(
        iterations=600, depth=6, learning_rate=0.05,
        l2_leaf_reg=3, scale_pos_weight=spw, verbose=0,
        random_seed=RANDOM_STATE,
    )
    t0 = time.time()
    cb.fit(Xtr, ytr)
    proba = cb.predict_proba(Xte)[:, 1]
    tag = "da" if da else "base"
    out_dir = os.path.join(MODELS_DIR, f"cb_{fp_name}_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    cb.save_model(os.path.join(out_dir, "cb.cbm"))
    test_df["proba"] = proba
    test_df.to_csv(os.path.join(out_dir, "external_pred.csv"), index=False)
    a_m, a_s, m_m, m_s = balanced_mcc_auc(proba, yte)
    print(f"  CatBoost on {fp_name:10s} ({tag})  학습 {time.time()-t0:.1f}s  AUC {a_m:.3f}±{a_s:.3f}  MCC {m_m:.3f}±{m_s:.3f}")
    return {"name": f"cb_{fp_name}_{tag}", "auc": a_m, "auc_std": a_s, "mcc": m_m, "mcc_std": m_s}


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    da_manifest = build_da_manifest()
    base_manifest = os.path.join(EXP, "exp_clean_full", "manifest.csv")

    summary = []
    # ⑦ 도메인 적응: 5종 FP × RF
    print("\n=== ⑦ 도메인 적응 RF on FP (학습 = exp_clean_full + chEMBL DA) ===")
    for fp in ["ecfp6", "avalon", "atompair", "tt", "pattern"]:
        summary.append(train_rf(fp, da_manifest))

    # ⑧ CatBoost — 도메인 적응 학습 (5종)
    print("\n=== ⑧ CatBoost on FP (도메인 적응 학습) ===")
    for fp in ["ecfp6", "avalon", "atompair", "tt", "pattern"]:
        summary.append(train_catboost(fp, da_manifest, da=True))

    # 비교용: CatBoost on base data (도메인 적응 없이) — AtomPair·TT 두개만
    print("\n=== (참고) CatBoost on FP (base = exp_clean_full만) ===")
    for fp in ["atompair", "tt"]:
        summary.append(train_catboost(fp, base_manifest, da=False))

    print("\n=== 요약 ===")
    for s in sorted(summary, key=lambda r: r["mcc"], reverse=True):
        print(f"  {s['name']:25s} AUC {s['auc']:.3f}  MCC {s['mcc']:.3f}±{s['mcc_std']:.3f}")
    with open(os.path.join(RESULTS, "domain_adapted_eval.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'domain_adapted_eval.json')}")


if __name__ == "__main__":
    main()
