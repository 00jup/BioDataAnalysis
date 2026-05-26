"""도메인별 모델 학습 — vivo 모델 + vitro 모델.

각 도메인마다:
  - FP 5종 (ECFP6, Avalon, AtomPair, TT, Pattern) × {RF, CatBoost} = 10 sub-model
  - train 에서 학습, val 에서 가중치 + threshold 최적화, test 에서 보고

저장:
  models/{vivo,vitro}/{rf,cb}_{fp}/model.{pkl,cbm}
  models/{vivo,vitro}/ensemble_meta.json  (가중치, threshold, val/test 결과)
  data/fp_cache/{ecfp6,avalon,atompair,tt,pattern}.parquet  (FP 캐시 보강)

사용:
    python src/train_domain_models.py vivo
    python src/train_domain_models.py vitro
    python src/train_domain_models.py both
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetTopologicalTorsionGenerator,
)
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
FP_DIR = os.path.join(PROJECT_ROOT, "data", "fp_cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42
FPS = {
    "ecfp6": 2048,
    "avalon": 512,
    "atompair": 2048,
    "tt": 2048,
    "pattern": 2048,
}
THRS = np.linspace(0.05, 0.95, 91)


def fp_of(name: str, mol) -> list[int]:
    nb = FPS[name]
    if name == "ecfp6":
        return list(GetMorganGenerator(radius=3, fpSize=nb).GetFingerprint(mol).GetOnBits())
    if name == "avalon":
        return list(pyAvalonTools.GetAvalonFP(mol, nBits=nb).GetOnBits())
    if name == "atompair":
        return list(GetAtomPairGenerator(fpSize=nb).GetFingerprint(mol).GetOnBits())
    if name == "tt":
        return list(GetTopologicalTorsionGenerator(fpSize=nb).GetFingerprint(mol).GetOnBits())
    if name == "pattern":
        return list(Chem.PatternFingerprint(mol, fpSize=nb).GetOnBits())
    raise ValueError(name)


def ensure_fp_cache(smiles_list: list[str], fp_name: str) -> pd.DataFrame:
    """SMILES 리스트의 FP 를 캐시에서 조회 + 없는 건 계산해 추가."""
    nb = FPS[fp_name]
    path = os.path.join(FP_DIR, f"{fp_name}.parquet")
    if os.path.exists(path):
        cache = pd.read_parquet(path).set_index("canonical_smiles")
        existing = set(cache.index)
    else:
        cache = None
        existing = set()
    missing = [s for s in smiles_list if s not in existing]
    if missing:
        print(f"    {fp_name}: {len(missing)} 분자 FP 추가 계산...")
        arr = np.zeros((len(missing), nb), dtype=np.uint8)
        for i, smi in enumerate(missing):
            mol = Chem.MolFromSmiles(smi)
            if mol is None: continue
            try: arr[i, fp_of(fp_name, mol)] = 1
            except Exception: pass
        cols = [f"{fp_name}_{j}" for j in range(nb)]
        new_df = pd.DataFrame(arr, columns=cols)
        new_df.insert(0, "canonical_smiles", missing)
        if cache is not None:
            merged = pd.concat([cache.reset_index(), new_df]).drop_duplicates("canonical_smiles")
        else:
            merged = new_df
        os.makedirs(FP_DIR, exist_ok=True)
        merged.to_parquet(path, index=False)
        cache = merged.set_index("canonical_smiles")
    return cache


def fp_xy(fp_name: str, df: pd.DataFrame, domain: str = "vivo", db: pd.DataFrame | None = None):
    """FP 추출. sample_weight 폐기 — 모든 라벨 동급 신뢰."""
    cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
    cols = [c for c in cache.columns]
    mask = df["canonical_smiles"].isin(cache.index)
    sub = df[mask].reset_index(drop=True)
    X = cache.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
    y = sub["label"].to_numpy(int)
    return X, y, sub


def train_sub_model(fp_name: str, kind: str, train_df, val_df, test_df, domain: str = "vivo", db=None):
    Xtr, ytr, _ = fp_xy(fp_name, train_df, domain, db=db)
    Xv, yv, _ = fp_xy(fp_name, val_df, domain, db=db)
    Xte, yte, _ = fp_xy(fp_name, test_df, domain, db=db)
    spw = float((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=500, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1)
    else:
        m = CatBoostClassifier(iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, scale_pos_weight=spw, verbose=0,
            random_seed=RANDOM_STATE)
    t0 = time.time()
    m.fit(Xtr, ytr)  # sample_weight 폐기 — class_weight 만 사용 (불균형 보정)
    pv = m.predict_proba(Xv)[:, 1]
    pte = m.predict_proba(Xte)[:, 1]
    return m, pv, pte, yv, yte, time.time() - t0


def optimize_ensemble(Xv: np.ndarray, yv: np.ndarray):
    """Nelder-Mead 가중치 + val MCC-optimal threshold."""
    def loss(w):
        w = np.abs(w); w = w / max(w.sum(), 1e-9)
        s = Xv @ w
        return -max(matthews_corrcoef(yv, (s >= t).astype(int)) for t in THRS)

    rng = np.random.default_rng(RANDOM_STATE)
    starts = [np.ones(Xv.shape[1])/Xv.shape[1]] + [rng.dirichlet(np.ones(Xv.shape[1])) for _ in range(5)]
    best_loss, best_w = 0.0, None
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead",
                     options={"xatol":5e-3,"fatol":1e-3,"maxiter":200})
        if r.fun < best_loss: best_loss, best_w = r.fun, r.x
    w = np.abs(best_w); w = w / w.sum()
    s_val = Xv @ w
    bt, bm = 0.5, -1.0
    for t in THRS:
        m = matthews_corrcoef(yv, (s_val >= t).astype(int))
        if m > bm: bm, bt = m, t
    return w, float(bt), float(bm)


def full_metrics(y, p, t):
    pred = (p >= t).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(fp + tn, 1)
    return {
        "n": int(len(y)), "pos": int(tp + fn), "neg": int(fp + tn),
        "threshold": float(t),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "tpr": float(tpr), "tnr": float(tnr),
        "precision": float(tp / max(tp + fp, 1)),
        "npv": float(tn / max(tn + fn, 1)),
        "balanced_accuracy": (tpr + tnr) / 2,
        "accuracy": float(accuracy_score(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, pred)),
        "auc": float(roc_auc_score(y, p)),
    }


def train_domain(domain: str):
    out_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"  {domain.upper()} 도메인 모델 학습 (class_weight balanced — sample_weight 폐기)")
    print(f"{'='*70}")

    # DB 로드 (sample_weight 계산용)
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))

    tr = pd.read_csv(os.path.join(TRAIN_DIR, f"{domain}.csv"))
    va = pd.read_csv(os.path.join(VAL_DIR, f"{domain}.csv"))
    te = pd.read_csv(os.path.join(TEST_DIR, f"{domain}.csv"))
    print(f"  train {len(tr)} (양성 {(tr.label==1).sum()} / 음성 {(tr.label==0).sum()})")
    print(f"  val   {len(va)}    test  {len(te)}\n")

    val_probs, test_probs = [], []
    yv_ref, yte_ref = None, None
    model_names = []
    for fp_name in FPS:
        for kind in ("rf", "cb"):
            name = f"{kind}_{fp_name}"
            model_names.append(name)
            m, pv, pte, yv, yte, elapsed = train_sub_model(fp_name, kind, tr, va, te, domain, db=db)
            val_probs.append(pv); test_probs.append(pte)
            if yv_ref is None: yv_ref, yte_ref = yv, yte
            sub_dir = os.path.join(out_dir, name)
            os.makedirs(sub_dir, exist_ok=True)
            if kind == "rf":
                joblib.dump(m, os.path.join(sub_dir, "model.pkl"))
            else:
                m.save_model(os.path.join(sub_dir, "model.cbm"))
            print(f"  {name:14s} {elapsed:5.1f}s  val AUC {roc_auc_score(yv,pv):.3f} MCC@? — test AUC {roc_auc_score(yte,pte):.3f}")

    print(f"\n[앙상블 최적화]")
    Xv = np.array(val_probs).T
    Xte = np.array(test_probs).T
    w, thr, val_mcc = optimize_ensemble(Xv, yv_ref)
    val_score = Xv @ w
    test_score = Xte @ w

    val_m = full_metrics(yv_ref, val_score, thr)
    test_m = full_metrics(yte_ref, test_score, thr)

    print(f"  가중치:")
    for n, ww in zip(model_names, w): print(f"    {n:14s} {ww:.3f}")
    print(f"  val threshold (MCC max): {thr:.3f}  → val MCC {val_m['mcc']:.3f}")
    print(f"\n  TEST 결과 (val 임계값 적용):")
    print(f"    N={test_m['n']} (양성 {test_m['pos']}, 음성 {test_m['neg']})")
    print(f"    TPR {test_m['tpr']:.3f}  TNR {test_m['tnr']:.3f}  bAcc {test_m['balanced_accuracy']:.3f}")
    print(f"    MCC {test_m['mcc']:.3f}  F1 {test_m['f1']:.3f}  AUC {test_m['auc']:.3f}")

    meta = {
        "domain": domain,
        "members": model_names,
        "weights": w.tolist(),
        "threshold": thr,
        "val_metrics": val_m,
        "test_metrics": test_m,
    }
    with open(os.path.join(out_dir, "ensemble_meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "both"
    targets = ["vivo", "vitro"] if arg == "both" else [arg]
    results = {}
    for d in targets:
        results[d] = train_domain(d)

    print(f"\n{'='*70}")
    print("  요약")
    print(f"{'='*70}")
    print(f"{'도메인':10s} {'AUC':>6s} {'MCC':>6s} {'TPR':>6s} {'TNR':>6s} {'F1':>6s}")
    for d, m in results.items():
        t = m["test_metrics"]
        print(f"{d:10s} {t['auc']:6.3f} {t['mcc']:6.3f} {t['tpr']:6.3f} {t['tnr']:6.3f} {t['f1']:6.3f}")


if __name__ == "__main__":
    main()
