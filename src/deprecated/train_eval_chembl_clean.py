"""깨끗한 chEMBL train/val/test 로 모델 학습 + 평가 + 앙상블 최적화.

방법:
  1. chEMBL train 만 으로 RF + CatBoost on 5 FP 학습 → 10 단일 모델
  2. chEMBL val 에 예측 → MCC 기준 threshold + Nelder-Mead 가중치 최적화
  3. chEMBL test 에서 최종 보고 (val 에서 결정된 threshold·가중 그대로)
  4. (참고) 기존 운영 앙상블도 같은 chEMBL test 에 적용 — 외부 비교

장점:
  - 진정한 holdout 평가: val·test 모두 학습 본 적 없음
  - test 는 threshold 도 본 적 없음 (val에서 결정)
"""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import RDLogger
from scipy.optimize import minimize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CC_DIR = os.path.join(PROJECT_ROOT, "data", "experiments", "chembl_clean")
FP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments", "fp_cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chembl_clean")
RESULTS = os.path.join(PROJECT_ROOT, "results")
RANDOM_STATE = 42
FPS = ["ecfp6", "avalon", "atompair", "tt", "pattern"]


def load_split() -> dict[str, pd.DataFrame]:
    return {
        s: pd.read_csv(os.path.join(CC_DIR, f"{s}.csv")) for s in ("train", "val", "test")
    }


def ensure_fp_cache(extra_smiles: list[str]) -> None:
    """train/val/test 의 SMILES 중 기존 캐시에 없는 게 있으면 보강."""
    from rdkit import Chem
    from rdkit.Avalon import pyAvalonTools
    from rdkit.Chem.rdFingerprintGenerator import (
        GetAtomPairGenerator,
        GetMorganGenerator,
        GetTopologicalTorsionGenerator,
    )

    def fp_ecfp6(mol):
        return GetMorganGenerator(radius=3, fpSize=2048).GetFingerprint(mol)

    def fp_avalon(mol):
        return pyAvalonTools.GetAvalonFP(mol, nBits=512)

    def fp_atompair(mol):
        return GetAtomPairGenerator(fpSize=2048).GetFingerprint(mol)

    def fp_tt(mol):
        return GetTopologicalTorsionGenerator(fpSize=2048).GetFingerprint(mol)

    def fp_pattern(mol):
        return Chem.PatternFingerprint(mol, fpSize=2048)

    FUNCS = {
        "ecfp6": (fp_ecfp6, 2048),
        "avalon": (fp_avalon, 512),
        "atompair": (fp_atompair, 2048),
        "tt": (fp_tt, 2048),
        "pattern": (fp_pattern, 2048),
    }
    for name, (func, nBits) in FUNCS.items():
        path = os.path.join(FP_DIR, f"{name}.parquet")
        cache = pd.read_parquet(path).set_index("canonical_smiles") if os.path.exists(path) else None
        missing = [s for s in extra_smiles if cache is None or s not in cache.index]
        if not missing:
            continue
        print(f"  {name}: {len(missing)} 추가 분자 FP 계산...")
        arr = np.zeros((len(missing), nBits), dtype=np.uint8)
        for i, smi in enumerate(missing):
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            try:
                fp = func(mol)
                arr[i, list(fp.GetOnBits())] = 1
            except Exception:
                pass
        cols = [f"{name}_{j}" for j in range(nBits)]
        new_df = pd.DataFrame(arr, columns=cols)
        new_df.insert(0, "canonical_smiles", missing)
        merged = pd.concat([cache.reset_index(), new_df]).drop_duplicates("canonical_smiles") if cache is not None else new_df
        merged.to_parquet(path, index=False)


def fp_xy(fp_name: str, df: pd.DataFrame):
    fp = pd.read_parquet(os.path.join(FP_DIR, f"{fp_name}.parquet")).set_index("canonical_smiles")
    fp_cols = [c for c in fp.columns if c != "canonical_smiles"]
    sub = df[df["canonical_smiles"].isin(fp.index)].reset_index(drop=True)
    X = fp.loc[sub["canonical_smiles"], fp_cols].to_numpy(dtype=np.uint8)
    y = sub["label"].to_numpy(int)
    return X, y, sub


def train_one(fp_name: str, kind: str, splits) -> dict:
    """RF 또는 CatBoost 학습 → val/test 확률 반환."""
    Xtr, ytr, _ = fp_xy(fp_name, splits["train"])
    Xv,  yv,  _ = fp_xy(fp_name, splits["val"])
    Xte, yte, _ = fp_xy(fp_name, splits["test"])

    spw = float((ytr == 0).sum()) / max(int(ytr.sum()), 1)
    t0 = time.time()
    if kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=500, max_depth=None, max_features="sqrt",
            min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
    else:
        clf = CatBoostClassifier(
            iterations=600, depth=6, learning_rate=0.05,
            l2_leaf_reg=3, scale_pos_weight=spw, verbose=0,
            random_seed=RANDOM_STATE,
        )
    clf.fit(Xtr, ytr)
    p_val = clf.predict_proba(Xv)[:, 1]
    p_test = clf.predict_proba(Xte)[:, 1]
    name = f"{kind}_{fp_name}"
    out_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    if kind == "rf":
        joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    else:
        clf.save_model(os.path.join(out_dir, "model.cbm"))

    # val/test 단독 평가
    def metrics(y, p, t=None):
        if t is None:
            from sklearn.metrics import roc_curve
            fpr, tpr, thr = roc_curve(y, p)
            t = float(thr[int(np.argmax(tpr - fpr))])
            t = min(max(t, 0.05), 0.95)
        pred = (p >= t).astype(int)
        return {
            "threshold": float(t),
            "auc": float(roc_auc_score(y, p)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
        }

    val_m = metrics(yv, p_val)
    test_m = metrics(yte, p_test, val_m["threshold"])  # val 임계값을 test 에 그대로
    print(f"  {name:14s} train {time.time()-t0:.1f}s  val AUC {val_m['auc']:.3f} MCC {val_m['mcc']:.3f}  → test AUC {test_m['auc']:.3f} MCC {test_m['mcc']:.3f}")
    return {
        "name": name, "kind": kind, "fp": fp_name,
        "val_proba": p_val.tolist(), "test_proba": p_test.tolist(),
        "val_metrics_at_youden": val_m, "test_metrics_at_val_thr": test_m,
    }


def ensemble_optimize(predictions, splits) -> dict:
    """val 에서 가중치 + threshold 결정 → test 평가."""
    val_y = splits["val"]["label"].to_numpy(int)
    test_y = splits["test"]["label"].to_numpy(int)
    names = [p["name"] for p in predictions]
    Xv = np.array([p["val_proba"] for p in predictions]).T   # (n_val, n_models)
    Xte = np.array([p["test_proba"] for p in predictions]).T

    THRS = np.linspace(0.05, 0.95, 91)

    def loss(w):
        w = np.abs(w); w = w / max(w.sum(), 1e-9)
        score = Xv @ w
        # val 에서 MCC 최대 threshold 의 MCC 값
        return -max(matthews_corrcoef(val_y, (score >= t).astype(int)) for t in THRS)

    rng = np.random.default_rng(RANDOM_STATE)
    starts = [np.ones(len(names)) / len(names)] + [rng.dirichlet(np.ones(len(names))) for _ in range(8)]
    best_loss, best_x = 0.0, None
    for x0 in starts:
        r = minimize(loss, x0, method="Nelder-Mead",
                     options={"xatol": 5e-3, "fatol": 1e-3, "maxiter": 300})
        if r.fun < best_loss:
            best_loss, best_x = r.fun, r.x
    w = np.abs(best_x); w = w / w.sum()
    val_score = Xv @ w
    test_score = Xte @ w

    # val 에서 MCC-optimal threshold 선정
    best_t, best_val_mcc = 0.5, -1.0
    for t in THRS:
        m = matthews_corrcoef(val_y, (val_score >= t).astype(int))
        if m > best_val_mcc:
            best_val_mcc, best_t = m, t

    def all_metrics(y, p, t):
        pred = (p >= t).astype(int)
        return {
            "threshold": float(t),
            "auc": float(roc_auc_score(y, p)),
            "pr_auc": float(average_precision_score(y, p)),
            "mcc": float(matthews_corrcoef(y, pred)),
            "f1": float(f1_score(y, pred, zero_division=0)),
            "accuracy": float(accuracy_score(y, pred)),
            "precision": float(precision_score(y, pred, zero_division=0)),
            "recall": float(recall_score(y, pred, zero_division=0)),
        }
    val_m = all_metrics(val_y, val_score, best_t)
    test_m = all_metrics(test_y, test_score, best_t)
    return {
        "members": names, "weights": w.tolist(),
        "val_chosen_threshold": float(best_t),
        "val_metrics": val_m,
        "test_metrics": test_m,
    }


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    splits = load_split()
    all_smi = sorted(set().union(*[set(splits[s]["canonical_smiles"]) for s in splits]))
    print(f"split 분자 수: train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}")
    print(f"전체 unique SMILES: {len(all_smi)}")

    # FP 캐시 보강
    print("\nFP 캐시 보강 (없는 분자만)...")
    ensure_fp_cache(all_smi)

    # 10 단일 모델 학습 (RF + CatBoost) × 5 FP
    print("\n=== 단일 모델 학습 ===")
    preds = []
    for fp in FPS:
        for kind in ("rf", "cb"):
            preds.append(train_one(fp, kind, splits))

    # 단일 비교
    print("\n=== 단일 모델 test 결과 (val 임계값 적용) ===")
    for p in sorted(preds, key=lambda r: r["test_metrics_at_val_thr"]["mcc"], reverse=True):
        t = p["test_metrics_at_val_thr"]
        print(f"  {p['name']:14s} AUC {t['auc']:.3f}  MCC {t['mcc']:.3f}  F1 {t['f1']:.3f}")

    # 앙상블 최적화
    print("\n=== 앙상블 가중치 최적화 (val에서 결정) ===")
    ens = ensemble_optimize(preds, splits)
    print(f"\n가중치:")
    for n, w in zip(ens["members"], ens["weights"]):
        print(f"  {n:14s} {w:.3f}")
    print(f"\nval (튜닝 사용)  : AUC {ens['val_metrics']['auc']:.3f}  MCC {ens['val_metrics']['mcc']:.3f}  threshold {ens['val_chosen_threshold']:.3f}")
    print(f"test (홀드아웃)  : AUC {ens['test_metrics']['auc']:.3f}  MCC {ens['test_metrics']['mcc']:.3f}  F1 {ens['test_metrics']['f1']:.3f}")
    print(f"                  Acc {ens['test_metrics']['accuracy']:.3f}  Prec {ens['test_metrics']['precision']:.3f}  Rec {ens['test_metrics']['recall']:.3f}")

    # 저장
    out = {
        "split_sizes": {s: len(splits[s]) for s in splits},
        "ensemble": ens,
        "singles": [{k: v for k, v in p.items() if k not in ("val_proba", "test_proba")} for p in preds],
    }
    with open(os.path.join(RESULTS, "chembl_clean_eval.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # 최종 test 예측 CSV
    test = splits["test"].copy()
    Xte = np.array([p["test_proba"] for p in preds]).T
    w = np.array(ens["weights"])
    test["proba_final"] = Xte @ w
    test.to_csv(os.path.join(MODELS_DIR, "test_pred.csv"), index=False)
    print(f"\n저장: {os.path.join(RESULTS, 'chembl_clean_eval.json')}, {os.path.join(MODELS_DIR, 'test_pred.csv')}")


if __name__ == "__main__":
    main()
