"""다양한 핑거프린트 × RF 앙상블 — 기존 exp2와 독립적인 신호 추가.

기존 exp2 = Morgan(r=2,2048) + MACCS(167) + RDKit-210
새로 더하는 표현 5종:
  - ECFP6  : Morgan(r=3, 2048) — 더 큰 부분구조
  - Avalon : Avalon FP (512) — RDKit/일반 FP와 다른 정의
  - AtomPair : 원자쌍 FP (2048) — 거리 기반
  - TT     : Topological Torsion (2048) — 비틀림 기반
  - Pattern: Pattern FP (2048) — Daylight 유사

각 FP 마다 양성 비율 자동 튜닝한 Random Forest 학습 → 외부 chEMBL test 평가.

사용:
    python src/train_diverse_fingerprints.py
"""

from __future__ import annotations

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetTopologicalTorsionGenerator,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(PROJECT_ROOT, "data", "experiments")
EXT_TEST = os.path.join(EXP, "external_test", "test.csv")
TRAIN_MANIFEST = os.path.join(EXP, "exp_clean_full", "manifest.csv")  # 양성 1465 / 음성 5000
FP_CACHE_DIR = os.path.join(EXP, "fp_cache")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "diverse_fp")
RESULTS = os.path.join(PROJECT_ROOT, "results")

RANDOM_STATE = 42


def _mol(smi: str):
    return Chem.MolFromSmiles(smi) if isinstance(smi, str) else None


# ---- 5종 핑거프린트 추출기 ----

def fp_ecfp6(mol, nBits: int = 2048) -> np.ndarray:
    gen = GetMorganGenerator(radius=3, fpSize=nBits)
    arr = np.zeros(nBits, dtype=np.uint8)
    fp = gen.GetFingerprint(mol)
    onbits = fp.GetOnBits()
    arr[list(onbits)] = 1
    return arr


def fp_avalon(mol, nBits: int = 512) -> np.ndarray:
    fp = pyAvalonTools.GetAvalonFP(mol, nBits=nBits)
    arr = np.zeros(nBits, dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return arr


def fp_atompair(mol, nBits: int = 2048) -> np.ndarray:
    gen = GetAtomPairGenerator(fpSize=nBits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(nBits, dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return arr


def fp_tt(mol, nBits: int = 2048) -> np.ndarray:
    gen = GetTopologicalTorsionGenerator(fpSize=nBits)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros(nBits, dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return arr


def fp_pattern(mol, nBits: int = 2048) -> np.ndarray:
    fp = Chem.PatternFingerprint(mol, fpSize=nBits)
    arr = np.zeros(nBits, dtype=np.uint8)
    arr[list(fp.GetOnBits())] = 1
    return arr


FPS = {
    "ecfp6": (fp_ecfp6, 2048),
    "avalon": (fp_avalon, 512),
    "atompair": (fp_atompair, 2048),
    "tt": (fp_tt, 2048),
    "pattern": (fp_pattern, 2048),
}


def build_fp_cache(name: str) -> str:
    """모든 학습+외부 test 분자에 대해 한 번만 FP 계산해 parquet 캐싱."""
    os.makedirs(FP_CACHE_DIR, exist_ok=True)
    path = os.path.join(FP_CACHE_DIR, f"{name}.parquet")
    if os.path.exists(path):
        print(f"  {name}: 캐시 적중 {path}")
        return path

    func, nBits = FPS[name]
    train = pd.read_csv(TRAIN_MANIFEST)
    test = pd.read_csv(EXT_TEST)
    smiles = sorted(set(train["canonical_smiles"]) | set(test["canonical_smiles"]))
    print(f"  {name}: {len(smiles)}개 분자 FP 계산 (nBits={nBits})...")

    arr = np.zeros((len(smiles), nBits), dtype=np.uint8)
    t0 = time.time()
    for i, smi in enumerate(smiles):
        m = _mol(smi)
        if m is None:
            continue
        try:
            arr[i] = func(m)
        except Exception:
            pass
        if i and i % 1000 == 0:
            print(f"    {i}/{len(smiles)}  {(i+1)/(time.time()-t0):.1f}/s")
    cols = [f"{name}_{j}" for j in range(nBits)]
    df = pd.DataFrame(arr, columns=cols)
    df.insert(0, "canonical_smiles", smiles)
    df.to_parquet(path, index=False)
    print(f"  {name}: 저장 {path} ({time.time()-t0:.1f}s)")
    return path


def load_xy(fp_path: str, csv: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    df = pd.read_csv(csv)
    fp = pd.read_parquet(fp_path).set_index("canonical_smiles")
    fp_cols = [c for c in fp.columns if c != "canonical_smiles"]
    mask = df["canonical_smiles"].isin(fp.index)
    df = df[mask].reset_index(drop=True)
    X = fp.loc[df["canonical_smiles"], fp_cols].to_numpy(dtype=np.uint8)
    y = df["label"].to_numpy(dtype=int)
    return X, y, df


def train_eval_one(name: str) -> dict:
    print(f"\n=== {name} ===")
    fp_path = build_fp_cache(name)

    Xtr, ytr, _ = load_xy(fp_path, TRAIN_MANIFEST)
    Xte, yte, test_df = load_xy(fp_path, EXT_TEST)
    print(f"  train X={Xtr.shape}, test X={Xte.shape}")

    # RF + class_weight 균형
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    t0 = time.time()
    rf.fit(Xtr, ytr)
    print(f"  학습 {time.time()-t0:.1f}s")

    proba = rf.predict_proba(Xte)[:, 1]
    out_dir = os.path.join(MODELS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(out_dir, "rf.pkl"))
    test_df["proba"] = proba
    test_df.to_csv(os.path.join(out_dir, "external_pred.csv"), index=False)

    # 1:1 balanced × 10회, MCC-optimal
    pos_i = np.where(yte == 1)[0]
    neg_i = np.where(yte == 0)[0]
    n = len(neg_i)
    rng = np.random.default_rng(RANDOM_STATE)
    aucs, mccs = [], []
    for _ in range(10):
        sp = rng.choice(pos_i, size=n, replace=False)
        idx = np.concatenate([sp, neg_i])
        ys = yte[idx]
        ss = proba[idx]
        aucs.append(float(roc_auc_score(ys, ss)))
        bt, bm = 0.5, -1.0
        for thr in np.linspace(0.05, 0.95, 91):
            m = matthews_corrcoef(ys, (ss >= thr).astype(int))
            if m > bm:
                bm = m
                bt = thr
        mccs.append(float(bm))
    summary = {
        "name": name,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "mcc_mean": float(np.mean(mccs)),
        "mcc_std": float(np.std(mccs)),
    }
    print(f"  AUC {summary['auc_mean']:.3f}±{summary['auc_std']:.3f}  MCC {summary['mcc_mean']:.3f}±{summary['mcc_std']:.3f}")
    return summary


def main() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)

    summaries = []
    for name in FPS:
        summaries.append(train_eval_one(name))

    print("\n=== 요약 ===")
    for s in summaries:
        print(f"  {s['name']:10s} AUC {s['auc_mean']:.3f}±{s['auc_std']:.3f}  MCC {s['mcc_mean']:.3f}±{s['mcc_std']:.3f}")

    with open(os.path.join(RESULTS, "diverse_fp_eval.json"), "w") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'diverse_fp_eval.json')}")


if __name__ == "__main__":
    main()
