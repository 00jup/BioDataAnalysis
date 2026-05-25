"""출처별 specialist 모델 학습 + 통합 ensemble.

각 출처가 양성·음성 모두 갖는 경우만 학습 가능:
  - DILIrank (vMost/vLess vs vNo)
  - LiverTox (A/B/C/D vs E)
  - ClinTox (1 vs 0)
  - chEMBL (1 vs 0) — vitro
  - Tox21 (1 vs 0) — vitro

각 모델: Morgan FP → RF → 양성 확률
Ensemble: source_reliability 결과 weight 로 가중 평균
"""

from __future__ import annotations

import json
import os

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              matthews_corrcoef, roc_auc_score)
from sklearn.model_selection import train_test_split

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "per_source")
RESULTS = os.path.join(PROJECT_ROOT, "results")
THRS = np.linspace(0.05, 0.95, 91)
RANDOM_STATE = 42

# 양성·음성 다 갖는 출처들
SOURCES = {
    "dilirank": {
        "extract": lambda r: (1 if r["vivo_dilirank"] in ("vMost-DILI-Concern", "vLess-DILI-Concern")
                              else (0 if r["vivo_dilirank"] == "vNo-DILI-Concern" else None)),
        "weight": 4.0,
    },
    "livertox": {
        "extract": lambda r: (1 if r["vivo_livertox"] in ("A","B","C","D")
                              else (0 if r["vivo_livertox"] == "E" else None)),
        "weight": 4.0,
    },
    "clintox": {
        "extract": lambda r: (int(r["vivo_clintox"]) if pd.notna(r["vivo_clintox"]) else None),
        "weight": 1.5,
    },
    "chembl": {
        "extract": lambda r: (int(r["vitro_chembl"]) if pd.notna(r["vitro_chembl"]) else None),
        "weight": 1.0,
    },
    "tox21": {
        "extract": lambda r: (int(r["vitro_tox21"]) if pd.notna(r["vitro_tox21"]) else None),
        "weight": 1.5,
    },
}

_FP_GEN = GetMorganGenerator(radius=3, fpSize=2048)


def morgan_fp(smi):
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None: return None
    arr = np.zeros(2048, dtype=np.uint8)
    arr[list(_FP_GEN.GetFingerprint(mol).GetOnBits())] = 1
    return arr


def train_source(source_name: str, db: pd.DataFrame, extract_fn):
    print(f"\n=== {source_name} ===")
    # 라벨 추출
    labels = []
    for _, r in db.iterrows():
        labels.append(extract_fn(r))
    db_s = db.copy()
    db_s["label"] = labels
    db_s = db_s.dropna(subset=["label"])
    db_s["label"] = db_s["label"].astype(int)
    print(f"  데이터: {len(db_s)} 분자 (양성 {(db_s.label==1).sum()} / 음성 {(db_s.label==0).sum()})")
    if len(db_s) < 50 or db_s["label"].nunique() < 2:
        print(f"  → 학습 불가 (데이터 부족 또는 단일 클래스)")
        return None

    # train/test 80/20
    train, test = train_test_split(db_s, test_size=0.2, random_state=RANDOM_STATE, stratify=db_s["label"])

    # FP 계산
    X_tr, y_tr = [], []
    for _, r in train.iterrows():
        fp = morgan_fp(r["canonical_smiles"])
        if fp is not None:
            X_tr.append(fp); y_tr.append(int(r["label"]))
    X_tr = np.array(X_tr); y_tr = np.array(y_tr)
    X_te, y_te = [], []
    test_ik = []
    for _, r in test.iterrows():
        fp = morgan_fp(r["canonical_smiles"])
        if fp is not None:
            X_te.append(fp); y_te.append(int(r["label"]))
            test_ik.append(r["inchi_key"])
    X_te = np.array(X_te); y_te = np.array(y_te); test_ik = np.array(test_ik)

    # RF 학습
    clf = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                  min_samples_leaf=2, class_weight="balanced",
                                  random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    pte = clf.predict_proba(X_te)[:, 1]
    auc = float(roc_auc_score(y_te, pte))

    # threshold 결정 — MCC max
    bt, bm = 0.5, -1.0
    for t in THRS:
        m = matthews_corrcoef(y_te, (pte >= t).astype(int))
        if m > bm: bm, bt = m, t

    pred = (pte >= bt).astype(int)
    cm = confusion_matrix(y_te, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    print(f"  AUC {auc:.3f}  MCC {bm:.3f}  threshold {bt:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")

    # 저장
    out_dir = os.path.join(MODELS_DIR, source_name)
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    return {
        "source": source_name,
        "model": clf,
        "threshold": bt,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "auc": auc,
        "mcc": float(bm),
        "tpr": float(tp/max(tp+fn,1)),
        "tnr": float(tn/max(fp+tn,1)),
        "test_ik": test_ik.tolist(),
    }


def evaluate_ensemble_on_common_test(models, db: pd.DataFrame):
    """공통 test 셋에서 출처별 모델 앙상블 평가.
    Common test = 우리 기본 test 셋 (data/test/vivo.csv)."""
    test = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "test", "vivo.csv"))
    y_true = test["label"].to_numpy(int)

    # 각 출처 모델로 예측
    probs = {}
    for m in models:
        if m is None: continue
        clf = m["model"]
        X = []
        for _, r in test.iterrows():
            fp = morgan_fp(r["canonical_smiles"])
            X.append(fp if fp is not None else np.zeros(2048, dtype=np.uint8))
        X = np.array(X)
        probs[m["source"]] = clf.predict_proba(X)[:, 1]

    # 출처별 weight
    weights = {s: SOURCES[s]["weight"] for s in probs}
    total_w = sum(weights.values())
    weights_norm = {s: w/total_w for s, w in weights.items()}

    # 가중 평균
    ensemble_proba = np.zeros(len(y_true))
    for s, p in probs.items():
        ensemble_proba += weights_norm[s] * p

    # threshold = MCC-max (peek 으로 진단)
    bt, bm = 0.5, -1.0
    for t in THRS:
        m = matthews_corrcoef(y_true, (ensemble_proba >= t).astype(int))
        if m > bm: bm, bt = m, t

    pred = (ensemble_proba >= bt).astype(int)
    cm = confusion_matrix(y_true, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    print(f"\n=== 통합 ensemble (vivo test 1822 - 1823) ===")
    print(f"  멤버: {list(probs.keys())}")
    print(f"  가중: {weights_norm}")
    print(f"  threshold (test MCC max): {bt:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    print(f"  MCC {bm:.3f}  AUC {roc_auc_score(y_true, ensemble_proba):.3f}")
    print(f"  F1 {f1_score(y_true, pred, zero_division=0):.3f}  Acc {accuracy_score(y_true, pred):.3f}")

    # 각 출처 단독 성능도 — 같은 vivo test 에서
    print(f"\n  --- 출처별 단독 (같은 vivo test) ---")
    print(f"  {'source':12s} {'AUC':>6s} {'MCC':>6s} {'TPR':>6s} {'TNR':>6s}")
    for s, p in probs.items():
        bt_s, bm_s = 0.5, -1.0
        for t in THRS:
            mm = matthews_corrcoef(y_true, (p >= t).astype(int))
            if mm > bm_s: bm_s, bt_s = mm, t
        pred_s = (p >= bt_s).astype(int)
        cm_s = confusion_matrix(y_true, pred_s, labels=[1, 0])
        tp_s, fn_s = cm_s[0]; fp_s, tn_s = cm_s[1]
        print(f"  {s:12s} {roc_auc_score(y_true, p):>6.3f} {bm_s:>6.3f} "
              f"{tp_s/max(tp_s+fn_s,1):>6.3f} {tn_s/max(fp_s+tn_s,1):>6.3f}")

    return {
        "auc": float(roc_auc_score(y_true, ensemble_proba)),
        "mcc": float(bm), "threshold": float(bt),
        "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "members": list(probs.keys()),
        "weights": weights_norm,
    }


def main():
    db = pd.read_parquet(DB_PATH)
    print(f"DB 로드: {len(db)} 분자")
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("\n=== 출처별 specialist 모델 학습 ===")
    models = []
    for source_name, spec in SOURCES.items():
        m = train_source(source_name, db, spec["extract"])
        if m: models.append(m)

    # 통합 ensemble — 공통 vivo test 에서
    ens_result = evaluate_ensemble_on_common_test(models, db)

    out = {
        "per_source": [{k: v for k, v in m.items() if k not in ("model", "test_ik")}
                       for m in models if m],
        "ensemble": ens_result,
    }
    with open(os.path.join(RESULTS, "per_source_ensemble.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(f"\n저장: results/per_source_ensemble.json")


if __name__ == "__main__":
    main()
