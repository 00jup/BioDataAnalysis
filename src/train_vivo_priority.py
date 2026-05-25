"""vivo 우선 + vitro fallback 통합 학습.

라벨 정의:
  - vivo 라벨 있으면 → vivo 라벨 사용 (vivo+ → 1, vivo- → 0)
  - vivo 라벨 없고 vitro 라벨 있으면 → vitro 라벨 사용 (보조)
  - 둘 다 없으면 → 제외

채점이 임상 vivo 기준이므로:
  - vivo-+vitro+ → 0 (vivo 답 우선)
  - vivo+ only → 1
  - vitro+ only → 1 (vivo 데이터 없으니 보조로 활용)
  - vitro- only → 0

같은 vivo test 에서 v1 baseline 과 직접 비교한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
SPLIT_DIR = os.path.join(PROJECT_ROOT, "data")


def build_vp_split():
    db = pd.read_parquet(DB_PATH)
    print(f"DB: {len(db)} 분자")

    db["vivo_lbl"]  = pd.to_numeric(db.get("vivo_label"),  errors="coerce")
    db["vitro_lbl"] = pd.to_numeric(db.get("vitro_label"), errors="coerce")
    has_any = db["vivo_lbl"].notna() | db["vitro_lbl"].notna()
    db_any = db[has_any].copy()

    # vivo 우선 — vivo 있으면 vivo, 없으면 vitro
    db_any["vp_label"] = np.where(
        db_any["vivo_lbl"].notna(),
        db_any["vivo_lbl"],
        db_any["vitro_lbl"],
    ).astype(int)
    db_any["label_source"] = np.where(db_any["vivo_lbl"].notna(), "vivo", "vitro_only")

    print(f"vp 가능 분자: {len(db_any)}")
    print(f"  양성 {(db_any.vp_label==1).sum()} / 음성 {(db_any.vp_label==0).sum()}")
    print(f"  vivo 라벨로: {(db_any.label_source=='vivo').sum()}")
    print(f"  vitro-only fallback: {(db_any.label_source=='vitro_only').sum()}")
    # 충돌 분포 — vivo 우선 정의로 라벨됨
    both = db_any[db_any.vivo_lbl.notna() & db_any.vitro_lbl.notna()]
    print(f"  vivo-+vitro+ {((both.vivo_lbl==0)&(both.vitro_lbl==1)).sum()} → label=0 (vivo 우선)")
    print(f"  vivo++vitro- {((both.vivo_lbl==1)&(both.vitro_lbl==0)).sum()} → label=1 (vivo 우선)")

    # 기존 vivo+vitro inchi_key 합집합으로 split
    splits = {}
    for sp in ("train", "val", "test"):
        v_iks  = set(pd.read_csv(f"{SPLIT_DIR}/{sp}/vivo.csv").inchi_key)
        vi_iks = set(pd.read_csv(f"{SPLIT_DIR}/{sp}/vitro.csv").inchi_key)
        splits[sp] = v_iks | vi_iks
    splits["val"]   -= splits["test"]
    splits["train"] -= (splits["test"] | splits["val"])

    for sp, iks in splits.items():
        sub = db_any[db_any.inchi_key.isin(iks)][
            ["inchi_key", "canonical_smiles", "name", "vp_label", "vivo_lbl", "vitro_lbl",
             "label_source"]
        ].copy()
        sub.columns = ["inchi_key", "canonical_smiles", "name", "label",
                       "vivo_label", "vitro_label", "label_source"]
        pos = (sub.label==1).sum(); neg = (sub.label==0).sum()
        print(f"  {sp}: {len(sub)} (양성 {pos} / 음성 {neg}) 비율 1:{neg/max(pos,1):.2f}")
        sp_dir = os.path.join(SPLIT_DIR, sp)
        sub.to_csv(os.path.join(sp_dir, "vp.csv"), index=False)


def train_vp():
    from src.train_domain_models import train_domain
    print("\n=== vp (vivo 우선) domain 학습 ===")
    return train_domain("vp")


def evaluate_on_vivo_test(vp_meta):
    """공평한 비교 — 같은 vivo test 에서 vp 모델 평가."""
    from src.train_domain_models import FPS, fp_xy
    import joblib
    from catboost import CatBoostClassifier
    from sklearn.metrics import (roc_auc_score, matthews_corrcoef, confusion_matrix,
                                  f1_score, accuracy_score)
    import numpy as np

    print("\n=== 같은 vivo test (1836) 에서 vp 모델 재평가 ===")
    vivo_test = pd.read_csv(f"{SPLIT_DIR}/test/vivo.csv")
    print(f"  vivo test: {len(vivo_test)} (양성 {(vivo_test.label==1).sum()})")

    db = pd.read_parquet(DB_PATH)
    out_dir = os.path.join(PROJECT_ROOT, "models", "vp")

    # 각 FP × kind 의 test prob 합산
    probs = []
    names = vp_meta["members"]
    weights = np.array(vp_meta["weights"])
    threshold = vp_meta["threshold"]

    for name in names:
        kind, fp_name = name.split("_", 1)
        # vivo test 의 FP
        X, y, _, _ = fp_xy(fp_name, vivo_test, "vp", db=db)
        sub_dir = os.path.join(out_dir, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sub_dir, "model.pkl"))
            p = m.predict_proba(X)[:, 1]
        else:
            m = CatBoostClassifier()
            m.load_model(os.path.join(sub_dir, "model.cbm"))
            p = m.predict_proba(X)[:, 1]
        probs.append(p)
        y_ref = y

    probs = np.array(probs).T
    score = probs @ weights
    pred = (score >= threshold).astype(int)
    cm = confusion_matrix(y_ref, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    auc = roc_auc_score(y_ref, score)
    mcc = matthews_corrcoef(y_ref, pred)
    f1 = f1_score(y_ref, pred, zero_division=0)
    acc = accuracy_score(y_ref, pred)
    print(f"  AUC {auc:.3f}  MCC {mcc:.3f}  F1 {f1:.3f}  Acc {acc:.3f}")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    return {
        "auc": float(auc), "mcc": float(mcc), "f1": float(f1), "acc": float(acc),
        "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
        "threshold": float(threshold), "n_test": int(len(y_ref)),
    }


def main():
    print("=== vivo 우선 + vitro fallback 학습 ===\n")
    build_vp_split()
    meta = train_vp()
    t = meta["test_metrics"]
    print(f"\n=== vp test (자체 split) ===")
    print(f"  AUC {t['auc']:.3f}  MCC {t['mcc']:.3f}  F1 {t['f1']:.3f}")
    print(f"  TPR {t['tpr']:.3f}  TNR {t['tnr']:.3f}  threshold {t['threshold']:.3f}")

    vivo_eval = evaluate_on_vivo_test(meta)

    # 비교 저장
    import json
    out = {
        "vp_self_test": t,
        "vp_on_vivo_test": vivo_eval,
        "v1_baseline_vivo": {"auc": 0.752, "mcc": 0.276},  # 기존 값
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results", "vp_vs_v1.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/vp_vs_v1.json")


if __name__ == "__main__":
    main()
