"""vivo + vitro 통합 학습 — 단일 라벨 (OR 합).

전략:
  - DB 의 vivo_label 과 vitro_label 둘 다 본다.
  - 통합 라벨 = max(vivo, vitro) — 어느 한쪽이라도 양성이면 양성.
  - 한쪽만 있는 분자도 사용 (NaN 한쪽은 fillna(0) 으로 처리).
  - 모델: RF + CatBoost × 5 FP (vivo·vitro 와 동일 hp).

비교 기준:
  - 같은 vivo test 와 vitro test 에서 측정.
  - 통합 라벨 test 도 별도 측정.
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
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "unified")
RANDOM_STATE = 42


def build_unified_split():
    """vivo + vitro 라벨을 OR 합해서 unified 데이터 만든다."""
    db = pd.read_parquet(DB_PATH)
    print(f"DB: {len(db)} 분자")

    # vivo/vitro 라벨 추출
    db["vivo_lbl"] = pd.to_numeric(db.get("vivo_label"), errors="coerce")
    db["vitro_lbl"] = pd.to_numeric(db.get("vitro_label"), errors="coerce")
    has_any = db["vivo_lbl"].notna() | db["vitro_lbl"].notna()
    db_any = db[has_any].copy()

    # 통합 라벨 = OR (NaN 은 0)
    db_any["unified"] = (
        (db_any["vivo_lbl"].fillna(0).astype(int) == 1)
        | (db_any["vitro_lbl"].fillna(0).astype(int) == 1)
    ).astype(int)
    print(f"통합 가능 분자: {len(db_any)} (양성 {(db_any.unified==1).sum()} / 음성 {(db_any.unified==0).sum()})")
    print(f"  vivo+: {(db_any.vivo_lbl==1).sum()}, vivo-: {(db_any.vivo_lbl==0).sum()}")
    print(f"  vitro+: {(db_any.vitro_lbl==1).sum()}, vitro-: {(db_any.vitro_lbl==0).sum()}")
    print(f"  vivo-only: {(db_any.vivo_lbl.notna() & db_any.vitro_lbl.isna()).sum()}")
    print(f"  vitro-only: {(db_any.vitro_lbl.notna() & db_any.vivo_lbl.isna()).sum()}")
    print(f"  both: {(db_any.vivo_lbl.notna() & db_any.vitro_lbl.notna()).sum()}")

    # 충돌 케이스
    both = db_any[db_any.vivo_lbl.notna() & db_any.vitro_lbl.notna()]
    conflict_neg_pos = ((both.vivo_lbl == 0) & (both.vitro_lbl == 1)).sum()
    conflict_pos_neg = ((both.vivo_lbl == 1) & (both.vitro_lbl == 0)).sum()
    print(f"  충돌: vivo-+vitro+ {conflict_neg_pos}, vivo++vitro- {conflict_pos_neg}")

    # split — vivo/vitro 의 기존 train/val/test inchi_key 와 호환되도록
    # 통합 split 은 union 하되 같은 inchi_key 가 train/test 안 겹치게 한다.
    # 기존 vivo+vitro test inchi_key 합집합을 test 로 사용
    splits = {}
    for sp in ("train", "val", "test"):
        v_iks = set(pd.read_csv(f"{SPLIT_DIR}/{sp}/vivo.csv").inchi_key)
        vi_iks = set(pd.read_csv(f"{SPLIT_DIR}/{sp}/vitro.csv").inchi_key)
        splits[sp] = v_iks | vi_iks
    # 충돌 (한 분자가 다른 split 에 동시 존재) — 더 보수적인 split 으로 이동 (test > val > train)
    splits["val"] -= splits["test"]
    splits["train"] -= (splits["test"] | splits["val"])

    out = {}
    for sp, iks in splits.items():
        sub = db_any[db_any.inchi_key.isin(iks)][["inchi_key", "canonical_smiles", "name",
                                                    "unified", "vivo_lbl", "vitro_lbl"]].copy()
        sub.columns = ["inchi_key", "canonical_smiles", "name", "label",
                       "vivo_label", "vitro_label"]
        out[sp] = sub
        print(f"  {sp}: {len(sub)} (양성 {(sub.label==1).sum()} / 음성 {(sub.label==0).sum()}) "
              f"비율 1:{(sub.label==0).sum()/max((sub.label==1).sum(),1):.2f}")
        sp_dir = os.path.join(SPLIT_DIR, sp)
        sub.to_csv(os.path.join(sp_dir, "unified.csv"), index=False)
    return out


def train_unified():
    """기존 train_domain 재활용 — domain 만 'unified' 로 호출."""
    from src.train_domain_models import train_domain
    print("\n=== unified domain 학습 (RF + CatBoost × 5 FP) ===")
    meta = train_domain("unified")
    return meta


def main():
    print("=== Unified vivo+vitro 통합 학습 ===\n")
    os.makedirs(MODELS_DIR, exist_ok=True)
    build_unified_split()
    meta = train_unified()
    t = meta["test_metrics"]
    print(f"\n=== unified test ===")
    print(f"  AUC {t['auc']:.3f}  MCC {t['mcc']:.3f}  F1 {t['f1']:.3f}")
    print(f"  TPR {t['tpr']:.3f}  TNR {t['tnr']:.3f}  threshold {t['threshold']:.3f}")


if __name__ == "__main__":
    main()
