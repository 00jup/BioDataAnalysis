"""학습 데이터 정제 — 약한 출처만 가진 분자 제거 후 재학습 (v4).

정제 기준 (vivo):
  KEEP 양성: DILIrank vMost/vLess OR LiverTox A/B/C/D OR ClinTox=1
  KEEP 음성: DILIrank vNo OR LiverTox E OR marketed_clean OR ClinTox=0
  DROP: 양성 중 SIDER/DILIst/Gold/TDC 단독만 (강한 출처 없음)

정제 기준 (vitro):
  현재 chEMBL/Tox21 만 있으므로 기존 그대로 (정제할 약한 출처 없음)

저장:
  models/{vivo,vitro}_filtered/ — 정제 학습 모델
  results/train_v4_filtered.json — 결과
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS = os.path.join(PROJECT_ROOT, "results")


STRONG_VIVO_POS_SOURCES = ("vivo_dilirank_pos", "vivo_livertox_A_to_D", "vivo_clintox_pos")
STRONG_VIVO_NEG_SOURCES = ("vivo_dilirank_vNo", "vivo_livertox_E", "vivo_clintox_neg", "vivo_marketed_clean_neg")


def has_strong_vivo_label(row, is_positive: bool) -> bool:
    """분자가 강한 vivo 출처에 의해 라벨된 경우만 True."""
    if is_positive:
        # 양성: 강한 양성 출처 필요
        if row.get("vivo_dilirank") in ("vMost-DILI-Concern", "vLess-DILI-Concern"):
            return True
        if row.get("vivo_livertox") in ("A", "B", "C", "D"):
            return True
        if row.get("vivo_clintox") == 1:
            return True
        return False
    else:
        # 음성: 강한 음성 출처 필요
        if row.get("vivo_dilirank") == "vNo-DILI-Concern":
            return True
        if row.get("vivo_livertox") == "E":
            return True
        if row.get("vivo_clintox") == 0:
            return True
        # marketed_clean_neg: 약하지만 0.97 agreement → 신뢰 OK (큰 비중)
        if row.get("vivo_marketed_clean_neg") == 1:
            return True
        return False


def filter_vivo_train(train_df: pd.DataFrame, db: pd.DataFrame) -> pd.DataFrame:
    """vivo train 에서 약한 출처 단독 분자 제거."""
    db_by_ik = db.set_index("inchi_key")
    keep_mask = []
    for _, r in train_df.iterrows():
        ik = r["inchi_key"]
        if ik not in db_by_ik.index:
            keep_mask.append(False); continue
        db_row = db_by_ik.loc[ik]
        if isinstance(db_row, pd.DataFrame): db_row = db_row.iloc[0]
        is_pos = r["label"] == 1
        keep_mask.append(has_strong_vivo_label(db_row, is_pos))
    return train_df[keep_mask].reset_index(drop=True)


def main():
    print("=== 학습 데이터 정제 + 재학습 (v4) ===\n")

    # vivo 정제
    db = pd.read_parquet(DB_PATH)
    tr_vivo = pd.read_csv(os.path.join(TRAIN_DIR, "vivo.csv"))
    print(f"vivo 원본 train: {len(tr_vivo)} (양성 {(tr_vivo.label==1).sum()} / 음성 {(tr_vivo.label==0).sum()})")
    tr_vivo_filtered = filter_vivo_train(tr_vivo, db)
    print(f"정제 후: {len(tr_vivo_filtered)} (양성 {(tr_vivo_filtered.label==1).sum()} / 음성 {(tr_vivo_filtered.label==0).sum()})")
    print(f"제거된 분자: {len(tr_vivo) - len(tr_vivo_filtered)} ({(len(tr_vivo)-len(tr_vivo_filtered))/len(tr_vivo)*100:.1f}%)")

    # 임시 저장 후 train_domain_models 재호출
    tmp_train = os.path.join(TRAIN_DIR, "vivo_filtered.csv")
    tr_vivo_filtered.to_csv(tmp_train, index=False)

    # 같은 학습 코드로 vivo_filtered 학습
    from src.train_domain_models import train_domain, MODELS_DIR as MD

    # train_domain 이 "vivo" 도메인을 보고 data/train/vivo.csv 를 읽는 식이라 패치 필요
    # 간단히 — train_filtered 라는 새 도메인 추가 (CSV 만 바꿔서)

    # vivo CSV 원본 백업 + filtered 로 교체 → 학습 → 복원
    orig_train = os.path.join(TRAIN_DIR, "vivo.csv")
    backup = orig_train + ".bak"
    os.rename(orig_train, backup)
    tr_vivo_filtered.to_csv(orig_train, index=False)
    # models 도 별도 경로로
    os.environ["MODEL_OUT_SUFFIX"] = "_filtered"

    print("\n=== vivo (filtered) 재학습 ===")
    meta_vivo = train_domain("vivo")
    # 결과 별도 폴더로 옮김
    import shutil
    src_dir = os.path.join(MD, "vivo")
    dst_dir = os.path.join(MD, "vivo_filtered")
    if os.path.exists(dst_dir): shutil.rmtree(dst_dir)
    shutil.copytree(src_dir, dst_dir)

    # CSV 원복
    os.remove(orig_train)
    os.rename(backup, orig_train)

    # vitro 는 정제 없이 그대로 (출처 2개라 정제할 게 없음)
    print(f"\n저장: models/vivo_filtered/")
    print(f"\n=== v4 결과 (vivo 만) ===")
    t = meta_vivo["test_metrics"]
    print(f"  TPR {t['tpr']:.3f}  TNR {t['tnr']:.3f}  bAcc {(t['tpr']+t['tnr'])/2:.3f}")
    print(f"  MCC {t['mcc']:.3f}  F1 {t['f1']:.3f}  AUC {t['auc']:.3f}")

    with open(os.path.join(RESULTS, "train_v4_filtered.json"), "w") as f:
        json.dump({"vivo_filtered": meta_vivo}, f, indent=2, ensure_ascii=False)
    print(f"저장: results/train_v4_filtered.json")


if __name__ == "__main__":
    main()
