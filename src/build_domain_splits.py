"""도메인별 train/val/test split 생성.

DB labels_db/full.parquet 에서:
  - vivo 모델용: vivo_label 있는 분자 → train/val/test
  - vitro 모델용: vitro_label 있는 분자 → train/val/test

요구:
  - 같은 도메인 안에서 train ∩ val ∩ test = 0 (InChIKey 기준)
  - 양 도메인 사이에는 겹쳐도 OK (각자 자기 라벨 사용)
  - stratified 70/15/15

저장:
  data/train/{vivo,vitro}.csv
  data/val/{vivo,vitro}.csv
  data/test/{vivo,vitro}.csv  (기존 professor_test.csv 는 보존)
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
VAL_DIR = os.path.join(PROJECT_ROOT, "data", "val")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
RANDOM_STATE = 42
SPLIT = (0.70, 0.15, 0.15)


def stratified_split(df: pd.DataFrame, label_col: str):
    """70/15/15 stratified."""
    train, temp = train_test_split(df, test_size=SPLIT[1] + SPLIT[2],
                                    random_state=RANDOM_STATE, stratify=df[label_col])
    val_frac = SPLIT[1] / (SPLIT[1] + SPLIT[2])
    val, test = train_test_split(temp, test_size=(1 - val_frac),
                                  random_state=RANDOM_STATE, stratify=temp[label_col])
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def build_domain_split(db: pd.DataFrame, domain: str):
    """domain = 'vivo' or 'vitro'."""
    label_col = f"{domain}_label"
    sub = db[db[label_col].notna()].copy()
    sub[label_col] = sub[label_col].astype(int)

    # 최소 정보: smiles + label + name + 그 도메인 confidence/sources
    cols = ["inchi_key", "canonical_smiles", "name", label_col,
            f"{domain}_confidence", f"{domain}_n_sources"]
    sub = sub[[c for c in cols if c in sub.columns]].rename(columns={label_col: "label"})
    sub = sub.dropna(subset=["canonical_smiles", "inchi_key"]).drop_duplicates("inchi_key")

    train, val, test = stratified_split(sub, "label")
    return train, val, test


def main():
    for d in (TRAIN_DIR, VAL_DIR, TEST_DIR):
        os.makedirs(d, exist_ok=True)

    print("=== 도메인별 train/val/test split ===\n")
    db = pd.read_parquet(DB_PATH)
    print(f"DB 전체: {len(db)} 분자\n")

    # vivo
    print("[vivo 도메인]")
    train, val, test = build_domain_split(db, "vivo")
    print(f"  train {len(train)}  val {len(val)}  test {len(test)}")
    print(f"  양성 분포: train {(train.label==1).sum()} / val {(val.label==1).sum()} / test {(test.label==1).sum()}")
    print(f"  음성 분포: train {(train.label==0).sum()} / val {(val.label==0).sum()} / test {(test.label==0).sum()}")
    train.to_csv(os.path.join(TRAIN_DIR, "vivo.csv"), index=False)
    val.to_csv(os.path.join(VAL_DIR, "vivo.csv"), index=False)
    test.to_csv(os.path.join(TEST_DIR, "vivo.csv"), index=False)

    # 누수 자체검증
    assert len(set(train.inchi_key) & set(val.inchi_key)) == 0
    assert len(set(train.inchi_key) & set(test.inchi_key)) == 0
    assert len(set(val.inchi_key) & set(test.inchi_key)) == 0
    print(f"  ✓ 누수 없음 (train∩val=val∩test=train∩test=0)")

    # vitro
    print("\n[vitro 도메인]")
    train, val, test = build_domain_split(db, "vitro")
    print(f"  train {len(train)}  val {len(val)}  test {len(test)}")
    print(f"  양성 분포: train {(train.label==1).sum()} / val {(val.label==1).sum()} / test {(test.label==1).sum()}")
    print(f"  음성 분포: train {(train.label==0).sum()} / val {(val.label==0).sum()} / test {(test.label==0).sum()}")
    train.to_csv(os.path.join(TRAIN_DIR, "vitro.csv"), index=False)
    val.to_csv(os.path.join(VAL_DIR, "vitro.csv"), index=False)
    test.to_csv(os.path.join(TEST_DIR, "vitro.csv"), index=False)

    assert len(set(train.inchi_key) & set(val.inchi_key)) == 0
    assert len(set(train.inchi_key) & set(test.inchi_key)) == 0
    assert len(set(val.inchi_key) & set(test.inchi_key)) == 0
    print(f"  ✓ 누수 없음")

    print(f"\n저장: data/{{train,val,test}}/{{vivo,vitro}}.csv")


if __name__ == "__main__":
    main()
