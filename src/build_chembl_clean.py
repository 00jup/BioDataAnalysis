"""chEMBL 원본에서 깨끗한 train/val/test 데이터셋 빌드.

요구:
  - train ∩ val ∩ test 모두 InChIKey 겹침 0
  - 기존 모델 학습 데이터와도 겹침 0 (선택적 — 보고서용)
  - 라벨 충돌 제거 (같은 분자가 양·음 둘 다 나타나면 제외)

소스:
  /Users/parkjeong-uk/Downloads/(~260515) 바데분 chEMBL 데이터 다운로드/
    - chembl_toxic_positive_set.csv          (1508 양성)
    - chembl_hepatotoxicity_compounds.csv    (활동 주석 다양)

양성 활동 주석:
  drug-induced liver injury reported, Toxic, Most-Dili-Concern, Less-Dili-Concern,
  HH: Evidence of human hepatotoxicity, DILI positive (training set / test set)

음성 활동 주석:
  no drug-induced liver injury reported, Non-Toxic, Non-toxic, No-Dili-Concern

애매 (제외):
  Ambiguous-Dili-Concern

분할: stratified 70/15/15, random_state=42.

저장: data/experiments/chembl_clean/{train,val,test,full}.csv
"""

from __future__ import annotations

import os

import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMBL_DIR = "/Users/parkjeong-uk/Downloads/(~260515) 바데분 chEMBL 데이터 다운로드"
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "experiments", "chembl_clean")

POS_TERMS = {
    "drug-induced liver injury reported",
    "Toxic",
    "Most-Dili-Concern",
    "Less-Dili-Concern",
    "HH: Evidence of human hepatotoxicity",
    "DILI positive (training set)",
    "DILI positive (test set)",
}
NEG_TERMS = {
    "no drug-induced liver injury reported",
    "Non-Toxic",
    "Non-toxic",
    "No-Dili-Concern",
}

RANDOM_STATE = 42


def canonicalize(smi: str) -> str | None:
    if not isinstance(smi, str) or not smi.strip():
        return None
    m = Chem.MolFromSmiles(smi)
    return Chem.MolToSmiles(m) if m else None


def inchikey(smi: str) -> str:
    m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    return Chem.MolToInchiKey(m) if m else ""


def collect_existing_training_ik() -> set[str]:
    """기존 모델이 학습에 본 모든 InChIKey 모음 — 누수 진단용."""
    seen = set()
    for p in [
        "data/experiments/exp2/manifest.csv",
        "data/experiments/exp_clean_strict/manifest.csv",
        "data/experiments/exp_clean_full/manifest.csv",
        "data/experiments/exp_clean_nosider/manifest.csv",
        "data/experiments/exp_clean_da/manifest.csv",
    ]:
        path = os.path.join(PROJECT_ROOT, p)
        if os.path.exists(path):
            seen |= set(pd.read_csv(path)["inchi_key"].dropna())
    return seen


def main() -> None:
    pos_set = pd.read_csv(os.path.join(CHEMBL_DIR, "chembl_toxic_positive_set.csv"))
    compounds = pd.read_csv(os.path.join(CHEMBL_DIR, "chembl_hepatotoxicity_compounds.csv"))

    # --- 양성 풀 ---
    p1 = pos_set.copy()
    p1["label"] = 1
    p2 = compounds[compounds["activity_comment"].isin(POS_TERMS)].copy()
    p2["label"] = 1
    pos = pd.concat([
        p1[["canonical_smiles", "molecule_chembl_id", "label"]],
        p2[["canonical_smiles", "molecule_chembl_id", "label"]],
    ], ignore_index=True)
    pos = pos.dropna(subset=["canonical_smiles"]).drop_duplicates("molecule_chembl_id")

    # --- 음성 풀 ---
    n_raw = compounds[compounds["activity_comment"].isin(NEG_TERMS)].copy()
    n_raw["label"] = 0
    neg = n_raw[["canonical_smiles", "molecule_chembl_id", "label"]].dropna(subset=["canonical_smiles"]).drop_duplicates("molecule_chembl_id")

    print(f"양성 후보 (원시): {len(pos)}, 음성 후보 (원시): {len(neg)}")

    # --- 같은 molecule_chembl_id 가 양·음 양쪽이면 제외 ---
    conflict = set(pos["molecule_chembl_id"]) & set(neg["molecule_chembl_id"])
    if conflict:
        pos = pos[~pos["molecule_chembl_id"].isin(conflict)]
        neg = neg[~neg["molecule_chembl_id"].isin(conflict)]
        print(f"chembl_id 라벨 충돌 {len(conflict)} 제거 → 양성 {len(pos)}, 음성 {len(neg)}")

    # --- canonicalize + InChIKey, 충돌 한번 더 ---
    pool = pd.concat([pos, neg], ignore_index=True)
    pool["canonical_smiles"] = pool["canonical_smiles"].map(canonicalize)
    pool = pool.dropna(subset=["canonical_smiles"])
    pool["inchi_key"] = pool["canonical_smiles"].map(inchikey)
    pool = pool[pool["inchi_key"].astype(bool)]

    # InChIKey 단위 충돌 (같은 분자 다른 chembl_id에서 다른 라벨) → 제외
    g = pool.groupby("inchi_key")["label"].nunique()
    ik_conf = set(g[g > 1].index)
    if ik_conf:
        pool = pool[~pool["inchi_key"].isin(ik_conf)]
        print(f"InChIKey 라벨 충돌 {len(ik_conf)} 제거")

    pool = pool.drop_duplicates("inchi_key").reset_index(drop=True)
    n_pos = int((pool["label"] == 1).sum())
    n_neg = int((pool["label"] == 0).sum())
    print(f"\n=== 정제 chEMBL 풀 ===")
    print(f"  총 {len(pool)} (양성 {n_pos} / 음성 {n_neg})")

    # --- 기존 모델 학습 데이터와의 겹침 진단 ---
    seen_train = collect_existing_training_ik()
    overlap = set(pool["inchi_key"]) & seen_train
    print(f"\n기존 모델 학습 데이터와 겹침: {len(overlap)} / {len(pool)} ({100*len(overlap)/len(pool):.1f}%)")
    pool["in_prev_train"] = pool["inchi_key"].isin(seen_train)

    # --- 분할: 70/15/15 stratified by label, 단 in_prev_train 도 균등 분포 시도 ---
    # 우선 fresh (기존 미사용) 와 seen 분리
    fresh = pool[~pool["in_prev_train"]].copy()
    print(f"  fresh (기존 학습 미사용): {len(fresh)} (양성 {int((fresh.label==1).sum())} / 음성 {int((fresh.label==0).sum())})")

    # fresh 만으로 70/15/15
    train, temp = train_test_split(fresh, test_size=0.30, random_state=RANDOM_STATE, stratify=fresh["label"])
    val, test = train_test_split(temp, test_size=0.50, random_state=RANDOM_STATE, stratify=temp["label"])

    # 누수 자체검증
    s_tr = set(train["inchi_key"]); s_v = set(val["inchi_key"]); s_te = set(test["inchi_key"])
    assert len(s_tr & s_v) == 0 and len(s_tr & s_te) == 0 and len(s_v & s_te) == 0, "분할 누수"
    print(f"\n=== 분할 (fresh 기준) ===")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        p = int((df.label==1).sum()); n = int((df.label==0).sum())
        print(f"  {name:5s} {len(df):5d} (양성 {p:4d} / 음성 {n:4d} / 양성 비율 {p/len(df):.2%})")

    os.makedirs(OUT_DIR, exist_ok=True)
    cols = ["canonical_smiles", "inchi_key", "label", "molecule_chembl_id"]
    pool[cols + ["in_prev_train"]].to_csv(os.path.join(OUT_DIR, "full_pool.csv"), index=False)
    fresh[cols].to_csv(os.path.join(OUT_DIR, "fresh_pool.csv"), index=False)
    train[cols].to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    val[cols].to_csv(os.path.join(OUT_DIR, "val.csv"), index=False)
    test[cols].to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)
    print(f"\n저장: {OUT_DIR}/")


if __name__ == "__main__":
    main()
