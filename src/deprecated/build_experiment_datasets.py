"""3개 실험 데이터셋 + 공통 외부 test 생성.

변형:
  exp1 - 양성: 저장소 hepatotoxic(lenient)        / 음성: marketed_clean 5,000
  exp2 - 양성: 저장소 + DILIst + GoldStandard union / 음성: marketed_clean 5,000
  exp3 - 양성: union                                / 음성: marketed_clean 전체

외부 test (3변형 공통, 고정):
  양성 - 팀원 chEMBL chembl_toxic_positive_set.csv 중 학습 양성과 안 겹치는 것
  음성 - chEMBL chembl_hepatotoxicity_compounds.csv 의 모순 없는 음성 중 학습과 안 겹치는 것

누수 차단: 학습 분자와 test 분자는 InChIKey 기준 상호 배타.
"""

from __future__ import annotations

import os

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(PROJECT_ROOT, "data", "marketed_drugs")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")

CHEMBL_DIR = "/Users/jeje/Downloads/(~260515) 바데분 chEMBL 데이터 다운로드"
CHEMBL_POS = os.path.join(CHEMBL_DIR, "chembl_toxic_positive_set.csv")
CHEMBL_COMPOUNDS = os.path.join(CHEMBL_DIR, "chembl_hepatotoxicity_compounds.csv")

NEG_NEG_TERMS = ["no drug-induced liver injury reported", "Non-Toxic"]
NEG_POS_TERMS = ["drug-induced liver injury reported", "Toxic"]

RANDOM_STATE = 42
VAL_RATIO = 0.15
NEG_SMALL = 5000


def to_inchikey(smiles) -> str:
    if not isinstance(smiles, str) or not smiles.strip():
        return ""
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def canonicalize(smiles) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _norm_df(df: pd.DataFrame, smiles_col: str) -> pd.DataFrame:
    """canonical_smiles + inchi_key 컬럼을 보장하고 무효 행 제거."""
    out = df.copy()
    out["canonical_smiles"] = out[smiles_col].apply(canonicalize)
    out = out[out["canonical_smiles"].notna()].copy()
    if "inchi_key" not in out.columns or out["inchi_key"].isna().any():
        out["inchi_key"] = out["canonical_smiles"].apply(to_inchikey)
    out = out[out["inchi_key"].astype(bool)]
    return out[["canonical_smiles", "inchi_key"]].drop_duplicates("inchi_key")


def build_external_test(train_pos_ik: set, train_neg_ik: set) -> pd.DataFrame:
    """학습과 안 겹치는 chEMBL 양성/음성으로 외부 test 구성."""
    pos = _norm_df(pd.read_csv(CHEMBL_POS), "canonical_smiles")
    pos = pos[~pos["inchi_key"].isin(train_pos_ik)].copy()
    pos["label"] = 1

    comp = pd.read_csv(CHEMBL_COMPOUNDS)
    neg_rows = comp[comp["activity_comment"].isin(NEG_NEG_TERMS)]
    pos_rows = comp[comp["activity_comment"].isin(NEG_POS_TERMS)]
    # 같은 분자에 toxic/non-toxic 모순 라벨이면 음성에서 제외
    clean_neg_mol = set(neg_rows["molecule_chembl_id"]) - set(pos_rows["molecule_chembl_id"])
    neg = _norm_df(neg_rows[neg_rows["molecule_chembl_id"].isin(clean_neg_mol)], "canonical_smiles")
    neg = neg[~neg["inchi_key"].isin(train_neg_ik)].copy()
    neg["label"] = 0

    test = pd.concat([pos, neg], ignore_index=True).drop_duplicates("inchi_key")
    test["source"] = "chembl_external"
    print(
        f"  외부 test: 양성 {int((test['label'] == 1).sum())}, 음성 {int((test['label'] == 0).sum())}"
    )
    return test[["canonical_smiles", "inchi_key", "label", "source"]]


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train, val = train_test_split(
        df, test_size=VAL_RATIO, random_state=RANDOM_STATE, stratify=df["label"]
    )
    return train.reset_index(drop=True), val.reset_index(drop=True)


def save_variant(name: str, pos: pd.DataFrame, neg: pd.DataFrame) -> None:
    combined = pd.concat([pos, neg], ignore_index=True).drop_duplicates("inchi_key")
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
    train, val = stratified_split(combined)

    vdir = os.path.join(OUT_DIR, name)
    os.makedirs(vdir, exist_ok=True)
    cols = ["canonical_smiles", "inchi_key", "label", "source"]
    train[cols].to_csv(os.path.join(vdir, "train.csv"), index=False)
    val[cols].to_csv(os.path.join(vdir, "val.csv"), index=False)
    combined[cols].to_csv(os.path.join(vdir, "manifest.csv"), index=False)
    n_pos = int((combined["label"] == 1).sum())
    n_neg = int((combined["label"] == 0).sum())
    print(f"  {name}: 양성 {n_pos} / 음성 {n_neg} (train {len(train)}, val {len(val)})")


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- 양성 소스 ---
    repo_pos = _norm_df(
        pd.read_csv(os.path.join(MD, "hepatotoxic", "hepatotoxic_all_lenient.csv")),
        "canonical_smiles",
    )
    ext_pos = _norm_df(
        pd.read_csv(os.path.join(MD, "hepatotoxic", "external", "external_positives.csv")),
        "canonical_smiles",
    )
    union_pos = pd.concat([repo_pos, ext_pos], ignore_index=True).drop_duplicates("inchi_key")
    print(
        f"양성 — 저장소 {len(repo_pos)}, 외부(DILIst+Gold) {len(ext_pos)}, union {len(union_pos)}"
    )

    # --- 음성 소스 ---
    neg_all = _norm_df(
        pd.read_csv(os.path.join(MD, "non_hepatotoxic", "marketed_clean.csv")),
        "canonical_smiles",
    )
    # 양성과 라벨 충돌하는 음성 제거
    neg_all = neg_all[~neg_all["inchi_key"].isin(set(union_pos["inchi_key"]))]
    print(f"음성 — marketed_clean (양성 충돌 제거 후) {len(neg_all)}")

    # --- 외부 test (학습 후보 전체와 배타) ---
    test = build_external_test(set(union_pos["inchi_key"]), set(neg_all["inchi_key"]))
    test_ik = set(test["inchi_key"])

    # 학습 풀에서 test 분자 제외
    union_pos = union_pos[~union_pos["inchi_key"].isin(test_ik)]
    repo_pos = repo_pos[~repo_pos["inchi_key"].isin(test_ik)]
    neg_all = neg_all[~neg_all["inchi_key"].isin(test_ik)]

    for d in (repo_pos, ext_pos, union_pos, neg_all):
        d["source"] = d.get("source", "")
    repo_pos = repo_pos.assign(label=1, source="repo")
    union_pos = union_pos.assign(label=1, source="union")
    neg_all = neg_all.assign(label=0, source="marketed_clean")

    neg_5000 = neg_all.sample(n=NEG_SMALL, random_state=RANDOM_STATE)

    print("\n변형 생성:")
    save_variant("exp1", repo_pos, neg_5000)
    save_variant("exp2", union_pos, neg_5000)
    save_variant("exp3", union_pos, neg_all)

    tdir = os.path.join(OUT_DIR, "external_test")
    os.makedirs(tdir, exist_ok=True)
    test.to_csv(os.path.join(tdir, "test.csv"), index=False)
    print(f"\n외부 test 저장: {os.path.join(tdir, 'test.csv')} ({len(test)}개)")


if __name__ == "__main__":
    main()
