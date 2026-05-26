"""v25 — ChEMBL + CTD 빼고 라벨 재합성.

feature-agreement < 0.5 인 출처 둘 다 제외:
  - ChEMBL (0.43) — oversensitive ALT
  - CTD (0.43) — oversensitive consensus 33% 충돌

남은 강한 출처만:
  - DILIrank, LiverTox, DailyMed, PubMed, FAERS, marketed_clean
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
OUT_DB = os.path.join(PROJECT_ROOT, "data", "labels_db", "full_no_chembl_ctd.parquet")
OUT_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v25_clean")


def vivo_decide_clean(row):
    """ChEMBL/CTD 제외 — 강한 출처만 가중치."""
    pos = 0; neg = 0
    if row.get("vivo_dilirank") in ("vMost-DILI-Concern", "vLess-DILI-Concern"):
        pos += 2
    lt = row.get("vivo_livertox")
    if lt in ("A", "B"): pos += 2
    elif lt == "C": pos += 1
    elif lt == "D": pos += 0.5
    dm = row.get("vivo_dailymed")
    if dm == "boxed_hepatotox": pos += 3
    elif dm == "warning_hepatotox": pos += 1.5
    elif dm == "adverse_hepatotox": pos += 0.5
    elif dm == "contraindication_hepatic": pos += 1
    pm = row.get("vivo_pubmed")
    if pm == "strong": pos += 2
    elif pm == "medium": pos += 1
    elif pm == "weak": pos += 0.5
    fa = row.get("vivo_faers")
    if fa == "strong": pos += 1.5
    elif fa == "medium": pos += 0.8
    elif fa == "weak": pos += 0.3
    # ChEMBL, CTD 제외 — 가중치 0

    if row.get("vivo_dilirank") == "vNo-DILI-Concern":
        neg += 2
    if lt == "E": neg += 2
    if pd.notna(row.get("vivo_marketed_clean_neg")) and row["vivo_marketed_clean_neg"] == 1 and pos == 0:
        neg += 1

    if row.get("vivo_dilirank") in ("vAmbig-DILI-Concern", "Ambiguous-DILI-Concern"):
        return None
    if pos == 0 and neg == 0:
        return None
    return 1 if pos > neg else (0 if neg > pos else None)


def main():
    db = pd.read_parquet(DB_PATH)
    db["vivo_label_clean"] = db.apply(vivo_decide_clean, axis=1)
    pos = (db.vivo_label_clean == 1).sum()
    neg = (db.vivo_label_clean == 0).sum()
    print(f"DB: {len(db)}")
    print(f"v25 clean (ChEMBL+CTD 제외):")
    print(f"  양성 {pos}, 음성 {neg}, 비율 1:{neg/max(pos,1):.2f}")

    db.to_parquet(OUT_DB, index=False)

    # 학습 데이터 — pure vivo (vitro 없음) + sanity 제외
    from src.train_v23_honest import SANITY_DRUGS
    from rdkit import Chem
    sanity_iks = set()
    sanity_csmi = set()
    for _, smi, _ in SANITY_DRUGS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        sanity_iks.add(Chem.MolToInchiKey(mol))
        sanity_csmi.add(Chem.MolToSmiles(mol))

    pure = db[db.vivo_label_clean.notna() & db.vitro_label.isna()].copy()
    pure = pure[~pure.inchi_key.isin(sanity_iks) & ~pure.canonical_smiles.isin(sanity_csmi)]
    pure["label"] = pure["vivo_label_clean"].astype(int)
    pure = pure.dropna(subset=["canonical_smiles", "label"])
    print(f"\nv25 pure vivo + sanity 제외: {len(pure)} (양성 {(pure.label==1).sum()})")

    os.makedirs(os.path.join(OUT_DATA, "vivo"), exist_ok=True)
    csv = os.path.join(OUT_DATA, "vivo", "all.csv")
    pure[["canonical_smiles", "label"]].to_csv(csv, index=False)
    print(f"저장: {csv}")


if __name__ == "__main__":
    main()
