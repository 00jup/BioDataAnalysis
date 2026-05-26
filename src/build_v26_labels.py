"""v26 — ChEMBL/CTD 가중치 0.5 (약하게 포함) + pure vivo + sanity 제외."""
from __future__ import annotations
import os, sys
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_v23_honest import SANITY_DRUGS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
OUT_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v26_balanced")


def vivo_decide_v26(row):
    """ChEMBL/CTD 가중치 0.5 — 약한 보조 신호."""
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
    # ChEMBL/CTD 가중치 낮춤 (0.5 — 약한 신호)
    ct = row.get("vivo_ctd")
    if ct == "strong": pos += 0.5
    elif ct == "medium": pos += 0.3
    elif ct == "weak": pos += 0.15
    ce = row.get("vivo_chembl")
    if pd.notna(ce) and int(ce) == 1: pos += 0.5
    elif pd.notna(ce) and int(ce) == 0: neg += 0.5  # ChEMBL 음성도 약한 음성

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
    db["v26_label"] = db.apply(vivo_decide_v26, axis=1)
    pos = (db.v26_label == 1).sum()
    neg = (db.v26_label == 0).sum()
    print(f"DB: {len(db)}")
    print(f"v26 (ChEMBL/CTD 가중치 0.5):")
    print(f"  양성 {pos}, 음성 {neg}, 비율 1:{neg/max(pos,1):.2f}")

    sanity_iks = set(); sanity_csmi = set()
    for _, smi, _ in SANITY_DRUGS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        sanity_iks.add(Chem.MolToInchiKey(mol))
        sanity_csmi.add(Chem.MolToSmiles(mol))

    # pure vivo + sanity 제외
    pure = db[db.v26_label.notna() & db.vitro_label.isna()].copy()
    pure = pure[~pure.inchi_key.isin(sanity_iks) & ~pure.canonical_smiles.isin(sanity_csmi)]
    pure["label"] = pure["v26_label"].astype(int)
    pure = pure.dropna(subset=["canonical_smiles", "label"])
    print(f"\nv26 pure vivo + sanity 제외: {len(pure)} (양성 {(pure.label==1).sum()})")

    os.makedirs(os.path.join(OUT_DATA, "vivo"), exist_ok=True)
    csv = os.path.join(OUT_DATA, "vivo", "all.csv")
    pure[["canonical_smiles", "label"]].to_csv(csv, index=False)
    print(f"저장: {csv}")


if __name__ == "__main__":
    main()
