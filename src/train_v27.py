"""v27 — 새 룰 (OR + manual curation) + pure vivo + sanity 제외 학습."""
from __future__ import annotations
import json, os, sys, subprocess, time
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_v23_honest import SANITY_DRUGS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v27")
SAVE = os.path.join(PROJECT_ROOT, "models", "chemprop_v27")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
SEED = 1111


def build_data():
    """새 DB (OR + manual curation) 에서 pure vivo + sanity 제외."""
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    print(f"DB: {len(db)}")
    sanity_iks = set(); sanity_csmi = set()
    for _, smi, _ in SANITY_DRUGS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        sanity_iks.add(Chem.MolToInchiKey(mol))
        sanity_csmi.add(Chem.MolToSmiles(mol))

    pure = db[db.vivo_label.notna() & db.vitro_label.isna()].copy()
    pure["label"] = pure["vivo_label"].astype(int)
    pure = pure[~pure.inchi_key.isin(sanity_iks) & ~pure.canonical_smiles.isin(sanity_csmi)]
    pure = pure.dropna(subset=["canonical_smiles", "label"])
    print(f"v27 학습 데이터: {len(pure)} (양성 {(pure.label==1).sum()}, 음성 {(pure.label==0).sum()})")
    os.makedirs(os.path.join(DATA, "vivo"), exist_ok=True)
    csv = os.path.join(DATA, "vivo", "all.csv")
    pure[["canonical_smiles", "label"]].to_csv(csv, index=False)
    return csv


def main():
    print(f"=== Chemprop v27 — 새 룰 (OR + manual curation) ===\n")
    csv = build_data()
    save = SAVE; os.makedirs(save, exist_ok=True)
    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv, "-s", "canonical_smiles", "--target-columns", "label",
        "-t", "classification", "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.70", "0.15", "0.15",
        "--data-seed", str(SEED), "--pytorch-seed", str(SEED),
        "--ensemble-size", "15",
        "--message-hidden-dim", "600",
        "--epochs", "40", "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "--save-smiles-splits",
        "-o", save,
    ]
    log = os.path.join(save, "train.log")
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")


if __name__ == "__main__":
    main()
