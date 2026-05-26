"""v28 — verified 라벨 학습 (Agent 검증 결과 통합).

약한 출처만 양성 1,661 → verified:
  494 양성 → 그대로 유지
  374 음성 → 음성으로 변경
  793 비약물 → 학습 제외 (None)
"""
from __future__ import annotations
import json, os, sys, subprocess, time
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_v23_honest import SANITY_DRUGS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v28")
SAVE = os.path.join(PROJECT_ROOT, "models", "chemprop_v28")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
SEED = 2222


def build_data():
    """v27 데이터 + verified 라벨 update."""
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    verified = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "labels_db",
                                          "conflicts", "verify_v28",
                                          "weak_positives_verified.csv"))
    print(f"DB: {len(db)}, verified: {len(verified)}")

    # verified 의 manual_label 로 vivo_label 업데이트
    verify_map = dict(zip(verified.inchi_key, verified.manual_label))
    db["vivo_label_v28"] = db["vivo_label"].copy()
    n_changed_to_neg = 0; n_changed_to_none = 0; n_kept = 0
    for ik, lab in verify_map.items():
        if ik not in db.inchi_key.values: continue
        mask = db.inchi_key == ik
        if pd.isna(lab):
            db.loc[mask, "vivo_label_v28"] = pd.NA
            n_changed_to_none += 1
        elif lab == 0:
            db.loc[mask, "vivo_label_v28"] = 0
            n_changed_to_neg += 1
        elif lab == 1:
            n_kept += 1  # 그대로
    print(f"  verified 1→1 유지: {n_kept}")
    print(f"  verified 1→0 변경: {n_changed_to_neg}")
    print(f"  verified 1→None 변경 (비약물): {n_changed_to_none}")

    sanity_iks = set(); sanity_csmi = set()
    for _, smi, _ in SANITY_DRUGS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        sanity_iks.add(Chem.MolToInchiKey(mol))
        sanity_csmi.add(Chem.MolToSmiles(mol))

    pure = db[db.vivo_label_v28.notna() & db.vitro_label.isna()].copy()
    pure["label"] = pure["vivo_label_v28"].astype(int)
    pure = pure[~pure.inchi_key.isin(sanity_iks) & ~pure.canonical_smiles.isin(sanity_csmi)]
    pure = pure.dropna(subset=["canonical_smiles", "label"])
    print(f"\nv28 학습: {len(pure)} (양성 {(pure.label==1).sum()}, 음성 {(pure.label==0).sum()})")
    os.makedirs(os.path.join(DATA, "vivo"), exist_ok=True)
    csv = os.path.join(DATA, "vivo", "all.csv")
    pure[["canonical_smiles", "label"]].to_csv(csv, index=False)
    return csv


def main():
    print("=== Chemprop v28 — verified 라벨 ===\n")
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
