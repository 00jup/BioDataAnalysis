"""v25 — ChEMBL+CTD 제외 + pure vivo + sanity 제외 + class_balance."""
from __future__ import annotations
import json, os, sys, subprocess, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v25_clean")
SAVE = os.path.join(PROJECT_ROOT, "models", "chemprop_v25")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
SEED = 789


def main():
    print(f"=== Chemprop v25 — ChEMBL/CTD 제외 + class_balance ===")
    save = SAVE; os.makedirs(save, exist_ok=True)
    cmd = [
        CHEMPROP_BIN, "train",
        "-i", os.path.join(DATA, "vivo", "all.csv"),
        "-s", "canonical_smiles", "--target-columns", "label",
        "-t", "classification", "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.70", "0.15", "0.15",
        "--data-seed", str(SEED), "--pytorch-seed", str(SEED),
        "--class-balance",       # 1:22.5 imbalance
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
