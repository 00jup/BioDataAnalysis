"""v26 학습 — ChEMBL/CTD 가중치 0.5 + pure vivo + sanity 제외."""
from __future__ import annotations
import json, os, sys, subprocess, time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v26_balanced")
SAVE = os.path.join(PROJECT_ROOT, "models", "chemprop_v26")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
SEED = 999


def main():
    print(f"=== Chemprop v26 — ChEMBL/CTD weight 0.5 ===")
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
    print(f"  설정: ensemble 15 + hidden 600 + featurizer (v17 동일)")
    print(f"  데이터: pure vivo + sanity 제외, ChEMBL/CTD weight 0.5")
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")


if __name__ == "__main__":
    main()
