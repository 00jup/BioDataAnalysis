"""엄격(gold) 라벨로 Chemprop 재학습 — 데모용.

v27 하이퍼파라미터 그대로 + --class-balance(양성 23%) + chemprop 내장 SCAFFOLD_BALANCED.
데이터: data/strict/vivo/all.csv   모델: models/chemprop_strict/

사용:
    python src/train_strict.py                  # 로컬 CPU
    python src/train_strict.py --accelerator gpu # Colab GPU
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
CSV = os.path.join(PROJECT_ROOT, "data", "strict", "vivo", "all.csv")
OUT = os.path.join(PROJECT_ROOT, "models", "chemprop_strict")


def main():
    import subprocess

    ap = argparse.ArgumentParser()
    ap.add_argument("--accelerator", default="cpu", help="cpu | gpu (Colab)")
    ap.add_argument("--ensemble-size", default="15")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    cmd = [
        CHEMPROP_BIN, "train",
        "-i", CSV,
        "-s", "canonical_smiles",
        "--target-columns", "label",
        "-t", "classification",
        "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        # 내장 scaffold-balanced split (v27 와 동일)
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.7", "0.15", "0.15",
        "--save-smiles-splits",
        # v27 하이퍼파라미터
        "--ensemble-size", args.ensemble_size,
        "--message-hidden-dim", "600",
        "--depth", "3",
        "--aggregation", "norm",
        "--aggregation-norm", "100",
        "--ffn-hidden-dim", "300",
        "--ffn-num-layers", "1",
        "--dropout", "0.0",
        "--epochs", "40",
        "--patience", "8",
        "--warmup-epochs", "2",
        "--init-lr", "0.0001",
        "--max-lr", "0.001",
        "--final-lr", "0.0001",
        "--batch-size", "64",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        # 불균형 대응 (양성 23%)
        "--class-balance",
        "--data-seed", "1111",
        "--pytorch-seed", "1111",
        "--accelerator", args.accelerator,
        "-o", OUT,
    ]
    if args.accelerator == "gpu":
        cmd += ["--devices", "1", "--num-workers", "4"]

    log = os.path.join(OUT, "train.log")
    t0 = time.time()
    print(f"학습 시작 ({time.strftime('%H:%M:%S')})  acc={args.accelerator}  → {OUT}")
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"학습 끝 ({(time.time() - t0) / 60:.1f}분, exit={r.returncode})  로그: {log}")
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()
