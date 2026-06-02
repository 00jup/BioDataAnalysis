"""Stacked DILI 추론 — chemprop v31 + ChemBERTa-zinc 앙상블.

SMILES → 간독성 1/0. honest stacking 으로 결정한 가중치/threshold 사용:
    P = 0.8 * chemprop_v31 + 0.2 * ChemBERTa-zinc,   pred = 1 if P >= 0.55
(scaffold_v3 val 에서 결정, test AUC 0.797 / MCC 0.423)

사용:
    python src/predict_stack.py "CC(=O)Nc1ccc(O)cc1"
    python src/predict_stack.py --file drugs.txt
    python src/predict_stack.py "SMILES" --threshold 0.55
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.standardize import standardize  # noqa: E402

CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
CP_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v31_class_expanded", "vivo")
CB_DIR = os.path.join(PROJECT_ROOT, "models", "chemberta_zinc_best")

# honest stacking 결과 (scaffold_v3 val 에서 결정)
W_CHEMPROP = 0.8
W_CHEMBERTA = 0.2
DEFAULT_THRESHOLD = 0.55
MAX_LEN = 200


def read_inputs(args):
    rows = []
    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "," in line:
                    name, smi = line.split(",", 1)
                    rows.append((name.strip(), smi.strip()))
                else:
                    rows.append(("", line))
    rows += [("", s) for s in args.smiles]
    return rows


def chemprop_predict(canon):
    if not canon:
        return {}
    uniq = sorted(set(canon))
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": uniq}).to_csv(f.name, index=False)
        inp = f.name
    outp = inp.replace(".csv", "_p.csv")
    r = subprocess.run(
        [CHEMPROP_BIN, "predict", "--test-path", inp, "-s", "canonical_smiles",
         "--model-paths", CP_DIR, "--preds-path", outp,
         "--molecule-featurizers", "v1_rdkit_2d_normalized", "--accelerator", "cpu"],
        capture_output=True,
    )
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode()[-1000:] + "\n")
        raise RuntimeError("chemprop predict 실패")
    df = pd.read_csv(outp)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    return dict(zip(df["canonical_smiles"], df[pcol].astype(float)))


_tok = None
_model = None


def chemberta_predict(canon):
    global _tok, _model
    if not canon:
        return {}
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(CB_DIR)
        _model = AutoModelForSequenceClassification.from_pretrained(CB_DIR).eval()
    uniq = sorted(set(canon))
    out = {}
    for i in range(0, len(uniq), 64):
        batch = uniq[i:i + 64]
        enc = _tok(batch, truncation=True, max_length=MAX_LEN, padding=True, return_tensors="pt")
        with torch.no_grad():
            prob = torch.softmax(_model(**enc).logits, dim=1)[:, 1].numpy()
        out.update(dict(zip(batch, prob)))
    return out


def main():
    ap = argparse.ArgumentParser(description="Stacked DILI 예측 (chemprop+ChemBERTa)")
    ap.add_argument("smiles", nargs="*")
    ap.add_argument("--file")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--out")
    args = ap.parse_args()

    rows = read_inputs(args)
    if not rows:
        ap.error("입력 SMILES 가 없다.")

    std = [(name, raw, standardize(raw)) for name, raw in rows]
    canon = [s[2][0] for s in std if s[2] is not None]
    print("  chemprop 예측 중…", file=sys.stderr)
    pcp = chemprop_predict(canon)
    print("  ChemBERTa 예측 중…", file=sys.stderr)
    pcb = chemberta_predict(canon)

    print(f"\n  STACKED: {W_CHEMPROP}*chemprop + {W_CHEMBERTA}*ChemBERTa  |  threshold {args.threshold}\n")
    print(f"  {'이름':<16}{'예측':>5}{'확률':>8}{'cp':>7}{'cb':>7}   판정")
    print("  " + "-" * 64)
    recs = []
    for name, raw, sr in std:
        if sr is None:
            print(f"  {name or '-':<16}{'?':>5}{'-':>8}{'-':>7}{'-':>7}   파싱 실패")
            continue
        c = sr[0]
        a, b = pcp.get(c, np.nan), pcb.get(c, np.nan)
        P = W_CHEMPROP * a + W_CHEMBERTA * b
        pred = int(P >= args.threshold)
        verdict = "간독성 위험 ⚠" if pred == 1 else "안전"
        print(f"  {name or '-':<16}{pred:>5}{P:>8.3f}{a:>7.3f}{b:>7.3f}   {verdict}")
        recs.append({"name": name, "input": raw, "canonical": c,
                     "prob": round(float(P), 4), "chemprop": round(float(a), 4),
                     "chemberta": round(float(b), 4), "pred": pred})
    print()
    if args.out and recs:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
            w.writeheader()
            w.writerows(recs)
        print(f"  저장: {args.out}\n")


if __name__ == "__main__":
    main()
