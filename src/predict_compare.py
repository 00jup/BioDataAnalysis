"""모델 비교 추론 — SMILES 하나에 대해 모든 모델 예측을 한 표로.

비교 모델: chemprop v27, chemprop v31, RFCB v31, ChemBERTa-zinc, STACKED(앙상블).
각 모델의 양성 확률 + 0/1 을 나란히 보여준다.

사용:
    python src/predict_compare.py "CC(=O)Nc1ccc(O)cc1"
    python src/predict_compare.py --file drugs.txt
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.standardize import standardize  # noqa: E402
from src.train_domain_models import ensure_fp_cache  # noqa: E402

CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
M = os.path.join(PROJECT_ROOT, "models")
CP_V27 = os.path.join(M, "chemprop_v27")
CP_V31 = os.path.join(M, "chemprop_v31_class_expanded", "vivo")
RFCB = os.path.join(M, "rfcb_v31", "vivo")
CB = os.path.join(M, "chemberta_zinc_best")

# 모델별 표시용 threshold (단일 0/1 판정 기준)
THR = {"chemprop_v27": 0.30, "chemprop_v31": 0.50, "rfcb_v31": 0.40,
       "chemberta": 0.45, "STACKED": 0.55}
# STACKED 가중치 (honest stacking 결과)
STACK_W = {"chemprop_v31": 0.8, "chemberta": 0.2}


def read_inputs(args):
    rows = []
    if args.file:
        with open(args.file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                name, smi = (line.split(",", 1) if "," in line else ("", line))
                rows.append((name.strip(), smi.strip()))
    rows += [("", s) for s in args.smiles]
    return rows


def chemprop_predict(canon, model_dir):
    uniq = sorted(set(canon))
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": uniq}).to_csv(f.name, index=False)
        inp = f.name
    outp = inp.replace(".csv", "_p.csv")
    subprocess.run(
        [CHEMPROP_BIN, "predict", "--test-path", inp, "-s", "canonical_smiles",
         "--model-paths", model_dir, "--preds-path", outp,
         "--molecule-featurizers", "v1_rdkit_2d_normalized", "--accelerator", "cpu"],
        capture_output=True, check=True,
    )
    df = pd.read_csv(outp)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    return dict(zip(df["canonical_smiles"], df[pcol].astype(float)))


def rfcb_predict(canon):
    import json
    meta = json.load(open(os.path.join(RFCB, "ensemble_meta.json")))
    w = np.array(meta["weights"])
    uniq = sorted(set(canon))
    probs = []
    for name in meta["members"]:
        kind, fp = name.split("_", 1)
        cache = ensure_fp_cache(uniq, fp)
        X = cache.loc[uniq, cache.columns.tolist()].to_numpy(dtype=np.uint8)
        sd = os.path.join(RFCB, name)
        m = joblib.load(os.path.join(sd, "model.pkl")) if kind == "rf" else CatBoostClassifier().__class__()
        if kind == "cb":
            m = CatBoostClassifier()
            m.load_model(os.path.join(sd, "model.cbm"))
        probs.append(m.predict_proba(X)[:, 1])
    return dict(zip(uniq, np.array(probs).T @ w))


_tok = _model = None


def chemberta_predict(canon):
    global _tok, _model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if _model is None:
        _tok = AutoTokenizer.from_pretrained(CB)
        _model = AutoModelForSequenceClassification.from_pretrained(CB).eval()
    uniq = sorted(set(canon))
    out = {}
    for i in range(0, len(uniq), 64):
        b = uniq[i:i + 64]
        enc = _tok(b, truncation=True, max_length=200, padding=True, return_tensors="pt")
        with torch.no_grad():
            out.update(dict(zip(b, torch.softmax(_model(**enc).logits, 1)[:, 1].numpy())))
    return out


def main():
    ap = argparse.ArgumentParser(description="모든 모델 비교 예측")
    ap.add_argument("smiles", nargs="*")
    ap.add_argument("--file")
    args = ap.parse_args()
    rows = read_inputs(args)
    if not rows:
        ap.error("입력 SMILES 가 없다.")

    std = [(n, raw, standardize(raw)) for n, raw in rows]
    canon = [s[2][0] for s in std if s[2] is not None]
    print("  예측 중 (chemprop×2, RFCB, ChemBERTa)…", file=sys.stderr)
    P = {
        "chemprop_v27": chemprop_predict(canon, CP_V27),
        "chemprop_v31": chemprop_predict(canon, CP_V31),
        "rfcb_v31": rfcb_predict(canon),
        "chemberta": chemberta_predict(canon),
    }
    cols = ["chemprop_v27", "chemprop_v31", "rfcb_v31", "chemberta", "STACKED"]

    for name, raw, sr in std:
        title = name or raw
        print(f"\n=== {title} ===")
        if sr is None:
            print("  SMILES 파싱 실패")
            continue
        c = sr[0]
        vals = {k: P[k].get(c, np.nan) for k in P}
        vals["STACKED"] = sum(STACK_W[k] * vals[k] for k in STACK_W)
        print(f"  {'모델':<14}{'확률':>8}{'판정':>6}")
        print("  " + "-" * 30)
        for k in cols:
            p = vals[k]
            pred = int(p >= THR[k])
            mark = "⚠1" if pred == 1 else " 0"
            star = "  ←최종" if k == "STACKED" else ""
            print(f"  {k:<14}{p:>8.3f}{mark:>6}{star}")
    print()


if __name__ == "__main__":
    main()
