"""Stacking 준비 — chemprop v31 + RFCB v31 의 val/test 예측을 scaffold_v3 split 에서 생성.

ChemBERTa(Colab) 예측과 '같은 split' 이라 honest stacking 에 바로 합칠 수 있다.
출력: results/stack_preds/{chemprop_v31,rfcb_v31}_{val,test}_pred.csv  (canonical_smiles,label,prob)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import ensure_fp_cache  # noqa: E402

CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v3", "vivo")
CP_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v31_class_expanded", "vivo")
RFCB_DIR = os.path.join(PROJECT_ROOT, "models", "rfcb_v31", "vivo")
OUT = os.path.join(PROJECT_ROOT, "results", "stack_preds")
os.makedirs(OUT, exist_ok=True)


def chemprop_predict(smiles):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles}).to_csv(f.name, index=False)
        inp = f.name
    outp = inp.replace(".csv", "_p.csv")
    subprocess.run(
        [CHEMPROP_BIN, "predict", "--test-path", inp, "-s", "canonical_smiles",
         "--model-paths", CP_DIR, "--preds-path", outp,
         "--molecule-featurizers", "v1_rdkit_2d_normalized", "--accelerator", "cpu"],
        capture_output=True, check=True,
    )
    df = pd.read_csv(outp)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    d = dict(zip(df["canonical_smiles"], df[pcol]))
    return np.array([d.get(s, np.nan) for s in smiles], dtype=float)


def rfcb_predict(smiles):
    meta = json.load(open(os.path.join(RFCB_DIR, "ensemble_meta.json")))
    w = np.array(meta["weights"])
    probs = []
    for name in meta["members"]:
        kind, fp = name.split("_", 1)
        cache = ensure_fp_cache(smiles, fp)
        X = cache.loc[smiles, cache.columns.tolist()].to_numpy(dtype=np.uint8)
        sd = os.path.join(RFCB_DIR, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sd, "model.pkl"))
        else:
            m = CatBoostClassifier()
            m.load_model(os.path.join(sd, "model.cbm"))
        probs.append(m.predict_proba(X)[:, 1])
    return np.array(probs).T @ w


def main():
    alldf = pd.read_csv(os.path.join(DATA, "all.csv"))
    sp = json.load(open(os.path.join(DATA, "splits.json")))
    sp = sp[0] if isinstance(sp, list) else sp
    for split in ("val", "test"):
        sub = alldf.iloc[sp[split]].reset_index(drop=True)
        smi = sub["canonical_smiles"].tolist()
        print(f"[{split}] n={len(smi)} — chemprop 예측…", flush=True)
        cp = chemprop_predict(smi)
        print(f"[{split}] RFCB 예측…", flush=True)
        rf = rfcb_predict(smi)
        for tag, prob in (("chemprop_v31", cp), ("rfcb_v31", rf)):
            out = pd.DataFrame({"canonical_smiles": smi, "label": sub["label"].astype(int), "prob": prob})
            path = os.path.join(OUT, f"{tag}_{split}_pred.csv")
            out.to_csv(path, index=False)
            print(f"  저장: {path}  (NaN {int(out.prob.isna().sum())})", flush=True)
    print("DONE")


if __name__ == "__main__":
    main()
