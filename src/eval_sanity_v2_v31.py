"""Sanity v2 (263 외부) 재평가 — Chemprop v31 + RF/CB v31.

v27 (기존) 대비 향상 측정.
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
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import ensure_fp_cache  # noqa: E402

CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
CHEMPROP_V31 = os.path.join(PROJECT_ROOT, "models", "chemprop_v31_class_expanded", "vivo")
RFCB_V31 = os.path.join(PROJECT_ROOT, "models", "rfcb_v31", "vivo")
SANITY = os.path.join(PROJECT_ROOT, "data", "sanity_v2", "external_sanity_200.csv")
RESULTS = os.path.join(PROJECT_ROOT, "results")


def chemprop_predict(smiles_list, model_dir):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles_list}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace(".csv", "_pred.csv")
    subprocess.run(
        [
            CHEMPROP_BIN,
            "predict",
            "--test-path",
            in_p,
            "-s",
            "canonical_smiles",
            "--model-paths",
            model_dir,
            "--preds-path",
            out_p,
            "--molecule-featurizers",
            "v1_rdkit_2d_normalized",
            "--accelerator",
            "cpu",
        ],
        capture_output=True,
        check=True,
    )
    df = pd.read_csv(out_p)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    by_smi = dict(zip(df["canonical_smiles"], df[pcol]))
    return np.array([by_smi.get(s, np.nan) for s in smiles_list], dtype=float)


def rfcb_predict(smiles_list, model_dir):
    meta = json.load(open(os.path.join(model_dir, "ensemble_meta.json")))
    w = np.array(meta["weights"])
    names = meta["members"]
    df = pd.DataFrame({"canonical_smiles": smiles_list})
    probs = []
    for name in names:
        kind, fp_name = name.split("_", 1)
        cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
        sub = df[df["canonical_smiles"].isin(cache.index)].reset_index(drop=True)
        X = cache.loc[sub["canonical_smiles"], cache.columns.tolist()].to_numpy(dtype=np.uint8)
        sd = os.path.join(model_dir, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sd, "model.pkl"))
        else:
            m = CatBoostClassifier()
            m.load_model(os.path.join(sd, "model.cbm"))
        probs.append(m.predict_proba(X)[:, 1])
    return np.array(probs).T @ w


def report(name, probs, labels):
    auc = roc_auc_score(labels, probs)
    print(f"\n=== {name} ===")
    print(f"AUC: {auc:.3f}")
    print(f"{'thr':>5s} {'TPR':>5s} {'TNR':>5s} {'MCC':>6s} {'Acc':>5s}")
    best_mcc = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.10, 0.90, 17):
        pred = (probs >= thr).astype(int)
        cm = confusion_matrix(labels, pred, labels=[1, 0])
        tp, fn = cm[0]
        fp, tn = cm[1]
        mcc = matthews_corrcoef(labels, pred)
        acc = (pred == labels).mean()
        if mcc > best_mcc:
            best_mcc, best_thr = mcc, thr
        print(
            f"{thr:>5.2f} {tp / max(tp + fn, 1):>5.3f} {tn / max(fp + tn, 1):>5.3f} {mcc:>+6.3f} {acc:>5.3f}"
        )
    print(f"  Best MCC: {best_mcc:+.3f} @ thr {best_thr:.2f}")
    return {"auc": auc, "best_mcc": best_mcc, "best_thr": best_thr}


def main():
    df = pd.read_csv(SANITY)
    labels = df.label.values
    smiles = df.canonical_smiles.tolist()
    print(f"Sanity v2: {len(df)} (양성 {(labels == 1).sum()} / 음성 {(labels == 0).sum()})\n")

    out = {}
    if os.path.exists(CHEMPROP_V31):
        print("[Chemprop v31 predict]")
        p_cp = chemprop_predict(smiles, CHEMPROP_V31)
        out["chemprop_v31"] = report("Chemprop v31 (class expanded)", p_cp, labels)
    else:
        print(f"⏳ {CHEMPROP_V31} 없음 — 학습 먼저.")

    if os.path.exists(RFCB_V31):
        print("\n[RF/CB v31 predict]")
        p_rf = rfcb_predict(smiles, RFCB_V31)
        out["rfcb_v31"] = report("RF/CB v31 (class expanded)", p_rf, labels)
    else:
        print(f"⏳ {RFCB_V31} 없음 — 학습 먼저.")

    # Ensemble (simple avg) if 둘 다 있음
    if "chemprop_v31" in out and "rfcb_v31" in out:
        p_avg = (p_cp + p_rf) / 2
        out["ensemble_avg"] = report("Ensemble (avg)", p_avg, labels)

    # Save
    with open(os.path.join(RESULTS, "sanity_v2_v31.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n저장: results/sanity_v2_v31.json")


if __name__ == "__main__":
    main()
