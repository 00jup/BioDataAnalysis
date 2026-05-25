"""Stacking — Chemprop scaffold + RF/CB scaffold ensemble.

val 에서 α 최적화 → test 평가.
진짜 OOD MCC 추가 향상 도전.
"""
from __future__ import annotations
import json, os, sys, subprocess, tempfile
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              confusion_matrix, f1_score)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import ensure_fp_cache, FPS
from src.train_rfcb_scaffold import load_scaffold_splits

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold")
CP_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold")
RFCB_DIR = os.path.join(PROJECT_ROOT, "models", "rfcb_scaffold")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def chemprop_predict_smiles(domain, smiles_list):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles_list}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace(".csv", "_pred.csv")
    subprocess.run([
        CHEMPROP_BIN, "predict",
        "--test-path", in_p, "-s", "canonical_smiles",
        "--model-paths", os.path.join(CP_DIR, domain),
        "--preds-path", out_p,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ], capture_output=True, check=True)
    df = pd.read_csv(out_p)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    return df[pcol].to_numpy(float)


def rfcb_predict_ensemble(domain, df):
    meta = json.load(open(os.path.join(RFCB_DIR, domain, "ensemble_meta.json")))
    weights = np.array(meta["weights"])
    members = meta["members"]
    probs = []
    for name in members:
        kind, fp_name = name.split("_", 1)
        cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
        cols = [c for c in cache.columns]
        sub = df[df["canonical_smiles"].isin(cache.index)].reset_index(drop=True)
        X = cache.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
        sd = os.path.join(RFCB_DIR, domain, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sd, "model.pkl"))
        else:
            m = CatBoostClassifier(); m.load_model(os.path.join(sd, "model.cbm"))
        probs.append(m.predict_proba(X)[:, 1])
    probs = np.array(probs).T
    return probs @ weights


def evaluate(y, p, thr=None, name=""):
    if thr is None:
        best_t, best_m = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            mc = matthews_corrcoef(y, (p >= t).astype(int))
            if mc > best_m: best_m, best_t = mc, t
        thr = best_t
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    auc = roc_auc_score(y, p)
    mcc = matthews_corrcoef(y, pred)
    print(f"  {name:18s} AUC {auc:.3f}  MCC {mcc:.3f}  thr {thr:.3f}  "
          f"TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    return {"auc": float(auc), "mcc": float(mcc), "threshold": float(thr),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1))}


def run(domain):
    print(f"\n{'='*70}\n  Stacking scaffold — {domain}\n{'='*70}")
    tr, va, te = load_scaffold_splits(domain)
    # chemprop predict — val + test
    p_cp_v = chemprop_predict_smiles(domain, va["canonical_smiles"].tolist())
    p_cp_t = chemprop_predict_smiles(domain, te["canonical_smiles"].tolist())
    # rfcb predict
    p_rf_v = rfcb_predict_ensemble(domain, va)
    p_rf_t = rfcb_predict_ensemble(domain, te)
    y_v = va.label.to_numpy(int); y_t = te.label.to_numpy(int)

    print(f"\n  [{domain} test 결과 비교]")
    out = {}
    out["chemprop_only"] = evaluate(y_t, p_cp_t, name="Chemprop only")
    out["rfcb_only"]     = evaluate(y_t, p_rf_t, name="RF/CB only")

    # stacking - val 에서 α 와 threshold 최적화
    best_alpha, best_thr, best_mcc = 0.5, 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        p = alpha * p_cp_v + (1 - alpha) * p_rf_v
        for t in np.linspace(0.05, 0.95, 91):
            mc = matthews_corrcoef(y_v, (p >= t).astype(int))
            if mc > best_mcc:
                best_mcc, best_alpha, best_thr = mc, alpha, t
    print(f"\n  val 최적: α={best_alpha:.2f} (chemprop), thr={best_thr:.3f}, val MCC={best_mcc:.3f}")

    p_ens_t = best_alpha * p_cp_t + (1 - best_alpha) * p_rf_t
    out["stacking_val_thr"]  = evaluate(y_t, p_ens_t, thr=best_thr, name="ENS (val thr)")
    out["stacking_peek_thr"] = evaluate(y_t, p_ens_t, name="ENS (PEEK)")
    out["alpha"] = float(best_alpha)
    return out


def main():
    out = {}
    for d in ("vivo", "vitro"):
        out[d] = run(d)
    with open(os.path.join(RESULTS, "stacking_scaffold.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/stacking_scaffold.json")


if __name__ == "__main__":
    main()
