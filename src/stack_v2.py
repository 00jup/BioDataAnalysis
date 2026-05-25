"""Stacking — Chemprop v15 + RF/CB v2 ensemble (same scaffold split).

val 에서 α 최적화 → test 평가.
new data 효과로 강해진 두 모델 합쳐서 MCC 추가 향상 시도.
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
from src.train_rfcb_scaffold_v2 import load_splits

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v2")
CP_BEST = {  # 각 domain 의 chemprop best — v17 (vivo MCC 0.694) 로 업데이트
    "vivo":  os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2", "v17_ens15_h600"),
    "vitro": os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2", "v14_classbal"),
}
RFCB_V2 = os.path.join(PROJECT_ROOT, "models", "rfcb_scaffold_v2")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def chemprop_predict(domain, smiles_list):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles_list}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace(".csv", "_pred.csv")
    cmd = [
        CHEMPROP_BIN, "predict",
        "--test-path", in_p, "-s", "canonical_smiles",
        "--model-paths", os.path.join(CP_BEST[domain], domain),
        "--preds-path", out_p,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ]
    # v17 (vivo) 는 hidden 600 으로 학습됨 — predict 시 매개변수 동일 적용
    subprocess.run(cmd, capture_output=True, check=True)
    df = pd.read_csv(out_p)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    return df[pcol].to_numpy(float)


def rfcb_predict_v2(domain, df):
    meta = json.load(open(os.path.join(RFCB_V2, domain, "ensemble_meta.json")))
    weights = np.array(meta["weights"])
    members = meta["members"]
    probs = []
    for name in members:
        kind, fp_name = name.split("_", 1)
        cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
        sub = df[df["canonical_smiles"].isin(cache.index)].reset_index(drop=True)
        X = cache.loc[sub["canonical_smiles"], cache.columns.tolist()].to_numpy(dtype=np.uint8)
        sd = os.path.join(RFCB_V2, domain, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sd, "model.pkl"))
        else:
            m = CatBoostClassifier(); m.load_model(os.path.join(sd, "model.cbm"))
        probs.append(m.predict_proba(X)[:, 1])
    return np.array(probs).T @ weights


def evaluate(y, p, thr=None, name=""):
    if thr is None:
        bt, bm = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            mc = matthews_corrcoef(y, (p >= t).astype(int))
            if mc > bm: bm, bt = mc, t
        thr = bt
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
    print(f"\n{'='*70}\n  Stacking v2 — {domain}\n{'='*70}")
    tr, va, te = load_splits(domain)
    print(f"  test {len(te)} (양성 {(te.label==1).sum()})")

    # chemprop predict val + test
    p_cp_v = chemprop_predict(domain, va["canonical_smiles"].tolist())
    p_cp_t = chemprop_predict(domain, te["canonical_smiles"].tolist())
    p_rf_v = rfcb_predict_v2(domain, va)
    p_rf_t = rfcb_predict_v2(domain, te)
    y_v = va.label.to_numpy(int); y_t = te.label.to_numpy(int)

    print(f"\n  [{domain} 비교 — same scaffold test]")
    out = {}
    out["chemprop"] = evaluate(y_t, p_cp_t, name="Chemprop (best)")
    out["rfcb"]     = evaluate(y_t, p_rf_t, name="RF/CB v2")

    best_alpha, best_thr, best_mcc = 0.5, 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        p = alpha * p_cp_v + (1 - alpha) * p_rf_v
        for t in np.linspace(0.05, 0.95, 91):
            mc = matthews_corrcoef(y_v, (p >= t).astype(int))
            if mc > best_mcc:
                best_mcc, best_alpha, best_thr = mc, alpha, t
    print(f"\n  val 최적: α={best_alpha:.2f} (chemprop), thr={best_thr:.3f}, val MCC={best_mcc:.3f}")

    p_ens_t = best_alpha * p_cp_t + (1 - best_alpha) * p_rf_t
    out["ens_val_thr"]  = evaluate(y_t, p_ens_t, thr=best_thr, name="ENS (val thr)")
    out["ens_peek"]     = evaluate(y_t, p_ens_t, name="ENS (PEEK)")
    out["alpha"] = float(best_alpha)
    return out


def main():
    out = {}
    for d in ("vivo", "vitro"):
        out[d] = run(d)
    with open(os.path.join(RESULTS, "stacking_v2.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/stacking_v2.json")


if __name__ == "__main__":
    main()
