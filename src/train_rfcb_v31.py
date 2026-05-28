"""RF/CB v31 — class expansion 데이터 사용. v2 와 동일 hyperparameter."""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import (  # noqa: E402
    FPS,
    ensure_fp_cache,
    full_metrics,
    optimize_ensemble,
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v3")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "rfcb_v31")
RESULTS = os.path.join(PROJECT_ROOT, "results")
RANDOM_STATE = 42


def fp_xy(fp_name, df):
    cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
    cols = list(cache.columns)
    sub = df[df["canonical_smiles"].isin(cache.index)].reset_index(drop=True)
    X = cache.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
    y = sub["label"].to_numpy(int)
    return X, y


def load_splits(domain):
    splits = json.load(open(os.path.join(DATA_DIR, domain, "splits.json")))
    s = splits[0] if isinstance(splits, list) else splits
    all_df = pd.read_csv(os.path.join(DATA_DIR, domain, "all.csv")).reset_index(drop=True)
    return all_df.iloc[s["train"]], all_df.iloc[s["val"]], all_df.iloc[s["test"]]


def train(domain):
    print(f"\n=== RF/CB v31 — {domain} (class expanded) ===")
    tr, va, te = load_splits(domain)
    print(f"  train {len(tr)} val {len(va)} test {len(te)}")
    print(f"  train 양성 {(tr.label == 1).sum()} / 음성 {(tr.label == 0).sum()}")

    out_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)
    val_probs, test_probs = [], []
    yv_ref = yte_ref = None
    names = []
    for fp_name in FPS:
        for kind in ("rf", "cb"):
            name = f"{kind}_{fp_name}"
            names.append(name)
            Xtr, ytr = fp_xy(fp_name, tr)
            Xv, yv = fp_xy(fp_name, va)
            Xte, yte = fp_xy(fp_name, te)
            t0 = time.time()
            if kind == "rf":
                m = RandomForestClassifier(
                    n_estimators=500,
                    max_features="sqrt",
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
                m.fit(Xtr, ytr)
            else:
                m = CatBoostClassifier(
                    iterations=500,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    verbose=0,
                    class_weights={0: 1, 1: 3},
                    random_state=RANDOM_STATE,
                )
                m.fit(Xtr, ytr)
            pv = m.predict_proba(Xv)[:, 1]
            pte = m.predict_proba(Xte)[:, 1]
            val_probs.append(pv)
            test_probs.append(pte)
            if yv_ref is None:
                yv_ref, yte_ref = yv, yte
            sd = os.path.join(out_dir, name)
            os.makedirs(sd, exist_ok=True)
            if kind == "rf":
                joblib.dump(m, os.path.join(sd, "model.pkl"))
            else:
                m.save_model(os.path.join(sd, "model.cbm"))
            print(
                f"  {name:14s} {time.time() - t0:5.1f}s  val AUC {roc_auc_score(yv, pv):.3f}  test AUC {roc_auc_score(yte, pte):.3f}"
            )

    Xv = np.array(val_probs).T
    Xte = np.array(test_probs).T
    w, thr, val_mcc = optimize_ensemble(Xv, yv_ref)
    test_m = full_metrics(yte_ref, Xte @ w, thr)
    print(
        f"  [{domain} RF/CB v31 test] AUC {test_m['auc']:.3f}  MCC {test_m['mcc']:.3f}  TPR {test_m['tpr']:.3f}  TNR {test_m['tnr']:.3f}"
    )
    meta = {
        "domain": domain,
        "members": names,
        "weights": w.tolist(),
        "threshold": thr,
        "test_metrics": test_m,
    }
    with open(os.path.join(out_dir, "ensemble_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return test_m


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    out = {"vivo": train("vivo")}
    with open(os.path.join(RESULTS, "rfcb_v31.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\n저장: results/rfcb_v31.json")


if __name__ == "__main__":
    main()
