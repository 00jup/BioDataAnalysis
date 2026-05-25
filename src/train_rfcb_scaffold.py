"""RF/CB scaffold split 학습 — Chemprop 와 동일 분리.

목적: Chemprop scaffold 모델과 stacking 위해 같은 split 으로 RF/CB 학습.
splits.json (chemprop 자동 저장) 의 train/val/test 인덱스를 그대로 사용.

학습 후 chemprop 와 stacking → MCC 천장 도전.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import pandas as pd
import joblib
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              confusion_matrix, f1_score)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import (ensure_fp_cache, FPS, full_metrics,
                                      optimize_ensemble)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold")
SCAFFOLD_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "rfcb_scaffold")
RESULTS = os.path.join(PROJECT_ROOT, "results")
RANDOM_STATE = 42


def load_scaffold_splits(domain: str):
    """chemprop 의 splits.json (인덱스) + all.csv (분자) → train/val/test df."""
    all_csv = os.path.join(DATA_DIR, domain, "all.csv")
    splits_json = os.path.join(SCAFFOLD_DIR, domain, "splits.json")
    df = pd.read_csv(all_csv).reset_index(drop=True)
    splits = json.load(open(splits_json))
    s = splits[0] if isinstance(splits, list) else splits
    tr = df.iloc[s["train"]].reset_index(drop=True)
    va = df.iloc[s["val"]].reset_index(drop=True)
    te = df.iloc[s["test"]].reset_index(drop=True)
    print(f"  [{domain}] train {len(tr)} val {len(va)} test {len(te)}")
    print(f"  train 양성 {(tr.label==1).sum()} / 음성 {(tr.label==0).sum()}")
    return tr, va, te


def fp_xy_simple(fp_name, df):
    """sample_weight 없이 단순 FP → X, y."""
    cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
    cols = [c for c in cache.columns]
    mask = df["canonical_smiles"].isin(cache.index)
    sub = df[mask].reset_index(drop=True)
    X = cache.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
    y = sub["label"].to_numpy(int)
    return X, y, sub


def train_sub(fp_name, kind, tr, va, te):
    Xtr, ytr, _ = fp_xy_simple(fp_name, tr)
    Xv,  yv, _ = fp_xy_simple(fp_name, va)
    Xte, yte, _ = fp_xy_simple(fp_name, te)
    if kind == "rf":
        m = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                    min_samples_leaf=2, class_weight="balanced",
                                    random_state=RANDOM_STATE, n_jobs=-1)
        m.fit(Xtr, ytr)
    else:
        m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                               loss_function="Logloss", verbose=0,
                               class_weights={0: 1, 1: 4},
                               random_state=RANDOM_STATE)
        m.fit(Xtr, ytr)
    return m, m.predict_proba(Xv)[:, 1], m.predict_proba(Xte)[:, 1], yv, yte


def train_domain(domain: str):
    print(f"\n{'='*70}\n  RF/CB scaffold — {domain}\n{'='*70}")
    tr, va, te = load_scaffold_splits(domain)
    out_dir = os.path.join(MODELS_DIR, domain)
    os.makedirs(out_dir, exist_ok=True)

    val_probs, test_probs = [], []
    yv_ref, yte_ref = None, None
    names = []
    for fp_name in FPS:
        for kind in ("rf", "cb"):
            name = f"{kind}_{fp_name}"
            names.append(name)
            t0 = time.time()
            m, pv, pte, yv, yte = train_sub(fp_name, kind, tr, va, te)
            val_probs.append(pv); test_probs.append(pte)
            if yv_ref is None: yv_ref, yte_ref = yv, yte
            sd = os.path.join(out_dir, name); os.makedirs(sd, exist_ok=True)
            if kind == "rf":
                joblib.dump(m, os.path.join(sd, "model.pkl"))
            else:
                m.save_model(os.path.join(sd, "model.cbm"))
            print(f"  {name:14s} {time.time()-t0:5.1f}s  val AUC {roc_auc_score(yv,pv):.3f}  test AUC {roc_auc_score(yte,pte):.3f}")

    # ensemble
    Xv = np.array(val_probs).T
    Xte = np.array(test_probs).T
    w, thr, val_mcc = optimize_ensemble(Xv, yv_ref)
    val_score = Xv @ w
    test_score = Xte @ w
    val_m = full_metrics(yv_ref, val_score, thr)
    test_m = full_metrics(yte_ref, test_score, thr)
    print(f"  val MCC max threshold {thr:.3f} → val MCC {val_m['mcc']:.3f}")
    print(f"  [{domain} RF/CB scaffold test] N={test_m['n']} (양성 {test_m['pos']})")
    print(f"  AUC {test_m['auc']:.3f}  MCC {test_m['mcc']:.3f}  TPR {test_m['tpr']:.3f}  TNR {test_m['tnr']:.3f}")

    meta = {"domain": domain, "members": names, "weights": w.tolist(),
            "threshold": thr, "val_metrics": val_m, "test_metrics": test_m}
    with open(os.path.join(out_dir, "ensemble_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return test_m


def main():
    os.makedirs(MODELS_DIR, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    out = {}
    for d in ("vivo", "vitro"):
        out[d] = train_domain(d)
    with open(os.path.join(RESULTS, "rfcb_scaffold.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/rfcb_scaffold.json")


if __name__ == "__main__":
    main()
