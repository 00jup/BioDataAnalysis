"""v9 ensemble — Chemprop + v1 RF/CB stacking.

vivo:  Chemprop (강) + RF/CB (보조) → 더 강한 vivo 모델
vitro: v1 RF/CB (강) + Chemprop (보조) → 균형 향상

방법:
  1. val 에서 두 모델 출력 → weighted average 최적화 (MCC max)
  2. val 에서 best threshold 선택 (MCC max + bAcc max 비교)
  3. test 에서 최종 평가
"""

from __future__ import annotations
import json, os, sys, subprocess
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_v9")
CHEMPROP_MODELS = os.path.join(PROJECT_ROOT, "models", "chemprop_v9")

PY = sys.executable
CHEMPROP_BIN = os.path.join(os.path.dirname(PY), "chemprop")


def chemprop_predict(domain: str, smiles_csv: str, out_path: str):
    """주어진 SMILES csv → 예측 확률 csv."""
    cmd = [
        CHEMPROP_BIN, "predict",
        "--test-path", smiles_csv,
        "-s", "canonical_smiles",
        "--model-paths", os.path.join(CHEMPROP_MODELS, domain),
        "--preds-path", out_path,
        "--accelerator", "cpu",
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    df = pd.read_csv(out_path).rename(columns={"label": "p_chemprop"})
    return df


def rfcb_predict(domain: str, df: pd.DataFrame):
    """v1 ensemble probability — load and apply members."""
    import joblib
    from catboost import CatBoostClassifier
    from src.train_domain_models import fp_xy

    db = pd.read_parquet(os.path.join(DATA_DIR, "labels_db", "full.parquet"))
    meta = json.load(open(os.path.join(MODELS_DIR, domain, "ensemble_meta.json")))
    weights = np.array(meta["weights"]); members = meta["members"]
    probs = []
    y_ref = None
    smiles_ref = None
    for name in members:
        kind, fp_name = name.split("_", 1)
        X, y, sub, _ = fp_xy(fp_name, df, domain, db=db)
        sd = os.path.join(MODELS_DIR, domain, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sd, "model.pkl"))
        else:
            m = CatBoostClassifier(); m.load_model(os.path.join(sd, "model.cbm"))
        p = m.predict_proba(X)[:, 1]
        probs.append(p)
        if y_ref is None: y_ref, smiles_ref = y, sub["canonical_smiles"].to_list()
    score = np.array(probs).T @ weights
    return pd.DataFrame({"canonical_smiles": smiles_ref, "label": y_ref, "p_rfcb": score})


def evaluate(y, p, name=""):
    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score, balanced_accuracy_score)
    auc = roc_auc_score(y, p)
    # MCC max
    best_t, best_m = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > best_m: best_m, best_t = m, t
    pred = (p >= best_t).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    print(f"  {name:14s} AUC {auc:.3f}  MCC {best_m:.3f}  thr {best_t:.3f}  "
          f"TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    return {"auc": float(auc), "mcc": float(best_m), "threshold": float(best_t),
            "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1))}


def optimize_stacking(y_val, p_cp_val, p_rf_val):
    """val 에서 alpha*chemprop + (1-alpha)*rfcb 의 alpha 최적화 (MCC max)."""
    from sklearn.metrics import matthews_corrcoef
    best_alpha, best_thr, best_mcc = 0.5, 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        p = alpha * p_cp_val + (1 - alpha) * p_rf_val
        for t in np.linspace(0.05, 0.95, 91):
            mcc = matthews_corrcoef(y_val, (p >= t).astype(int))
            if mcc > best_mcc:
                best_mcc, best_alpha, best_thr = mcc, alpha, t
    return best_alpha, best_thr, best_mcc


def run_domain(domain: str):
    print(f"\n{'='*70}\n  {domain} — Chemprop + v1 RF/CB stacking\n{'='*70}")
    val_df = pd.read_csv(os.path.join(DATA_DIR, "val", f"{domain}.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test", f"{domain}.csv"))

    # Chemprop 예측 — val 새로, test 는 캐시 있으면 재활용
    print("  [1/4] Chemprop val 예측")
    val_smiles_path = os.path.join(CHEMPROP_DATA, domain, "val.csv")
    val_pred_path = os.path.join(CHEMPROP_MODELS, domain, "val_pred.csv")
    val_cp = chemprop_predict(domain, val_smiles_path, val_pred_path)
    print("  [2/4] Chemprop test 예측 (캐시 또는 재계산)")
    te_smiles_path = os.path.join(CHEMPROP_DATA, domain, "test.csv")
    te_pred_path = os.path.join(CHEMPROP_MODELS, domain, "test_pred.csv")
    if os.path.exists(te_pred_path):
        te_cp = pd.read_csv(te_pred_path).rename(columns={"label": "p_chemprop"})
    else:
        te_cp = chemprop_predict(domain, te_smiles_path, te_pred_path)

    print("  [3/4] RF/CB val + test 예측")
    val_rfcb = rfcb_predict(domain, val_df)
    test_rfcb = rfcb_predict(domain, test_df)

    # merge
    val_merged = val_rfcb.merge(val_cp, on="canonical_smiles", how="inner")
    te_merged = test_rfcb.merge(te_cp, on="canonical_smiles", how="inner")
    print(f"  val merged: {len(val_merged)}, test merged: {len(te_merged)}")

    y_val = val_merged["label"].to_numpy(int)
    p_cp_v = val_merged["p_chemprop"].to_numpy(float)
    p_rf_v = val_merged["p_rfcb"].to_numpy(float)
    y_te = te_merged["label"].to_numpy(int)
    p_cp_t = te_merged["p_chemprop"].to_numpy(float)
    p_rf_t = te_merged["p_rfcb"].to_numpy(float)

    # 단일 모델 비교 (test)
    print(f"\n  [4/4] test 결과 (각 모델 + ensemble)")
    out = {}
    out["chemprop_only"] = evaluate(y_te, p_cp_t, "Chemprop only")
    out["rfcb_only"]     = evaluate(y_te, p_rf_t, "RF/CB only")

    # Stacking — val 에서 alpha 최적화
    alpha, thr, val_mcc = optimize_stacking(y_val, p_cp_v, p_rf_v)
    print(f"\n  [stacking] val 최적: alpha={alpha:.2f} (chemprop), thr={thr:.3f}, val MCC={val_mcc:.3f}")
    p_ens_t = alpha * p_cp_t + (1 - alpha) * p_rf_t
    # 학습된 threshold 그대로 test 적용 (peek 아님)
    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score)
    pred = (p_ens_t >= thr).astype(int)
    cm = confusion_matrix(y_te, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    auc = roc_auc_score(y_te, p_ens_t)
    mcc = matthews_corrcoef(y_te, pred)
    tpr = tp/max(tp+fn,1); tnr = tn/max(fp+tn,1)
    print(f"  {'ENS (val thr)':14s} AUC {auc:.3f}  MCC {mcc:.3f}  thr {thr:.3f}  "
          f"TPR {tpr:.3f}  TNR {tnr:.3f}")
    out["stacking"] = {"alpha": float(alpha), "threshold": float(thr),
                       "auc": float(auc), "mcc": float(mcc),
                       "tpr": float(tpr), "tnr": float(tnr)}

    # peek (test MCC max) — 천장 확인
    out["stacking_peek"] = evaluate(y_te, p_ens_t, "ENS (PEEK)")
    return out


def main():
    out = {}
    for d in ("vivo", "vitro"):
        out[d] = run_domain(d)
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "ensemble_v9.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/ensemble_v9.json")


if __name__ == "__main__":
    main()
