"""v23 honest 평가 — Chemprop v23 + RF/CB v3 stacking + 진짜 외부 sanity."""
from __future__ import annotations
import json, os, sys, subprocess, tempfile, joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              confusion_matrix, f1_score)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import ensure_fp_cache
from src.train_v23_honest import SANITY_DRUGS

CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
CHEMPROP_V23 = os.path.join(PROJECT_ROOT, "models", "chemprop_v23")
RFCB_V3 = os.path.join(PROJECT_ROOT, "models", "rfcb_v3")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_v23_honest")
RESULTS = os.path.join(PROJECT_ROOT, "results")


def chemprop_predict(smiles_list):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles_list}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace(".csv", "_pred.csv")
    subprocess.run([
        CHEMPROP_BIN, "predict",
        "--test-path", in_p, "-s", "canonical_smiles",
        "--model-paths", CHEMPROP_V23,
        "--preds-path", out_p,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ], capture_output=True, check=True)
    df = pd.read_csv(out_p)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    by_smi = dict(zip(df["canonical_smiles"], df[pcol]))
    return np.array([by_smi.get(s, np.nan) for s in smiles_list], dtype=float)


def rfcb_predict(smiles_list):
    meta = json.load(open(os.path.join(RFCB_V3, "ensemble_meta.json")))
    weights = np.array(meta["weights"]); members = meta["members"]
    df = pd.DataFrame({"canonical_smiles": smiles_list})
    probs = []
    for name in members:
        kind, fp_name = name.split("_", 1)
        cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
        sub = df[df["canonical_smiles"].isin(cache.index)].reset_index(drop=True)
        X = cache.loc[sub["canonical_smiles"], cache.columns.tolist()].to_numpy(dtype=np.uint8)
        sd = os.path.join(RFCB_V3, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sd, "model.pkl"))
        else:
            m = CatBoostClassifier(); m.load_model(os.path.join(sd, "model.cbm"))
        probs.append(m.predict_proba(X)[:, 1])
    return np.array(probs).T @ weights


def main():
    # 데이터 — chemprop v23 의 splits + sanity
    val_csv = os.path.join(CHEMPROP_V23, "val_smiles.csv")
    test_csv = os.path.join(CHEMPROP_V23, "test_smiles.csv")
    all_csv = os.path.join(DATA_DIR, "vivo", "all.csv")

    val = pd.read_csv(val_csv).merge(pd.read_csv(all_csv), on="canonical_smiles", how="left")
    test = pd.read_csv(test_csv).merge(pd.read_csv(all_csv), on="canonical_smiles", how="left")

    val_x = val["canonical_smiles"].tolist(); val_y = val["label"].to_numpy(int)
    test_x = test["canonical_smiles"].tolist(); test_y = test["label"].to_numpy(int)
    san_x = [d[1] for d in SANITY_DRUGS]; san_y = np.array([d[2] for d in SANITY_DRUGS])

    print(f"val {len(val_x)}, test {len(test_x)}, sanity {len(san_x)} (진짜 외부)")

    # 예측 6번
    print("\n[Chemprop val]"); val_p_cp = chemprop_predict(val_x)
    print("[Chemprop test]"); test_p_cp = chemprop_predict(test_x)
    print("[Chemprop sanity]"); san_p_cp = chemprop_predict(san_x)
    print("[RF/CB val]"); val_p_rf = rfcb_predict(val_x)
    print("[RF/CB test]"); test_p_rf = rfcb_predict(test_x)
    print("[RF/CB sanity]"); san_p_rf = rfcb_predict(san_x)

    # 단일 모델 (val thr → test 적용)
    print(f"\n=== 단일 모델 baseline (honest) ===")
    print(f"  {'model':<14s} {'val_thr':>7s} {'test_MCC':>9s} {'test_AUC':>9s} {'san_MCC':>8s} {'san_AUC':>8s}")
    results = {}
    for nm, pv, pt, ps in [("Chemprop v23", val_p_cp, test_p_cp, san_p_cp),
                            ("RF/CB v3", val_p_rf, test_p_rf, san_p_rf)]:
        bt, bm = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            m = matthews_corrcoef(val_y, (pv >= t).astype(int))
            if m > bm: bm, bt = m, t
        test_mcc = matthews_corrcoef(test_y, (pt >= bt).astype(int))
        test_auc = roc_auc_score(test_y, pt)
        # sanity: 같은 val thr
        san_mcc = matthews_corrcoef(san_y, (ps >= bt).astype(int))
        san_auc = roc_auc_score(san_y, ps)
        # sanity TPR/TNR
        spred = (ps >= bt).astype(int)
        stp = ((spred==1)&(san_y==1)).sum(); sfn = ((spred==0)&(san_y==1)).sum()
        stn = ((spred==0)&(san_y==0)).sum(); sfp = ((spred==1)&(san_y==0)).sum()
        s_tpr = stp/max(stp+sfn,1); s_tnr = stn/max(sfp+stn,1)
        print(f"  {nm:<14s} {bt:>7.3f} {test_mcc:>9.3f} {test_auc:>9.3f} {san_mcc:>8.3f} {san_auc:>8.3f}  (sanity TPR {s_tpr:.2f} TNR {s_tnr:.2f})")
        results[nm] = {"val_thr": bt, "test_mcc": test_mcc, "test_auc": test_auc,
                        "san_mcc": san_mcc, "san_auc": san_auc,
                        "san_tpr": s_tpr, "san_tnr": s_tnr}

    # Stacking — val 에서 α + thr 결정
    print(f"\n=== Honest Stacking (val α/thr → test + sanity) ===")
    best_a, best_t, best_mcc = 0.5, 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        p_val = alpha * val_p_cp + (1 - alpha) * val_p_rf
        for thr in np.linspace(0.05, 0.95, 91):
            m = matthews_corrcoef(val_y, (p_val >= thr).astype(int))
            if m > best_mcc:
                best_mcc, best_a, best_t = m, alpha, thr
    print(f"  Best on val: α={best_a:.2f}, thr={best_t:.3f}, val MCC={best_mcc:.3f}")
    p_test_s = best_a * test_p_cp + (1 - best_a) * test_p_rf
    pred = (p_test_s >= best_t).astype(int)
    cm = confusion_matrix(test_y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    test_mcc_s = matthews_corrcoef(test_y, pred)
    test_auc_s = roc_auc_score(test_y, p_test_s)
    test_tpr_s = tp/max(tp+fn,1); test_tnr_s = tn/max(fp+tn,1)
    print(f"  test scaffold (honest): MCC {test_mcc_s:.3f}, AUC {test_auc_s:.3f}, TPR {test_tpr_s:.3f}, TNR {test_tnr_s:.3f}")

    # sanity 같은 α/thr
    p_san_s = best_a * san_p_cp + (1 - best_a) * san_p_rf
    pred_s = (p_san_s >= best_t).astype(int)
    cm_s = confusion_matrix(san_y, pred_s, labels=[1, 0])
    tps, fns = cm_s[0]; fps, tns = cm_s[1]
    san_mcc_s = matthews_corrcoef(san_y, pred_s)
    san_auc_s = roc_auc_score(san_y, p_san_s)
    san_tpr_s = tps/max(tps+fns,1); san_tnr_s = tns/max(fps+tns,1)
    print(f"  sanity (진짜 외부, val α/thr): MCC {san_mcc_s:.3f}, AUC {san_auc_s:.3f}, TPR {san_tpr_s:.3f}, TNR {san_tnr_s:.3f}")

    print(f"\n  prob 분포 (sanity 20 약물):")
    for d, p_cp, p_rf, p_st in zip(SANITY_DRUGS, san_p_cp, san_p_rf, p_san_s):
        n, smi, t = d
        print(f"  {n:<22s} {t:>5d}  cp {p_cp:>5.2f}  rf {p_rf:>5.2f}  stack {p_st:>5.2f}")

    out = {"single": results, "stacking_honest": {
        "alpha": float(best_a), "threshold": float(best_t),
        "val_mcc": float(best_mcc),
        "test_mcc": float(test_mcc_s), "test_auc": float(test_auc_s),
        "test_tpr": float(test_tpr_s), "test_tnr": float(test_tnr_s),
        "sanity_mcc": float(san_mcc_s), "sanity_auc": float(san_auc_s),
        "sanity_tpr": float(san_tpr_s), "sanity_tnr": float(san_tnr_s),
    }}
    with open(os.path.join(RESULTS, "v23_honest_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/v23_honest_eval.json")


if __name__ == "__main__":
    main()
