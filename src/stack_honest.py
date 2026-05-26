"""Honest Stacking — val 에서 α/threshold 결정 → test 평가 (no peek).

이전 stack_final.py 의 PEEK 문제 수정:
  - val 에서 α + threshold 둘 다 최적화
  - 그 결정 으로 test 평가
  - 진짜 일반화 MCC 측정
"""
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

CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
CHEMPROP_V17 = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2",
                              "v17_ens15_h600", "vivo")
RFCB_V2 = os.path.join(PROJECT_ROOT, "models", "rfcb_scaffold_v2", "vivo")
RESULTS = os.path.join(PROJECT_ROOT, "results")


def chemprop_predict(smiles_list):
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles_list}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace(".csv", "_pred.csv")
    subprocess.run([
        CHEMPROP_BIN, "predict",
        "--test-path", in_p, "-s", "canonical_smiles",
        "--model-paths", CHEMPROP_V17,
        "--preds-path", out_p,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ], capture_output=True, check=True)
    df = pd.read_csv(out_p)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    by_smi = dict(zip(df["canonical_smiles"], df[pcol]))
    return np.array([by_smi.get(s, np.nan) for s in smiles_list], dtype=float)


def rfcb_predict(smiles_list):
    meta = json.load(open(os.path.join(RFCB_V2, "ensemble_meta.json")))
    weights = np.array(meta["weights"]); members = meta["members"]
    df = pd.DataFrame({"canonical_smiles": smiles_list})
    probs = []
    for name in members:
        kind, fp_name = name.split("_", 1)
        cache = ensure_fp_cache(df["canonical_smiles"].tolist(), fp_name)
        sub = df[df["canonical_smiles"].isin(cache.index)].reset_index(drop=True)
        X = cache.loc[sub["canonical_smiles"], cache.columns.tolist()].to_numpy(dtype=np.uint8)
        sd = os.path.join(RFCB_V2, name)
        if kind == "rf":
            m = joblib.load(os.path.join(sd, "model.pkl"))
        else:
            m = CatBoostClassifier(); m.load_model(os.path.join(sd, "model.cbm"))
        probs.append(m.predict_proba(X)[:, 1])
    return np.array(probs).T @ weights


def main():
    print("=== Honest Stacking — val 에서 결정 → test 평가 ===\n")

    # val + test 가져오기
    base = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2",
                         "v12_baseline", "vivo")
    all_csv = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v2",
                                         "vivo", "all.csv"))

    val_smi = pd.read_csv(f"{base}/val_smiles.csv").merge(all_csv, on="canonical_smiles", how="left")
    test_smi = pd.read_csv(f"{base}/test_smiles.csv").merge(all_csv, on="canonical_smiles", how="left")

    val_x = val_smi["canonical_smiles"].tolist()
    val_y = val_smi["label"].to_numpy(int)
    test_x = test_smi["canonical_smiles"].tolist()
    test_y = test_smi["label"].to_numpy(int)
    print(f"val: {len(val_x)}, test: {len(test_x)}")

    # 양 모델 predict
    print("\n[Chemprop val 예측]")
    val_p_cp = chemprop_predict(val_x)
    print("[Chemprop test 예측]")
    test_p_cp = chemprop_predict(test_x)
    print("[RF/CB val 예측]")
    val_p_rf = rfcb_predict(val_x)
    print("[RF/CB test 예측]")
    test_p_rf = rfcb_predict(test_x)

    # === val 에서 α + threshold 최적화 ===
    print("\n=== α/threshold 결정 (val MCC max) ===")
    best_a, best_t, best_mcc = 0.5, 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        p_val = alpha * val_p_cp + (1 - alpha) * val_p_rf
        for thr in np.linspace(0.05, 0.95, 91):
            mcc = matthews_corrcoef(val_y, (p_val >= thr).astype(int))
            if mcc > best_mcc:
                best_mcc, best_a, best_t = mcc, alpha, thr
    print(f"  Best on val: α={best_a:.2f}, thr={best_t:.3f}, val MCC={best_mcc:.3f}")

    # === test 평가 (val 결정 적용) ===
    print(f"\n=== test 평가 (val α/thr 적용) ===")
    p_test = best_a * test_p_cp + (1 - best_a) * test_p_rf
    pred = (p_test >= best_t).astype(int)
    cm = confusion_matrix(test_y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    auc = roc_auc_score(test_y, p_test)
    mcc = matthews_corrcoef(test_y, pred)
    tpr = tp/max(tp+fn,1); tnr = tn/max(fp+tn,1)
    print(f"  test scaffold (val α/thr): MCC {mcc:.3f}, AUC {auc:.3f}")
    print(f"  TPR {tpr:.3f}  TNR {tnr:.3f}")

    # 단일 모델 baseline (test, 같은 thr/α 안 적용 — 각자 best thr 로 비교)
    print(f"\n=== 단일 모델 baseline (각자 val thr) ===")
    for nm, pv, pt in [("Chemprop", val_p_cp, test_p_cp),
                        ("RF/CB v2", val_p_rf, test_p_rf)]:
        # val 에서 thr 결정
        bt, bm = 0.5, -1.0
        for t in np.linspace(0.05, 0.95, 91):
            m = matthews_corrcoef(val_y, (pv >= t).astype(int))
            if m > bm: bm, bt = m, t
        # test 에 적용
        pred = (pt >= bt).astype(int)
        mcc_t = matthews_corrcoef(test_y, pred)
        auc_t = roc_auc_score(test_y, pt)
        print(f"  {nm:<10s} val_thr={bt:.3f}, test MCC {mcc_t:.3f}, AUC {auc_t:.3f}")

    # PEEK 비교 (test 에서 직접)
    print(f"\n=== PEEK 비교 (test 에서 직접 — cheating 참고) ===")
    p_test_peek = best_a * test_p_cp + (1 - best_a) * test_p_rf
    bt_p, bm_p = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        m = matthews_corrcoef(test_y, (p_test_peek >= t).astype(int))
        if m > bm_p: bm_p, bt_p = m, t
    # 최고 α 도 peek
    best_a_p, best_t_p, bm_pp = 0.5, 0.5, -1.0
    for alpha in np.linspace(0, 1, 21):
        p_test_pp = alpha * test_p_cp + (1 - alpha) * test_p_rf
        for t in np.linspace(0.05, 0.95, 91):
            m = matthews_corrcoef(test_y, (p_test_pp >= t).astype(int))
            if m > bm_pp:
                bm_pp, best_a_p, best_t_p = m, alpha, t
    print(f"  Stacking val_a/thr   test MCC: {mcc:.3f}  (honest)")
    print(f"  Stacking val_a, peek thr test MCC: {bm_p:.3f}")
    print(f"  Stacking peek α + peek thr test MCC: {bm_pp:.3f}  (cheating)")

    with open(os.path.join(RESULTS, "stack_honest.json"), "w") as f:
        json.dump({
            "honest_best_alpha": float(best_a),
            "honest_best_threshold": float(best_t),
            "honest_val_mcc": float(best_mcc),
            "honest_test_mcc": float(mcc),
            "honest_test_auc": float(auc),
            "honest_test_tpr": float(tpr),
            "honest_test_tnr": float(tnr),
            "peek_test_mcc": float(bm_pp),
            "peek_alpha": float(best_a_p),
            "peek_threshold": float(best_t_p),
        }, f, indent=2)
    print(f"\n저장: results/stack_honest.json")


if __name__ == "__main__":
    main()
