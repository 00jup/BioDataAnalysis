"""Final Stacking — Chemprop v17 + RF/CB v2 ensemble.

전략:
  1. α (0.0~1.0) 21 단계 sweep
  2. 각 α 에 대해 scaffold OOD + sanity 두 평가
  3. 두 평가 모두에서 좋은 α + threshold 선택

목적: 채점 모델 후보 최종 결정.
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


# === Predict ===
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


# === 평가 데이터 ===
def get_scaffold_test():
    """v17 scaffold test (2,066 분자)."""
    test_csv = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2",
                             "v12_baseline", "vivo", "test_smiles.csv")
    all_csv = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v2", "vivo", "all.csv")
    te = pd.read_csv(test_csv).merge(pd.read_csv(all_csv), on="canonical_smiles", how="left")
    return te["canonical_smiles"].tolist(), te["label"].to_numpy(int)


def get_sanity():
    """의학 라벨 20 약물 + 진짜 외부 3."""
    drugs = [
        ('Acetaminophen', 'CC(=O)Nc1ccc(O)cc1', 1),
        ('Isoniazid', 'NNC(=O)c1ccncc1', 1),
        ('Valproic acid', 'CCCC(CCC)C(=O)O', 1),
        ('Troglitazone', 'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O', 1),
        ('Diclofenac', 'OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl', 1),
        ('Halothane', 'FC(F)(F)C(Cl)Br', 1),
        ('Ketoconazole', 'CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1', 1),
        ('Methotrexate', 'CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1', 1),
        ('Amoxicillin-clav', 'CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O', 1),
        ('Nitrofurantoin', 'O=C1OCC(N1\\N=C\\c1ccc(o1)[N+](=O)[O-])', 1),
        ('Aspirin', 'CC(=O)Oc1ccccc1C(=O)O', 1),
        ('Ibuprofen', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1', 1),
        ('Atorvastatin', 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O', 1),
        ('Metformin', 'CN(C)C(=N)NC(=N)N', 1),
        ('Lisinopril', 'NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O', 1),
        ('Amlodipine', 'CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl', 1),
        ('Omeprazole', 'COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1', 1),
        ('Loratadine', 'CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1', 0),
        ('Cetirizine', 'OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1', 0),
        ('Levothyroxine', 'Oc1cc(I)c(Oc2cc(I)c(CC(N)C(=O)O)cc2I)c(I)c1', 0),
    ]
    smiles = [d[1] for d in drugs]
    labels = np.array([d[2] for d in drugs])
    return smiles, labels


def best_thr_mcc(y, p):
    bt, bm = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > bm: bm, bt = m, t
    return bt, bm


def metrics_at_thr(y, p, thr):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    return {
        "tpr": float(tp/max(tp+fn,1)),
        "tnr": float(tn/max(fp+tn,1)),
        "mcc": float(matthews_corrcoef(y, pred)),
    }


def main():
    # 1. Predict 둘 다
    print("=== 1. Predict 양 모델 (scaffold + sanity) ===\n")
    print("[Scaffold test 2,066 분자]")
    scf_smi, scf_y = get_scaffold_test()
    print(f"  로드: {len(scf_smi)} (양성 {scf_y.sum()})")
    print("[Chemprop v17 예측 중...]")
    scf_p_cp = chemprop_predict(scf_smi)
    print("[RF/CB v2 예측 중...]")
    scf_p_rf = rfcb_predict(scf_smi)

    print("\n[Sanity 20 약물]")
    san_smi, san_y = get_sanity()
    print("[Chemprop v17 예측 중...]")
    san_p_cp = chemprop_predict(san_smi)
    print("[RF/CB v2 예측 중...]")
    san_p_rf = rfcb_predict(san_smi)

    # 단일 모델 baseline
    print(f"\n=== 2. 단일 모델 baseline ===")
    print(f"  {'model':<20s} {'scaffold AUC':>13s} {'scaffold MCC':>13s} {'sanity AUC':>11s} {'sanity MCC':>11s}")
    cp_thr, cp_scf_mcc = best_thr_mcc(scf_y, scf_p_cp)
    cp_san_thr, cp_san_mcc = best_thr_mcc(san_y, san_p_cp)
    rf_thr, rf_scf_mcc = best_thr_mcc(scf_y, scf_p_rf)
    rf_san_thr, rf_san_mcc = best_thr_mcc(san_y, san_p_rf)
    print(f"  {'Chemprop v17':<20s} {roc_auc_score(scf_y, scf_p_cp):>13.3f} {cp_scf_mcc:>13.3f} "
          f"{roc_auc_score(san_y, san_p_cp):>11.3f} {cp_san_mcc:>11.3f}")
    print(f"  {'RF/CB v2':<20s} {roc_auc_score(scf_y, scf_p_rf):>13.3f} {rf_scf_mcc:>13.3f} "
          f"{roc_auc_score(san_y, san_p_rf):>11.3f} {rf_san_mcc:>11.3f}")

    # 3. α sweep
    print(f"\n=== 3. α (chemprop weight) sweep — 21 단계 ===")
    print(f"  α    scaf_AUC scaf_MCC | san_AUC san_MCC | 평균_MCC")
    print("-" * 75)
    out = []
    for alpha in np.linspace(0, 1, 21):
        p_scf = alpha * scf_p_cp + (1 - alpha) * scf_p_rf
        p_san = alpha * san_p_cp + (1 - alpha) * san_p_rf
        scf_auc = roc_auc_score(scf_y, p_scf)
        san_auc = roc_auc_score(san_y, p_san)
        _, scf_m = best_thr_mcc(scf_y, p_scf)
        _, san_m = best_thr_mcc(san_y, p_san)
        avg = (scf_m + san_m) / 2
        out.append({"alpha": float(alpha), "scaf_auc": float(scf_auc),
                     "scaf_mcc": float(scf_m), "san_auc": float(san_auc),
                     "san_mcc": float(san_m), "avg_mcc": float(avg)})
        marker = " ⭐" if alpha in (0.0, 0.5, 1.0) else ""
        print(f"  {alpha:.2f}   {scf_auc:>7.3f}  {scf_m:>7.3f}  | "
              f"{san_auc:>7.3f} {san_m:>7.3f} | {avg:>7.3f}{marker}")

    # 4. Best α
    best_avg = max(out, key=lambda x: x["avg_mcc"])
    best_san = max(out, key=lambda x: x["san_mcc"])
    best_scf = max(out, key=lambda x: x["scaf_mcc"])
    print(f"\n=== 4. Best α ===")
    print(f"  평균 MCC 최고: α={best_avg['alpha']:.2f}, scaf {best_avg['scaf_mcc']:.3f}, "
          f"san {best_avg['san_mcc']:.3f}, avg {best_avg['avg_mcc']:.3f}")
    print(f"  Sanity MCC 최고: α={best_san['alpha']:.2f}, san {best_san['san_mcc']:.3f}")
    print(f"  Scaffold MCC 최고: α={best_scf['alpha']:.2f}, scaf {best_scf['scaf_mcc']:.3f}")

    # 5. Best α 의 details (threshold + TPR/TNR)
    print(f"\n=== 5. 평균 MCC 최고 α={best_avg['alpha']:.2f} 의 세부 ===")
    a = best_avg["alpha"]
    p_scf = a * scf_p_cp + (1 - a) * scf_p_rf
    p_san = a * san_p_cp + (1 - a) * san_p_rf
    for nm, y, p in [("scaffold", scf_y, p_scf), ("sanity", san_y, p_san)]:
        t, mcc = best_thr_mcc(y, p)
        m = metrics_at_thr(y, p, t)
        auc = roc_auc_score(y, p)
        print(f"  [{nm}] thr={t:.3f}, AUC={auc:.3f}, MCC={mcc:.3f}, "
              f"TPR={m['tpr']:.3f}, TNR={m['tnr']:.3f}")

    with open(os.path.join(RESULTS, "stack_final.json"), "w") as f:
        json.dump({"sweep": out, "best_avg": best_avg,
                    "best_sanity": best_san, "best_scaffold": best_scf}, f, indent=2)
    print(f"\n저장: results/stack_final.json")


if __name__ == "__main__":
    main()
