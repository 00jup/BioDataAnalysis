"""v23 honest — sanity 약물 완전 제외 + 새 scaffold split + 두 모델 새 학습.

조건:
  1. 학습 데이터에서 sanity 20 약물 (InChIKey) 다 제외
     → train/val/test 어디에도 sanity 약물 없음
     → sanity 평가가 진짜 외부 검증
  2. 새 scaffold-balanced split (다른 seed)
  3. Chemprop v23 학습 (v17 같은 hp)
  4. RF/CB v3 학습 (RF/CB v2 같은 hp)
  5. Honest stacking 평가 (val α/thr 결정 → test)
  6. Sanity 평가 = 진짜 외부
"""
from __future__ import annotations
import json, os, sys, subprocess, tempfile, time, joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                              confusion_matrix, f1_score)

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import ensure_fp_cache, FPS, optimize_ensemble, full_metrics

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_v23_honest")
CHEMPROP_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v23")
RFCB_DIR = os.path.join(PROJECT_ROOT, "models", "rfcb_v3")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
RANDOM_STATE = 123  # 다른 seed (v17 = 42)


# === Sanity 약물 (학습 제외 대상) ===
SANITY_DRUGS = [
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


def get_sanity_iks():
    """sanity 약물의 InChIKey."""
    iks = set()
    canonical_smiles_set = set()
    for name, smi, _ in SANITY_DRUGS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        iks.add(Chem.MolToInchiKey(mol))
        canonical_smiles_set.add(Chem.MolToSmiles(mol))
    return iks, canonical_smiles_set


def build_data_excl_sanity():
    """학습 데이터에서 sanity 약물 InChIKey 제외."""
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    vivo = db[db.vivo_label.notna()][["canonical_smiles", "vivo_label", "inchi_key"]].rename(
        columns={"vivo_label": "label"})
    vivo["label"] = vivo["label"].astype(int)
    vivo = vivo.dropna(subset=["canonical_smiles", "label"])

    sanity_iks, sanity_csmi = get_sanity_iks()
    before = len(vivo)
    vivo_filtered = vivo[
        (~vivo.inchi_key.isin(sanity_iks)) &
        (~vivo.canonical_smiles.isin(sanity_csmi))
    ].copy()
    excluded = before - len(vivo_filtered)
    print(f"학습 데이터: {before} → {len(vivo_filtered)} (sanity {excluded} 제외)")

    os.makedirs(os.path.join(DATA_DIR, "vivo"), exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "vivo", "all.csv")
    vivo_filtered[["canonical_smiles", "label"]].to_csv(csv_path, index=False)
    return csv_path, len(vivo_filtered)


# === Chemprop v23 ===
def train_chemprop_v23(csv_path):
    print(f"\n=== Chemprop v23 (sanity 제외 + 새 scaffold + seed {RANDOM_STATE}) ===")
    save_dir = CHEMPROP_DIR; os.makedirs(save_dir, exist_ok=True)
    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv_path, "-s", "canonical_smiles",
        "--target-columns", "label", "-t", "classification", "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.70", "0.15", "0.15",
        "--data-seed", str(RANDOM_STATE),
        "--pytorch-seed", str(RANDOM_STATE),
        "--ensemble-size", "15",
        "--message-hidden-dim", "600",
        "--epochs", "40", "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "--save-smiles-splits",
        "-o", save_dir,
    ]
    log_path = os.path.join(save_dir, "train.log")
    t0 = time.time()
    print(f"  hp: ensemble 15 + hidden 600 + featurizer (v17 동일)")
    print(f"  seed: {RANDOM_STATE}")
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    return r.returncode == 0


# === RF/CB v3 ===
def train_rfcb_v3():
    """Chemprop 의 splits.json 재사용 → 같은 train/val/test."""
    print(f"\n=== RF/CB v3 (chemprop v23 와 같은 split) ===")
    splits = json.load(open(os.path.join(CHEMPROP_DIR, "splits.json")))
    s = splits[0] if isinstance(splits, list) else splits
    all_df = pd.read_csv(os.path.join(DATA_DIR, "vivo", "all.csv")).reset_index(drop=True)
    tr = all_df.iloc[s["train"]]
    va = all_df.iloc[s["val"]]
    te = all_df.iloc[s["test"]]
    print(f"  split: train {len(tr)}, val {len(va)}, test {len(te)}")

    save_dir = RFCB_DIR; os.makedirs(save_dir, exist_ok=True)
    val_probs, test_probs = [], []
    yv_ref, yte_ref = None, None
    names = []
    for fp_name in FPS:
        for kind in ("rf", "cb"):
            name = f"{kind}_{fp_name}"; names.append(name)
            cache = ensure_fp_cache(tr["canonical_smiles"].tolist(), fp_name)
            ensure_fp_cache(va["canonical_smiles"].tolist(), fp_name)
            ensure_fp_cache(te["canonical_smiles"].tolist(), fp_name)
            cache = ensure_fp_cache((tr.canonical_smiles.tolist() +
                                       va.canonical_smiles.tolist() +
                                       te.canonical_smiles.tolist()), fp_name)
            cols = cache.columns.tolist()
            def fp_xy(df):
                mask = df["canonical_smiles"].isin(cache.index)
                sub = df[mask].reset_index(drop=True)
                X = cache.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
                y = sub["label"].to_numpy(int)
                return X, y
            Xtr, ytr = fp_xy(tr)
            Xv, yv = fp_xy(va)
            Xte, yte = fp_xy(te)
            t0 = time.time()
            if kind == "rf":
                m = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                             min_samples_leaf=2, class_weight="balanced",
                                             random_state=RANDOM_STATE, n_jobs=-1)
                m.fit(Xtr, ytr)
            else:
                m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                          loss_function="Logloss", verbose=0,
                                          class_weights={0: 1, 1: 3},
                                          random_state=RANDOM_STATE)
                m.fit(Xtr, ytr)
            pv = m.predict_proba(Xv)[:, 1]
            pte = m.predict_proba(Xte)[:, 1]
            val_probs.append(pv); test_probs.append(pte)
            if yv_ref is None: yv_ref, yte_ref = yv, yte
            sd = os.path.join(save_dir, name); os.makedirs(sd, exist_ok=True)
            if kind == "rf":
                joblib.dump(m, os.path.join(sd, "model.pkl"))
            else:
                m.save_model(os.path.join(sd, "model.cbm"))
            print(f"  {name:14s} {time.time()-t0:5.1f}s  val AUC {roc_auc_score(yv,pv):.3f}  test AUC {roc_auc_score(yte,pte):.3f}")

    Xv = np.array(val_probs).T
    Xte = np.array(test_probs).T
    w, thr, val_mcc = optimize_ensemble(Xv, yv_ref)
    test_m = full_metrics(yte_ref, Xte @ w, thr)
    print(f"  [RF/CB v3 test] AUC {test_m['auc']:.3f}, MCC {test_m['mcc']:.3f}, TPR {test_m['tpr']:.3f}, TNR {test_m['tnr']:.3f}")
    meta = {"members": names, "weights": w.tolist(), "threshold": thr, "test_metrics": test_m}
    with open(os.path.join(save_dir, "ensemble_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return test_m


def main():
    print("=== v23 honest 학습 — sanity 제외 + 두 모델 새 학습 ===\n")
    csv_path, n = build_data_excl_sanity()
    if not train_chemprop_v23(csv_path):
        print("Chemprop 학습 실패"); return
    train_rfcb_v3()
    print(f"\n학습 완료 — 평가는 별도 스크립트 실행")


if __name__ == "__main__":
    main()
