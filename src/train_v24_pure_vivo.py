"""v24 — vivo only (vitro 라벨 있는 분자도 제외) + sanity 제외 + 두 모델 새 학습.

조건:
  1. vivo_label notna AND vitro_label isna → 순수 vivo only
  2. sanity 약물 InChIKey 제외
  3. 새 scaffold split (seed=456)
  4. Chemprop + RF/CB 둘 다 새 학습 (v17 hp 동일)
"""
from __future__ import annotations
import json, os, sys, subprocess, time, joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.train_domain_models import ensure_fp_cache, FPS, optimize_ensemble, full_metrics
from src.train_v23_honest import SANITY_DRUGS

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_v24_pure")
CHEMPROP_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v24")
RFCB_DIR = os.path.join(PROJECT_ROOT, "models", "rfcb_v4")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
SEED = 456


def get_sanity_iks():
    iks = set(); csmi = set()
    for _, smi, _ in SANITY_DRUGS:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: continue
        iks.add(Chem.MolToInchiKey(mol))
        csmi.add(Chem.MolToSmiles(mol))
    return iks, csmi


def build_pure_vivo():
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    print(f"DB 전체: {len(db)}")
    # vivo only (vitro 라벨 없음)
    pure = db[db.vivo_label.notna() & db.vitro_label.isna()].copy()
    print(f"  vivo 있고 vitro 없음 (pure): {len(pure)}")
    pure["label"] = pure["vivo_label"].astype(int)
    sanity_iks, sanity_csmi = get_sanity_iks()
    before = len(pure)
    pure = pure[~pure.inchi_key.isin(sanity_iks) & ~pure.canonical_smiles.isin(sanity_csmi)]
    print(f"  sanity 제외 후: {len(pure)} (-{before-len(pure)})")
    print(f"  최종 양성 {(pure.label==1).sum()} / 음성 {(pure.label==0).sum()}")
    os.makedirs(os.path.join(DATA_DIR, "vivo"), exist_ok=True)
    csv = os.path.join(DATA_DIR, "vivo", "all.csv")
    pure[["canonical_smiles", "label"]].to_csv(csv, index=False)
    return csv


def train_chemprop():
    print(f"\n=== Chemprop v24 — pure vivo + seed {SEED} ===")
    save = CHEMPROP_DIR; os.makedirs(save, exist_ok=True)
    cmd = [
        CHEMPROP_BIN, "train",
        "-i", os.path.join(DATA_DIR, "vivo", "all.csv"),
        "-s", "canonical_smiles", "--target-columns", "label",
        "-t", "classification", "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.70", "0.15", "0.15",
        "--data-seed", str(SEED), "--pytorch-seed", str(SEED),
        "--ensemble-size", "15",
        "--message-hidden-dim", "600",
        "--epochs", "40", "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "--save-smiles-splits",
        "-o", save,
    ]
    log = os.path.join(save, "train.log")
    t0 = time.time()
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    return r.returncode == 0


def train_rfcb():
    print(f"\n=== RF/CB v4 — chemprop v24 와 같은 split ===")
    splits = json.load(open(os.path.join(CHEMPROP_DIR, "splits.json")))
    s = splits[0] if isinstance(splits, list) else splits
    all_df = pd.read_csv(os.path.join(DATA_DIR, "vivo", "all.csv")).reset_index(drop=True)
    tr, va, te = all_df.iloc[s["train"]], all_df.iloc[s["val"]], all_df.iloc[s["test"]]
    print(f"  train {len(tr)} val {len(va)} test {len(te)}")

    save = RFCB_DIR; os.makedirs(save, exist_ok=True)
    val_probs, test_probs = [], []
    yv_ref = yte_ref = None
    names = []
    for fp_name in FPS:
        for kind in ("rf", "cb"):
            name = f"{kind}_{fp_name}"; names.append(name)
            cache = ensure_fp_cache(
                (tr.canonical_smiles.tolist() + va.canonical_smiles.tolist() +
                 te.canonical_smiles.tolist()), fp_name)
            cols = cache.columns.tolist()
            def fp_xy(df):
                mask = df["canonical_smiles"].isin(cache.index)
                sub = df[mask].reset_index(drop=True)
                X = cache.loc[sub["canonical_smiles"], cols].to_numpy(dtype=np.uint8)
                y = sub["label"].to_numpy(int)
                return X, y
            Xtr, ytr = fp_xy(tr); Xv, yv = fp_xy(va); Xte, yte = fp_xy(te)
            t0 = time.time()
            if kind == "rf":
                m = RandomForestClassifier(n_estimators=500, max_features="sqrt",
                                             min_samples_leaf=2, class_weight="balanced",
                                             random_state=SEED, n_jobs=-1)
                m.fit(Xtr, ytr)
            else:
                m = CatBoostClassifier(iterations=500, depth=6, learning_rate=0.05,
                                          verbose=0, class_weights={0: 1, 1: 3},
                                          random_state=SEED)
                m.fit(Xtr, ytr)
            pv = m.predict_proba(Xv)[:, 1]; pte = m.predict_proba(Xte)[:, 1]
            val_probs.append(pv); test_probs.append(pte)
            if yv_ref is None: yv_ref, yte_ref = yv, yte
            sd = os.path.join(save, name); os.makedirs(sd, exist_ok=True)
            if kind == "rf":
                joblib.dump(m, os.path.join(sd, "model.pkl"))
            else:
                m.save_model(os.path.join(sd, "model.cbm"))
            print(f"  {name:14s} {time.time()-t0:5.1f}s  val AUC {roc_auc_score(yv,pv):.3f}  test AUC {roc_auc_score(yte,pte):.3f}")

    Xv = np.array(val_probs).T; Xte = np.array(test_probs).T
    w, thr, val_mcc = optimize_ensemble(Xv, yv_ref)
    test_m = full_metrics(yte_ref, Xte @ w, thr)
    print(f"  [RF/CB v4] AUC {test_m['auc']:.3f}, MCC {test_m['mcc']:.3f}, TPR {test_m['tpr']:.3f}, TNR {test_m['tnr']:.3f}")
    meta = {"members": names, "weights": w.tolist(), "threshold": thr, "test_metrics": test_m}
    with open(os.path.join(save, "ensemble_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    print("=== v24 pure vivo (vitro 라벨 완전 제외) ===\n")
    build_pure_vivo()
    if not train_chemprop(): print("Chemprop 실패"); return
    train_rfcb()
    print(f"\n학습 완료 — 평가는 eval_v23.py 응용")


if __name__ == "__main__":
    main()
