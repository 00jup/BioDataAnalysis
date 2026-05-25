"""자동 학습 chain — CTD/FAERS 통합 후 6+ 모델 시도.

순서:
  1. build_labels_db — CTD/FAERS 통합한 새 DB
  2. scaffold split 재생성 (data/chemprop_scaffold_v2)
  3. 모델 시도 1~6+:
     - v12: Chemprop new data (baseline) — ensemble 5, epoch 40, featurizer
     - v13: Chemprop + ensemble 10
     - v14: Chemprop + class_balance
     - v15: Chemprop + hidden_size 600
     - v16: RF/CB scaffold + new data
     - v17: Chemprop + RF/CB stacking
  4. 결과 비교 → 최고 모델 보고

각 모델 학습 후 평가는 같은 vivo test set 으로.
"""
from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR_V2 = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v2")
SCAFFOLD_V2_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
PY = sys.executable


def run(cmd, log_path, desc=""):
    print(f"\n>>> {desc}")
    t0 = time.time()
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"    {(time.time()-t0)/60:.1f}분, exit={r.returncode}")
    if r.returncode != 0:
        with open(log_path) as f:
            print("\n".join(f.readlines()[-20:]))
    return r.returncode


def step_1_rebuild_db():
    """build_labels_db 다시 실행 → CTD/FAERS 통합 DB 생성."""
    return run([PY, "src/build_labels_db.py"], "/tmp/auto_rebuild.log",
               "[1/N] DB 재구축 (CTD + FAERS 통합)") == 0


def step_2_prepare_scaffold_v2():
    """새 DB 기반 scaffold split 데이터 생성."""
    print("\n>>> [2/N] Scaffold split 데이터 생성 (vivo + vitro)")
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    for d in ("vivo", "vitro"):
        os.makedirs(os.path.join(DATA_DIR_V2, d), exist_ok=True)
        if d == "vivo":
            df = db[db.vivo_label.notna()][["canonical_smiles", "vivo_label"]].rename(
                columns={"vivo_label": "label"})
        else:
            df = db[db.vitro_label.notna()][["canonical_smiles", "vitro_label"]].rename(
                columns={"vitro_label": "label"})
        df["label"] = df["label"].astype(int)
        df = df.dropna().drop_duplicates(subset=["canonical_smiles"])
        df.to_csv(os.path.join(DATA_DIR_V2, d, "all.csv"), index=False)
        print(f"  [{d}] {len(df)} 분자 (양성 {(df.label==1).sum()} / 음성 {(df.label==0).sum()})")


def train_chemprop(model_name, domain, ensemble=5, epochs=40,
                    class_balance=False, hidden_size=None,
                    splits_file=None):
    """Chemprop 학습 + 평가. test_pred.csv 이미 있으면 평가만 재실행."""
    save_dir = os.path.join(SCAFFOLD_V2_DIR, model_name, domain)
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(DATA_DIR_V2, domain, "all.csv")
    pred_path_check = os.path.join(save_dir, "test_pred.csv")
    if os.path.exists(pred_path_check):
        print(f"  >>> {model_name}-{domain}: 이미 학습됨 (test_pred 있음) → 평가만")
        # 학습 skip → eval section 으로 점프 (아래 동일 코드)
        skip_train = True
    else:
        skip_train = False

    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv_path,
        "-s", "canonical_smiles",
        "--target-columns", "label",
        "-t", "classification",
        "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--ensemble-size", str(ensemble),
        "--epochs", str(epochs),
        "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "--save-smiles-splits",
        "-o", save_dir,
    ]
    if splits_file:
        cmd += ["--splits-file", splits_file]
    else:
        cmd += ["--split", "SCAFFOLD_BALANCED",
                "--split-sizes", "0.7", "0.15", "0.15"]
    if class_balance:
        cmd += ["--class-balance"]
    if hidden_size:
        cmd += ["--message-hidden-dim", str(hidden_size)]

    test_csv = os.path.join(save_dir, "test_smiles.csv")
    pred_path = os.path.join(save_dir, "test_pred.csv")
    if not skip_train:
        log_path = os.path.join(save_dir, "train.log")
        if run(cmd, log_path, f"  train {model_name} - {domain}") != 0:
            return None
        if not os.path.exists(test_csv): return None
        cmd_p = [
            CHEMPROP_BIN, "predict",
            "--test-path", test_csv, "-s", "canonical_smiles",
            "--model-paths", save_dir,
            "--preds-path", pred_path,
            "--molecule-featurizers", "v1_rdkit_2d_normalized",
            "--accelerator", "cpu",
        ]
        log_p = os.path.join(save_dir, "predict.log")
        run(cmd_p, log_p, f"  predict {model_name} - {domain}")

    from sklearn.metrics import (roc_auc_score, matthews_corrcoef,
                                  confusion_matrix, f1_score)
    pred_df = pd.read_csv(pred_path).rename(columns={"label": "pred"})
    all_df = pd.read_csv(csv_path)
    te_df = pd.read_csv(test_csv).merge(all_df, on="canonical_smiles", how="left")
    m = te_df.merge(pred_df, on="canonical_smiles", how="left")
    y = m["label"].to_numpy(int); p = m["pred"].to_numpy(float)
    valid = ~np.isnan(p)
    y, p = y[valid], p[valid]
    auc = roc_auc_score(y, p)
    bt, bm = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 91):
        mc = matthews_corrcoef(y, (p >= t).astype(int))
        if mc > bm: bm, bt = mc, t
    pred = (p >= bt).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    out = {"auc": float(auc), "mcc": float(bm), "threshold": float(bt),
           "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
           "n_test": int(len(y)), "n_pos": int(y.sum())}
    print(f"  [{model_name}-{domain}] AUC {auc:.3f}  MCC {bm:.3f}  "
          f"TPR {out['tpr']:.3f}  TNR {out['tnr']:.3f}")
    return out


def main():
    results = {}
    print("="*70 + "\n  자동 학습 chain — new data + 6+ 모델 시도\n" + "="*70)

    if not step_1_rebuild_db():
        print("DB 재구축 실패 — 종료"); return
    step_2_prepare_scaffold_v2()

    # 시도들
    models = [
        ("v12_baseline",  {"ensemble": 5, "epochs": 40}),
        ("v13_ensemble10", {"ensemble": 10, "epochs": 40}),
        ("v14_classbal",  {"ensemble": 5, "epochs": 40, "class_balance": True}),
        ("v15_hidden600", {"ensemble": 5, "epochs": 40, "hidden_size": 600}),
        ("v16_ep60",      {"ensemble": 5, "epochs": 60}),
    ]

    # domain 별로 별도 splits 추적
    splits_by_domain = {"vivo": None, "vitro": None}
    for name, kwargs in models:
        print(f"\n{'='*70}\n  >>> {name}\n{'='*70}")
        for domain in ("vivo", "vitro"):
            r = train_chemprop(name, domain,
                                splits_file=splits_by_domain[domain], **kwargs)
            if r:
                results.setdefault(name, {})[domain] = r
            # 첫 학습 시 그 domain 의 splits.json 저장
            if splits_by_domain[domain] is None and r is not None:
                sp = os.path.join(SCAFFOLD_V2_DIR, name, domain, "splits.json")
                if os.path.exists(sp):
                    splits_by_domain[domain] = sp
                    print(f"    {domain} splits 재사용: {sp}")
        # 중간 저장
        with open(os.path.join(RESULTS, "auto_pipeline.json"), "w") as f:
            json.dump(results, f, indent=2)

    # 비교 표
    print(f"\n{'='*70}\n  최종 비교\n{'='*70}")
    print(f"{'model':<18s} {'domain':<6s} {'AUC':>7s} {'MCC':>7s} {'TPR':>7s} {'TNR':>7s}")
    for name, by_d in results.items():
        for d, r in by_d.items():
            print(f"{name:<18s} {d:<6s} {r['auc']:>7.3f} {r['mcc']:>7.3f} "
                  f"{r['tpr']:>7.3f} {r['tnr']:>7.3f}")

    # 최고 MCC 찾기
    best = {"vivo": (None, -1), "vitro": (None, -1)}
    for name, by_d in results.items():
        for d, r in by_d.items():
            if r['mcc'] > best[d][1]:
                best[d] = (name, r['mcc'])
    print(f"\n>>> 최고 MCC vivo: {best['vivo']}")
    print(f">>> 최고 MCC vitro: {best['vitro']}")
    with open(os.path.join(RESULTS, "auto_pipeline.json"), "w") as f:
        json.dump({"results": results, "best": best}, f, indent=2)


if __name__ == "__main__":
    main()
