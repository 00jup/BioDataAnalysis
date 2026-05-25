"""채점 시나리오 통합 평가 — DB lookup + Chemprop fallback.

진짜 채점:
  SMILES → InChIKey → DB lookup
    hit:   DB vivo_label 그대로 사용
    miss:  Chemprop 모델로 fallback (threshold 0.20)

Test set 1836 분자 적용 후 정확도 측정.

추가로: out-of-DB 시나리오 — DB hit 모두 강제 miss → 순수 모델 성능.
"""

from __future__ import annotations
import json, os, sys, subprocess, tempfile
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

CP_THR = 0.20
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RESULTS = os.path.join(PROJECT_ROOT, "results")


def main():
    db = pd.read_parquet(os.path.join(DATA_DIR, "labels_db", "full.parquet"))
    db_lookup = db.set_index("inchi_key")
    test = pd.read_csv(os.path.join(DATA_DIR, "test", "vivo.csv"))
    print(f"test: {len(test)} 분자 (양성 {(test.label==1).sum()})")

    # === 시나리오 1: DB lookup priority + Chemprop fallback ===
    # DB hit: db 의 vivo_label 사용
    # miss:   chemprop 예측 + thr 0.20
    preds_db_hits = 0
    preds_model = 0
    preds_unknown = 0
    pred_label = []
    pred_source = []

    # chemprop 결과 (이미 있음)
    cp = pd.read_csv(os.path.join(PROJECT_ROOT, "models", "chemprop_v9", "vivo",
                                   "test_pred.csv")).rename(columns={"label": "pred_cp"})
    cp_map = dict(zip(cp.canonical_smiles, cp.pred_cp))

    for _, r in test.iterrows():
        ik = r["inchi_key"]
        if ik in db_lookup.index:
            db_row = db_lookup.loc[ik]
            if isinstance(db_row, pd.DataFrame): db_row = db_row.iloc[0]
            vl = db_row.get("vivo_label")
            if pd.notna(vl):
                pred_label.append(int(vl))
                pred_source.append("db")
                preds_db_hits += 1
                continue
        # fallback to chemprop
        p = cp_map.get(r["canonical_smiles"])
        if p is None:
            pred_label.append(0)  # default neg
            pred_source.append("unknown")
            preds_unknown += 1
        else:
            pred_label.append(int(float(p) >= CP_THR))
            pred_source.append("model")
            preds_model += 1

    y = test.label.to_numpy(int); pred = np.array(pred_label)
    src = np.array(pred_source)
    print(f"\n=== 시나리오 1: DB lookup priority + Chemprop fallback (thr {CP_THR}) ===")
    print(f"  DB hit:    {preds_db_hits}/{len(y)} ({100*preds_db_hits/len(y):.1f}%)")
    print(f"  model:     {preds_model}/{len(y)} ({100*preds_model/len(y):.1f}%)")
    print(f"  unknown:   {preds_unknown}/{len(y)}")

    from sklearn.metrics import (matthews_corrcoef, confusion_matrix,
                                  f1_score, accuracy_score, balanced_accuracy_score)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    mcc = matthews_corrcoef(y, pred)
    print(f"\n  N={len(y)} (양성 {(y==1).sum()}, 음성 {(y==0).sum()})")
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    print(f"  MCC {mcc:.3f}  F1 {f1_score(y, pred):.3f}  Acc {accuracy_score(y, pred):.3f}")
    print(f"  bAcc {balanced_accuracy_score(y, pred):.3f}")

    # 분해: DB hit 정확도 vs model 정확도
    print(f"\n  분해:")
    for sname in ("db", "model", "unknown"):
        mask = src == sname
        if mask.sum() == 0: continue
        acc = (pred[mask] == y[mask]).mean()
        pos = (y[mask] == 1).sum()
        print(f"    {sname:>10s}: N={mask.sum():>4d}, 양성 {pos:>3d}, accuracy {acc:.3f}")

    # === 시나리오 2: 순수 Chemprop (DB miss 가정) ===
    print(f"\n=== 시나리오 2: 순수 Chemprop only (모든 DB 무시) ===")
    pred_only = np.array([int(cp_map.get(s, 0) >= CP_THR) for s in test.canonical_smiles])
    cm = confusion_matrix(y, pred_only, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    mcc = matthews_corrcoef(y, pred_only)
    print(f"  TPR {tp/max(tp+fn,1):.3f}  TNR {tn/max(fp+tn,1):.3f}")
    print(f"  MCC {mcc:.3f}  F1 {f1_score(y, pred_only):.3f}  Acc {accuracy_score(y, pred_only):.3f}")

    # === 저장 ===
    out = {
        "scenario_1_lookup_priority": {
            "db_hits": preds_db_hits, "model": preds_model, "unknown": preds_unknown,
            "n_test": len(y), "pos": int(y.sum()),
            "mcc": float(matthews_corrcoef(y, pred)),
            "tpr": float(tp/max(tp+fn,1)) if (cm := confusion_matrix(y, pred, labels=[1,0])) is not None else None,
        },
        "scenario_2_chemprop_only_thr020": {
            "mcc": float(mcc), "tpr": float(tp/max(tp+fn,1)), "tnr": float(tn/max(fp+tn,1)),
            "threshold": CP_THR,
        },
    }
    with open(os.path.join(RESULTS, "lookup_pipeline_eval.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n저장: results/lookup_pipeline_eval.json")


if __name__ == "__main__":
    main()
