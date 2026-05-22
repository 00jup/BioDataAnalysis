"""파이프라인 최종 평가 — 3 룰 × 여러 시나리오.

평가 셋:
  - data/test/vivo.csv  (vivo 라벨 기준)
  - data/test/vitro.csv (vitro 라벨 기준)
  - data/test/professor_test.csv (라벨 없음, 예측만)

각 셋 × 3 룰 (vivo_priority / weighted / consensus) 비교.
"""

from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              matthews_corrcoef, precision_score,
                              recall_score, roc_auc_score)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
RESULTS = os.path.join(PROJECT_ROOT, "results")
RULES = ("vivo_priority", "weighted", "consensus")


def evaluate_one(pipe, test_df: pd.DataFrame, label_col: str = "label") -> dict:
    has_label = label_col in test_df.columns
    rows = []
    t0 = time.time()
    for i, row in test_df.iterrows():
        r = pipe.predict(row["canonical_smiles"])
        rec = {
            "smiles": row["canonical_smiles"], "pred": r.label, "score": r.score,
            "source": r.source, "p_vivo": r.model_p_vivo, "p_vitro": r.model_p_vitro,
            "db_hit": r.db_hit,
        }
        if has_label:
            rec["actual"] = int(row[label_col])
        rows.append(rec)
        if i % 200 == 0 and i > 0:
            print(f"    {i}/{len(test_df)} ({(i+1)/(time.time()-t0):.1f}/s)")
    pred_df = pd.DataFrame(rows)
    if not has_label:
        return {"n": len(pred_df), "preds_only": True,
                "pred_distribution": pred_df["pred"].value_counts(dropna=False).to_dict(),
                "source_distribution": pred_df["source"].value_counts().to_dict()}

    # 라벨 있을 때 — 메트릭
    decided = pred_df[pred_df["pred"].notna()].copy()
    y = decided["actual"].astype(int).to_numpy()
    p = decided["pred"].astype(int).to_numpy()
    cm = confusion_matrix(y, p, labels=[1,0])
    tp, fn = cm[0]; fp, tn = cm[1]
    n_p, n_n = tp+fn, fp+tn
    metrics = {
        "n_total": int(len(pred_df)),
        "n_decided": int(len(decided)),
        "n_abstain": int(len(pred_df) - len(decided)),
        "n_pos": int(n_p), "n_neg": int(n_n),
        "tp": int(tp), "fn": int(fn), "fp": int(fp), "tn": int(tn),
        "tpr": float(tp/max(n_p,1)),
        "tnr": float(tn/max(n_n,1)),
        "precision": float(tp/max(tp+fp,1)),
        "npv": float(tn/max(tn+fn,1)),
        "accuracy": float(accuracy_score(y, p)),
        "f1": float(f1_score(y, p, zero_division=0)),
        "mcc": float(matthews_corrcoef(y, p)),
        "source_distribution": pred_df["source"].value_counts().to_dict(),
    }
    # AUC — score 있을 때만 (db_lookup 은 score=0/1 이므로 약간 왜곡 가능)
    try:
        s = decided["score"].astype(float).to_numpy()
        metrics["auc"] = float(roc_auc_score(y, s))
    except Exception:
        metrics["auc"] = None
    return metrics


def main():
    sys.path.insert(0, PROJECT_ROOT)
    from src.predict_pipeline import HepatotoxPipeline

    os.makedirs(RESULTS, exist_ok=True)
    all_results = {}

    use_lookup = "--no-lookup" not in sys.argv
    mode_tag = "lookup+model" if use_lookup else "ML-only"

    for rule in RULES:
        print(f"\n{'='*70}\n  RULE: {rule}  [{mode_tag}]\n{'='*70}")
        pipe = HepatotoxPipeline(rule=rule, use_lookup=use_lookup)
        rule_results = {}

        # vivo test
        df_v = pd.read_csv(os.path.join(TEST_DIR, "vivo.csv"))
        print(f"\n[vivo.csv {len(df_v)}개] 평가...")
        rule_results["vivo_test"] = evaluate_one(pipe, df_v)
        m = rule_results["vivo_test"]
        print(f"  n={m['n_decided']}/{m['n_total']} (abstain {m['n_abstain']})")
        print(f"  TPR {m['tpr']:.3f}  TNR {m['tnr']:.3f}  bAcc {(m['tpr']+m['tnr'])/2:.3f}  MCC {m['mcc']:.3f}  AUC {m.get('auc', 0):.3f}")
        print(f"  source: {m['source_distribution']}")

        # vitro test
        df_vi = pd.read_csv(os.path.join(TEST_DIR, "vitro.csv"))
        print(f"\n[vitro.csv {len(df_vi)}개] 평가...")
        rule_results["vitro_test"] = evaluate_one(pipe, df_vi)
        m = rule_results["vitro_test"]
        print(f"  n={m['n_decided']}/{m['n_total']} (abstain {m['n_abstain']})")
        print(f"  TPR {m['tpr']:.3f}  TNR {m['tnr']:.3f}  bAcc {(m['tpr']+m['tnr'])/2:.3f}  MCC {m['mcc']:.3f}  AUC {m.get('auc', 0):.3f}")

        # professor test (labels 모름)
        prof_path = os.path.join(TEST_DIR, "professor_test.csv")
        if os.path.exists(prof_path):
            df_p = pd.read_csv(prof_path)
            print(f"\n[professor_test.csv {len(df_p)}개] 예측만 (라벨 없음)...")
            # canonical_smiles 컬럼 없으면 SMILES 컬럼 추정
            smi_col = "canonical_smiles" if "canonical_smiles" in df_p.columns else ("SMILES" if "SMILES" in df_p.columns else df_p.columns[0])
            df_p = df_p.rename(columns={smi_col: "canonical_smiles"})
            rule_results["professor_test"] = evaluate_one(pipe, df_p)
            m = rule_results["professor_test"]
            print(f"  예측 분포: {m['pred_distribution']}")
            print(f"  source: {m['source_distribution']}")

        all_results[rule] = rule_results

    out = os.path.join(RESULTS, f"pipeline_eval{'_ml_only' if not use_lookup else ''}.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {out}")

    # 종합 표
    print(f"\n{'='*90}")
    print(f"  종합 비교 (룰별 × 테스트셋)")
    print(f"{'='*90}")
    print(f"{'rule':18s} {'set':12s} {'n':>5s} {'abstain':>7s} {'TPR':>6s} {'TNR':>6s} {'MCC':>6s} {'AUC':>6s} {'F1':>6s}")
    for rule, rr in all_results.items():
        for set_name, m in rr.items():
            if m.get("preds_only"): continue
            auc = m.get("auc")
            auc_s = f"{auc:.3f}" if auc is not None else "  —  "
            print(f"{rule:18s} {set_name:12s} {m['n_total']:>5d} {m['n_abstain']:>7d} {m['tpr']:>6.3f} {m['tnr']:>6.3f} {m['mcc']:>6.3f} {auc_s:>6s} {m['f1']:>6.3f}")


if __name__ == "__main__":
    main()
