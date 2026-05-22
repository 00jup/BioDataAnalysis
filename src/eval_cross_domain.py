"""크로스 도메인 평가 — 모델이 자기 도메인 밖에서 얼마나 일반화 하나.

매트릭스:
                 vivo test       vitro test
  vivo model    [ self ]        [ cross 1 ]
  vitro model   [ cross 2 ]     [ self ]

추가:
  - 두 모델 평균 (no rule)
  - 룰별 결합 (vivo_priority / weighted / consensus)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              matthews_corrcoef, roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict_pipeline import DomainModel

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")
RESULTS = os.path.join(PROJECT_ROOT, "results")
THRS = np.linspace(0.05, 0.95, 91)


def predict_all(model: DomainModel, smiles_list: list[str]) -> np.ndarray:
    """분자 리스트 → 모델 확률 (model.predict 이 1 분자씩이라 loop)."""
    out = np.zeros(len(smiles_list))
    for i, s in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            out[i] = 0.5
        else:
            out[i] = model.predict(mol)
    return out


def metrics_at(y, p, t):
    pred = (p >= t).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]; fp, tn = cm[1]
    tpr = tp / max(tp+fn, 1); tnr = tn / max(fp+tn, 1)
    return {
        "threshold": float(t),
        "tpr": float(tpr), "tnr": float(tnr),
        "bacc": (tpr + tnr) / 2,
        "mcc": float(matthews_corrcoef(y, pred)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "auc": float(roc_auc_score(y, p)),
    }


def best_threshold(y, p):
    bt, bm = 0.5, -1.0
    for t in THRS:
        m = matthews_corrcoef(y, (p >= t).astype(int))
        if m > bm: bm, bt = m, t
    return float(bt), float(bm)


def main():
    print("=== 모델 로드 ===")
    vivo = DomainModel("vivo")
    vitro = DomainModel("vitro")

    results = {}
    for test_name, csv in [("vivo_test", "vivo.csv"), ("vitro_test", "vitro.csv")]:
        df = pd.read_csv(os.path.join(TEST_DIR, csv))
        y = df["label"].to_numpy(int)
        smi = df["canonical_smiles"].tolist()
        print(f"\n=== {test_name} ({len(df)}개, 양성 {y.sum()}, 음성 {(1-y).sum()}) ===")

        # vivo 모델 예측
        print(f"  vivo model 예측 중...")
        p_vivo = predict_all(vivo, smi)
        # vitro 모델 예측
        print(f"  vitro model 예측 중...")
        p_vitro = predict_all(vitro, smi)

        # 각 모델: 자체 threshold (학습 시 결정) 사용
        t_v = vivo.meta["threshold"]
        t_vi = vitro.meta["threshold"]

        # 그리고 test 에서 MCC-optimal threshold 도 별도 (peek, 진단)
        bt_v, _ = best_threshold(y, p_vivo)
        bt_vi, _ = best_threshold(y, p_vitro)

        results[test_name] = {
            "n": int(len(df)), "pos": int(y.sum()), "neg": int((1-y).sum()),
            "vivo_model_at_train_thr":  metrics_at(y, p_vivo, t_v),
            "vivo_model_at_peek_thr":   metrics_at(y, p_vivo, bt_v),
            "vitro_model_at_train_thr": metrics_at(y, p_vitro, t_vi),
            "vitro_model_at_peek_thr":  metrics_at(y, p_vitro, bt_vi),
            "average_at_0.5":           metrics_at(y, (p_vivo+p_vitro)/2, 0.5),
        }
        # 평균의 best threshold
        avg = (p_vivo+p_vitro)/2
        bt_a, _ = best_threshold(y, avg)
        results[test_name]["average_at_peek_thr"] = metrics_at(y, avg, bt_a)

    # ---- 출력 ----
    print(f"\n{'='*90}")
    print(f"  크로스 도메인 결과 (train threshold = 학습시 val MCC-max)")
    print(f"{'='*90}")
    print(f"{'test':12s} {'model':28s} {'thr':>5s} {'TPR':>6s} {'TNR':>6s} {'bAcc':>6s} {'MCC':>6s} {'AUC':>6s}")
    for ts, m in results.items():
        for mk in ("vivo_model_at_train_thr", "vitro_model_at_train_thr", "average_at_0.5"):
            r = m[mk]
            print(f"{ts:12s} {mk:28s} {r['threshold']:>5.3f} {r['tpr']:>6.3f} {r['tnr']:>6.3f} {r['bacc']:>6.3f} {r['mcc']:>6.3f} {r['auc']:>6.3f}")
        print()

    # ---- 핵심 매트릭스 ----
    print(f"{'='*90}")
    print(f"  요약 매트릭스 (train threshold, MCC)")
    print(f"{'='*90}")
    print(f"{'':18s} {'vivo test':>15s} {'vitro test':>15s}")
    print(f"{'vivo model':18s} {results['vivo_test']['vivo_model_at_train_thr']['mcc']:>15.3f} {results['vitro_test']['vivo_model_at_train_thr']['mcc']:>15.3f}")
    print(f"{'vitro model':18s} {results['vivo_test']['vitro_model_at_train_thr']['mcc']:>15.3f} {results['vitro_test']['vitro_model_at_train_thr']['mcc']:>15.3f}")
    print(f"{'avg (no rule)':18s} {results['vivo_test']['average_at_0.5']['mcc']:>15.3f} {results['vitro_test']['average_at_0.5']['mcc']:>15.3f}")

    print(f"\n{'='*90}")
    print(f"  요약 매트릭스 (AUC — threshold-free)")
    print(f"{'='*90}")
    print(f"{'':18s} {'vivo test':>15s} {'vitro test':>15s}")
    print(f"{'vivo model':18s} {results['vivo_test']['vivo_model_at_train_thr']['auc']:>15.3f} {results['vitro_test']['vivo_model_at_train_thr']['auc']:>15.3f}")
    print(f"{'vitro model':18s} {results['vivo_test']['vitro_model_at_train_thr']['auc']:>15.3f} {results['vitro_test']['vitro_model_at_train_thr']['auc']:>15.3f}")

    with open(os.path.join(RESULTS, "cross_domain_eval.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n저장: {os.path.join(RESULTS, 'cross_domain_eval.json')}")


if __name__ == "__main__":
    main()
