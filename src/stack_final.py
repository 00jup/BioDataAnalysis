"""Honest stacking — chemprop v31 + RFCB v31 + ChemBERTa-zinc-base (scaffold_v3, vivo).

val 에서 가중치(simplex)+threshold 를 MCC 최대로 찾고 → test 에 적용 (정직한 일반화).
개별 모델 대비 stacking 이득을 출력한다.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, matthews_corrcoef, roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SP = os.path.join(ROOT, "results", "stack_preds")
CB = "/Users/parkjeong-uk/Downloads/drive-download-20260601T160309Z-3-001"

SOURCES = {
    "chemprop": os.path.join(SP, "chemprop_v31_{}_pred.csv"),
    "rfcb": os.path.join(SP, "rfcb_v31_{}_pred.csv"),
    "chemberta": os.path.join(CB, "ChemBERTa-zinc-base-v1_{}_pred.csv"),
}
NAMES = list(SOURCES)


def load(split):
    base = None
    for n, patt in SOURCES.items():
        d = pd.read_csv(patt.format(split))[["canonical_smiles", "label", "prob"]].rename(columns={"prob": n})
        base = d[["canonical_smiles", "label", n]] if base is None else base.merge(
            d[["canonical_smiles", n]], on="canonical_smiles", how="inner")
    return base


def best_thr(prob, y):
    b = (-1.0, 0.5)
    for t in np.linspace(0.05, 0.95, 19):
        m = matthews_corrcoef(y, (prob >= t).astype(int))
        if m > b[0]:
            b = (m, t)
    return b


def stats(prob, y, thr):
    pred = (prob >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[1, 0])
    tp, fn = cm[0]
    fp, tn = cm[1]
    return dict(auc=roc_auc_score(y, prob), mcc=matthews_corrcoef(y, pred),
               tpr=tp / max(tp + fn, 1), tnr=tn / max(fp + tn, 1))


def main():
    val, test = load("val"), load("test")
    print(f"정렬: val {len(val)} / test {len(test)} (공통 분자)")
    Xv, yv = val[NAMES].to_numpy(), val["label"].to_numpy(int)
    Xt, yt = test[NAMES].to_numpy(), test["label"].to_numpy(int)

    print("\n=== 개별 (test) ===")
    for i, n in enumerate(NAMES):
        m, t = best_thr(Xv[:, i], yv)  # val 에서 thr 결정 → test 적용 (honest)
        s = stats(Xt[:, i], yt, t)
        print(f"  {n:<10} AUC {s['auc']:.3f}  MCC {s['mcc']:+.3f}  (TPR {s['tpr']:.2f}/TNR {s['tnr']:.2f}) @thr{t:.2f}")

    # weight simplex grid + threshold on val
    grid = np.round(np.arange(0, 1.0001, 0.05), 2)
    best = (-1.0, None, 0.5)
    for w1 in grid:
        for w2 in grid:
            w3 = round(1 - w1 - w2, 2)
            if w3 < -1e-9:
                continue
            w = np.array([w1, w2, w3])
            m, t = best_thr(Xv @ w, yv)
            if m > best[0]:
                best = (m, w, t)
    _, w, thr = best
    s = stats(Xt @ w, yt, thr)
    print("\n=== STACKED (honest: val→test) ===")
    print(f"  weights {dict(zip(NAMES, w.round(2)))}  thr {thr:.2f}")
    print(f"  test AUC {s['auc']:.3f}  MCC {s['mcc']:+.3f}  TPR {s['tpr']:.2f}  TNR {s['tnr']:.2f}")
    print(f"\n  (단일 최고 chemprop v31: AUC 0.788 / MCC 0.436 대비)")


if __name__ == "__main__":
    main()
