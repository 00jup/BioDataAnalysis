"""Balanced training — 음성 다운샘플링으로 1:1 학습 (v5).

vivo: 양성 815 + 음성 815 (랜덤 샘플) = 1,630 분자, 1:1 balanced
vitro: 양성 2,070 + 음성 2,070 = 4,140 분자 (이미 1:1.8 이라 효과 작을 듯)

같은 imbalanced test 에서 평가 → TPR ↑ 기대.
"""

from __future__ import annotations

import os
import sys

import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RANDOM_STATE = 42


def make_balanced(train_df: pd.DataFrame, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """양성:음성 1:1 — 음성을 양성 수만큼 다운샘플."""
    pos = train_df[train_df.label == 1]
    neg = train_df[train_df.label == 0]
    n = min(len(pos), len(neg))
    pos_s = pos.sample(n=n, random_state=seed)
    neg_s = neg.sample(n=n, random_state=seed)
    return pd.concat([pos_s, neg_s], ignore_index=True).sample(frac=1, random_state=seed).reset_index(drop=True)


def main():
    print("=== Balanced training (v5) — 1:1 다운샘플 ===\n")

    import shutil
    from src.train_domain_models import train_domain

    for domain in ("vivo", "vitro"):
        orig = os.path.join(TRAIN_DIR, f"{domain}.csv")
        backup = orig + ".bak_v5"

        # 백업 후 balanced 로 교체
        tr = pd.read_csv(orig)
        bal = make_balanced(tr)
        print(f"\n[{domain}]  원본 {len(tr)} (양성 {(tr.label==1).sum()} / 음성 {(tr.label==0).sum()})")
        print(f"          → 1:1  {len(bal)} (양성 {(bal.label==1).sum()} / 음성 {(bal.label==0).sum()})")

        os.rename(orig, backup)
        bal.to_csv(orig, index=False)

        try:
            meta = train_domain(domain)
            # models/{domain}/ → models/{domain}_balanced/
            src_dir = os.path.join(MODELS_DIR, domain)
            dst_dir = os.path.join(MODELS_DIR, f"{domain}_balanced")
            if os.path.exists(dst_dir):
                shutil.rmtree(dst_dir)
            shutil.copytree(src_dir, dst_dir)
            t = meta["test_metrics"]
            print(f"\n[{domain} balanced]  TPR {t['tpr']:.3f}  TNR {t['tnr']:.3f}  MCC {t['mcc']:.3f}  AUC {t['auc']:.3f}  F1 {t['f1']:.3f}")
        finally:
            os.remove(orig)
            os.rename(backup, orig)


if __name__ == "__main__":
    main()
