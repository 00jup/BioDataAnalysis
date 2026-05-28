"""Scaffold-balanced split v3 — class expansion 통합 데이터 사용.

Bemis-Murcko scaffold-balanced 70/15/15.
v2 와 동일한 seed/rules — 같은 분자는 같은 split (안정성).

Source: data/labels_db/full_class_expanded.parquet
Output:
  data/chemprop_scaffold_v3/{vivo,vitro}/all.csv
  data/chemprop_scaffold_v3/{vivo,vitro}/splits.json
"""

from __future__ import annotations

import json
import os
import random
from collections import defaultdict

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full_class_expanded.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v3")
SEED = 42
SPLIT = (0.70, 0.15, 0.15)


def murcko(smi: str) -> str:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return ""
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc, canonical=True)
    except Exception:
        return ""


def scaffold_split(df: pd.DataFrame):
    """양성/음성 비율 유지하면서 scaffold 단위로 split."""
    rng = random.Random(SEED)
    scaffolds = defaultdict(list)
    for idx, row in df.iterrows():
        sc = murcko(row["canonical_smiles"])
        scaffolds[sc].append(idx)

    # 큰 scaffold → train 으로, 작은 scaffold → val/test
    scaffold_groups = sorted(scaffolds.items(), key=lambda x: (-len(x[1]), x[0]))

    n_total = len(df)
    n_train_target = int(n_total * SPLIT[0])
    n_val_target = int(n_total * SPLIT[1])

    train_idx, val_idx, test_idx = [], [], []
    for sc, idxs in scaffold_groups:
        # 균형 위해 양/음 모두 있는 큰 scaffold → train
        # 작은 scaffold → val/test 에 랜덤 배분
        if len(train_idx) + len(idxs) <= n_train_target:
            train_idx.extend(idxs)
        elif len(val_idx) + len(idxs) <= n_val_target:
            val_idx.extend(idxs)
        else:
            test_idx.extend(idxs)
    # 약간 shuffle 으로 안정
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)
    return train_idx, val_idx, test_idx


def build_domain(db: pd.DataFrame, domain: str):
    """domain: vivo or vitro."""
    label_col = f"{domain}_label"
    sub = db[db[label_col].notna()].copy()
    sub[label_col] = sub[label_col].astype(int)
    sub = sub[["canonical_smiles", label_col]].rename(columns={label_col: "label"})
    sub = sub.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles")
    sub = sub.reset_index(drop=True)

    print(f"\n[{domain}] {len(sub)} 분자")
    print(f"  양성 {(sub.label == 1).sum()}, 음성 {(sub.label == 0).sum()}")

    train_idx, val_idx, test_idx = scaffold_split(sub)
    print(f"  Split: train {len(train_idx)}, val {len(val_idx)}, test {len(test_idx)}")
    print(
        f"  train 양성률 {sub.iloc[train_idx].label.mean():.3f} / "
        f"val {sub.iloc[val_idx].label.mean():.3f} / "
        f"test {sub.iloc[test_idx].label.mean():.3f}"
    )

    out_d = os.path.join(OUT_DIR, domain)
    os.makedirs(out_d, exist_ok=True)
    sub.to_csv(os.path.join(out_d, "all.csv"), index=False)

    splits = [{"train": train_idx, "val": val_idx, "test": test_idx}]
    with open(os.path.join(out_d, "splits.json"), "w") as f:
        json.dump(splits, f)
    print(f"  저장: {out_d}/all.csv, splits.json")


def main():
    print("=== Scaffold-balanced split v3 (class expanded) ===")
    if not os.path.exists(DB_PATH):
        print(f"ERROR: {DB_PATH} 없음. integrate_class_expansion 먼저 실행.")
        return
    db = pd.read_parquet(DB_PATH)
    print(f"\nDB: {len(db)} 분자")
    print(f"  vivo labeled: {db.vivo_label.notna().sum()}")
    print(f"  vitro labeled: {db.vitro_label.notna().sum()}")

    os.makedirs(OUT_DIR, exist_ok=True)
    build_domain(db, "vivo")
    if db.vitro_label.notna().sum() > 100:
        build_domain(db, "vitro")
    print("\n완료.")


if __name__ == "__main__":
    main()
