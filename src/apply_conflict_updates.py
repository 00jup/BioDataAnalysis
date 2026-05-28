"""28개 conflict (0 → 1) label update.

LiverTox 가 명확히 hepatotoxic (A/B/C) 으로 분류한 분자.
우리 DB 가 source 부족으로 음성 분류한 것 수정.

Update:
  - data/labels_db/full_class_expanded.parquet (vivo_label 0→1)
  - data/chemprop_scaffold_v3/vivo/all.csv (label 0→1)
"""

from __future__ import annotations

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFLICTS = os.path.join(PROJECT_ROOT, "data", "labels_db", "class_expansion_conflicts.csv")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full_class_expanded.parquet")
DB_OUT = os.path.join(PROJECT_ROOT, "data", "labels_db", "full_class_expanded_v2.parquet")
SCAFFOLD = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v3", "vivo", "all.csv")
SCAFFOLD_OUT = SCAFFOLD  # 같은 위치 update


def main():
    # 28개 0→1 만
    c = pd.read_csv(CONFLICTS)
    update = c[(c.db_label == 0) & (c.label == 1)].copy()
    print(f"=== Conflict update: {len(update)}개 (DB 0 → LiverTox 1) ===\n")
    print("Update list:")
    for _, r in update.iterrows():
        print(f"  {r['name']:25s}  inchi_key={r['inchi_key'][:14]}...")

    update_keys = set(update["inchi_key"])

    # DB update
    db = pd.read_parquet(DB_PATH)
    print(
        f"\n[DB before] vivo 양성: {(db.vivo_label == 1).sum()} / 음성: {(db.vivo_label == 0).sum()}"
    )
    mask = db["inchi_key"].isin(update_keys)
    print(f"매칭 DB rows: {mask.sum()}")
    db.loc[mask, "vivo_label"] = 1.0
    print(
        f"[DB after]  vivo 양성: {(db.vivo_label == 1).sum()} / 음성: {(db.vivo_label == 0).sum()}"
    )
    db.to_parquet(DB_OUT, index=False)
    print(f"저장: {DB_OUT}")

    # Scaffold split CSV update (label 만)
    if os.path.exists(SCAFFOLD):
        # InChIKey 가 all.csv 에 없음 → SMILES 매칭
        sc = pd.read_csv(SCAFFOLD)
        print(
            f"\n[Scaffold all.csv before] 양성 {(sc.label == 1).sum()} / 음성 {(sc.label == 0).sum()}"
        )
        # update 분자의 canonical_smiles 가져오기
        update_smis = set()
        for ikey in update_keys:
            sub = db[db.inchi_key == ikey]
            if len(sub) > 0:
                update_smis.add(sub.iloc[0]["canonical_smiles"])
        smi_mask = sc["canonical_smiles"].isin(update_smis)
        print(f"매칭 scaffold rows: {smi_mask.sum()}")
        sc.loc[smi_mask, "label"] = 1
        print(
            f"[Scaffold all.csv after]  양성 {(sc.label == 1).sum()} / 음성 {(sc.label == 0).sum()}"
        )
        sc.to_csv(SCAFFOLD_OUT, index=False)
        print(f"저장: {SCAFFOLD_OUT}")


if __name__ == "__main__":
    main()
