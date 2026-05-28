"""Class expansion 데이터 통합 → 학습 DB 에 신규 분자 추가.

Step:
  1. data/class_expansion/*.csv 모두 로드
  2. canonical_smiles 표준화 (RDKit MolStandardize)
  3. InChIKey 생성
  4. 학습 DB (full.parquet) 와 비교 → 신규만 추출
  5. 기존 분자는 label 보강 (학습 label != class_expansion label 인 경우 conflict)
  6. labels_db 에 통합 → full_class_expanded.parquet

산출물:
  data/labels_db/full_class_expanded.parquet
  data/labels_db/class_expansion_added.csv (신규)
  data/labels_db/class_expansion_conflicts.csv (label 불일치)
"""

from __future__ import annotations

import glob
import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPANSION_DIR = os.path.join(PROJECT_ROOT, "data", "class_expansion")
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
OUT_DB = os.path.join(PROJECT_ROOT, "data", "labels_db", "full_class_expanded.parquet")
ADDED_CSV = os.path.join(PROJECT_ROOT, "data", "labels_db", "class_expansion_added.csv")
CONFLICT_CSV = os.path.join(PROJECT_ROOT, "data", "labels_db", "class_expansion_conflicts.csv")

# Standardization chain
_normalizer = rdMolStandardize.Normalizer()
_uncharger = rdMolStandardize.Uncharger()
_lfc = rdMolStandardize.LargestFragmentChooser()


def standardize(smi: str) -> tuple[str | None, str | None]:
    """SMILES → (canonical SMILES, InChIKey)."""
    if not smi or not isinstance(smi, str):
        return None, None
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None, None
        m = _normalizer.normalize(m)
        m = _lfc.choose(m)
        m = _uncharger.uncharge(m)
        canon = Chem.MolToSmiles(m, canonical=True)
        ikey = Chem.MolToInchiKey(m)
        return canon, ikey
    except Exception:
        return None, None


def main():
    print("=== Class expansion 통합 ===\n")

    # Step 1: 모든 expansion CSV 로드
    files = sorted(glob.glob(os.path.join(EXPANSION_DIR, "*.csv")))
    print(f"발견 CSV: {len(files)}")
    for f in files:
        print(f"  {os.path.basename(f)}")

    dfs = []
    for f in files:
        # csv module 로 직접 (quote 처리)
        try:
            df = pd.read_csv(f, engine="python", quoting=0)  # QUOTE_MINIMAL
        except Exception:
            # 더 robust 한 방법: 첫 6개 컬럼만 split, 나머지는 reasoning 으로 join
            import csv

            rows = []
            with open(f) as fp:
                reader = csv.reader(fp)
                header = next(reader)
                for r in reader:
                    if len(r) > 7:
                        # reasoning 컬럼 (마지막) 에 쉼표 → join
                        r = r[:6] + [",".join(r[6:])]
                    elif len(r) < 7:
                        # 빈 reasoning?
                        r = r + [""] * (7 - len(r))
                    rows.append(r)
            df = pd.DataFrame(rows, columns=header)
        df["source_file"] = os.path.basename(f)
        dfs.append(df)
        print(f"  {os.path.basename(f)}: {len(df)} rows")

    if not dfs:
        print("CSV 없음.")
        return

    combined = pd.concat(dfs, ignore_index=True)
    print(f"\n총 expansion rows: {len(combined)}")

    # Step 2: canonical_smiles 컬럼 정리
    smi_col = None
    for c in ("canonical_smiles", "smiles", "SMILES"):
        if c in combined.columns:
            smi_col = c
            break
    if not smi_col:
        print("ERROR: SMILES 컬럼 없음.")
        return
    print(f"SMILES 컬럼: {smi_col}")

    # Step 3: standardize
    print("\n표준화 중...")
    out = combined[smi_col].apply(standardize)
    combined["canon"] = [t[0] for t in out]
    combined["inchi_key"] = [t[1] for t in out]
    n_valid = combined["canon"].notna().sum()
    print(f"  유효 SMILES: {n_valid} / {len(combined)}")
    combined = combined.dropna(subset=["canon", "inchi_key"]).reset_index(drop=True)

    # label 정리
    if "label" not in combined.columns:
        print("ERROR: label 컬럼 없음.")
        return
    # 0/1 만 유지 (uncertain 제외)
    combined["label"] = pd.to_numeric(combined["label"], errors="coerce")
    valid = combined["label"].isin([0, 1])
    print(f"  0/1 label 유효: {valid.sum()} / {len(combined)} (uncertain 제외)")
    combined = combined[valid].copy()
    combined["label"] = combined["label"].astype(int)

    # InChIKey duplicate (같은 분자, source 다른 경우 → majority vote)
    print(f"\nInChIKey deduplication ({len(combined)})")
    agg = (
        combined.groupby("inchi_key")
        .agg(
            name=("name", "first"),
            canonical_smiles=("canon", "first"),
            label=("label", lambda x: int(x.mean() >= 0.5)),
            n_sources=("source_file", "nunique"),
            sources=("source_file", lambda x: ";".join(sorted(set(x)))),
            references=(
                "reference_url" if "reference_url" in combined.columns else "name",
                lambda x: ";".join(str(v) for v in x if pd.notna(v))[:500],
            ),
        )
        .reset_index()
    )
    print(f"  unique InChIKey: {len(agg)}")

    # Step 4: 학습 DB 와 비교
    db = pd.read_parquet(DB_PATH)
    print(f"\n학습 DB: {len(db)} unique molecules")
    print(f"  vivo labeled: {db.vivo_label.notna().sum()}")
    db_keys = set(db["inchi_key"])
    print(f"  unique InChIKey in DB: {len(db_keys)}")

    # Step 5: 신규 vs 기존 분류
    agg["in_db"] = agg["inchi_key"].isin(db_keys)
    new_mols = agg[~agg["in_db"]].copy()
    existing = agg[agg["in_db"]].copy()
    print(f"\n신규 분자 (DB 에 없음): {len(new_mols)}")
    print(f"  양성: {(new_mols.label == 1).sum()}, 음성: {(new_mols.label == 0).sum()}")
    print(f"기존 분자 (DB 에 있음): {len(existing)}")

    # 기존 분자 label conflict 체크
    existing_merged = existing.merge(db[["inchi_key", "vivo_label"]], on="inchi_key", how="left")
    existing_merged["db_label"] = existing_merged["vivo_label"]
    conflict = existing_merged[
        existing_merged["db_label"].notna()
        & (existing_merged["label"] != existing_merged["db_label"])
    ].copy()
    print(f"  Label 불일치 (DB vs expansion): {len(conflict)}")

    # Save 신규
    new_mols[
        ["name", "canonical_smiles", "inchi_key", "label", "n_sources", "sources", "references"]
    ].to_csv(ADDED_CSV, index=False)
    print(f"\n저장: {ADDED_CSV}")

    # Save 불일치
    if len(conflict) > 0:
        conflict[
            ["name", "canonical_smiles", "inchi_key", "label", "db_label", "sources", "references"]
        ].to_csv(CONFLICT_CSV, index=False)
        print(f"저장: {CONFLICT_CSV}")

    # Step 6: 통합 DB 만들기
    # 신규 분자만 추가 (label = vivo_label 으로 설정)
    add_rows = new_mols[["name", "canonical_smiles", "inchi_key", "label"]].copy()
    add_rows["vivo_label"] = add_rows["label"].astype(float)
    add_rows["vitro_label"] = None
    add_rows["source"] = "class_expansion"

    # DB columns alignment
    common_cols = [c for c in db.columns if c in add_rows.columns]
    print(f"\n공통 컬럼: {common_cols}")
    for c in db.columns:
        if c not in add_rows.columns:
            add_rows[c] = None
    add_rows = add_rows[db.columns]

    db_new = pd.concat([db, add_rows], ignore_index=True)
    print(f"\n새 DB: {len(db_new)} (원본 {len(db)} + 신규 {len(add_rows)})")
    print(f"  vivo labeled: {db_new.vivo_label.notna().sum()}")
    print(f"  양성: {(db_new.vivo_label == 1).sum()}, 음성: {(db_new.vivo_label == 0).sum()}")

    db_new.to_parquet(OUT_DB, index=False)
    print(f"\n저장: {OUT_DB}")


if __name__ == "__main__":
    main()
