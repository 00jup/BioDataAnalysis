"""Strict 라벨 재정의 — 진짜 강한 vivo DILI 만 양성.

기존: 어떤 hepatic 신호도 양성 (sensitive)
신규: 강한 임상 신호만 양성 (specific)

양성 조건 (OR):
  - DILIrank vMost-DILI-Concern
  - LiverTox A/B
  - DailyMed boxed_hepatotox
  - PubMed strong (≥20 papers)
  - CTD strong (marker/mechanism direct evidence)

음성 조건 (양성 아니고 OR):
  - DILIrank vNo-DILI-Concern
  - LiverTox E
  - DailyMed contraindication 만 또는 신호 없음
  - 다른 출처 약한/없음

Ambiguous (vAmbig 등): 제외

산출:
  data/labels_db/full_strict.parquet
  data/chemprop_scaffold_strict/vivo/all.csv
"""
from __future__ import annotations
import os, json
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
OUT_DB = os.path.join(PROJECT_ROOT, "data", "labels_db", "full_strict.parquet")
OUT_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_strict")


def vivo_strict_label(row):
    """Strict 양성: 진짜 강한 DILI 신호만."""
    # 양성 조건
    is_pos = (
        row.get("vivo_dilirank") == "vMost-DILI-Concern"
        or row.get("vivo_livertox") in ("A", "B")
        or row.get("vivo_dailymed") == "boxed_hepatotox"
        or row.get("vivo_pubmed") == "strong"
        or row.get("vivo_ctd") == "strong"
    )
    if is_pos:
        return 1
    # 명시적 음성 조건
    is_neg = (
        row.get("vivo_dilirank") == "vNo-DILI-Concern"
        or row.get("vivo_livertox") == "E"
        or (pd.notna(row.get("vivo_marketed_clean_neg")) and row.get("vivo_marketed_clean_neg") == 1)
    )
    if is_neg:
        return 0
    # 약한 신호만 있으면 음성 (sanity check 안전 약물 의도)
    has_weak_signal = (
        row.get("vivo_dilirank") in ("vLess-DILI-Concern",)
        or row.get("vivo_livertox") in ("C", "D")
        or row.get("vivo_dailymed") in ("warning_hepatotox", "adverse_hepatotox",
                                          "contraindication_hepatic")
        or row.get("vivo_pubmed") in ("medium", "weak")
        or row.get("vivo_ctd") in ("medium", "weak")
        or pd.notna(row.get("vivo_faers"))  # FAERS — 환자 보고만으로는 약함
    )
    if has_weak_signal:
        return 0  # 약한 신호 → 음성 (specific 정의)
    # Ambiguous → None
    if row.get("vivo_dilirank") in ("vAmbig-DILI-Concern", "Ambiguous-DILI-Concern"):
        return None
    return None  # 신호 없음 → 라벨 없음


def main():
    db = pd.read_parquet(DB_PATH)
    print(f"DB: {len(db)} 분자")

    db["vivo_strict"] = db.apply(vivo_strict_label, axis=1)
    pos = (db.vivo_strict == 1).sum()
    neg = (db.vivo_strict == 0).sum()
    none = db.vivo_strict.isna().sum()
    print(f"\nStrict 라벨:")
    print(f"  양성 (vMost / LiverTox A/B / boxed / CTD strong / PubMed strong): {pos}")
    print(f"  음성 (vNo / LiverTox E / marketed_clean / 약한 신호): {neg}")
    print(f"  None (ambiguous / 신호 없음): {none}")
    print(f"  사용 가능: {pos + neg} (비율 1:{neg/max(pos,1):.2f})")

    # 양성 출처 분포
    pos_db = db[db.vivo_strict == 1]
    print(f"\n양성 출처별 분포 (다중 가능):")
    print(f"  DILIrank vMost: {(pos_db.vivo_dilirank == 'vMost-DILI-Concern').sum()}")
    print(f"  LiverTox A: {(pos_db.vivo_livertox == 'A').sum()}")
    print(f"  LiverTox B: {(pos_db.vivo_livertox == 'B').sum()}")
    print(f"  DailyMed boxed: {(pos_db.vivo_dailymed == 'boxed_hepatotox').sum()}")
    print(f"  PubMed strong: {(pos_db.vivo_pubmed == 'strong').sum()}")
    print(f"  CTD strong: {(pos_db.vivo_ctd == 'strong').sum()}")

    db.to_parquet(OUT_DB, index=False)
    print(f"\n저장: {OUT_DB}")

    # scaffold split 용 data
    os.makedirs(os.path.join(OUT_DATA, "vivo"), exist_ok=True)
    strict_df = db[db.vivo_strict.notna()][["canonical_smiles", "vivo_strict"]].rename(
        columns={"vivo_strict": "label"})
    strict_df["label"] = strict_df["label"].astype(int)
    strict_df = strict_df.dropna().drop_duplicates(subset=["canonical_smiles"])
    out_csv = os.path.join(OUT_DATA, "vivo", "all.csv")
    strict_df.to_csv(out_csv, index=False)
    print(f"\nvivo all.csv: {len(strict_df)} 분자 (양성 {(strict_df.label==1).sum()})")
    print(f"  → {out_csv}")


if __name__ == "__main__":
    main()
