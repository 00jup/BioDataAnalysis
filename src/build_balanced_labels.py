"""양성/음성 균형 라벨 — Strict 와 Sensitive 사이.

양성 조건 (강한 + 중간 신호):
  - DILIrank vMost 또는 vLess
  - LiverTox A/B/C
  - DailyMed boxed/warning (adverse 제외)
  - PubMed strong (≥20)
  - CTD strong (marker/mechanism)

음성 조건:
  - DILIrank vNo
  - LiverTox E
  - marketed_clean_neg (시판 안전 약)
  - 약한 신호만 (LiverTox D, DailyMed adverse, PubMed weak/medium, CTD medium/weak, FAERS)
"""
from __future__ import annotations
import os
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
OUT_DB = os.path.join(PROJECT_ROOT, "data", "labels_db", "full_balanced.parquet")
OUT_DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_balanced")


def label(row):
    # 양성 — 강한+중간 신호
    is_pos = (
        row.get("vivo_dilirank") in ("vMost-DILI-Concern", "vLess-DILI-Concern")
        or row.get("vivo_livertox") in ("A", "B", "C")
        or row.get("vivo_dailymed") in ("boxed_hepatotox", "warning_hepatotox")
        or row.get("vivo_pubmed") == "strong"
        or row.get("vivo_ctd") == "strong"
    )
    if is_pos:
        return 1
    # 음성 — 명시적 안전
    is_neg = (
        row.get("vivo_dilirank") == "vNo-DILI-Concern"
        or row.get("vivo_livertox") == "E"
        or (pd.notna(row.get("vivo_marketed_clean_neg")) and row.get("vivo_marketed_clean_neg") == 1)
    )
    if is_neg:
        return 0
    # 약한 신호 → 음성 (안전 약 학습 의도)
    has_weak = (
        row.get("vivo_livertox") == "D"
        or row.get("vivo_dailymed") in ("adverse_hepatotox", "contraindication_hepatic")
        or row.get("vivo_pubmed") in ("weak", "medium")
        or row.get("vivo_ctd") in ("medium", "weak")
        or pd.notna(row.get("vivo_faers"))
    )
    if has_weak:
        return 0
    if row.get("vivo_dilirank") in ("vAmbig-DILI-Concern", "Ambiguous-DILI-Concern"):
        return None
    return None


def main():
    db = pd.read_parquet(DB_PATH)
    db["vivo_balanced"] = db.apply(label, axis=1)
    pos = (db.vivo_balanced == 1).sum()
    neg = (db.vivo_balanced == 0).sum()
    none = db.vivo_balanced.isna().sum()
    print(f"DB: {len(db)}")
    print(f"Balanced 라벨:")
    print(f"  양성: {pos}")
    print(f"  음성: {neg}")
    print(f"  None: {none}")
    print(f"  비율 1:{neg/max(pos,1):.2f}")
    db.to_parquet(OUT_DB, index=False)

    os.makedirs(os.path.join(OUT_DATA, "vivo"), exist_ok=True)
    sub = db[db.vivo_balanced.notna()][["canonical_smiles", "vivo_balanced"]].rename(
        columns={"vivo_balanced": "label"})
    sub["label"] = sub["label"].astype(int)
    sub = sub.dropna().drop_duplicates(subset=["canonical_smiles"])
    out_csv = os.path.join(OUT_DATA, "vivo", "all.csv")
    sub.to_csv(out_csv, index=False)
    print(f"\nvivo all.csv: {len(sub)} 분자 (양성 {(sub.label==1).sum()})")
    print(f"  → {out_csv}")


if __name__ == "__main__":
    main()
