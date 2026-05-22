"""CLI 진입점: 전체 파이프라인 orchestration."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

from src.marketed_drugs import (
    chembl,
    drugbank,
    drugcentral,
    flag,
    inxight,
    kegg,
    merge,
    pubchem,
    split,
    validate,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "marketed_drugs"
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
REFERENCE_DIR = DATA_ROOT / "reference"


def _ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, REFERENCE_DIR):
        d.mkdir(parents=True, exist_ok=True)


SOURCES = {
    "chembl": chembl.collect_chembl,
    "drugcentral": drugcentral.collect_drugcentral,
    "pubchem": pubchem.collect_pubchem,
    "inxight": inxight.collect_inxight,
    "kegg": kegg.collect_kegg,
    "drugbank": drugbank.collect_drugbank,
}

# 환경변수로 일부 소스 비활성화 가능 (예: SKIP_SOURCES=kegg,inxight)
SKIPPED = set(s.strip().lower() for s in os.environ.get("SKIP_SOURCES", "").split(",") if s.strip())


def collect_one(name: str, force: bool = False) -> pd.DataFrame | None:
    """단일 소스 수집 (캐시 활용)."""
    cache_path = RAW_DIR / f"{name}.csv"
    if cache_path.exists() and not force:
        cached = pd.read_csv(cache_path)
        if not cached.empty:
            logger.info("%s: 캐시 사용 (%s, %d개)", name, cache_path, len(cached))
            return cached
        logger.warning("%s: 빈 캐시 무시 후 재수집", name)
    try:
        df = SOURCES[name]()
        if df is None or df.empty:
            logger.warning("%s: 빈 결과 — 캐시 저장 skip", name)
            return None
        df.to_csv(cache_path, index=False)
        return df
    except Exception as e:
        logger.error("%s 수집 실패: %s", name, e)
        return None


def run_pipeline(skip_cache: bool = False, only_source: str | None = None) -> int:
    _ensure_dirs()

    raw_by_source: dict[str, pd.DataFrame] = {}
    sources_to_run = [only_source] if only_source else list(SOURCES.keys())
    for name in sources_to_run:
        if name in SKIPPED:
            logger.info("%s: SKIP_SOURCES 환경변수에 의해 skip", name)
            continue
        df = collect_one(name, force=skip_cache)
        if df is not None and not df.empty:
            raw_by_source[name] = df

    if only_source:
        logger.info("단일 소스 모드 종료")
        return 0

    tier1_sources = {"chembl", "drugcentral", "pubchem", "inxight", "kegg"}
    successful = sum(1 for n in raw_by_source if n in tier1_sources)
    if successful < 3:
        logger.error("Tier 1 소스 %d개 성공 (최소 3개 필요)", successful)
        return 1

    merged_path = PROCESSED_DIR / "merged.csv"
    if merged_path.exists() and not skip_cache:
        merged = pd.read_csv(merged_path)
        logger.info("Merge 캐시 사용")
    else:
        merged = merge.merge_sources(raw_by_source)
        chembl_w = raw_by_source.get("chembl")
        drugbank_w = raw_by_source.get("drugbank")
        if drugbank_w is not None and not drugbank_w.empty:
            drugbank_w = drugbank_w.assign(
                withdrawn=drugbank_w["name"].astype(str).str.lower().str.contains("withdrawn")
            )
        merged, removed = merge.filter_withdrawn(
            merged, chembl_withdrawn=chembl_w, drugbank_status=drugbank_w
        )
        merged.to_csv(merged_path, index=False)
        removed.to_csv(PROCESSED_DIR / "withdrawn_filtered.csv", index=False)

    dilirank_path = REFERENCE_DIR / "dilirank.csv"
    if not dilirank_path.exists():
        logger.warning("DILIrank CSV 없음 → 빈 DF 사용. %s에 수동 다운로드 필요.", dilirank_path)
        dilirank_df = pd.DataFrame(
            columns=["LTKBID", "Compound Name", "Severity Class", "InChIKey"]
        )
    else:
        dilirank_df = flag.load_dilirank(str(dilirank_path))

    tdc_train_path = PROJECT_ROOT / "data" / "train" / "dili_train.csv"
    tdc_pos = flag.load_tdc_pos_inchikeys(str(tdc_train_path)) if tdc_train_path.exists() else set()
    flagged = flag.flag_hepatotoxic(merged, dilirank_df, tdc_pos)

    split.split_by_hepatotoxic(flagged, DATA_ROOT)

    ok = validate.validate(DATA_ROOT)
    return 0 if ok else 2


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="시판 약물 수집 파이프라인")
    parser.add_argument("--skip-cache", action="store_true", help="raw 캐시 무시")
    parser.add_argument("--source", choices=list(SOURCES.keys()), help="단일 소스만 실행")
    args = parser.parse_args()
    return run_pipeline(skip_cache=args.skip_cache, only_source=args.source)


if __name__ == "__main__":
    sys.exit(main())
