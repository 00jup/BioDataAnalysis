"""DrugBank Open Data 어휘 (Tier 2 enrichment, SMILES 미제공)."""

from __future__ import annotations

import io
import logging
import zipfile

import pandas as pd

from src.marketed_drugs._http import fetch_bytes, fetch_text

logger = logging.getLogger(__name__)

DRUGBANK_OPEN_DATA_URL = "https://go.drugbank.com/releases/latest/downloads/all-drugbank-vocabulary"


def _read_csv_from_zip_or_text(content: bytes | str) -> pd.DataFrame:
    """ZIP이면 내부 CSV 추출, 일반 텍스트면 그대로 파싱."""
    if isinstance(content, bytes):
        # ZIP signature check
        if content[:2] == b"PK":
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not csv_names:
                    raise ValueError("ZIP 안에 CSV 없음")
                with zf.open(csv_names[0]) as f:
                    return pd.read_csv(f)
        return pd.read_csv(io.BytesIO(content))
    return pd.read_csv(io.StringIO(content))


def collect_drugbank() -> pd.DataFrame:
    """DrugBank Open Data Vocabulary 수집 (ZIP 처리, SMILES 없음, InChIKey 보강)."""
    try:
        content = fetch_bytes(DRUGBANK_OPEN_DATA_URL)
        df_raw = _read_csv_from_zip_or_text(content)
    except Exception as e:
        # 폴백: 텍스트 시도
        logger.warning("DrugBank ZIP 처리 실패 (%s), 텍스트 시도", e)
        text = fetch_text(DRUGBANK_OPEN_DATA_URL)
        df_raw = _read_csv_from_zip_or_text(text)

    df = pd.DataFrame(
        {
            "drugbank_id": df_raw["DrugBank ID"].astype(str),
            "name": df_raw["Common name"].astype(str),
            "smiles": "",
            "inchi_key": df_raw.get("Standard InChI Key", "").astype(str),
            "unii": df_raw.get("UNII", "").astype(str),
        }
    )
    logger.info("DrugBank: %d 항목 (enrichment 전용, SMILES 미제공)", len(df))
    return df
