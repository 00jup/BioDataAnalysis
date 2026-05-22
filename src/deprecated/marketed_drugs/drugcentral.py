"""DrugCentral CSV dump에서 시판약 SMILES 수집."""

from __future__ import annotations

import io
import logging

import pandas as pd

from src.marketed_drugs._http import fetch_text

logger = logging.getLogger(__name__)

DRUGCENTRAL_URL = "https://unmtid-dbs.net/download/DrugCentral/2021_09_01/structures.smiles.tsv"


def collect_drugcentral() -> pd.DataFrame:
    """DrugCentral structures.smiles.tsv 다운로드 후 파싱."""
    text = fetch_text(DRUGCENTRAL_URL)
    df_raw = pd.read_csv(io.StringIO(text), sep="\t")

    df = pd.DataFrame(
        {
            "drugcentral_id": df_raw["ID"].astype(str),
            "name": df_raw.get("INN", df_raw.get("name", "")).astype(str).str.strip(),
            "smiles": df_raw["SMILES"].astype(str),
            "inchi_key": df_raw.get("InChIKey", "").astype(str),
        }
    )
    df = df[df["smiles"].notna() & (df["smiles"] != "") & (df["smiles"] != "nan")]
    logger.info("DrugCentral: %d 약물 수집", len(df))
    return df.reset_index(drop=True)
