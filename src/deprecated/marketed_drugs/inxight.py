"""NCATS Inxight Drugs에서 marketed 약물 수집."""

from __future__ import annotations

import logging
import time

import pandas as pd
from tqdm import tqdm

from src.marketed_drugs._http import fetch_json

logger = logging.getLogger(__name__)

INXIGHT_API = "https://drugs.ncats.io/api/v1/substances"
PAGE_SIZE = 100
RATE_LIMIT_DELAY = 0.5


def collect_inxight() -> pd.DataFrame:
    """Inxight substances API에서 marketed=Marketed인 항목 수집."""
    rows: list[dict] = []
    skip = 0
    pbar = tqdm(desc="Inxight Marketed", unit="drug")

    while True:
        params = {
            "filter": 'marketingStatus.includes("Marketed")',
            "top": PAGE_SIZE,
            "skip": skip,
        }
        data = fetch_json(INXIGHT_API, params=params)
        content = data.get("content", [])
        if not content:
            break

        if pbar.total is None:
            pbar.total = data.get("total", 0)

        for item in content:
            structure = item.get("structure") or {}
            smiles = structure.get("smiles", "") if isinstance(structure, dict) else ""
            if not smiles:
                continue
            rows.append(
                {
                    "unii": item.get("approvalID", ""),
                    "name": item.get("_name", ""),
                    "smiles": smiles,
                    "inchi_key": (
                        structure.get("stdInchiKey", "") if isinstance(structure, dict) else ""
                    ),
                    "marketing_status": ";".join(item.get("marketingStatus", [])),
                }
            )
            pbar.update(1)

        if len(content) < PAGE_SIZE:
            break
        skip += PAGE_SIZE
        time.sleep(RATE_LIMIT_DELAY)

    pbar.close()
    df = pd.DataFrame(rows)
    logger.info("Inxight: %d 약물 수집", len(df))
    return df
