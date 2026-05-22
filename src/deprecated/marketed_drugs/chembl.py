"""ChEMBL phase 4 (시판 승인 약물) 수집."""

from __future__ import annotations

import logging
import time

import pandas as pd
from tqdm import tqdm

from src.marketed_drugs._http import fetch_json

logger = logging.getLogger(__name__)

CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data/molecule.json"
PAGE_SIZE = 1000
RATE_LIMIT_DELAY = 1.0


def collect_chembl() -> pd.DataFrame:
    """ChEMBL phase 4 약물 전체를 페이징으로 수집."""
    rows: list[dict] = []
    offset = 0
    total = None

    pbar = tqdm(desc="ChEMBL phase 4", unit="drug")
    while True:
        params = {"max_phase": 4, "limit": PAGE_SIZE, "offset": offset}
        data = fetch_json(CHEMBL_API, params=params)

        molecules = data.get("molecules", [])
        if not molecules:
            break

        if total is None:
            total = data.get("page_meta", {}).get("total_count")
            if total:
                pbar.total = total

        for m in molecules:
            structures = m.get("molecule_structures") or {}
            smiles = structures.get("canonical_smiles") if isinstance(structures, dict) else None
            if not smiles:
                continue
            rows.append(
                {
                    "chembl_id": m.get("molecule_chembl_id"),
                    "pref_name": m.get("pref_name"),
                    "smiles": smiles,
                    "max_phase": m.get("max_phase"),
                    "withdrawn_flag": bool(m.get("withdrawn_flag", False)),
                }
            )
            pbar.update(1)

        next_url = data.get("page_meta", {}).get("next")
        if not next_url:
            break

        offset += PAGE_SIZE
        time.sleep(RATE_LIMIT_DELAY)

    pbar.close()
    df = pd.DataFrame(rows)
    logger.info("ChEMBL: %d 약물 수집", len(df))
    return df
