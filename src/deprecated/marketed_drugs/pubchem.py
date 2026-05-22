"""PubChem FDA Approved Drugs 분류 수집."""

from __future__ import annotations

import logging
import time
from urllib.parse import quote

import pandas as pd

from src.marketed_drugs._http import fetch_json

logger = logging.getLogger(__name__)

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
# DrugBank source — ~13.5k entries (approved/investigational/etc)
# (GSRS는 168k로 너무 큼. NINDS는 1k로 작음. DrugBank는 균형이 좋음)
SOURCE_NAME = "DrugBank"
RATE_LIMIT_DELAY = 0.2
BATCH_SIZE = 200


def _fetch_cids() -> list[int]:
    """GSRS substance source에서 CID 리스트 조회 (SID→CID 매핑 평탄화)."""
    url = f"{PUBCHEM_BASE}/substance/sourceall/{quote(SOURCE_NAME)}/cids/JSON"
    data = fetch_json(url)
    info_list = data.get("InformationList", {}).get("Information", [])
    cids: set[int] = set()
    for entry in info_list:
        for cid in entry.get("CID", []) or []:
            if cid:
                cids.add(int(cid))
    return sorted(cids)


def _fetch_properties(cids: list[int]) -> list[dict]:
    """CID 배치별로 SMILES + InChIKey + Title 조회.

    PubChem 2024+에서 CanonicalSMILES/IsomericSMILES 폐기되고 SMILES로 통합.
    """
    rows: list[dict] = []
    for i in range(0, len(cids), BATCH_SIZE):
        batch = cids[i : i + BATCH_SIZE]
        cid_str = ",".join(str(c) for c in batch)
        url = f"{PUBCHEM_BASE}/compound/cid/{cid_str}/property/Title,SMILES,InChIKey/JSON"
        data = fetch_json(url)
        props = data.get("PropertyTable", {}).get("Properties", [])
        rows.extend(props)
        time.sleep(RATE_LIMIT_DELAY)
    return rows


def collect_pubchem() -> pd.DataFrame:
    """PubChem DrugBank source 전체 수집."""
    cids = _fetch_cids()
    logger.info("PubChem: %d CID 발견", len(cids))

    props = _fetch_properties(cids)

    df = pd.DataFrame(
        [
            {
                "cid": p.get("CID"),
                "name": p.get("Title", ""),
                "smiles": p.get("SMILES") or p.get("ConnectivitySMILES") or "",
                "inchi_key": p.get("InChIKey", ""),
            }
            for p in props
        ]
    )
    if not df.empty:
        df = df[df["smiles"].notna() & (df["smiles"] != "")]
    logger.info("PubChem: %d 약물 수집", len(df))
    return df.reset_index(drop=True)
