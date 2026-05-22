"""KEGG DRUG에서 시판약 SMILES 수집."""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.marketed_drugs._http import fetch_json, fetch_text

logger = logging.getLogger(__name__)

KEGG_LIST_URL = "https://rest.kegg.jp/list/drug"
KEGG_GET_URL = "https://rest.kegg.jp/get/dr:{}"
PUBCHEM_PROP_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/"
    "CanonicalSMILES,InChIKey/JSON"
)
RATE_LIMIT_DELAY = 0.34
# KEGG 전체 ~12000개 (각 약물마다 GET + PubChem fallback 호출)
# Default 1500개로 제한 (약 15-20분). 환경변수로 override 가능.
DEFAULT_LIMIT = int(os.environ.get("KEGG_LIMIT", "1500"))


def parse_kegg_list(text: str) -> list[dict]:
    """KEGG list/drug 응답 파싱."""
    rows = []
    for line in text.strip().splitlines():
        parts = line.split("\t")
        if len(parts) < 2:
            continue
        kegg_id = parts[0].replace("dr:", "")
        name = parts[1].strip()
        rows.append({"kegg_id": kegg_id, "name": name})
    return rows


def parse_kegg_entry(text: str) -> dict[str, Any]:
    """KEGG get 응답에서 ATC, PubChem CID 등 추출."""
    info: dict[str, Any] = {"atc_code": "", "pubchem_cid": ""}
    in_dblinks = False
    for line in text.splitlines():
        if line.startswith("ATC_CODE"):
            atc = line.split("ATC_CODE", 1)[1].strip()
            info["atc_code"] = ";".join(atc.split())
        elif line.startswith("DBLINKS"):
            in_dblinks = True
            content = line.split("DBLINKS", 1)[1].strip()
            if "PubChem:" in content:
                info["pubchem_cid"] = content.split("PubChem:", 1)[1].strip().split()[0]
        elif in_dblinks and line.startswith(" "):
            if "PubChem:" in line:
                info["pubchem_cid"] = line.split("PubChem:", 1)[1].strip().split()[0]
        else:
            in_dblinks = False
    return info


def _smiles_from_pubchem_cid(cid: str) -> str:
    """PubChem CID로 SMILES 조회 (KEGG MOL 미제공 시 폴백)."""
    if not cid:
        return ""
    try:
        data = fetch_json(PUBCHEM_PROP_URL.format(cid))
        props = data.get("PropertyTable", {}).get("Properties", [])
        if props:
            return props[0].get("CanonicalSMILES", "")
    except Exception as e:
        logger.warning("PubChem lookup 실패 (CID=%s): %s", cid, e)
    return ""


def collect_kegg(limit: int | None = None) -> pd.DataFrame:
    """KEGG DRUG 수집. limit=None이면 DEFAULT_LIMIT(1500) 사용."""
    if limit is None:
        limit = DEFAULT_LIMIT
    list_text = fetch_text(KEGG_LIST_URL)
    entries = parse_kegg_list(list_text)
    if limit and limit > 0:
        entries = entries[:limit]
    logger.info("KEGG list: %d entries", len(entries))

    rows: list[dict] = []
    pbar = tqdm(entries, desc="KEGG DRUG", unit="drug")
    for ent in pbar:
        try:
            entry_text = fetch_text(KEGG_GET_URL.format(ent["kegg_id"]))
            info = parse_kegg_entry(entry_text)
            smiles = _smiles_from_pubchem_cid(info.get("pubchem_cid", ""))
            if not smiles:
                continue
            rows.append(
                {
                    "kegg_id": ent["kegg_id"],
                    "name": ent["name"],
                    "smiles": smiles,
                    "inchi_key": "",
                    "atc_code": info.get("atc_code", ""),
                }
            )
        except Exception as e:
            logger.warning("KEGG %s 수집 실패: %s", ent["kegg_id"], e)
        time.sleep(RATE_LIMIT_DELAY)

    df = pd.DataFrame(rows)
    logger.info("KEGG: %d 약물 수집", len(df))
    return df
