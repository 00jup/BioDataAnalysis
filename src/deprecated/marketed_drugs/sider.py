"""SIDER (Side Effect Resource)에서 간 관련 부작용 가진 약물 추출."""

from __future__ import annotations

import gzip
import logging
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.marketed_drugs._http import fetch_bytes, fetch_json

logger = logging.getLogger(__name__)

SIDER_SE_URL = "http://sideeffects.embl.de/media/download/meddra_all_se.tsv.gz"
SIDER_DRUGS_URL = "http://sideeffects.embl.de/media/download/drug_names.tsv"

PUBCHEM_PROP_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{}/property/SMILES,InChIKey,Title/JSON"
)

# 간 관련 부작용 키워드 (대소문자 무시)
LIVER_KEYWORDS = [
    r"\bhepatic\b",
    r"\bhepatotox",
    r"\bhepatitis\b",
    r"\bhepatomegaly\b",
    r"\bhepatocellular\b",
    r"\bcholestat",
    r"\bcholestasis\b",
    r"\bjaundice\b",
    r"\bliver\b",
    r"\bbilirubin",
    r"\btransaminase",
    r"\balanine aminotransferase",
    r"\baspartate aminotransferase",
    r"\bgamma.?glutamyl",
    r"\balkaline phosphatase",
    r"\balt increased",
    r"\bast increased",
    r"\bggt increased",
    r"hepatorenal",
]
LIVER_RE = re.compile("|".join(LIVER_KEYWORDS), re.IGNORECASE)

# SIDER STITCH ID format: "CID" + flag(1 or 0) + 8-digit padded PubChem CID
# 예: "CID100002244" → flag=1(flat), CID=2244 (aspirin)
CID_PATTERN = re.compile(r"CID[01](\d{8})")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PUBCHEM_RATE_DELAY = 0.2
PUBCHEM_BATCH = 200


def _stitch_to_cid(stitch_id: str) -> int | None:
    """STITCH ID (CID0... or CID1...) → PubChem CID."""
    match = CID_PATTERN.match(stitch_id)
    if not match:
        return None
    return int(match.group(1))


def download_sider(out_dir: Path | None = None) -> tuple[Path, Path]:
    """SIDER 파일 다운로드 (있으면 캐시 사용)."""
    out_dir = out_dir or (PROJECT_ROOT / "data" / "marketed_drugs" / "reference" / "sider")
    out_dir.mkdir(parents=True, exist_ok=True)

    se_path = out_dir / "meddra_all_se.tsv"
    drugs_path = out_dir / "drug_names.tsv"

    if not se_path.exists():
        logger.info("SIDER side effects 다운로드 중...")
        gz_content = fetch_bytes(SIDER_SE_URL)
        decompressed = gzip.decompress(gz_content)
        se_path.write_bytes(decompressed)
    else:
        logger.info("SIDER 캐시 사용: %s", se_path)

    if not drugs_path.exists():
        logger.info("SIDER drug names 다운로드 중...")
        from src.marketed_drugs._http import fetch_text

        text = fetch_text(SIDER_DRUGS_URL)
        drugs_path.write_text(text)

    return se_path, drugs_path


def find_liver_drugs(se_path: Path, drugs_path: Path) -> pd.DataFrame:
    """간 관련 부작용 가진 약물의 PubChem CID 추출."""
    logger.info("SIDER 부작용 파싱: %s", se_path)
    se = pd.read_csv(
        se_path,
        sep="\t",
        header=None,
        names=["cid_flat", "cid_stereo", "umls_lt", "type", "umls_pt", "side_effect"],
        dtype=str,
    )
    logger.info("총 %d 부작용 entries", len(se))

    # PT (Preferred Term) 레벨만 사용 (LLT는 중복 많음)
    pt = se[se["type"] == "PT"].copy()

    # 간 관련 필터
    liver_mask = pt["side_effect"].fillna("").apply(lambda s: bool(LIVER_RE.search(s)))
    liver_pt = pt[liver_mask].copy()
    logger.info(
        "간 관련 PT entries: %d (고유 부작용 %d)",
        len(liver_pt),
        liver_pt["side_effect"].nunique(),
    )

    # 약물별 그룹: side effects 합쳐서 표시
    drug_terms = (
        liver_pt.groupby("cid_flat")["side_effect"]
        .apply(lambda x: ";".join(sorted(set(x))))
        .reset_index()
        .rename(columns={"side_effect": "sider_terms"})
    )

    # CID 변환
    drug_terms["cid"] = drug_terms["cid_flat"].apply(_stitch_to_cid)
    drug_terms = drug_terms[drug_terms["cid"].notna()].copy()
    drug_terms["cid"] = drug_terms["cid"].astype(int)

    # drug_names 조인
    drugs = pd.read_csv(
        drugs_path, sep="\t", header=None, names=["cid_flat", "drug_name"], dtype=str
    )
    drug_terms = drug_terms.merge(drugs, on="cid_flat", how="left")

    logger.info("간 부작용 가진 약물: %d", len(drug_terms))
    return drug_terms[["cid", "drug_name", "sider_terms"]].drop_duplicates(subset="cid")


def fetch_smiles_for_cids(cids: list[int]) -> pd.DataFrame:
    """PubChem batch로 CID → SMILES + InChIKey + Title."""
    rows = []
    pbar = tqdm(range(0, len(cids), PUBCHEM_BATCH), desc="PubChem SMILES", unit="batch")
    for i in pbar:
        batch = cids[i : i + PUBCHEM_BATCH]
        cid_str = ",".join(str(c) for c in batch)
        url = PUBCHEM_PROP_URL.format(cid_str)
        try:
            data = fetch_json(url)
            props = data.get("PropertyTable", {}).get("Properties", [])
            rows.extend(props)
        except Exception as e:
            logger.warning("PubChem batch 실패 (CIDs %s): %s", batch[0], e)
        time.sleep(PUBCHEM_RATE_DELAY)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    smiles_col = (
        df["SMILES"]
        if "SMILES" in df.columns
        else df["ConnectivitySMILES"]
        if "ConnectivitySMILES" in df.columns
        else pd.Series([""] * len(df))
    )
    return pd.DataFrame(
        {
            "cid": df["CID"].astype(int),
            "name_pubchem": df["Title"] if "Title" in df.columns else "",
            "smiles": smiles_col.fillna("").astype(str),
            "inchi_key": df["InChIKey"].fillna("").astype(str) if "InChIKey" in df.columns else "",
        }
    )


def collect_sider_hepatotoxic(
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """SIDER에서 간독성 약물 + SMILES 추출하여 저장."""
    se_path, drugs_path = download_sider()
    liver_drugs = find_liver_drugs(se_path, drugs_path)

    cids = liver_drugs["cid"].tolist()
    logger.info("PubChem에서 SMILES 조회: %d CIDs", len(cids))
    smiles_df = fetch_smiles_for_cids(cids)

    if smiles_df.empty:
        logger.error("SMILES 조회 실패")
        return pd.DataFrame()

    merged = liver_drugs.merge(smiles_df, on="cid", how="left")
    merged = merged[merged["smiles"].notna() & (merged["smiles"] != "")].copy()

    # canonical SMILES 변환 + dedup
    from src.marketed_drugs.merge import canonicalize_smiles, smiles_to_inchikey

    merged["canonical_smiles"] = merged["smiles"].apply(canonicalize_smiles)
    merged = merged[merged["canonical_smiles"].notna()].copy()
    merged["inchi_key"] = merged.apply(
        lambda r: r["inchi_key"] if r["inchi_key"] else smiles_to_inchikey(r["canonical_smiles"]),
        axis=1,
    )
    merged = merged.drop_duplicates(subset="canonical_smiles").reset_index(drop=True)

    out_df = pd.DataFrame(
        {
            "name": merged["name_pubchem"].fillna(merged["drug_name"]),
            "smiles": merged["smiles"],
            "canonical_smiles": merged["canonical_smiles"],
            "inchi_key": merged["inchi_key"],
            "sources": "sider",
            "source_ids": merged["cid"].astype(str),
            "sider_terms": merged["sider_terms"],
            "in_dilirank": 0,
            "dilirank_category": "unknown",
            "in_tdc_dili_pos": 0,
            "known_hepatotoxic": 1,
        }
    )

    if output_csv:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        out_df.to_csv(output_csv, index=False)
        logger.info("SIDER hepatotoxic 저장: %s (%d개)", output_csv, len(out_df))

    return out_df


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    out = PROJECT_ROOT / "data" / "marketed_drugs" / "hepatotoxic" / "sider_hepatotoxic.csv"
    df = collect_sider_hepatotoxic(out)
    return 0 if not df.empty else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
