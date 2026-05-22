"""DILIrank 2.0 다운로드 + 이름 매칭으로 InChIKey 채움."""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.marketed_drugs._http import fetch_bytes, fetch_json

logger = logging.getLogger(__name__)

# FDA DILIrank 2.0 직접 다운로드 (Excel)
DILIRANK_FDA_URL = "https://www.fda.gov/media/113052/download?attachment"

# PubChem name → property
PUBCHEM_NAME_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{}/property/InChIKey,SMILES/JSON"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _normalize_name(name: str) -> str:
    """약물명 정규화 (lowercase, salt/hydrate 제거, whitespace 정리)."""
    if not isinstance(name, str):
        return ""
    n = name.lower().strip()
    # 흔한 salt/형태 접미사 제거
    suffixes = [
        " hydrochloride",
        " sulfate",
        " sodium",
        " potassium",
        " calcium",
        " phosphate",
        " maleate",
        " citrate",
        " fumarate",
        " hydrobromide",
        " mesylate",
        " tartrate",
        " acetate",
        " hcl",
    ]
    for s in suffixes:
        if n.endswith(s):
            n = n[: -len(s)].strip()
            break
    # 괄호 안 내용 제거
    n = re.sub(r"\s*\([^)]*\)\s*", " ", n).strip()
    return n


def download_dilirank_excel(local_path: Path | None = None) -> pd.DataFrame:
    """FDA에서 DILIrank 2.0 Excel 다운로드 + 파싱.

    local_path가 주어지면 다운로드 대신 로컬 파일 사용 (FDA rate limit 회피).
    """
    import io

    if local_path and Path(local_path).exists():
        logger.info("DILIrank 로컬 파일 사용: %s", local_path)
        with open(local_path, "rb") as f:
            content = f.read()
    else:
        logger.info("DILIrank 2.0 Excel 다운로드 중...")
        content = fetch_bytes(DILIRANK_FDA_URL)
    df = pd.read_excel(io.BytesIO(content), header=1)
    # 컬럼 정규화
    df.columns = [c.strip() for c in df.columns]
    logger.info("DILIrank 다운로드: %d entries", len(df))

    # vDILI-Concern 표기 통일 (vMOST → vMost 등)
    def normalize_class(s: str) -> str:
        if not isinstance(s, str):
            return "unknown"
        s = s.strip()
        # "vMOST-DILI-concern" → "vMost-DILI-Concern"
        s = re.sub(r"vMOST", "vMost", s)
        s = re.sub(r"-concern$", "-Concern", s)
        return s

    df["Severity Class"] = df["vDILI-Concern"].apply(normalize_class)
    df = df.rename(columns={"CompoundName": "Compound Name"})
    return df[["LTKBID", "Compound Name", "Severity Class"]].copy()


def match_inchikeys_by_name(
    dilirank_df: pd.DataFrame, marketed_all_path: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """marketed_all.csv의 (name, inchi_key) 매핑으로 InChIKey 채움.

    반환: (matched_df, unmatched_df)
    """
    if not marketed_all_path.exists():
        logger.warning("marketed_all.csv 없음: %s — InChIKey 매칭 스킵", marketed_all_path)
        dilirank_df["InChIKey"] = ""
        return dilirank_df, pd.DataFrame()

    marketed = pd.read_csv(marketed_all_path)
    marketed["name_norm"] = marketed["name"].apply(_normalize_name)
    name_to_ik = (
        marketed[marketed["inchi_key"].astype(str).str.strip() != ""]
        .drop_duplicates(subset="name_norm")
        .set_index("name_norm")["inchi_key"]
        .to_dict()
    )
    logger.info("marketed_all 매칭 풀: %d unique 정규화 이름", len(name_to_ik))

    dilirank_df["name_norm"] = dilirank_df["Compound Name"].apply(_normalize_name)
    dilirank_df["InChIKey"] = dilirank_df["name_norm"].map(name_to_ik).fillna("")

    matched = dilirank_df[dilirank_df["InChIKey"] != ""].copy()
    unmatched = dilirank_df[dilirank_df["InChIKey"] == ""].copy()
    logger.info(
        "이름 매칭: %d/%d (matched), %d unmatched",
        len(matched),
        len(dilirank_df),
        len(unmatched),
    )
    return matched, unmatched


def fill_unmatched_via_pubchem(unmatched: pd.DataFrame) -> pd.DataFrame:
    """매칭 안 된 약물을 PubChem name lookup으로 InChIKey 보강."""
    from urllib.parse import quote

    rows = unmatched.copy()
    pbar = tqdm(rows.iterrows(), total=len(rows), desc="PubChem name lookup")
    for idx, row in pbar:
        name = row["Compound Name"]
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            url = PUBCHEM_NAME_URL.format(quote(name))
            data = fetch_json(url)
            props = data.get("PropertyTable", {}).get("Properties", [])
            if props:
                ik = props[0].get("InChIKey", "")
                if ik:
                    rows.at[idx, "InChIKey"] = ik
        except Exception:
            pass
        time.sleep(0.2)  # PubChem rate limit
    pbar.close()
    filled = rows[rows["InChIKey"] != ""]
    logger.info("PubChem 보강: %d → %d InChIKey 채움", len(unmatched), len(filled))
    return rows


def save_dilirank(
    output_path: Path,
    use_pubchem_fallback: bool = True,
    local_excel: Path | None = None,
) -> int:
    """DILIrank 다운로드 + 이름 매칭 + 저장."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = download_dilirank_excel(local_path=local_excel)

    marketed_all = PROJECT_ROOT / "data" / "marketed_drugs" / "all" / "marketed_all.csv"
    matched, unmatched = match_inchikeys_by_name(df, marketed_all)

    if use_pubchem_fallback and not unmatched.empty:
        logger.info("PubChem fallback으로 %d개 보강 시도", len(unmatched))
        unmatched = fill_unmatched_via_pubchem(unmatched)

    full = pd.concat([matched, unmatched], ignore_index=True)
    full = full[["LTKBID", "Compound Name", "Severity Class", "InChIKey"]]
    full.to_csv(output_path, index=False)

    matched_count = int((full["InChIKey"] != "").sum())
    logger.info(
        "저장: %s (%d/%d InChIKey 매칭)",
        output_path,
        matched_count,
        len(full),
    )

    counts = full["Severity Class"].value_counts()
    for cat, count in counts.items():
        with_ik = int((full[full["Severity Class"] == cat]["InChIKey"] != "").sum())
        logger.info("  %s: %d (InChIKey %d)", cat, count, with_ik)
    return matched_count


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    import argparse

    parser = argparse.ArgumentParser(description="DILIrank 다운로드 + 매칭")
    parser.add_argument(
        "--local-excel",
        default=None,
        help="FDA에서 미리 받아둔 .xlsx 파일 경로 (rate limit 회피용)",
    )
    parser.add_argument("--no-pubchem", action="store_true", help="PubChem 보강 스킵")
    args = parser.parse_args()

    output = PROJECT_ROOT / "data" / "marketed_drugs" / "reference" / "dilirank.csv"
    local = Path(args.local_excel) if args.local_excel else None
    return (
        0
        if save_dilirank(
            output,
            use_pubchem_fallback=not args.no_pubchem,
            local_excel=local,
        )
        > 0
        else 1
    )


if __name__ == "__main__":
    import sys

    sys.exit(main())
