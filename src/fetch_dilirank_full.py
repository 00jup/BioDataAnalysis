"""FDA DILIrank 전체 (~1,337 rows) 다운로드 + 정제 + PubChem SMILES 매핑."""

from __future__ import annotations

import os
import re
import time

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dilirank")
URL = "https://www.fda.gov/media/113052/download"
HEADERS = {"User-Agent": "Mozilla/5.0 (research; ai.jeje.labs@gmail.com)"}
SLEEP_PUBCHEM = 0.35

# FDA 표기 → DB 표준 표기
CAT_MAP = {
    "vmost-dili-concern": "vMost-DILI-Concern",
    "vless-dili-concern": "vLess-DILI-Concern",
    "vno-dili-concern": "vNo-DILI-Concern",
    "ambiguous-dili-concern": "Ambiguous-DILI-Concern",
}


def download_excel(path: str) -> bool:
    if os.path.exists(path):
        print(f"  이미 있음: {path}")
        return True
    print(f"[1/3] FDA DILIrank Excel 다운로드 → {path}")
    try:
        r = requests.get(URL, headers=HEADERS, timeout=60, allow_redirects=True)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  완료, {len(r.content) / 1024:.1f} KB")
        return True
    except Exception as e:
        print(f"  실패: {e}")
        return False


def parse(xlsx_path: str) -> pd.DataFrame:
    print("\n[2/3] Excel 파싱 (header=1)")
    df = pd.read_excel(xlsx_path, sheet_name="version 2", header=1)
    df = df.rename(
        columns={
            "CompoundName": "name",
            "vDILI-Concern": "dilirank_category_raw",
            "SeverityClass": "severity_class",
            "LabelSection": "label_section",
            "LTKBID": "ltkb_id",
        }
    )
    df = df.dropna(subset=["name"]).reset_index(drop=True)
    df["dilirank_category"] = (
        df["dilirank_category_raw"].astype(str).str.strip().str.lower().map(CAT_MAP)
    )
    print(f"  rows: {len(df)}")
    print("  카테고리 분포:")
    print(df["dilirank_category"].value_counts(dropna=False).to_dict())
    return df


def lookup_smiles(names: list[str]) -> dict[str, str]:
    import pubchempy as pcp

    out = {}
    for i, name in enumerate(names):
        if not isinstance(name, str) or not name.strip():
            continue
        try:
            res = pcp.get_compounds(name.strip(), "name")
            if res:
                out[name] = res[0].canonical_smiles
        except Exception:
            try:
                simple = re.sub(r"\s*\(.*\)", "", name).strip()
                if simple and simple != name:
                    res = pcp.get_compounds(simple, "name")
                    if res:
                        out[name] = res[0].canonical_smiles
            except Exception:
                pass
        time.sleep(SLEEP_PUBCHEM)
        if i % 50 == 0 and i > 0:
            print(f"    PubChem: {i}/{len(names)} 매핑 {len(out)}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    xlsx_path = os.path.join(OUT_DIR, "dilirank_full.xlsx")
    if not download_excel(xlsx_path):
        return

    df = parse(xlsx_path)

    print(f"\n[3/3] PubChem SMILES 매핑 ({len(df)} 분자)")
    name_to_smi = lookup_smiles(df["name"].astype(str).tolist())
    df["canonical_smiles"] = df["name"].map(name_to_smi)
    df_with_smi = df.dropna(subset=["canonical_smiles"]).copy()

    csv_path = os.path.join(OUT_DIR, "dilirank_full.csv")
    df_with_smi.to_csv(csv_path, index=False)
    print(f"\n저장: {csv_path}  ({len(df_with_smi)} / {len(df)} SMILES 매핑)")
    print(df_with_smi["dilirank_category"].value_counts(dropna=False).to_dict())


if __name__ == "__main__":
    main()
