"""CTD (Comparative Toxicogenomics Database) — chemical-disease 매핑.

CTD 는 NIH/EPA 가 후원하는 큐레이션된 chemical-disease 매핑 DB.
간 관련 질병에 대해 'M' (Marker/mechanism), 'T' (Therapeutic), 또는
inferred 관계로 분류.

전략:
  1. CTD bulk download: chemicals_diseases.tsv.gz (약 50MB)
  2. 간 관련 disease MeSH ID 필터:
     - MESH:D056486 (Drug-Induced Liver Injury)
     - MESH:D008113 (Liver Diseases)
     - MESH:D017093 (Liver Failure, Acute)
     - MESH:D056487 (Chemical and Drug Induced Liver Injury, Chronic)
     - MESH:D058186 (Acute Kidney Injury - 제외)
     - MESH:D015431 (Weight Loss - 제외)
  3. 'Direct Evidence' 가 marker/mechanism (M) 인 chemical 만 (강한 증거)
  4. PubChem 으로 SMILES 매핑

저장:
  data/raw/ctd/ctd_dili_raw.csv (원본)
  data/raw/ctd/ctd_dili.csv (SMILES 매핑 후)
"""

from __future__ import annotations

import gzip
import os
import time

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "ctd")
CTD_URL = "https://ctdbase.org/reports/CTD_chemicals_diseases.tsv.gz"
HEADERS = {"User-Agent": "research scraper / ai.jeje.labs@gmail.com"}
SLEEP_PUBCHEM = 0.30

# 간독성 관련 MeSH ID
LIVER_MESH = {
    "MESH:D056486": "Drug-Induced Liver Injury",
    "MESH:D008113": "Liver Diseases",
    "MESH:D017093": "Liver Failure, Acute",
    "MESH:D056487": "Chemical and Drug Induced Liver Injury, Chronic",
    "MESH:D005764": "Hepatic Cholestasis (drug-induced)",
    "MESH:D006501": "Hepatic Encephalopathy",
    "MESH:D009503": "Necrosis (간 관련)",
    "MESH:D008106": "Liver Cirrhosis",
    "MESH:D006505": "Hepatitis",
    "MESH:D023281": "Cholestasis, Intrahepatic",
}


def download_ctd() -> str:
    """CTD bulk file 다운로드 (이미 있으면 재사용)."""
    out_gz = os.path.join(OUT_DIR, "CTD_chemicals_diseases.tsv.gz")
    if os.path.exists(out_gz) and os.path.getsize(out_gz) > 1_000_000:
        print(f"  캐시 사용: {out_gz} ({os.path.getsize(out_gz) / 1e6:.1f}MB)")
        return out_gz
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"  CTD 다운로드: {CTD_URL}")
    r = requests.get(CTD_URL, stream=True, headers=HEADERS, timeout=120)
    r.raise_for_status()
    with open(out_gz, "wb") as f:
        total = 0
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
            total += len(chunk)
            if total % (10 * 1024 * 1024) < 8192:
                print(f"    {total / 1e6:.1f}MB 다운로드")
    print(f"  → {out_gz} ({total / 1e6:.1f}MB)")
    return out_gz


def parse_ctd(path: str) -> pd.DataFrame:
    """간독성 관련 chemical-disease 매핑 추출."""
    print("  TSV 파싱 (메모리 큰 파일 — chunk 로드)")
    rows = []
    # CTD format: # 으로 시작하는 주석, 헤더 정의는 ChemicalName, ChemicalID, CasRN, DiseaseName, DiseaseID, DirectEvidence, InferenceGeneSymbol, InferenceScore, OmimIDs, PubMedIDs
    cols = [
        "ChemicalName",
        "ChemicalID",
        "CasRN",
        "DiseaseName",
        "DiseaseID",
        "DirectEvidence",
        "InferenceGeneSymbol",
        "InferenceScore",
        "OmimIDs",
        "PubMedIDs",
    ]
    with gzip.open(path, "rt", encoding="utf-8") as f:
        chunks = pd.read_csv(
            f,
            sep="\t",
            comment="#",
            names=cols,
            low_memory=False,
            chunksize=200_000,
            on_bad_lines="skip",
        )
        for ck in chunks:
            sub = ck[ck["DiseaseID"].isin(LIVER_MESH.keys())].copy()
            if not sub.empty:
                rows.append(sub)
    if not rows:
        print("  liver disease 관련 row 없음")
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    print(f"  liver 관련: {len(df)} 행")

    # 강한 증거 (DirectEvidence='marker/mechanism' 또는 'therapeutic') 우선
    df["evidence_strong"] = df["DirectEvidence"].fillna("").str.contains("marker", case=False)
    # inferred 도 PubMed ID 가 많으면 (>=3) 보조 신호
    df["n_pmid"] = df["PubMedIDs"].fillna("").str.count(r"\|") + 1
    df.loc[df["PubMedIDs"].isna() | (df["PubMedIDs"] == ""), "n_pmid"] = 0
    print(f"  evidence_strong: {df.evidence_strong.sum()}")
    print(f"  n_pmid 분포: median {df.n_pmid.median():.0f}, max {df.n_pmid.max()}")
    return df


def aggregate_chemicals(df: pd.DataFrame) -> pd.DataFrame:
    """chemical 별로 통합. strong evidence 카운트 + total PMID."""
    agg = (
        df.groupby(["ChemicalName", "ChemicalID", "CasRN"])
        .agg(
            strong_evidence=("evidence_strong", "sum"),
            n_evidence=("DiseaseID", "count"),
            max_pmid=("n_pmid", "max"),
            diseases=("DiseaseName", lambda x: ";".join(sorted(set(x)))),
        )
        .reset_index()
    )
    print(f"  unique chemicals: {len(agg)}")
    print(f"  strong evidence ≥ 1: {(agg.strong_evidence >= 1).sum()}")
    return agg


def lookup_smiles(names: list[str]) -> dict[str, str]:
    import pubchempy as pcp

    out = {}
    for i, name in enumerate(names):
        if not name:
            continue
        try:
            res = pcp.get_compounds(name, "name")
            if res:
                out[name] = res[0].canonical_smiles
        except Exception:
            pass
        time.sleep(SLEEP_PUBCHEM)
        if i % 100 == 0 and i > 0:
            print(f"    PubChem: {i}/{len(names)} 매핑 {len(out)}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== CTD chemical-disease 매핑 추출 ===\n")
    path = download_ctd()
    df = parse_ctd(path)
    if df.empty:
        print("liver 관련 데이터 없음 — 종료")
        return
    agg = aggregate_chemicals(df)

    raw_path = os.path.join(OUT_DIR, "ctd_dili_raw.csv")
    agg.to_csv(raw_path, index=False)
    print(f"\nraw 저장: {raw_path}")

    # 강한 증거 또는 evidence count ≥ 5 만 유지
    filtered = agg[(agg.strong_evidence >= 1) | (agg.n_evidence >= 5)].copy()
    print(f"\nfilter (strong ≥1 OR n_evidence ≥5): {len(filtered)}")

    # PubChem SMILES 매핑
    print(f"\nPubChem SMILES 매핑 ({len(filtered)} chemicals)")
    name_to_smi = lookup_smiles(filtered["ChemicalName"].tolist())
    filtered["canonical_smiles"] = filtered["ChemicalName"].map(name_to_smi)
    final = filtered.dropna(subset=["canonical_smiles"]).copy()

    out_path = os.path.join(OUT_DIR, "ctd_dili.csv")
    final.rename(columns={"ChemicalName": "name"}).to_csv(out_path, index=False)
    print(f"\n저장: {out_path}  ({len(final)} / {len(filtered)} SMILES 매핑)")


if __name__ == "__main__":
    main()
