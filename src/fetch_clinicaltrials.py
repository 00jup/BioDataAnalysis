"""clinicaltrials.gov — 임상시험의 hepatic AE 보고.

전략:
  1. ClinicalTrials.gov API v2 (https://clinicaltrials.gov/api/v2/studies)
  2. condition = hepatotoxicity / drug-induced liver injury
  3. 또는 intervention 의 약물명 추출 + adverse events 필터
  4. 약물별 hepatic AE 발생률 집계

저장: data/raw/clinicaltrials/ct_dili.csv
"""

from __future__ import annotations

import os
import time
from collections import Counter

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "clinicaltrials")
HEADERS = {"User-Agent": "Mozilla/5.0"}
API = "https://clinicaltrials.gov/api/v2/studies"
SLEEP = 0.3
SLEEP_PUBCHEM = 0.35

HEPATIC_TERMS = [
    "drug-induced liver injury",
    "hepatotoxicity",
    "hepatic failure",
    "hepatitis",
    "elevated liver enzymes",
    "transaminase increase",
    "alanine aminotransferase increased",
    "hepatocellular injury",
    "cholestasis",
    "jaundice",
]


def search_studies(query, page_size=100, max_pages=20):
    """ClinicalTrials AE 가 있는 study + 약물명 추출.

    v2 API 의 fields 옵션 복잡 — 전체 study 받고 manual parse.
    """
    all_studies = []
    next_token = None
    for p in range(max_pages):
        params = {"query.cond": query, "pageSize": page_size}
        if next_token:
            params["pageToken"] = next_token
        r = requests.get(API, params=params, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            print(f"  API 오류 {r.status_code}: {r.text[:200]}")
            break
        data = r.json()
        studies = data.get("studies", [])
        if not studies:
            break
        all_studies.extend(studies)
        next_token = data.get("nextPageToken")
        time.sleep(SLEEP)
        if not next_token:
            break
    print(f"  {len(all_studies)} studies 수집")
    return all_studies


def extract_drugs(studies):
    drug_counter = Counter()
    for s in studies:
        prot = s.get("protocolSection", {})
        intervs = prot.get("armsInterventionsModule", {}).get("interventions", [])
        for iv in intervs:
            name = iv.get("name", "").strip().lower()
            if name and len(name) > 2 and "placebo" not in name and "control" not in name:
                drug_counter[name] += 1
    return drug_counter


def lookup_smiles(names):
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
        if i % 50 == 0 and i > 0:
            print(f"    PubChem: {i}/{len(names)} 매핑 {len(out)}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== ClinicalTrials.gov — hepatic AE 약물 수집 ===\n")
    all_studies = []
    for q in ["drug-induced liver injury", "hepatotoxicity", "hepatic failure"]:
        print(f"[검색] {q}")
        all_studies.extend(search_studies(q, max_pages=5))

    # 약물명 집계
    drug_counter = extract_drugs(all_studies)
    df = pd.DataFrame([{"name": n, "n_trials": c} for n, c in drug_counter.most_common() if c >= 2])
    print(f"\n수집 unique 약물 (≥2 trials): {len(df)}")
    df.to_csv(os.path.join(OUT_DIR, "ct_dili_raw.csv"), index=False)

    # SMILES 매핑
    print(f"\nPubChem SMILES 매핑 ({len(df)})")
    name_to_smi = lookup_smiles(df["name"].head(500).tolist())  # 상위 500만
    df["canonical_smiles"] = df["name"].map(name_to_smi)
    final = df.dropna(subset=["canonical_smiles"]).copy()
    final.to_csv(os.path.join(OUT_DIR, "ct_dili.csv"), index=False)
    print(f"저장: {OUT_DIR}/ct_dili.csv ({len(final)})")


if __name__ == "__main__":
    main()
