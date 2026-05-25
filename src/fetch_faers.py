"""FAERS (FDA Adverse Event Reporting System) — openFDA API.

DailyMed/openFDA Drug Label 과 다른 데이터.
FAERS 는 환자/의료진이 직접 신고한 ADR.

전략:
  1. openFDA /drug/event.json 으로 hepatic AE 신고된 약물 검색
  2. reaction term 들 (hepatic_failure, hepatitis 등) 별로 검색
  3. 약물명 (medicinalproduct) 별 hepatic AE 신고 수 집계
  4. 신고 수 ≥ 50 이면 양성 후보

저장:
  data/raw/faers/faers_dili_raw.csv (집계)
  data/raw/faers/faers_dili.csv     (SMILES 매핑 후)
"""
from __future__ import annotations
import json, os, time
from collections import Counter
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "faers")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14.4; rv:124.0) Gecko/20100101 Firefox/124.0"}
SLEEP_API = 0.30
SLEEP_PUBCHEM = 0.35

# MedDRA-like hepatic reaction term — FAERS 에서 자주 사용
HEPATIC_TERMS = [
    "hepatic failure", "drug-induced liver injury", "hepatocellular injury",
    "hepatitis", "jaundice cholestatic", "hepatic necrosis", "liver injury",
    "hepatic enzyme increased", "hepatotoxicity", "cholestasis",
    "alanine aminotransferase increased",
]


def search_faers_reaction(term: str, limit: int = 100) -> dict:
    """특정 reaction term 으로 발생한 약물 집계 (openFDA count API).

    openfda.generic_name.exact 사용 — FDA 정제 generic name 기준.
    (medicinalproduct.exact 는 403 차단됨)
    """
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f'patient.reaction.reactionmeddrapt:"{term}"',
        "count": "patient.drug.openfda.generic_name.exact",
        "limit": limit,
    }
    r = requests.get(url, params=params, headers=HEADERS, timeout=60)
    if r.status_code == 404:
        return {"results": []}
    r.raise_for_status()
    return r.json()


def collect_all_terms() -> pd.DataFrame:
    """각 reaction term 별 약물 집계 → 통합."""
    drug_counter = Counter()
    drug_reactions = {}
    for term in HEPATIC_TERMS:
        print(f"\n[FAERS] {term} ...")
        try:
            data = search_faers_reaction(term)
        except Exception as e:
            print(f"  API 오류: {e}")
            continue
        results = data.get("results", [])
        print(f"  {len(results)} unique 약물")
        for r in results:
            name = r.get("term", "").strip().lower()
            count = int(r.get("count", 0))
            if not name or count < 5: continue
            drug_counter[name] += count
            drug_reactions.setdefault(name, set()).add(term)
        time.sleep(SLEEP_API)

    rows = []
    for name, total in drug_counter.most_common():
        if total < 50:  # 최소 50건 이상 (noise 필터)
            continue
        rows.append({
            "name": name,
            "n_reports": total,
            "n_reaction_types": len(drug_reactions.get(name, set())),
            "reactions": ";".join(sorted(drug_reactions.get(name, set()))),
        })
    return pd.DataFrame(rows)


def lookup_smiles(names: list[str]) -> dict[str, str]:
    import pubchempy as pcp
    out = {}
    for i, name in enumerate(names):
        if not name: continue
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
    print("=== FAERS openFDA hepatic AE 추출 ===\n")
    raw = collect_all_terms()
    print(f"\n=== 수집 완료: {len(raw)} 약물 (≥50 보고) ===")
    if raw.empty:
        return

    print(f"분포:")
    print(f"  보고 수 평균: {raw.n_reports.mean():.0f}")
    print(f"  보고 수 중간값: {raw.n_reports.median():.0f}")
    print(f"  보고 수 ≥1000: {(raw.n_reports>=1000).sum()}")
    print(f"  reaction type 다양성 평균: {raw.n_reaction_types.mean():.1f}")

    raw_path = os.path.join(OUT_DIR, "faers_dili_raw.csv")
    raw.to_csv(raw_path, index=False)
    print(f"raw 저장: {raw_path}")

    # PubChem SMILES
    print(f"\nPubChem SMILES 매핑 ({len(raw)})")
    name_to_smi = lookup_smiles(raw["name"].tolist())
    raw["canonical_smiles"] = raw["name"].map(name_to_smi)
    final = raw.dropna(subset=["canonical_smiles"]).copy()

    out_path = os.path.join(OUT_DIR, "faers_dili.csv")
    final.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}  ({len(final)} / {len(raw)} SMILES 매핑)")


if __name__ == "__main__":
    main()
