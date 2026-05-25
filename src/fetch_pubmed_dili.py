"""PubMed 에서 DILI 관련 약물 추출 (NCBI E-utilities + MeSH 태그).

전략 — Free-text NLP 대신 MeSH 태깅 활용 (신뢰도 ↑):
  1. PubMed 에서 'Chemical and Drug Induced Liver Injury' MeSH 태그 논문 검색
  2. 각 논문의 MeSH 헤딩 추출 (drug + chemical compound 자동 태깅됨)
  3. 약물 빈도 카운트 (몇 개 DILI 논문에 등장?)
  4. PubChem 으로 SMILES 매핑

저장:
  data/raw/pubmed/pubmed_dili_raw.csv  (논문-약물 매핑)
  data/raw/pubmed/pubmed_dili.csv      (약물별 통합, SMILES 매핑)
"""

from __future__ import annotations

import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "pubmed")
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": "research scraper / ai.jeje.labs@gmail.com"}
SLEEP_NCBI = 0.35
SLEEP_PUBCHEM = 0.35

# DILI MeSH 검색 (공신력 strict)
#   - DILI 가 논문의 Major Topic 인 것만 ([majr])
#   - 신뢰성 있는 publication type 만 (Case Reports, Clinical Trial, RCT, Meta-Analysis, Systematic Review)
#   - 사람 데이터만 ("humans"[MeSH Terms])
DILI_QUERY = (
    "((Chemical and Drug Induced Liver Injury[majr]) OR "
    "(Drug-Induced Liver Injury[majr])) "
    "AND ("
    "Case Reports[ptyp] OR "
    "Clinical Trial[ptyp] OR "
    "Randomized Controlled Trial[ptyp] OR "
    "Meta-Analysis[ptyp] OR "
    "Systematic Review[ptyp]"
    ") "
    "AND humans[MeSH Terms]"
)


def esearch_dili_pmids(max_results: int = 10000) -> list[str]:
    """DILI 논문 PMID 전체 조회."""
    print(f"[1/4] PubMed DILI 논문 검색 (MeSH: {DILI_QUERY})")
    pmids = []
    retstart = 0
    batch = 1000
    while True:
        try:
            r = requests.get(f"{EUTILS}/esearch.fcgi", params={
                "db": "pubmed", "term": DILI_QUERY,
                "retmax": batch, "retstart": retstart, "retmode": "json",
            }, headers=HEADERS, timeout=30)
            data = r.json()
        except Exception as e:
            print(f"  retstart={retstart} 에러 (계속): {e}")
            time.sleep(2.0)
            retstart += batch
            if retstart > 50000: break
            continue
        batch_ids = data["esearchresult"]["idlist"]
        pmids += batch_ids
        total = int(data["esearchresult"]["count"])
        print(f"  retstart={retstart}, batch={len(batch_ids)} 누적 {len(pmids)} / 총 {total}")
        if not batch_ids or len(pmids) >= max_results or len(pmids) >= total:
            break
        retstart += batch
        time.sleep(SLEEP_NCBI)
    print(f"  → {len(pmids)} PMIDs 확보")
    return pmids[:max_results]


def efetch_articles(pmids: list[str]) -> list[dict]:
    """배치별 efetch — 논문 메타데이터 + MeSH 헤딩."""
    print(f"\n[2/4] 각 논문의 MeSH 헤딩 + 약물명 추출 ({len(pmids)}개)")
    out = []
    batch = 100
    for i in range(0, len(pmids), batch):
        chunk = pmids[i:i+batch]
        r = requests.post(f"{EUTILS}/efetch.fcgi", data={
            "db": "pubmed", "id": ",".join(chunk),
            "rettype": "xml", "retmode": "xml",
        }, headers=HEADERS, timeout=60)
        try:
            root = ET.fromstring(r.text)
        except Exception as e:
            print(f"  XML 파싱 실패 batch {i}: {e}")
            continue
        for art in root.findall(".//PubmedArticle"):
            pmid = art.findtext(".//PMID")
            title = art.findtext(".//ArticleTitle") or ""
            year = art.findtext(".//PubDate/Year") or ""
            # Publication type
            ptypes = [p.text for p in art.findall(".//PublicationType") if p.text]
            # MeSH 헤딩 — drug/chemical 만 추출
            drug_terms = []
            for mh in art.findall(".//MeshHeading"):
                desc = mh.find("DescriptorName")
                if desc is None: continue
                desc_text = desc.text or ""
                major = desc.get("MajorTopicYN", "N")
                drug_terms.append({"name": desc_text, "major": major})
            # ChemicalList — 약물·화합물 명시적 태깅
            chemicals = []
            for chem in art.findall(".//Chemical/NameOfSubstance"):
                chemicals.append(chem.text)
            out.append({
                "pmid": pmid, "title": title, "year": year,
                "pub_types": ";".join(ptypes),
                "mesh_drugs": ";".join(d["name"] for d in drug_terms if d["name"]),
                "chemicals": ";".join(c for c in chemicals if c),
            })
        time.sleep(SLEEP_NCBI)
        if i % 1000 == 0:
            print(f"  처리 {i+len(chunk)}/{len(pmids)}")
    return out


def aggregate_drugs(articles: list[dict]) -> pd.DataFrame:
    """약물별로 등장 횟수 + 관련 PMID 집계."""
    print(f"\n[3/4] 약물별 통합")
    drug_counter = Counter()
    drug_pmids = defaultdict(list)
    for a in articles:
        # Chemicals (구체적 화학물질명) 우선
        for c in a["chemicals"].split(";"):
            c = c.strip()
            if c and len(c) > 2:
                drug_counter[c] += 1
                drug_pmids[c].append(a["pmid"])
    rows = []
    for drug, n in drug_counter.most_common():
        rows.append({
            "name": drug,
            "n_papers": n,
            "pmid_examples": ";".join(drug_pmids[drug][:5]),
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
            try:
                # 첫 단어만
                simple = re.split(r"[\s,;]", name.strip())[0]
                if simple and simple != name:
                    res = pcp.get_compounds(simple, "name")
                    if res: out[name] = res[0].canonical_smiles
            except Exception:
                pass
        time.sleep(SLEEP_PUBCHEM)
        if i % 50 == 0 and i > 0:
            print(f"    PubChem: {i}/{len(names)} 매핑 {len(out)}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pmids = esearch_dili_pmids(max_results=9500)  # JSON boundary issue 우회
    articles = efetch_articles(pmids)
    # raw 저장
    pd.DataFrame(articles).to_csv(os.path.join(OUT_DIR, "pubmed_dili_papers.csv"), index=False)
    print(f"\nraw 논문 저장 — {len(articles)} 논문")

    drugs = aggregate_drugs(articles)
    drugs.to_csv(os.path.join(OUT_DIR, "pubmed_dili_drugs_raw.csv"), index=False)
    print(f"raw 약물 저장 — {len(drugs)} unique 약물")
    print(f"상위 약물: {drugs.head(10)['name'].tolist()}")

    # 최소 3 papers 이상 — 약물명이 짧고 일반적인 것 (carbon dioxide 같은) 줄임
    drugs_filtered = drugs[drugs["n_papers"] >= 3].copy()
    print(f"\nn_papers >= 3: {len(drugs_filtered)} 약물")

    print(f"\n[4/4] PubChem SMILES 매핑 ({len(drugs_filtered)} 약물)")
    name_to_smi = lookup_smiles(drugs_filtered["name"].tolist())
    drugs_filtered["canonical_smiles"] = drugs_filtered["name"].map(name_to_smi)
    final = drugs_filtered.dropna(subset=["canonical_smiles"]).copy()
    final.to_csv(os.path.join(OUT_DIR, "pubmed_dili.csv"), index=False)
    print(f"\n저장: data/raw/pubmed/pubmed_dili.csv  ({len(final)} / {len(drugs_filtered)} SMILES 매핑)")
    print(f"등장 빈도 분포 — 평균 {final['n_papers'].mean():.1f}, 최대 {final['n_papers'].max()}")


if __name__ == "__main__":
    main()
