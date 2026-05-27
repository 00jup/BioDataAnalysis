"""LiverTox (NIH NCBI Bookshelf) 스크래퍼 — E-utilities 기반.

전략:
  1. esearch — LiverTox(NBK547852) 안에 "Likelihood Score" 포함 챕터 ID 1,195개 확보
  2. esummary — ID → 챕터 제목 + bookshelf slug
  3. efetch/HTML fetch — 각 챕터 페이지 → likelihood 추출
  4. PubChem — 약물명 → canonical SMILES

저장:
  data/raw/livertox/livertox_raw.csv  (raw 추출)
  data/raw/livertox/livertox.csv      (SMILES 매핑 완료)
"""

from __future__ import annotations

import os
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "livertox")

BOOK_ACCN = "NBK547852"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
HEADERS = {"User-Agent": "research scraper / contact: ai.jeje.labs@gmail.com"}
SLEEP_NCBI = 0.35  # NCBI 권장: <3 req/sec
SLEEP_PUBCHEM = 0.35

LIKELIHOOD_REGEX = re.compile(
    r"Likelihood\s*[Ss]core[:\s]+([A-EHX](?:\s*\([^)]*\))?|\*)\b", re.IGNORECASE
)


def esearch_likelihood() -> list[str]:
    """LiverTox 내 'Likelihood Score' 포함 페이지 ID 전체."""
    r = requests.get(
        f"{EUTILS}/esearch.fcgi",
        params={
            "db": "books",
            "term": f"{BOOK_ACCN}[bksaccn] AND Likelihood Score[text]",
            "retmax": "5000",
            "retmode": "json",
        },
        headers=HEADERS,
        timeout=30,
    )
    return r.json()["esearchresult"]["idlist"]


def esummary_batch(ids: list[str]) -> list[dict]:
    """배치 ID → 챕터 메타. 핵심: chapteraccessionid (NBK URL) + id 경로에서 약물명 추출.

    esummary 가 section 단위 결과를 주는데, 챕터 단위로 dedup 필요.
    """
    out = {}  # key = chapter accession (NBK), value = {drug_name, accn, uid}
    for i in range(0, len(ids), 100):
        chunk = ids[i : i + 100]
        r = requests.get(
            f"{EUTILS}/esummary.fcgi",
            params={
                "db": "books",
                "id": ",".join(chunk),
                "retmode": "json",
            },
            headers=HEADERS,
            timeout=60,
        )
        data = r.json().get("result", {})
        for uid in chunk:
            if uid not in data:
                continue
            d = data[uid]
            chap_accn = d.get("chapteraccessionid", "")  # 진짜 약물 페이지 NBK
            if not chap_accn:
                continue
            # id 경로: "livertox/{DrugName}/sec/.../PMC"
            id_path = d.get("id", "")
            parts = id_path.split("/")
            drug_name = parts[1] if len(parts) > 1 else ""
            if chap_accn not in out:
                out[chap_accn] = {
                    "uid": uid,
                    "name": drug_name,
                    "accn": chap_accn,
                    "section_title": d.get("title", ""),
                }
        time.sleep(SLEEP_NCBI)
        print(f"    esummary: {i + len(chunk)}/{len(ids)} (unique chapters {len(out)})")
    return list(out.values())


def fetch_chapter(accn: str) -> tuple[str, str | None]:
    """챕터 HTML fetch → 본문 + likelihood."""
    if not accn:
        return "", None
    url = f"https://www.ncbi.nlm.nih.gov/books/{accn}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        text = soup.get_text(" ", strip=True)
        m = LIKELIHOOD_REGEX.search(text)
        if m:
            raw = m.group(1).strip()
            letter = raw[0].upper() if raw else None
            if letter in ("A", "B", "C", "D", "E", "H", "X"):
                return text[:300], letter
        return text[:300], None
    except Exception:
        return "", None


def lookup_smiles_batch(names: list[str]) -> dict[str, str]:
    """약물명 리스트 → SMILES dict (PubChem)."""
    import pubchempy as pcp

    out = {}
    for i, name in enumerate(names):
        if not name:
            continue
        # 1) 그대로
        try:
            res = pcp.get_compounds(name, "name")
            if res:
                out[name] = res[0].canonical_smiles
                time.sleep(SLEEP_PUBCHEM)
                continue
        except Exception:
            pass
        # 2) 첫 단어만 (예: "Acetaminophen (Paracetamol)" → "Acetaminophen")
        try:
            simple = re.sub(r"\s*\(.*\)", "", name).strip()
            if simple and simple != name:
                res = pcp.get_compounds(simple, "name")
                if res:
                    out[name] = res[0].canonical_smiles
                    time.sleep(SLEEP_PUBCHEM)
                    continue
        except Exception:
            pass
        time.sleep(SLEEP_PUBCHEM)
        if i % 50 == 0 and i > 0:
            print(f"    PubChem: {i}/{len(names)} 매핑됨 {len(out)}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/4] esearch — LiverTox 내 likelihood 챕터 ID")
    ids = esearch_likelihood()
    print(f"  {len(ids)} 챕터 발견")

    print("\n[2/4] esummary — ID → 제목 + accession")
    summaries = esummary_batch(ids)
    print(f"  {len(summaries)} 챕터 메타 수집")

    print(f"\n[3/4] 챕터별 fetch + likelihood 추출 (unique 챕터 {len(summaries)})")
    rows = []
    for i, s in enumerate(summaries):
        text, lik = fetch_chapter(s["accn"])
        s["likelihood"] = lik
        s["snippet"] = text
        rows.append(s)
        time.sleep(SLEEP_NCBI)
        if i % 50 == 0 and i > 0:
            n_lik = sum(1 for r in rows if r.get("likelihood"))
            print(f"  {i}/{len(summaries)}  likelihood 추출 {n_lik}")

    raw_df = pd.DataFrame(rows)
    raw_path = os.path.join(OUT_DIR, "livertox_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    n_lik = int(raw_df["likelihood"].notna().sum())
    print(f"\n  raw 저장: {raw_path} (전체 {len(raw_df)}, likelihood {n_lik})")
    print(f"  Likelihood 분포: {raw_df['likelihood'].value_counts(dropna=False).to_dict()}")

    print("\n[4/4] PubChem SMILES 매핑")
    likelihood_rows = raw_df[raw_df["likelihood"].notna()].copy()
    names = likelihood_rows["name"].tolist()
    name_to_smi = lookup_smiles_batch(names)

    likelihood_rows["canonical_smiles"] = likelihood_rows["name"].map(name_to_smi)
    final = likelihood_rows.dropna(subset=["canonical_smiles"]).copy()
    final = final[["name", "canonical_smiles", "likelihood", "accn", "uid"]]

    final_path = os.path.join(OUT_DIR, "livertox.csv")
    final.to_csv(final_path, index=False)
    print(f"\n저장: {final_path} ({len(final)} 약물 SMILES + likelihood)")
    print("  Likelihood 분포 (SMILES 매핑 후):")
    print(f"  {final['likelihood'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
