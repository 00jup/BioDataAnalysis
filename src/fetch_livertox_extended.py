"""LiverTox 확장 — NCBI Bookshelf 의 LiverTox 의 전체 chapter scrape.

현재 우리 LiverTox: 851 분자
LiverTox 전체: 약 1,200+ chapter
→ 추가 ~300-400 분자 가능

NCBI Bookshelf API + LiverTox 의 ID: NBK547852 (root)
각 chapter 의 URL pattern: NBK<id>
각 chapter 내 Likelihood Score (A/B/C/D/E)
"""
from __future__ import annotations
import json, os, re, time
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "livertox_extended")
HEADERS = {"User-Agent": "Mozilla/5.0 research"}
SLEEP = 0.4
SLEEP_PUBCHEM = 0.35

ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
BOOKURL = "https://www.ncbi.nlm.nih.gov/books"


def get_livertox_chapter_ids():
    """LiverTox 의 모든 chapter ID."""
    print("LiverTox chapter ID 수집...")
    ids = []
    retstart = 0
    while True:
        r = requests.get(ESEARCH, params={
            "db": "books", "term": '"LiverTox"[BOOK]',
            "retmax": 500, "retstart": retstart, "retmode": "json",
        }, headers=HEADERS, timeout=30)
        try:
            data = r.json()
        except Exception:
            break
        batch = data["esearchresult"]["idlist"]
        ids += batch
        total = int(data["esearchresult"]["count"])
        if not batch or len(ids) >= total:
            break
        retstart += 500
        time.sleep(SLEEP)
        if retstart > 3000: break
    print(f"  {len(ids)} LiverTox chapter ID 확보")
    return ids


def fetch_chapter_metadata(ids):
    """각 chapter 의 메타 (title + accession ID)."""
    print(f"Chapter 메타 fetch (batch)...")
    out = []
    batch = 100
    for i in range(0, len(ids), batch):
        chunk = ids[i:i+batch]
        r = requests.get(ESUMMARY, params={
            "db": "books", "id": ",".join(chunk), "retmode": "json",
        }, headers=HEADERS, timeout=30)
        try:
            data = r.json()
        except Exception:
            time.sleep(2.0); continue
        for uid in chunk:
            item = data["result"].get(uid)
            if not item: continue
            # title 에서 약물명 추출 (보통 "Drug Name" 또는 "Drug Name: ...")
            title = item.get("title", "")
            # filter — root, prefix, intro chapter 제외
            if title in ("LiverTox", "INTRODUCTION", "INDEX"): continue
            out.append({
                "id": uid,
                "title": title,
                "accession": item.get("chapteraccessionid") or item.get("publisherreport"),
            })
        time.sleep(SLEEP)
        if i % 500 == 0 and i > 0:
            print(f"    {i}/{len(ids)}")
    return out


def fetch_chapter_likelihood(accession):
    """단일 chapter 의 본문에서 Likelihood Score 추출."""
    if not accession: return None
    url = f"{BOOKURL}/{accession}/"
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200: return None
        # Likelihood Score: A, B, C, D, E
        # HTML 에서 "Likelihood score: <strong>X" pattern 찾기
        # 또는 "LIKELIHOOD SCORE\nDescription: <X>"
        text = r.text
        # 패턴 1: "Likelihood Score: X"
        m = re.search(r"Likelihood Score[:\s]*<[^>]+>([A-E])(?:\s|<|,)", text)
        if m: return m.group(1)
        m = re.search(r"Likelihood Score[:\s]+([A-E])(?:\s|<|,|—|-|:)", text)
        if m: return m.group(1)
        # 패턴 2: "LIKELIHOOD CATEGORY: X (description)"
        m = re.search(r"LIKELIHOOD\s+(?:SCORE|CATEGORY)[:\s]+([A-E])", text)
        if m: return m.group(1)
        # 패턴 3: 영문 description
        for ll, score in [("Well known to cause clinically apparent liver injury", "A"),
                          ("Well-known cause of clinically apparent liver injury", "A"),
                          ("Highly likely or known cause of clinically apparent", "A"),
                          ("Likely or known cause of clinically apparent", "B"),
                          ("Probable cause of clinically apparent", "C"),
                          ("Possible but rare cause", "D"),
                          ("Unlikely cause of clinically apparent liver injury", "E"),
                          ("Unproven but suspected", "C"),
                          ("Probable but rare cause", "D")]:
            if ll.lower() in text.lower():
                return score
        return None
    except Exception:
        return None


def lookup_smiles(names):
    import pubchempy as pcp
    out = {}
    for i, name in enumerate(names):
        if not name: continue
        try:
            res = pcp.get_compounds(name, "name")
            if res: out[name] = res[0].canonical_smiles
        except Exception:
            pass
        time.sleep(SLEEP_PUBCHEM)
        if i % 50 == 0 and i > 0:
            print(f"    PubChem: {i}/{len(names)} 매핑 {len(out)}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== LiverTox 전체 chapter 확장 scrape ===\n")
    ids = get_livertox_chapter_ids()
    metas = fetch_chapter_metadata(ids)
    print(f"\nchapter 메타: {len(metas)}")

    # 각 chapter 의 likelihood
    print(f"\n각 chapter 본문에서 Likelihood Score 추출 ({len(metas)})")
    out = []
    for i, meta in enumerate(metas):
        score = fetch_chapter_likelihood(meta["accession"])
        if score:
            # 약물명 = title (filter)
            name = meta["title"].split(":")[0].split("—")[0].strip()
            if len(name) < 2 or len(name) > 80: continue
            out.append({
                "name": name.lower(),
                "likelihood": score,
                "accession": meta["accession"],
            })
        time.sleep(SLEEP)
        if i % 50 == 0 and i > 0:
            print(f"    {i}/{len(metas)} 처리, 추출 {len(out)}")

    df = pd.DataFrame(out).drop_duplicates(subset=["name"])
    print(f"\n수집 완료: {len(df)} unique (likelihood 있는 약물)")
    print(f"  분포: {df['likelihood'].value_counts().to_dict()}")
    df.to_csv(os.path.join(OUT_DIR, "livertox_extended_raw.csv"), index=False)

    # PubChem SMILES
    print(f"\nPubChem SMILES 매핑 ({len(df)})")
    name_to_smi = lookup_smiles(df["name"].tolist())
    df["canonical_smiles"] = df["name"].map(name_to_smi)
    final = df.dropna(subset=["canonical_smiles"]).copy()
    final.to_csv(os.path.join(OUT_DIR, "livertox_extended.csv"), index=False)
    print(f"\n저장: {OUT_DIR}/livertox_extended.csv ({len(final)} / {len(df)})")


if __name__ == "__main__":
    main()
