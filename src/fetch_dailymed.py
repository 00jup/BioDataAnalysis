"""DailyMed / openFDA Drug Label 에서 hepatotoxicity 정보 추출.

전략:
  1. openFDA /drug/label.json API 로 약물 라벨 수집
  2. 각 라벨에서 hepatic/liver 관련 경고 추출
  3. 심각도 분류:
     - BOXED_WARNING + 간 관련 → "boxed_hepatotox" (가장 강한 양성)
     - WARNINGS_AND_PRECAUTIONS 간 관련 → "warning_hepatotox"
     - ADVERSE_REACTIONS 간 관련 → "adverse_hepatotox"
     - 명시 없음 → "no_signal"
  4. 약물명 → PubChem SMILES 매핑

저장:
  data/raw/dailymed/dailymed_raw.csv (스크래핑 원본)
  data/raw/dailymed/dailymed.csv      (SMILES 매핑 후)
"""

from __future__ import annotations

import os
import re
import time

import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "dailymed")
HEADERS = {"User-Agent": "research scraper / ai.jeje.labs@gmail.com"}
SLEEP_API = 0.3
SLEEP_PUBCHEM = 0.35

# 간독성 관련 키워드 (NLP 추출용)
LIVER_TERMS = [
    "hepatotoxic",
    "hepatic failure",
    "hepatic injury",
    "hepatitis",
    "liver injury",
    "liver damage",
    "liver failure",
    "liver toxic",
    "hepatocellular",
    "cholestasis",
    "jaundice",
    "hepatorenal",
    "transaminase",
    "ALT",
    "AST elevation",
    "GGT",
    "bilirubin",
    "elevated liver enzymes",
    "drug-induced liver",
]
LIVER_REGEX = re.compile("|".join(LIVER_TERMS), re.IGNORECASE)


def search_openfda(query: str, limit: int = 100, skip: int = 0) -> dict:
    """openFDA Drug Label API 검색."""
    url = "https://api.fda.gov/drug/label.json"
    params = {"search": query, "limit": limit, "skip": skip}
    r = requests.get(url, params=params, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        return {"results": [], "meta": {"results": {"total": 0}}}
    r.raise_for_status()
    return r.json()


def classify_label(item: dict) -> dict | None:
    """라벨 1개에서 간독성 신호 추출."""
    # 약물명 (active ingredient)
    name = None
    if item.get("openfda", {}).get("generic_name"):
        name = item["openfda"]["generic_name"][0]
    elif item.get("openfda", {}).get("brand_name"):
        name = item["openfda"]["brand_name"][0]
    elif item.get("openfda", {}).get("substance_name"):
        name = item["openfda"]["substance_name"][0]
    if not name:
        return None

    # 섹션별 간독성 키워드 검색
    severity = "no_signal"
    snippet = ""
    sources = []

    # 1) Boxed Warning — 가장 강한 신호
    box_text = " ".join(item.get("boxed_warning", []))
    if LIVER_REGEX.search(box_text):
        severity = "boxed_hepatotox"
        m = LIVER_REGEX.search(box_text)
        snippet = box_text[max(0, m.start() - 60) : m.end() + 60]
        sources.append("boxed_warning")

    # 2) Warnings & Precautions
    if severity == "no_signal":
        warn_text = " ".join(item.get("warnings_and_cautions", []) + item.get("warnings", []))
        if LIVER_REGEX.search(warn_text):
            severity = "warning_hepatotox"
            m = LIVER_REGEX.search(warn_text)
            snippet = warn_text[max(0, m.start() - 60) : m.end() + 60]
            sources.append("warnings")

    # 3) Adverse Reactions
    if severity == "no_signal":
        adv_text = " ".join(item.get("adverse_reactions", []))
        if LIVER_REGEX.search(adv_text):
            severity = "adverse_hepatotox"
            m = LIVER_REGEX.search(adv_text)
            snippet = adv_text[max(0, m.start() - 60) : m.end() + 60]
            sources.append("adverse_reactions")

    # 4) Contraindications (hepatic impairment 등 — 보조 신호)
    contra_text = " ".join(item.get("contraindications", []))
    if "hepatic" in contra_text.lower() or "liver" in contra_text.lower():
        if severity == "no_signal":
            severity = "contraindication_hepatic"
        sources.append("contraindications")

    return {
        "name": name.strip().lower(),
        "severity": severity,
        "snippet": snippet.strip() if snippet else "",
        "sources": ";".join(sources),
        "set_id": item.get("set_id"),
        "spl_id": item.get("id"),
    }


def collect_labels() -> pd.DataFrame:
    """간독성 관련 라벨 모두 수집 (4 섹션 별로 검색)."""
    queries = [
        "boxed_warning:(hepatotoxic OR hepatic OR liver)",
        "warnings_and_cautions:(hepatotoxic OR hepatitis OR liver_injury OR hepatic_failure)",
        "warnings:(hepatotoxic OR hepatic_injury)",
        "adverse_reactions:(hepatitis OR hepatotoxicity OR liver_injury)",
        # 음성 신호: 간 관련 contraindication
        "contraindications:(hepatic_impairment OR liver_failure)",
    ]
    all_results = {}  # set_id → record (dedup)
    for q in queries:
        print(f"\n[검색] {q[:60]}...")
        skip = 0
        while True:
            try:
                data = search_openfda(q, limit=100, skip=skip)
            except Exception as e:
                print(f"  API 오류: {e}")
                break
            results = data.get("results", [])
            if not results:
                break
            for r in results:
                rec = classify_label(r)
                if rec and rec["set_id"]:
                    sid = rec["set_id"]
                    if sid in all_results:
                        # severity 가 더 강하면 교체
                        prev = all_results[sid]
                        order = {
                            "no_signal": 0,
                            "contraindication_hepatic": 1,
                            "adverse_hepatotox": 2,
                            "warning_hepatotox": 3,
                            "boxed_hepatotox": 4,
                        }
                        if order[rec["severity"]] > order[prev["severity"]]:
                            all_results[sid] = rec
                    else:
                        all_results[sid] = rec
            skip += 100
            print(f"  skip={skip - 100} → {len(results)} (누적 unique {len(all_results)})")
            time.sleep(SLEEP_API)
            if skip >= 1000:  # openFDA 1000 limit
                break
    return pd.DataFrame(list(all_results.values()))


def lookup_smiles(names: list[str]) -> dict[str, str]:
    """약물명 → canonical SMILES (PubChem)."""
    import pubchempy as pcp

    out = {}
    seen = set()
    unique_names = list(set(names))
    for i, name in enumerate(unique_names):
        if name in seen:
            continue
        seen.add(name)
        try:
            res = pcp.get_compounds(name, "name")
            if res:
                out[name] = res[0].canonical_smiles
        except Exception:
            pass
        time.sleep(SLEEP_PUBCHEM)
        if i % 100 == 0 and i > 0:
            print(f"    PubChem: {i}/{len(unique_names)} 매핑 {len(out)}")
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== openFDA Drug Label 간독성 정보 수집 ===\n")
    raw_df = collect_labels()
    print(f"\n=== 수집 완료: {len(raw_df)} 약물 라벨 ===")
    print(f"심각도 분포: {raw_df['severity'].value_counts().to_dict()}")

    raw_path = os.path.join(OUT_DIR, "dailymed_raw.csv")
    raw_df.to_csv(raw_path, index=False)
    print(f"raw 저장: {raw_path}")

    # PubChem SMILES 매핑
    print(f"\n=== PubChem 매핑 ({raw_df['name'].nunique()} 약물) ===")
    name_to_smi = lookup_smiles(raw_df["name"].tolist())
    raw_df["canonical_smiles"] = raw_df["name"].map(name_to_smi)
    final = raw_df.dropna(subset=["canonical_smiles"]).copy()
    final = final[
        ["name", "canonical_smiles", "severity", "sources", "snippet", "set_id", "spl_id"]
    ]
    final_path = os.path.join(OUT_DIR, "dailymed.csv")
    final.to_csv(final_path, index=False)
    print(f"\n저장: {final_path}  ({len(final)} / {len(raw_df)} SMILES 매핑)")
    print(f"심각도 분포 (매핑 후): {final['severity'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
