"""Open Targets — drug-adverse event 큐레이션.

Open Targets 는 EMBL-EBI + Wellcome Sanger + GSK 가 공동 운영하는
큐레이션 drug-disease DB. 공개 GraphQL API 제공.

전략:
  1. Open Targets bulk download (parquet 형식)
     - drug index: ChEMBL ID + name + structure
     - drug-AE 매핑: significantAdverseEvents
  2. hepatic AE 관련 term 필터:
     - hepatic failure, hepatotoxicity, drug-induced liver injury
     - hepatitis, cholestasis, jaundice
  3. ChEMBL ID → SMILES (ChEMBL local 또는 UniChem)
  4. n_significantAEs 강도별로 라벨 생성

저장:
  data/raw/open_targets/opentargets_dili.csv
"""
from __future__ import annotations
import gzip, json, os, re, time
import pandas as pd
import requests

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "open_targets")
HEADERS = {"User-Agent": "Mozilla/5.0 research"}

# Open Targets v24.06 (최신 stable, parquet 다운로드)
OT_BASE = "https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/24.06"
URL_AE_BY_DRUG = f"{OT_BASE}/output/etl/parquet/significantAdverseDrugReactions"
URL_DRUG = f"{OT_BASE}/output/etl/parquet/molecule"

# GraphQL fallback (작은 query)
GQL_URL = "https://api.platform.opentargets.org/api/v4/graphql"

LIVER_TERMS = re.compile(
    r"(hepat|liver|cholestasis|jaundice|drug-induced liver|hepatic failure"
    r"|hepatitis|hepatocellular|hepatorenal|biliary)",
    re.IGNORECASE,
)


def graphql_drug_aes(chembl_id: str) -> dict:
    """Drug 의 adverse events 추출 (GraphQL)."""
    query = """
    query DrugAEs($id: String!) {
      drug(chemblId: $id) {
        id
        name
        adverseEvents(page: {index: 0, size: 100}) {
          count
          rows { name criticalValue logLR }
        }
      }
    }
    """
    r = requests.post(GQL_URL, json={"query": query, "variables": {"id": chembl_id}},
                       headers=HEADERS, timeout=30)
    if r.status_code != 200:
        return None
    return r.json().get("data", {}).get("drug")


def search_drug_by_name(name: str) -> str | None:
    """약물 이름으로 ChEMBL ID 검색."""
    # 새 OT GraphQL API 의 search query 형식
    query = """
    query Search($q: String!) {
      search(queryString: $q, entityNames: ["drug"], page: {index: 0, size: 3}) {
        hits {
          id
          name
          entity
          object {
            ... on Drug {
              id
              name
            }
          }
        }
      }
    }
    """
    try:
        r = requests.post(GQL_URL, json={"query": query, "variables": {"q": name}},
                           headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        data = r.json()
        if "errors" in data: return None
        hits = data.get("data", {}).get("search", {}).get("hits", []) or []
        for h in hits:
            entity = h.get("entity")
            hid = h.get("id", "")
            if entity == "drug" and hid.startswith("CHEMBL"):
                return hid
        return None
    except Exception:
        return None


def fetch_via_graphql(names: list[str]) -> pd.DataFrame:
    """이름으로 ChEMBL ID 찾고 → adverse events → liver 필터."""
    rows = []
    for i, name in enumerate(names):
        chembl_id = search_drug_by_name(name)
        if not chembl_id:
            continue
        try:
            data = graphql_drug_aes(chembl_id)
        except Exception:
            time.sleep(1.0)
            continue
        if not data: continue
        aes = data.get("adverseEvents", {}).get("rows", [])
        liver_aes = [a for a in aes if LIVER_TERMS.search(a.get("name", ""))]
        if not liver_aes:
            continue
        rows.append({
            "chembl_id": chembl_id,
            "name": data.get("name", name),
            "n_total_AE": data.get("adverseEvents", {}).get("count", 0),
            "n_liver_AE": len(liver_aes),
            "max_logLR": max([float(a.get("logLR", 0)) for a in liver_aes]),
            "liver_AEs": ";".join(a["name"] for a in liver_aes[:10]),
        })
        time.sleep(0.5)
        if i % 50 == 0 and i > 0:
            print(f"  진행 {i}/{len(names)} 매핑 {len(rows)}")
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== Open Targets — drug adverse events (hepatic 필터) ===\n")

    # 우리 marketed unknown 분자 중 name 있는 것
    mc = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "raw", "marketed",
                                    "marketed_clean.csv"))
    print(f"전체 marketed: {len(mc)}")
    # 우리 DB 와 매칭 — unknown 만 (DILIrank 평가 안 됨)
    db = pd.read_parquet(os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet"))
    db_iks = set(db[db.vivo_label.notna()].inchi_key)
    # marketed 중 우리 DB 에 vivo_label 없는 것
    target = mc[~mc.inchi_key.isin(db_iks)].copy()
    target = target[target.name.notna() & (target.name.str.len() > 2)]
    print(f"대상 (DB 라벨 없음): {len(target)}")
    # 최대 2000개 정도로 제한 — 시간 제약
    target = target.drop_duplicates(subset=["inchi_key"]).head(2000)
    print(f"이번 query: {len(target)} (GraphQL)")

    df = fetch_via_graphql(target["name"].tolist())
    if df.empty:
        print("hepatic AE 매핑된 약물 없음")
        return
    print(f"\n매핑된 hepatic AE 약물: {len(df)}")

    # SMILES 매핑 — marketed_clean 에서 직접
    df_with_smi = df.merge(
        mc[["name", "canonical_smiles", "inchi_key"]],
        on="name", how="left").dropna(subset=["canonical_smiles"])
    print(f"SMILES 매칭: {len(df_with_smi)}")

    # 강도 코드
    def code(row):
        n = row["n_liver_AE"]; lr = row["max_logLR"]
        if n >= 5 or lr >= 4.0: return "strong"
        if n >= 2 or lr >= 2.0: return "medium"
        return "weak"
    df_with_smi["vivo_open_targets"] = df_with_smi.apply(code, axis=1)

    out_path = os.path.join(OUT_DIR, "opentargets_dili.csv")
    df_with_smi.to_csv(out_path, index=False)
    print(f"\n저장: {out_path}")
    print(f"강도 분포: {df_with_smi.vivo_open_targets.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
