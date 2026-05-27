"""통합 라벨 DB 구축 — OR 룰 + curated 충돌 lookup (가중치 코드 폐기).

라벨링 룰
=========
in vivo (임상 데이터):
  1. 양성 신호 only → 1
  2. 음성 신호 only → 0
  3. 양성 + 음성 충돌 → conflicts_curated.csv 의 manual_label
     - DILIrank/LiverTox/DM Boxed 보유: 공식 라벨 그대로
     - 약 지식 사전 매칭: 잘 알려진 시판약 양성/음성 분류
     - 비약물 화학물질 또는 무명 화합물: 제외 (None)
  4. DILIrank=Ambiguous: 제외
  5. 신호 없음: 제외

in vitro (어세이): Tox21 만 사용. chEMBL 은 vivo 로 재분류됨.

저장 형식
=========
data/labels_db/full.parquet — 핵심 컬럼:
  inchi_key, canonical_smiles, name,
  vivo_dilirank, vivo_livertox, vivo_dailymed, vivo_pubmed, vivo_ctd,
  vivo_faers, vivo_chembl, vivo_marketed_clean_neg,
  vivo_label,    ← 최종 vivo 라벨 (1/0/None)
  vitro_tox21,
  vitro_label    ← 최종 vitro 라벨 (1/0/None)

제거된 컬럼 (가중치 룰 폐기):
  vivo_confidence, vivo_n_sources, vitro_confidence, vitro_n_sources,
  label_vivo_priority, label_weighted, label_consensus, final_source, conflict
"""

from __future__ import annotations

import os

import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(PROJECT_ROOT, "data", "raw")
EXP = os.path.join(PROJECT_ROOT, "data", "experiments")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "labels_db")


# ---- SMILES Standardize (RDKit MolStandardize 체인) ----
def standardize(smi: str) -> tuple[str, str] | None:
    """SMILES → (canonical_smiles, inchi_key). 표준화 실패 시 None."""
    if not isinstance(smi, str) or not smi.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smi.strip())
        if mol is None:
            return None
        # 1. 큰 단편만 보관 (염 제거)
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = rdMolStandardize.Cleanup(mol)
        chooser = rdMolStandardize.LargestFragmentChooser()
        mol = chooser.choose(mol)
        # 2. 전하 중화
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)
        # 3. canonical
        canon = Chem.MolToSmiles(mol, canonical=True)
        ik = Chem.MolToInchiKey(mol)
        if not canon or not ik:
            return None
        return canon, ik
    except Exception:
        return None


def _std_df(df: pd.DataFrame, smi_col: str = "canonical_smiles") -> pd.DataFrame:
    out = df.copy()
    out["__pair"] = out[smi_col].map(standardize)
    out = out[out["__pair"].notna()].copy()
    out["canonical_smiles"] = out["__pair"].map(lambda p: p[0])
    out["inchi_key"] = out["__pair"].map(lambda p: p[1])
    return out.drop(columns=["__pair"])


# ---- 소스별 로더 ----
def load_dilirank_full() -> pd.DataFrame:
    """FDA DILIrank 전체 (raw/dilirank/dilirank_full.csv) — 모든 카테고리."""
    p = pd.read_csv(os.path.join(RAW, "dilirank", "dilirank_full.csv"))
    p = p.dropna(subset=["canonical_smiles", "dilirank_category"])
    p["vivo_dilirank"] = p["dilirank_category"]
    return p[["canonical_smiles", "name", "vivo_dilirank"]]


def load_livertox() -> pd.DataFrame:
    """LiverTox Likelihood Score (A-E)."""
    p = pd.read_csv(os.path.join(RAW, "livertox", "livertox.csv"))
    p = p.dropna(subset=["canonical_smiles", "likelihood"])
    p["vivo_livertox"] = p["likelihood"]
    return p[["canonical_smiles", "name", "vivo_livertox"]]


def load_dailymed() -> pd.DataFrame:
    """DailyMed/openFDA Drug Label — hepatotox 심각도별 라벨."""
    p = pd.read_csv(os.path.join(RAW, "dailymed", "dailymed.csv"))
    p = p.dropna(subset=["canonical_smiles", "severity"])
    # severity 별 코드화
    p["vivo_dailymed"] = p["severity"]
    return p[["canonical_smiles", "name", "vivo_dailymed"]]


def load_pubmed() -> pd.DataFrame:
    """PubMed DILI MeSH 기반 — n_papers 빈도로 신뢰도 코드화."""
    p = pd.read_csv(os.path.join(RAW, "pubmed", "pubmed_dili.csv"))
    p = p.dropna(subset=["canonical_smiles", "n_papers"])

    # n_papers 별 강도
    #   >= 20 = strong (Acetaminophen 같은 well-known hepatotox)
    #   5-19 = medium
    #   3-4  = weak
    def code(n):
        if n >= 20:
            return "strong"
        if n >= 5:
            return "medium"
        return "weak"

    p["vivo_pubmed"] = p["n_papers"].apply(code)
    p["pubmed_n_papers"] = p["n_papers"]
    return p[["canonical_smiles", "name", "vivo_pubmed", "pubmed_n_papers"]]


def load_ctd() -> pd.DataFrame:
    """CTD chemical-disease 매핑 (NIH/EPA 큐레이션).

    강한 증거 (strong_evidence ≥ 1) 또는 다수 evidence (n_evidence ≥ 5) 보유한
    chemical 만. liver disease 관련 화합물 ~3,000개.
    """
    path = os.path.join(RAW, "ctd", "ctd_dili.csv")
    if not os.path.exists(path):
        return pd.DataFrame(
            columns=["canonical_smiles", "name", "vivo_ctd", "ctd_strong", "ctd_pmid"]
        )
    p = pd.read_csv(path)
    p = p.dropna(subset=["canonical_smiles"])

    # 강도 코드화
    def code(row):
        if row.get("strong_evidence", 0) >= 1:
            return "strong"
        if row.get("n_evidence", 0) >= 10:
            return "medium"
        return "weak"

    p["vivo_ctd"] = p.apply(code, axis=1)
    p["ctd_strong"] = p.get("strong_evidence", 0)
    p["ctd_pmid"] = p.get("max_pmid", 0)
    return p[["canonical_smiles", "name", "vivo_ctd", "ctd_strong", "ctd_pmid"]]


def load_faers() -> pd.DataFrame:
    """FAERS (FDA Adverse Event Reporting) — 환자 보고 hepatic AE.

    n_reports 별 강도 — 보고 수가 많을수록 강한 신호.
    """
    path = os.path.join(RAW, "faers", "faers_dili.csv")
    if not os.path.exists(path):
        return pd.DataFrame(columns=["canonical_smiles", "name", "vivo_faers", "faers_n_reports"])
    p = pd.read_csv(path)
    p = p.dropna(subset=["canonical_smiles"])

    def code(n):
        if n >= 5000:
            return "strong"
        if n >= 1000:
            return "medium"
        return "weak"

    p["vivo_faers"] = p["n_reports"].apply(code)
    p["faers_n_reports"] = p["n_reports"]
    return p[["canonical_smiles", "name", "vivo_faers", "faers_n_reports"]]


def load_sider_strict() -> pd.DataFrame:
    """SIDER 간 부작용 strict (간 키워드 매칭)."""
    p = pd.read_csv(os.path.join(RAW, "sider", "sider_liver_strict.csv"))
    p = p.dropna(subset=["canonical_smiles", "inchi_key"])
    p["vivo_sider_liver"] = 1
    return p[["canonical_smiles", "inchi_key", "name", "vivo_sider_liver"]]


def load_sider_lenient() -> pd.DataFrame:
    """SIDER hepatotoxic lenient (더 넓은 매칭, 229 추가)."""
    p = pd.read_csv(os.path.join(RAW, "sider", "sider_hepatotoxic_lenient.csv"))
    p = p.dropna(subset=["canonical_smiles", "inchi_key"])
    p["vivo_sider_hepatotox"] = 1
    return p[["canonical_smiles", "inchi_key", "name", "vivo_sider_hepatotox"]]


def load_dilist_gold() -> pd.DataFrame:
    """external_positives.csv (DILIst + Gold 분리)."""
    p = pd.read_csv(os.path.join(RAW, "dilist_gold", "external_positives.csv"))
    p = p.dropna(subset=["canonical_smiles", "inchi_key"])
    p["vivo_dilist"] = (p["source_label"] == "dilist").astype("Int64")
    p["vivo_gold"] = (p["source_label"] == "goldstandard").astype("Int64")
    return p[["canonical_smiles", "inchi_key", "name", "vivo_dilist", "vivo_gold"]]


def load_tdc_dili() -> pd.DataFrame:
    p = pd.read_csv(os.path.join(RAW, "tdc_dili", "tdc_dili.csv"))
    p = p.dropna(subset=["canonical_smiles", "inchi_key"])
    p["vivo_tdc_dili"] = 1
    return p[["canonical_smiles", "inchi_key", "name", "vivo_tdc_dili"]]


def load_marketed_clean() -> pd.DataFrame:
    """시판약 음성 풀 — DILIrank 카테고리도 함께."""
    p = pd.read_csv(os.path.join(RAW, "marketed", "marketed_clean.csv"))
    p = p.dropna(subset=["canonical_smiles", "inchi_key"]).drop_duplicates("inchi_key")
    p["vivo_marketed_clean_neg"] = (p["dilirank_category"] != "Ambiguous-DILI-Concern").astype(
        "Int64"
    )
    p["vivo_dilirank"] = p["dilirank_category"]
    return p[["canonical_smiles", "inchi_key", "name", "vivo_marketed_clean_neg", "vivo_dilirank"]]


def load_clintox():
    """ClinTox from raw/clintox/clintox.csv."""
    data = pd.read_csv(os.path.join(RAW, "clintox", "clintox.csv"))
    rows = []
    for _, r in data.iterrows():
        std = standardize(r["Drug"])
        if std:
            rows.append(
                {
                    "canonical_smiles": std[0],
                    "inchi_key": std[1],
                    "name": r.get("Drug_ID", ""),
                    "vivo_clintox": int(r["Y"]),
                }
            )
    return pd.DataFrame(rows).drop_duplicates("inchi_key")


def load_chembl() -> pd.DataFrame:
    """chEMBL hepatotoxicity — 사람 (Homo sapiens) 임상 데이터.

    confirmed via assay_organism = 'Homo sapiens' (7,881개, 100%),
    activity_comment = 'HH: Evidence of human hepatotoxicity' 등.
    이전엔 vitro 로 잘못 분류 — vivo 가 맞다.

    한 컬럼 'vivo_chembl' 로 라벨 부여.
    """
    POS_TERMS = {
        "drug-induced liver injury reported",
        "Toxic",
        "Most-Dili-Concern",
        "Less-Dili-Concern",
        "HH: Evidence of human hepatotoxicity",
        "DILI positive (training set)",
        "DILI positive (test set)",
    }
    NEG_TERMS = {
        "no drug-induced liver injury reported",
        "Non-Toxic",
        "Non-toxic",
        "No-Dili-Concern",
    }

    pos_set = pd.read_csv(os.path.join(RAW, "chembl", "chembl_toxic_positive_set.csv"))
    compounds = pd.read_csv(os.path.join(RAW, "chembl", "chembl_hepatotoxicity_compounds.csv"))

    p1 = pos_set.copy()
    p1["label"] = 1
    p2 = compounds[compounds["activity_comment"].isin(POS_TERMS)].copy()
    p2["label"] = 1
    n = compounds[compounds["activity_comment"].isin(NEG_TERMS)].copy()
    n["label"] = 0

    pool = (
        pd.concat(
            [
                p1[["canonical_smiles", "molecule_chembl_id", "label"]],
                p2[["canonical_smiles", "molecule_chembl_id", "label"]],
                n[["canonical_smiles", "molecule_chembl_id", "label"]],
            ],
            ignore_index=True,
        )
        .dropna(subset=["canonical_smiles"])
        .drop_duplicates(["molecule_chembl_id", "label"])
    )

    # molecule_chembl_id 단위 라벨 충돌 제거
    g = pool.groupby("molecule_chembl_id")["label"].nunique()
    pool = pool[~pool["molecule_chembl_id"].isin(set(g[g > 1].index))]

    pool["vivo_chembl"] = pool["label"].astype("Int64")
    pool["name"] = pool["molecule_chembl_id"]
    return pool[["canonical_smiles", "name", "vivo_chembl"]].drop_duplicates("canonical_smiles")


def load_tox21() -> pd.DataFrame:
    """Tox21 통합 from raw/tox21/tox21_combined_5tasks.csv."""
    data = pd.read_csv(os.path.join(RAW, "tox21", "tox21_combined_5tasks.csv"))
    rows = []
    for _, r in data.iterrows():
        std = standardize(r["smiles"])
        if std:
            rows.append(
                {
                    "canonical_smiles": std[0],
                    "inchi_key": std[1],
                    "name": "",
                    "vitro_tox21": int(r["label_any_pos"]),
                    "tox21_pos_count": int(r["n_assays_pos"]),
                    "tox21_total_assays": int(r["n_assays_tested"]),
                }
            )
    df = pd.DataFrame(rows).drop_duplicates("inchi_key")
    print(
        f"  Tox21 통합: {len(df)} 분자 (양성 {int((df.vitro_tox21 == 1).sum())} / 음성 {int((df.vitro_tox21 == 0).sum())})"
    )
    return df


def merge_and_label(loaded: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """InChIKey 기준으로 모든 출처 병합 후 vivo/vitro 라벨 결정."""
    # 1. 모든 SMILES 표준화 (출처별 std → 같은 InChIKey 가 다른 SMILES 가질 수 있음)
    for name, df in loaded.items():
        df = _std_df(df, "canonical_smiles")
        loaded[name] = df

    # 2. InChIKey unique 집합
    all_ik = set()
    for df in loaded.values():
        all_ik |= set(df["inchi_key"])

    # 3. 각 분자에 대해 모든 라벨 수집
    records = []
    smiles_per_ik = {}
    name_per_ik = {}
    for ik in all_ik:
        rec = {"inchi_key": ik}
        for sname, df in loaded.items():
            row = df[df["inchi_key"] == ik]
            if len(row) > 0:
                # 컬럼 채워주기
                for col in row.columns:
                    if col not in ("inchi_key",):
                        v = row.iloc[0][col]
                        if col == "canonical_smiles":
                            smiles_per_ik.setdefault(ik, v)
                        elif col == "name":
                            if isinstance(v, str) and v and ik not in name_per_ik:
                                name_per_ik[ik] = v
                        else:
                            rec[col] = v
        rec["canonical_smiles"] = smiles_per_ik.get(ik, "")
        rec["name"] = name_per_ik.get(ik, "")
        records.append(rec)
    df = pd.DataFrame(records)

    # 4. vivo / vitro 라벨 컬럼 정렬 (없는 컬럼은 NaN)
    vivo_cols = [
        "vivo_dilirank",
        "vivo_livertox",
        "vivo_dailymed",
        "vivo_pubmed",
        "vivo_ctd",
        "vivo_faers",
        "vivo_chembl",  # chEMBL human assay 추가
        "vivo_dilist",
        "vivo_gold",
        "vivo_sider_liver",
        "vivo_sider_hepatotox",
        "vivo_tdc_dili",
        "vivo_clintox",
        "vivo_marketed_clean_neg",
    ]
    vitro_cols = ["vitro_tox21"]  # chEMBL 은 vivo 로 옮김
    for c in vivo_cols + vitro_cols + ["tox21_pos_count", "tox21_total_assays"]:
        if c not in df.columns:
            df[c] = pd.NA

    # 5. vivo label 결정 — OR 룰 + 충돌은 curated lookup
    #
    # 새 룰 (가중치 코드 폐기):
    #   - 양성 신호만 있음 → 1
    #   - 음성 신호만 있음 → 0
    #   - 둘 다 있음 (충돌) → conflicts_curated.csv 에서 manual_label 조회
    #   - 둘 다 없음 / DILIrank=Ambiguous → None
    #
    def has_pos_signal(row) -> bool:
        if row.get("vivo_dilirank") in ("vMost-DILI-Concern", "vLess-DILI-Concern"):
            return True
        if row.get("vivo_livertox") in ("A", "B", "C", "D"):
            return True
        if row.get("vivo_dailymed") in (
            "boxed_hepatotox",
            "warning_hepatotox",
            "adverse_hepatotox",
        ):
            return True
        if row.get("vivo_pubmed") in ("strong", "medium", "weak"):
            return True
        if row.get("vivo_ctd") in ("strong", "medium", "weak"):
            return True
        if row.get("vivo_faers") in ("strong", "medium", "weak"):
            return True
        ce = row.get("vivo_chembl")
        if pd.notna(ce) and int(ce) == 1:
            return True
        return False

    def has_neg_signal(row) -> bool:
        if row.get("vivo_dilirank") == "vNo-DILI-Concern":
            return True
        if row.get("vivo_livertox") == "E":
            return True
        ce = row.get("vivo_chembl")
        if pd.notna(ce) and int(ce) == 0:
            return True
        if pd.notna(row.get("vivo_marketed_clean_neg")) and row["vivo_marketed_clean_neg"] == 1:
            return True
        return False

    # 충돌 curated lookup 로드
    curated_path = os.path.join(OUT_DIR, "conflicts", "conflicts_curated.csv")
    curated_map: dict[str, float | None] = {}
    if os.path.exists(curated_path):
        cur = pd.read_csv(curated_path)
        for _, r in cur.iterrows():
            lab = r["manual_label"]
            curated_map[r["inchi_key"]] = None if pd.isna(lab) else int(lab)
        print(
            f"  curated 충돌 라벨: {len(curated_map)}건 로드 (양성 "
            f"{sum(1 for v in curated_map.values() if v == 1)}, "
            f"음성 {sum(1 for v in curated_map.values() if v == 0)}, "
            f"제외 {sum(1 for v in curated_map.values() if v is None)})"
        )
    else:
        print("  ⚠️  curated 파일 없음 — 충돌 케이스 전부 제외됨")

    def vivo_decide(row):
        # Ambiguous → 제외
        if row.get("vivo_dilirank") == "Ambiguous-DILI-Concern":
            return None
        pos = has_pos_signal(row)
        neg = has_neg_signal(row)
        if not pos and not neg:
            return None
        if pos and not neg:
            return 1
        if neg and not pos:
            return 0
        # 충돌 — curated lookup
        return curated_map.get(row["inchi_key"], None)

    df["vivo_label"] = df.apply(vivo_decide, axis=1)

    # 6. vitro label 결정 (Tox21 만)
    def vitro_decide(row):
        if pd.notna(row.get("vitro_tox21")):
            return int(row["vitro_tox21"])
        return None

    df["vitro_label"] = df.apply(vitro_decide, axis=1)
    return df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== 통합 라벨 DB 빌드 (data/raw/ → data/labels_db/) ===\n")
    print("[1/10] DILIrank 전체 (FDA, 1,223)")
    drk = load_dilirank_full()
    print(f"  {len(drk)} 분자")
    print(f"  카테고리: {drk['vivo_dilirank'].value_counts().to_dict()}")

    print("\n[2/10] LiverTox (NIH, A-E)")
    lt = load_livertox()
    print(f"  {len(lt)} 분자")
    print(f"  Likelihood: {lt['vivo_livertox'].value_counts().to_dict()}")

    # 약한 출처 (Gold/DILIst/SIDER/TDC/ClinTox) — _unreliable 로 분리 (학습 미사용)
    # source_reliability 분석 결과 AUC ≤ 0.58 → 노이즈 우세 → 제외
    # raw 데이터는 data/raw/_unreliable/ 에 보존
    dl = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_dilist", "vivo_gold"])
    sd_s = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_sider_liver"])
    sd_l = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_sider_hepatotox"])
    td = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_tdc_dili"])
    ct = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_clintox"])

    print("\n[3/10] 시판약 음성 풀")
    mc = load_marketed_clean()
    print(f"  {len(mc)} 분자")

    print("[9/10] chEMBL (in vitro)")
    ch = load_chembl()
    print(
        f"  {len(ch)} 분자 (양성 {int((ch.vivo_chembl == 1).sum())} / 음성 {int((ch.vivo_chembl == 0).sum())}) — vivo 로 재분류"
    )

    print("[10/11] Tox21 통합 (in vitro)")
    tx = load_tox21()

    print("\n[11/12] DailyMed / openFDA 라벨")
    dm = load_dailymed()
    print(f"  {len(dm)} 분자 (심각도: {dm['vivo_dailymed'].value_counts().to_dict()})")

    print("\n[12/14] PubMed DILI MeSH (strict: Major Topic + 임상 publication types)")
    pm = load_pubmed()
    print(f"  {len(pm)} 분자 (강도: {pm['vivo_pubmed'].value_counts().to_dict()})")

    print("\n[13/14] CTD (NIH/EPA chemical-disease 매핑)")
    ctd = load_ctd()
    if len(ctd) > 0:
        print(f"  {len(ctd)} 분자 (강도: {ctd['vivo_ctd'].value_counts().to_dict()})")
    else:
        print("  데이터 없음 — fetch_ctd.py 실행 필요")

    print("\n[14/14] FAERS (FDA 환자 보고 hepatic AE)")
    fa = load_faers()
    if len(fa) > 0:
        print(f"  {len(fa)} 분자 (강도: {fa['vivo_faers'].value_counts().to_dict()})")
    else:
        print("  데이터 없음 — fetch_faers.py 실행 필요")

    # 순서 중요 — merge 시 뒤에 오는 게 같은 컬럼 덮어씀
    # marketed_clean 의 dilirank_category 보다 dilirank_full 우선
    loaded = {
        "marketed_clean": mc,
        "dilirank_full": drk,
        "livertox": lt,
        "dilist_gold": dl,
        "sider_strict": sd_s,
        "sider_lenient": sd_l,
        "tdc": td,
        "clintox": ct,
        "dailymed": dm,
        "pubmed": pm,
        "ctd": ctd,
        "faers": fa,
        "chembl": ch,
        "tox21": tx,
    }

    print("\n=== 병합 + 라벨 결정 ===")
    db = merge_and_label(loaded)
    print(f"\n전체 unique InChIKey: {len(db)}")

    # 요약
    print("\n=== vivo 라벨 분포 ===")
    print(db["vivo_label"].value_counts(dropna=False).to_dict())
    print("\n=== vitro 라벨 분포 ===")
    print(db["vitro_label"].value_counts(dropna=False).to_dict())

    # 저장
    out = os.path.join(OUT_DIR, "full.parquet")
    db.to_parquet(out, index=False)
    print(f"\n저장: {out}")

    # CSV 도 같이 (보기 편하게)
    db.to_csv(os.path.join(OUT_DIR, "full.csv"), index=False)
    print(f"      {os.path.join(OUT_DIR, 'full.csv')}")


if __name__ == "__main__":
    main()
