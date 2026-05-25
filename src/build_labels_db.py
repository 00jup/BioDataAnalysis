"""통합 라벨 DB 구축 — in vivo (임상) + in vitro (어세이) 라벨 분리 보관.

소스
====
in vivo (임상/사람 데이터, 우선)
  - DILIrank vMost / vLess / vNo / Ambiguous (FDA 공식)
  - DILIst + GoldStandard (외부 양성)
  - SIDER 간 부작용 (시판 후 부작용)
  - TDC DILI (학술 정제)
  - ClinTox (MoleculeNet, 임상시험 실패)

in vitro (어세이 데이터)
  - chEMBL hepatotoxicity (어세이 결과)
  - Tox21 간 관련 어세이 (HepG2, 미토콘드리아, BSEP 등)

저장 형식
=========
data/labels_db/full.parquet — 컬럼:
  inchi_key, canonical_smiles, name,
  vivo_dilirank, vivo_dilist, vivo_gold, vivo_sider_liver, vivo_tdc_dili,
  vivo_clintox, vivo_marketed_clean_neg,
  vivo_label, vivo_n_sources, vivo_confidence,
  vitro_chembl, vitro_tox21,
  vitro_label, vitro_n_sources, vitro_confidence,
  final_label, final_source, conflict
"""

from __future__ import annotations

import os
from collections import defaultdict

import numpy as np
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
        cleaner = rdMolStandardize.CleanupParameters()
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
        if n >= 20: return "strong"
        if n >= 5: return "medium"
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
        return pd.DataFrame(columns=["canonical_smiles", "name", "vivo_ctd",
                                      "ctd_strong", "ctd_pmid"])
    p = pd.read_csv(path)
    p = p.dropna(subset=["canonical_smiles"])
    # 강도 코드화
    def code(row):
        if row.get("strong_evidence", 0) >= 1: return "strong"
        if row.get("n_evidence", 0) >= 10: return "medium"
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
        return pd.DataFrame(columns=["canonical_smiles", "name", "vivo_faers",
                                      "faers_n_reports"])
    p = pd.read_csv(path)
    p = p.dropna(subset=["canonical_smiles"])
    def code(n):
        if n >= 5000: return "strong"
        if n >= 1000: return "medium"
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
    p["vivo_marketed_clean_neg"] = (p["dilirank_category"] != "Ambiguous-DILI-Concern").astype("Int64")
    p["vivo_dilirank"] = p["dilirank_category"]
    return p[["canonical_smiles", "inchi_key", "name", "vivo_marketed_clean_neg", "vivo_dilirank"]]


def load_clintox():
    """ClinTox from raw/clintox/clintox.csv."""
    data = pd.read_csv(os.path.join(RAW, "clintox", "clintox.csv"))
    rows = []
    for _, r in data.iterrows():
        std = standardize(r["Drug"])
        if std:
            rows.append({"canonical_smiles": std[0], "inchi_key": std[1],
                         "name": r.get("Drug_ID", ""), "vivo_clintox": int(r["Y"])})
    return pd.DataFrame(rows).drop_duplicates("inchi_key")


def load_chembl() -> pd.DataFrame:
    """chEMBL — in vitro 라벨 (raw 에서 직접 재정제)."""
    POS_TERMS = {
        "drug-induced liver injury reported", "Toxic", "Most-Dili-Concern",
        "Less-Dili-Concern", "HH: Evidence of human hepatotoxicity",
        "DILI positive (training set)", "DILI positive (test set)",
    }
    NEG_TERMS = {"no drug-induced liver injury reported", "Non-Toxic", "Non-toxic", "No-Dili-Concern"}

    pos_set = pd.read_csv(os.path.join(RAW, "chembl", "chembl_toxic_positive_set.csv"))
    compounds = pd.read_csv(os.path.join(RAW, "chembl", "chembl_hepatotoxicity_compounds.csv"))

    p1 = pos_set.copy(); p1["label"] = 1
    p2 = compounds[compounds["activity_comment"].isin(POS_TERMS)].copy(); p2["label"] = 1
    n = compounds[compounds["activity_comment"].isin(NEG_TERMS)].copy(); n["label"] = 0

    pool = pd.concat([
        p1[["canonical_smiles","molecule_chembl_id","label"]],
        p2[["canonical_smiles","molecule_chembl_id","label"]],
        n[["canonical_smiles","molecule_chembl_id","label"]],
    ], ignore_index=True).dropna(subset=["canonical_smiles"]).drop_duplicates(["molecule_chembl_id","label"])

    # molecule_chembl_id 단위 라벨 충돌 제거
    g = pool.groupby("molecule_chembl_id")["label"].nunique()
    pool = pool[~pool["molecule_chembl_id"].isin(set(g[g>1].index))]

    pool["vitro_chembl"] = pool["label"].astype("Int64")
    pool["name"] = pool["molecule_chembl_id"]
    return pool[["canonical_smiles","name","vitro_chembl"]].drop_duplicates("canonical_smiles")


def load_tox21() -> pd.DataFrame:
    """Tox21 통합 from raw/tox21/tox21_combined_5tasks.csv."""
    data = pd.read_csv(os.path.join(RAW, "tox21", "tox21_combined_5tasks.csv"))
    rows = []
    for _, r in data.iterrows():
        std = standardize(r["smiles"])
        if std:
            rows.append({"canonical_smiles": std[0], "inchi_key": std[1], "name": "",
                         "vitro_tox21": int(r["label_any_pos"]),
                         "tox21_pos_count": int(r["n_assays_pos"]),
                         "tox21_total_assays": int(r["n_assays_tested"])})
    df = pd.DataFrame(rows).drop_duplicates("inchi_key")
    print(f"  Tox21 통합: {len(df)} 분자 (양성 {int((df.vitro_tox21==1).sum())} / 음성 {int((df.vitro_tox21==0).sum())})")
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
    vivo_cols = ["vivo_dilirank", "vivo_livertox", "vivo_dailymed", "vivo_pubmed",
                 "vivo_ctd", "vivo_faers",
                 "vivo_dilist", "vivo_gold",
                 "vivo_sider_liver", "vivo_sider_hepatotox",
                 "vivo_tdc_dili", "vivo_clintox", "vivo_marketed_clean_neg"]
    vitro_cols = ["vitro_chembl", "vitro_tox21"]
    for c in vivo_cols + vitro_cols + ["tox21_pos_count", "tox21_total_assays"]:
        if c not in df.columns:
            df[c] = pd.NA

    # 5. vivo label 결정
    def vivo_decide(row):
        # 양성 신호
        pos = 0
        if row["vivo_dilirank"] in ("vMost-DILI-Concern", "vLess-DILI-Concern"):
            pos += 2  # FDA 공식 = 강한 신호
        # LiverTox Likelihood Score: A/B = 강한 양성, C = 양성, D = 약한 양성
        lt = row["vivo_livertox"]
        if lt in ("A", "B"): pos += 2
        elif lt == "C":     pos += 1
        elif lt == "D":     pos += 0.5
        # DailyMed FDA 라벨 — 심각도별 점수
        dm = row.get("vivo_dailymed")
        if dm == "boxed_hepatotox": pos += 3       # FDA Boxed Warning (가장 강한 양성)
        elif dm == "warning_hepatotox": pos += 1.5  # Warnings & Precautions
        elif dm == "adverse_hepatotox": pos += 0.5  # Adverse Reactions
        elif dm == "contraindication_hepatic": pos += 1  # 간 손상 환자 금기 = 간에 부담
        # PubMed DILI MeSH (Major Topic + Case Reports/Clinical Trial/Meta-Analysis)
        pm = row.get("vivo_pubmed")
        if pm == "strong": pos += 2          # n_papers >= 20 (Acetaminophen 등 well-known)
        elif pm == "medium": pos += 1        # 5-19 papers
        elif pm == "weak": pos += 0.5        # 3-4 papers
        # CTD (NIH/EPA chemical-disease 큐레이션)
        ct = row.get("vivo_ctd")
        if ct == "strong": pos += 2          # marker/mechanism direct evidence
        elif ct == "medium": pos += 1        # multiple PMID evidence (≥10)
        elif ct == "weak": pos += 0.3
        # FAERS (FDA 환자 보고)
        fa = row.get("vivo_faers")
        if fa == "strong": pos += 1.5        # n_reports ≥ 5000
        elif fa == "medium": pos += 0.8      # n_reports 1000-5000
        elif fa == "weak": pos += 0.3        # n_reports 50-1000
        # 약한 출처들 (DILIst/Gold/SIDER/TDC/ClinTox) — _unreliable 로 분리
        # 가중치 0 (사실상 미사용). 컬럼은 NaN 이므로 if pd.notna 통과 안 함.
        if pd.notna(row["vivo_dilist"]) and row["vivo_dilist"] == 1: pos += 0
        if pd.notna(row["vivo_gold"]) and row["vivo_gold"] == 1: pos += 0
        if pd.notna(row["vivo_sider_liver"]) and row["vivo_sider_liver"] == 1: pos += 0
        elif pd.notna(row["vivo_sider_hepatotox"]) and row["vivo_sider_hepatotox"] == 1: pos += 0
        if pd.notna(row["vivo_tdc_dili"]) and row["vivo_tdc_dili"] == 1: pos += 0
        if pd.notna(row["vivo_clintox"]) and row["vivo_clintox"] == 1: pos += 0

        # 음성 신호
        neg = 0
        if row["vivo_dilirank"] == "vNo-DILI-Concern":
            neg += 2  # FDA 공식
        if lt == "E": neg += 2  # LiverTox unlikely — 강한 음성 신호
        if pd.notna(row["vivo_marketed_clean_neg"]) and row["vivo_marketed_clean_neg"] == 1 and pos == 0:
            neg += 1  # 시판 안전약 신호 (양성 신호 없을 때만)
        if pd.notna(row["vivo_clintox"]) and row["vivo_clintox"] == 0 and pos == 0:
            neg += 1  # ClinTox FDA approved + no fail

        # Ambiguous → vivo_label None
        if row["vivo_dilirank"] == "Ambiguous-DILI-Concern":
            return None, "ambiguous", 0
        if pos == 0 and neg == 0:
            return None, "no_signal", 0

        label = 1 if pos > neg else (0 if neg > pos else None)
        # 충돌
        if pos > 0 and neg > 0:
            # DILIrank 가 vNo 인데 다른 출처가 양성이면? — vivo_dilirank 가 우선
            if row["vivo_dilirank"] == "vNo-DILI-Concern" and pos >= 2:
                conf = "low"
                label = 0  # FDA vNo 우선
            else:
                conf = "low"
        elif pos > 0:
            conf = "high" if pos >= 3 else "med"
        else:
            conf = "high" if neg >= 2 else "med"

        n_sources = sum([
            pd.notna(row[c]) and row[c] not in (None, "Ambiguous-DILI-Concern")
            for c in vivo_cols
        ])
        return label, conf, n_sources

    df[["vivo_label", "vivo_confidence", "vivo_n_sources"]] = df.apply(
        lambda r: pd.Series(vivo_decide(r)), axis=1)

    # 6. vitro label 결정 (chEMBL > Tox21)
    def vitro_decide(row):
        labels = []
        if pd.notna(row["vitro_chembl"]): labels.append(int(row["vitro_chembl"]))
        if pd.notna(row["vitro_tox21"]):  labels.append(int(row["vitro_tox21"]))
        if not labels:
            return None, None, 0
        # 합의 — 어느 하나라도 양성이면 양성 (보수적)
        if 1 in labels:
            label = 1
        elif all(l == 0 for l in labels):
            label = 0
        else:
            label = None
        conf = "high" if len(labels) == 2 else "med"
        return label, conf, len(labels)

    df[["vitro_label", "vitro_confidence", "vitro_n_sources"]] = df.apply(
        lambda r: pd.Series(vitro_decide(r)), axis=1)

    # 7. 3가지 룰별 라벨 + final_source + conflict
    CONF_SCORE = {"high": 3, "med": 2, "low": 1, None: 0}

    def rules(row):
        v = row["vivo_label"]; vi = row["vitro_label"]
        v_conf = CONF_SCORE.get(row["vivo_confidence"], 0)
        vi_conf = CONF_SCORE.get(row["vitro_confidence"], 0)

        # ── 룰 1: 임상우선 ─────────────────────────────
        if pd.notna(v):
            r1 = int(v)
        elif pd.notna(vi):
            r1 = int(vi)
        else:
            r1 = None

        # ── 룰 2: 신뢰도가중 ──────────────────────────
        # vivo+ → +v_conf,  vivo- → -v_conf,  None → 0
        s = 0
        if pd.notna(v): s += v_conf if int(v) == 1 else -v_conf
        if pd.notna(vi): s += vi_conf if int(vi) == 1 else -vi_conf
        if s > 0: r2 = 1
        elif s < 0: r2 = 0
        else: r2 = None  # 동점 또는 둘 다 None

        # ── 룰 3: 양쪽합의 ────────────────────────────
        if pd.notna(v) and pd.notna(vi):
            if int(v) == int(vi):
                r3 = int(v)
            else:
                r3 = None  # 충돌 → 모델로 fallback
        else:
            r3 = None  # 한쪽만 있으면 합의 X → 모델로 fallback

        # final_source 표시
        if pd.notna(v) and pd.notna(vi):
            if int(v) == int(vi):
                source = "both_agree"
                conflict = False
            else:
                source = "conflict"
                conflict = True
        elif pd.notna(v):
            source = "vivo_only"
            conflict = False
        elif pd.notna(vi):
            source = "vitro_only"
            conflict = False
        else:
            source = "no_label"
            conflict = False

        return r1, r2, r3, source, conflict

    df[["label_vivo_priority", "label_weighted", "label_consensus",
        "final_source", "conflict"]] = df.apply(
        lambda r: pd.Series(rules(r)), axis=1)
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
    dl = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name",
                                "vivo_dilist", "vivo_gold"])
    sd_s = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_sider_liver"])
    sd_l = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_sider_hepatotox"])
    td = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_tdc_dili"])
    ct = pd.DataFrame(columns=["canonical_smiles", "inchi_key", "name", "vivo_clintox"])

    print("\n[3/10] 시판약 음성 풀")
    mc = load_marketed_clean()
    print(f"  {len(mc)} 분자")

    print("[9/10] chEMBL (in vitro)")
    ch = load_chembl()
    print(f"  {len(ch)} 분자 (양성 {int((ch.vitro_chembl==1).sum())} / 음성 {int((ch.vitro_chembl==0).sum())})")

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
    loaded = {"marketed_clean": mc, "dilirank_full": drk, "livertox": lt,
              "dilist_gold": dl, "sider_strict": sd_s, "sider_lenient": sd_l,
              "tdc": td, "clintox": ct, "dailymed": dm, "pubmed": pm,
              "ctd": ctd, "faers": fa,
              "chembl": ch, "tox21": tx}

    print("\n=== 병합 + 라벨 결정 ===")
    db = merge_and_label(loaded)
    print(f"\n전체 unique InChIKey: {len(db)}")

    # 요약
    print(f"\n=== vivo 라벨 분포 ===")
    print(db["vivo_label"].value_counts(dropna=False).to_dict())
    print(f"\n=== vitro 라벨 분포 ===")
    print(db["vitro_label"].value_counts(dropna=False).to_dict())
    print(f"\n=== final_source 분포 ===")
    print(db["final_source"].value_counts(dropna=False).to_dict())
    print(f"\n=== rule 1: vivo_priority 라벨 분포 ===")
    print(db["label_vivo_priority"].value_counts(dropna=False).to_dict())
    print(f"\n=== rule 2: weighted 라벨 분포 ===")
    print(db["label_weighted"].value_counts(dropna=False).to_dict())
    print(f"\n=== rule 3: consensus 라벨 분포 ===")
    print(db["label_consensus"].value_counts(dropna=False).to_dict())
    print(f"\n=== 라벨 충돌 (vivo ≠ vitro) ===  {int(db['conflict'].sum())} 건")

    # 저장
    out = os.path.join(OUT_DIR, "full.parquet")
    db.to_parquet(out, index=False)
    print(f"\n저장: {out}")

    # CSV 도 같이 (보기 편하게)
    db.to_csv(os.path.join(OUT_DIR, "full.csv"), index=False)
    print(f"      {os.path.join(OUT_DIR, 'full.csv')}")


if __name__ == "__main__":
    main()
