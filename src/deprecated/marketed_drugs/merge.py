"""canonical SMILES 통합, dedup, withdrawn 필터."""

from __future__ import annotations

import logging

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize.rdMolStandardize import LargestFragmentChooser, Uncharger

logger = logging.getLogger(__name__)
RDLogger.logger().setLevel(RDLogger.ERROR)

_chooser = LargestFragmentChooser()
_uncharger = Uncharger()


SOURCE_ID_COL = {
    "chembl": "chembl_id",
    "drugcentral": "drugcentral_id",
    "pubchem": "cid",
    "inxight": "unii",
    "kegg": "kegg_id",
    "drugbank": "drugbank_id",
}


def canonicalize_smiles(smiles: str | None) -> str | None:
    """SMILES → canonical SMILES (largest fragment, neutralized, RDKit canonical)."""
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        if "." in smiles:
            mol = _chooser.choose(mol)
        mol = _uncharger.uncharge(mol)
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def smiles_to_inchikey(smiles: str | None) -> str:
    """SMILES → InChIKey. 실패 시 빈 문자열."""
    if not smiles:
        return ""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        return Chem.MolToInchiKey(mol)
    except Exception:
        return ""


def _normalize_source_df(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """소스별 df를 (source, source_id, name, smiles) 표준 스키마로."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["source", "source_id", "name", "smiles"])
    id_col = SOURCE_ID_COL.get(name)
    out = pd.DataFrame(
        {
            "source": name,
            "source_id": df[id_col].astype(str) if id_col in df.columns else "",
            "name": df["name"].astype(str)
            if "name" in df.columns
            else df.get("pref_name", "").astype(str),
            "smiles": df["smiles"].astype(str) if "smiles" in df.columns else "",
        }
    )
    out = out[out["smiles"].notna() & (out["smiles"] != "")]
    return out


def merge_sources(raw_by_source: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """6개 소스 DataFrame을 canonical SMILES 기준으로 통합 + dedup."""
    pieces = []
    for name, df in raw_by_source.items():
        norm = _normalize_source_df(name, df)
        pieces.append(norm)

    if not pieces:
        return pd.DataFrame()

    combined = pd.concat(pieces, ignore_index=True)
    combined["canonical_smiles"] = combined["smiles"].apply(canonicalize_smiles)
    combined = combined[combined["canonical_smiles"].notna()].copy()
    combined["inchi_key"] = combined["canonical_smiles"].apply(smiles_to_inchikey)

    grouped = combined.groupby("canonical_smiles", as_index=False).agg(
        name=("name", "first"),
        smiles=("smiles", "first"),
        inchi_key=("inchi_key", "first"),
        sources=("source", lambda s: ";".join(sorted(set(s)))),
        source_ids=("source_id", lambda s: ";".join(s.astype(str).tolist())),
    )

    assert grouped["canonical_smiles"].is_unique, "canonical_smiles must be unique after merge"

    logger.info("Merge: %d 소스 행 → %d 고유 분자", len(combined), len(grouped))
    return grouped.reset_index(drop=True)


def filter_withdrawn(
    df: pd.DataFrame,
    chembl_withdrawn: pd.DataFrame | None = None,
    drugbank_status: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """ChEMBL withdrawn_flag, DrugBank withdrawn으로 제외. (kept, removed) 반환."""
    withdrawn_keys: set[str] = set()

    if chembl_withdrawn is not None and not chembl_withdrawn.empty:
        if "withdrawn_flag" in chembl_withdrawn.columns:
            for _, row in chembl_withdrawn[chembl_withdrawn["withdrawn_flag"]].iterrows():
                ck = canonicalize_smiles(row.get("smiles", ""))
                if ck:
                    ik = smiles_to_inchikey(ck)
                    if ik:
                        withdrawn_keys.add(ik)

    if drugbank_status is not None and not drugbank_status.empty:
        if "withdrawn" in drugbank_status.columns:
            for _, row in drugbank_status[drugbank_status["withdrawn"]].iterrows():
                ik = row.get("inchi_key", "")
                if ik:
                    withdrawn_keys.add(ik)

    is_withdrawn = df["inchi_key"].isin(withdrawn_keys)
    kept = df[~is_withdrawn].reset_index(drop=True)
    removed = df[is_withdrawn].reset_index(drop=True)
    logger.info("Withdrawn 제외: %d → %d (제거 %d)", len(df), len(kept), len(removed))
    return kept, removed
