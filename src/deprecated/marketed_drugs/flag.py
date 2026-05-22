"""DILIrank + TDC DILI Y=1으로 hepatotoxic 플래그."""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

HEPATOTOXIC_CATEGORIES = {"vMost-DILI-Concern", "vLess-DILI-Concern"}


def flag_hepatotoxic(
    df: pd.DataFrame,
    dilirank: pd.DataFrame,
    tdc_pos_inchikeys: set[str],
) -> pd.DataFrame:
    """`marketed_all.csv` 스키마에 hepatotoxic 플래그 컬럼 추가."""
    out = df.copy()

    dili_map: dict[str, str] = {}
    for _, row in dilirank.iterrows():
        ik = row.get("InChIKey", "")
        cat = row.get("Severity Class", "")
        if ik and cat:
            dili_map[ik] = cat

    out["in_dilirank"] = out["inchi_key"].isin(dili_map.keys()).astype(int)
    out["dilirank_category"] = out["inchi_key"].map(dili_map).fillna("unknown")
    out["in_tdc_dili_pos"] = out["inchi_key"].isin(tdc_pos_inchikeys).astype(int)

    in_dili_hepatotoxic = out["dilirank_category"].isin(HEPATOTOXIC_CATEGORIES)
    out["known_hepatotoxic"] = (in_dili_hepatotoxic | (out["in_tdc_dili_pos"] == 1)).astype(int)

    logger.info(
        "Flag: %d/%d hepatotoxic (DILIrank match=%d, TDC=%d)",
        int(out["known_hepatotoxic"].sum()),
        len(out),
        int(out["in_dilirank"].sum()),
        int(out["in_tdc_dili_pos"].sum()),
    )
    return out


def load_dilirank(csv_path: str) -> pd.DataFrame:
    """DILIrank CSV 로드."""
    return pd.read_csv(csv_path)


def load_tdc_pos_inchikeys(tdc_train_path: str) -> set[str]:
    """기존 TDC DILI train CSV에서 Y=1 약물의 InChIKey 추출."""
    from src.marketed_drugs.merge import smiles_to_inchikey

    df = pd.read_csv(tdc_train_path)
    label_col = "Label" if "Label" in df.columns else "Y"
    smiles_col = "SMILES" if "SMILES" in df.columns else "Drug"
    pos = df[df[label_col] == 1]
    keys = set()
    for s in pos[smiles_col]:
        ik = smiles_to_inchikey(s)
        if ik:
            keys.add(ik)
    return keys
