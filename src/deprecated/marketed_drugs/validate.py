"""수집 결과 sanity check."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

KNOWN_DRUGS = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Acetaminophen": "CC(=O)Nc1ccc(O)cc1",
    "Metformin": "CN(C)C(=N)N=C(N)N",
    "Caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}


def validate(marketed_root: Path) -> bool:
    """간단한 sanity check. 실패 시 False 반환."""
    from src.marketed_drugs.merge import smiles_to_inchikey

    all_path = marketed_root / "all" / "marketed_all.csv"
    clean_path = marketed_root / "non_hepatotoxic" / "marketed_clean.csv"

    if not all_path.exists() or not clean_path.exists():
        logger.error("출력 CSV 누락: %s, %s", all_path, clean_path)
        return False

    df_all = pd.read_csv(all_path)
    df_clean = pd.read_csv(clean_path)
    ok = True

    print(f"✓ marketed_all.csv: {len(df_all):,} 약물")
    print(f"✓ marketed_clean.csv: {len(df_clean):,} 약물 (hepatotoxic 제외)")

    for name, smiles in KNOWN_DRUGS.items():
        ik = smiles_to_inchikey(smiles)
        present = (df_all["inchi_key"] == ik).any()
        symbol = "✓" if present else "✗"
        print(f"  {symbol} {name} ({ik})")
        if not present:
            ok = False

    if not df_all["canonical_smiles"].is_unique:
        logger.error("canonical_smiles unique invariant 위반")
        ok = False
    else:
        print("✓ canonical_smiles unique invariant")

    hep_ratio = df_all["known_hepatotoxic"].mean() if "known_hepatotoxic" in df_all.columns else 0
    print(f"  Hepatotoxic 비율: {hep_ratio:.1%}")
    if not (0.02 <= hep_ratio <= 0.20):
        logger.warning("Hepatotoxic 비율 비정상: %.1f%%", hep_ratio * 100)

    return ok


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    DATA_ROOT = PROJECT_ROOT / "data" / "marketed_drugs"
    ok = validate(DATA_ROOT)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
