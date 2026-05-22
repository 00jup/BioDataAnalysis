"""SMILES 표준화 모듈 — RDKit MolStandardize 체인.

체인: parse → cleanup → 큰 단편 추출 → 전하 중화 → canonical → InChIKey.

추론 시 입력 SMILES 와 학습 데이터의 형식을 동일하게 만들어
lookup·예측의 안정성을 보장한다.

사용:
    from src.standardize import standardize, smiles_to_inchikey
    canon, ikey = standardize("CC(=O)Nc1ccc(O)cc1.HCl")
    # → ('CC(=O)Nc1ccc(O)cc1', 'RZVAJINKPMORJF-UHFFFAOYSA-N')
"""

from __future__ import annotations

from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.logger().setLevel(RDLogger.ERROR)

_LARGEST = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize(smiles: str) -> tuple[str, str] | None:
    """SMILES → (canonical_smiles, inchi_key).

    실패 시 None. 실패 케이스:
      - 빈 문자열·None
      - RDKit parsing 실패
      - InChIKey 생성 실패
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles.strip())
        if mol is None:
            return None
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        canon = Chem.MolToSmiles(mol, canonical=True)
        ikey = Chem.MolToInchiKey(mol)
        if not canon or not ikey:
            return None
        return canon, ikey
    except Exception:
        return None


def standardize_batch(smiles_list: list[str]) -> list[tuple[str, str] | None]:
    """배치 처리 — 같은 입력 순서 보존, 실패는 None."""
    return [standardize(s) for s in smiles_list]


def smiles_to_inchikey(smiles: str) -> str | None:
    """SMILES → InChIKey 만 추출 (lookup 키용)."""
    r = standardize(smiles)
    return r[1] if r else None


def smiles_to_canonical(smiles: str) -> str | None:
    """SMILES → canonical SMILES."""
    r = standardize(smiles)
    return r[0] if r else None
