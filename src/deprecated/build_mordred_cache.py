"""실험용 feature cache 구축 — Morgan FP + MACCS + RDKit descriptor 전체.

exp1/2/3 의 train/val + 외부 test 에 등장하는 모든 분자에 대해
한 번만 계산해 parquet 로 캐싱한다.

feature 구성: Morgan(2048) + MACCS(167) + RDKit descriptor(210) = 2425
(Mordred 는 이 머신에서 분자당 수 초로 너무 느려 RDKit 내장 descriptor 로 대체)

사용법:
    python src/build_mordred_cache.py
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
CACHE_PATH = os.path.join(EXP_DIR, "feature_cache.parquet")

_DESC_NAMES = [n for n, _ in Descriptors.descList]
_DESC_FUNCS = [f for _, f in Descriptors.descList]


def collect_smiles() -> list[str]:
    files = []
    for exp in ("exp1", "exp2", "exp3"):
        files += [os.path.join(EXP_DIR, exp, "train.csv"), os.path.join(EXP_DIR, exp, "val.csv")]
    files.append(os.path.join(EXP_DIR, "external_test", "test.csv"))
    smiles = set()
    for f in files:
        smiles.update(pd.read_csv(f)["canonical_smiles"].dropna().tolist())
    return sorted(smiles)


def _descriptors(mol) -> list[float]:
    out = []
    for func in _DESC_FUNCS:
        try:
            out.append(float(func(mol)))
        except Exception:
            out.append(0.0)
    return out


def main() -> None:
    smiles = collect_smiles()
    print(f"고유 분자 {len(smiles)}개 — feature 계산 시작", flush=True)

    mols, valid_smiles = [], []
    for smi in smiles:
        m = Chem.MolFromSmiles(smi)
        if m is not None:
            mols.append(m)
            valid_smiles.append(smi)
    print(f"  유효 분자 {len(mols)}개", flush=True)

    morgan = np.array(
        [np.array(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)) for m in mols]
    )
    maccs = np.array([np.array(MACCSkeys.GenMACCSKeys(m)) for m in mols])
    desc = np.array([_descriptors(m) for m in mols], dtype=np.float64)
    print(f"  Morgan {morgan.shape}, MACCS {maccs.shape}, RDKit desc {desc.shape}", flush=True)

    cols = (
        [f"Morgan_{i}" for i in range(morgan.shape[1])]
        + [f"MACCS_{i}" for i in range(maccs.shape[1])]
        + [f"RDKit_{n}" for n in _DESC_NAMES]
    )
    full = np.hstack([morgan, maccs, desc])
    full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)

    out = pd.DataFrame(full, columns=cols, index=valid_smiles)
    out.index.name = "canonical_smiles"
    out.to_parquet(CACHE_PATH)
    print(f"저장: {CACHE_PATH}  shape={out.shape}", flush=True)


if __name__ == "__main__":
    main()
