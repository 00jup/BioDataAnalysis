"""실험용 Mordred feature cache 구축 — Morgan FP + MACCS + Mordred 2D descriptor.

원래 계획대로 Mordred ~1600개 2D descriptor 를 사용한다.
RDKit 210 descriptor 버전(feature_cache.parquet)과 별도 캐시로 보관해
RDKit vs Mordred 성능을 비교할 수 있게 한다.

거대 분자(예: 407원자 oligonucleotide)는 Mordred 가 분자당 2분 이상 걸리므로
분자별 타임아웃(TIMEOUT 초)을 두고, 초과 분자는 Mordred 값 NaN→0 으로 채운다.
BLAS 스레드를 1로 고정해 워커 7개 × BLAS 다중스레드 과다생성을 막는다.

feature 구성: Morgan(2048) + MACCS(167) + Mordred 2D(~1613)
출력: data/experiments/feature_cache_mordred.parquet
진행률: data/experiments/.mordred_progress.txt

사용법:
    python src/build_mordred_features.py
"""

from __future__ import annotations

import os

# numpy/BLAS 멀티스레드 비활성화 — 워커별 1스레드로 깨끗한 7-way 병렬 (import 전에 설정)
for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_v] = "1"

import signal  # noqa: E402
from multiprocessing import Pool  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from mordred import Calculator, descriptors  # noqa: E402
from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, MACCSkeys  # noqa: E402

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
CACHE_PATH = os.path.join(EXP_DIR, "feature_cache_mordred.parquet")
PROGRESS_PATH = os.path.join(EXP_DIR, ".mordred_progress.txt")
NPROC = max(1, (os.cpu_count() or 2) - 1)
TIMEOUT = 12  # 분자당 Mordred 계산 제한 시간(초)

_CALC: Calculator | None = None


class _Timeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _Timeout()


def _init_worker() -> None:
    global _CALC
    RDLogger.logger().setLevel(RDLogger.ERROR)
    _CALC = Calculator(descriptors, ignore_3D=True)
    signal.signal(signal.SIGALRM, _alarm_handler)


def _compute(smi: str):
    """SMILES 한 개의 Mordred descriptor 값 리스트 반환. 타임아웃/실패 시 None."""
    signal.alarm(TIMEOUT)
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return smi, None
        res = _CALC(mol)
        vals = []
        for v in res:
            try:
                vals.append(float(v))
            except Exception:
                vals.append(np.nan)
        return smi, vals
    except _Timeout:
        return smi, None
    except Exception:
        return smi, None
    finally:
        signal.alarm(0)


def collect_smiles() -> list[str]:
    files = []
    for exp in ("exp1", "exp2", "exp3"):
        files += [os.path.join(EXP_DIR, exp, "train.csv"), os.path.join(EXP_DIR, exp, "val.csv")]
    files.append(os.path.join(EXP_DIR, "external_test", "test.csv"))
    smiles = set()
    for f in files:
        smiles.update(pd.read_csv(f)["canonical_smiles"].dropna().tolist())
    return sorted(smiles)


def main() -> None:
    smiles = collect_smiles()
    print(
        f"고유 분자 {len(smiles)}개 — feature 계산 시작 (nproc={NPROC}, timeout={TIMEOUT}s)",
        flush=True,
    )

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
    print(f"  Morgan {morgan.shape}, MACCS {maccs.shape}", flush=True)

    base_calc = Calculator(descriptors, ignore_3D=True)
    desc_names = [str(d) for d in base_calc.descriptors]
    n_desc = len(desc_names)
    print(f"  Mordred 2D descriptor {n_desc}개 — 병렬 계산 시작", flush=True)

    total = len(valid_smiles)
    results: dict[str, list | None] = {}
    n_timeout = 0
    with Pool(NPROC, initializer=_init_worker) as pool:
        for done, (smi, vals) in enumerate(
            pool.imap_unordered(_compute, valid_smiles, chunksize=4), start=1
        ):
            results[smi] = vals
            if vals is None:
                n_timeout += 1
            if done % 100 == 0 or done == total:
                with open(PROGRESS_PATH, "w") as f:
                    f.write(f"{done}/{total}  timeout/실패={n_timeout}")
                print(f"  진행 {done}/{total}  timeout/실패={n_timeout}", flush=True)

    mord = np.array(
        [results[s] if results[s] is not None else [np.nan] * n_desc for s in valid_smiles],
        dtype=np.float64,
    )
    print(f"  Mordred desc {mord.shape}  (timeout/실패 {n_timeout}개는 NaN→0)", flush=True)

    cols = (
        [f"Morgan_{i}" for i in range(morgan.shape[1])]
        + [f"MACCS_{i}" for i in range(maccs.shape[1])]
        + [f"Mordred_{c}" for c in desc_names]
    )
    full = np.hstack([morgan, maccs, mord])
    full = np.nan_to_num(full, nan=0.0, posinf=0.0, neginf=0.0)

    out = pd.DataFrame(full, columns=cols, index=valid_smiles)
    out.index.name = "canonical_smiles"
    out.to_parquet(CACHE_PATH)
    print(f"저장: {CACHE_PATH}  shape={out.shape}", flush=True)


if __name__ == "__main__":
    main()
