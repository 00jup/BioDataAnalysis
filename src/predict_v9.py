"""최종 통합 예측 파이프라인 (v9).

채점 시나리오:
  SMILES → standardize → InChIKey
    → DB lookup hit?
        yes: DB vivo_label 사용 (양성/음성)
        no:  Chemprop v9 vivo 예측 (threshold 0.20)

사용:
    from src.predict_v9 import predict_smiles, predict_batch
    predict_smiles("CC(=O)Nc1ccc(O)cc1")  # acetaminophen
"""

from __future__ import annotations
import os, sys, json, subprocess, tempfile
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
CHEMPROP_VIVO = os.path.join(PROJECT_ROOT, "models", "chemprop_v9", "vivo")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
VIVO_THR = 0.20  # threshold_sweep 결과 — TPR/TNR 균형

# DB 캐시 — 모듈 로드 시 1회
_DB = None
_DB_BY_IK = None

def _load_db():
    global _DB, _DB_BY_IK
    if _DB is None:
        _DB = pd.read_parquet(DB_PATH)
        _DB_BY_IK = _DB.set_index("inchi_key")
    return _DB_BY_IK


_STANDARDIZER = rdMolStandardize.CleanupParameters()
_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()


def standardize(smiles: str) -> tuple[str | None, str | None]:
    """SMILES → (canonical_smiles, inchi_key)."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST_FRAG.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        csmi = Chem.MolToSmiles(mol)
        ik = Chem.MolToInchiKey(mol)
        return csmi, ik
    except Exception:
        return None, None


def chemprop_predict_batch(smiles_list: list[str]) -> list[float]:
    """Chemprop v9 vivo 예측 — 배치."""
    if not smiles_list: return []
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles_list}).to_csv(f.name, index=False)
        in_path = f.name
    out_path = in_path.replace(".csv", "_pred.csv")
    cmd = [
        CHEMPROP_BIN, "predict",
        "--test-path", in_path,
        "-s", "canonical_smiles",
        "--model-paths", CHEMPROP_VIVO,
        "--preds-path", out_path,
        "--accelerator", "cpu",
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    df = pd.read_csv(out_path)
    pred_col = [c for c in df.columns if c != "canonical_smiles"][0]
    return df[pred_col].astype(float).tolist()


def predict_batch(smiles_list: list[str]) -> pd.DataFrame:
    """배치 예측 — lookup priority + Chemprop fallback.

    반환:
      DataFrame[input_smiles, canonical_smiles, inchi_key, label, score, source]
        source ∈ {"db", "chemprop", "invalid"}
    """
    db_by_ik = _load_db()
    rows = []
    miss_idx = []
    miss_smiles = []

    for i, smi in enumerate(smiles_list):
        csmi, ik = standardize(smi)
        if csmi is None:
            rows.append({"input_smiles": smi, "canonical_smiles": None,
                         "inchi_key": None, "label": None, "score": None,
                         "source": "invalid"})
            continue
        if ik in db_by_ik.index:
            db_row = db_by_ik.loc[ik]
            if isinstance(db_row, pd.DataFrame): db_row = db_row.iloc[0]
            vl = db_row.get("vivo_label")
            if pd.notna(vl):
                rows.append({"input_smiles": smi, "canonical_smiles": csmi,
                             "inchi_key": ik, "label": int(vl), "score": float(vl),
                             "source": "db"})
                continue
        # miss → chemprop
        rows.append({"input_smiles": smi, "canonical_smiles": csmi,
                     "inchi_key": ik, "label": None, "score": None,
                     "source": "chemprop"})
        miss_idx.append(i)
        miss_smiles.append(csmi)

    # batch chemprop on misses
    if miss_smiles:
        probs = chemprop_predict_batch(miss_smiles)
        for i, p in zip(miss_idx, probs):
            rows[i]["score"] = float(p)
            rows[i]["label"] = int(p >= VIVO_THR)

    return pd.DataFrame(rows)


def predict_smiles(smiles: str) -> dict:
    df = predict_batch([smiles])
    return df.iloc[0].to_dict()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # CLI: python -m src.predict_v9 <smiles_or_csv>
        target = sys.argv[1]
        if os.path.exists(target):
            inp = pd.read_csv(target)
            smiles_col = next((c for c in inp.columns if "smi" in c.lower()), inp.columns[0])
            print(f"입력 {len(inp)} SMILES from {target} (컬럼: {smiles_col})")
            out = predict_batch(inp[smiles_col].tolist())
            out_csv = target.replace(".csv", "_predicted.csv")
            out.to_csv(out_csv, index=False)
            print(f"저장: {out_csv}")
            print(out.head())
            print(f"\nsource 분포: {out['source'].value_counts().to_dict()}")
        else:
            r = predict_smiles(target)
            print(json.dumps(r, indent=2, ensure_ascii=False))
    else:
        # 데모 — sanity check 약물 일부
        demo = [
            ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
            ("Isoniazid",     "NNC(=O)c1ccncc1"),
            ("Aspirin",       "CC(=O)Oc1ccccc1C(=O)O"),
            ("Metformin",     "CN(C)C(=N)NC(=N)N"),
        ]
        names = [d[0] for d in demo]
        smiles = [d[1] for d in demo]
        out = predict_batch(smiles)
        print(out)
