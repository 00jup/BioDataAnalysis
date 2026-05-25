"""최종 통합 예측 파이프라인 (final).

채점 시나리오:
  SMILES → standardize → InChIKey → DB lookup
    hit:   DB vivo_label 즉시
    miss:  Chemprop v17 (vivo, MCC 0.694) 로 fallback (threshold MCC max)

모델 정보:
  - vivo:  Chemprop v17 (ensemble 15 + hidden 600 + v1_rdkit_2d_normalized)
           scaffold-balanced split, AUC 0.947, MCC 0.694
  - vitro: 보조 — RF/CB v2 (AUC 0.762, MCC 0.382)

사용:
    .venv/bin/python -m src.predict_final --smiles "CC(=O)Nc1ccc(O)cc1"
    .venv/bin/python -m src.predict_final --csv input.csv
"""
from __future__ import annotations
import os, sys, json, subprocess, tempfile, argparse
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
CHEMPROP_VIVO = os.path.join(PROJECT_ROOT, "models", "chemprop_scaffold_v2",
                              "v17_ens15_h600", "vivo")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")
VIVO_THR = 0.35  # v17 의 자체 best MCC threshold

# DB cache
_DB_BY_IK = None
def _load_db():
    global _DB_BY_IK
    if _DB_BY_IK is None:
        _DB_BY_IK = pd.read_parquet(DB_PATH).set_index("inchi_key")
    return _DB_BY_IK


_LARGEST_FRAG = rdMolStandardize.LargestFragmentChooser()
_UNCHARGER = rdMolStandardize.Uncharger()

def standardize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None, None
        mol = rdMolStandardize.Cleanup(mol)
        mol = _LARGEST_FRAG.choose(mol)
        mol = _UNCHARGER.uncharge(mol)
        return Chem.MolToSmiles(mol), Chem.MolToInchiKey(mol)
    except Exception:
        return None, None


def chemprop_predict_batch(smiles_list):
    if not smiles_list: return []
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": smiles_list}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace(".csv", "_pred.csv")
    cmd = [
        CHEMPROP_BIN, "predict",
        "--test-path", in_p, "-s", "canonical_smiles",
        "--model-paths", CHEMPROP_VIVO,
        "--preds-path", out_p,
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    df = pd.read_csv(out_p)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    # 입력 순서대로 정렬
    by_smi = dict(zip(df["canonical_smiles"], df[pcol]))
    return [by_smi.get(s) for s in smiles_list]


def predict_batch(smiles_list):
    """배치 예측 — DB lookup priority + Chemprop v17 fallback.

    반환: DataFrame[input_smiles, canonical_smiles, inchi_key,
                    label, score, source]
    """
    db = _load_db()
    rows = []
    miss_idx, miss_smiles = [], []
    for i, smi in enumerate(smiles_list):
        csmi, ik = standardize(smi)
        if csmi is None:
            rows.append({"input_smiles": smi, "canonical_smiles": None,
                          "inchi_key": None, "label": None, "score": None,
                          "source": "invalid"})
            continue
        if ik in db.index:
            r = db.loc[ik]
            if isinstance(r, pd.DataFrame): r = r.iloc[0]
            vl = r.get("vivo_label")
            if pd.notna(vl):
                rows.append({"input_smiles": smi, "canonical_smiles": csmi,
                              "inchi_key": ik, "label": int(vl), "score": float(vl),
                              "source": "db"})
                continue
        rows.append({"input_smiles": smi, "canonical_smiles": csmi,
                      "inchi_key": ik, "label": None, "score": None,
                      "source": "chemprop_v17"})
        miss_idx.append(i); miss_smiles.append(csmi)

    if miss_smiles:
        probs = chemprop_predict_batch(miss_smiles)
        for i, p in zip(miss_idx, probs):
            if p is None: continue
            rows[i]["score"] = float(p)
            rows[i]["label"] = int(float(p) >= VIVO_THR)

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", help="단일 SMILES 예측")
    ap.add_argument("--csv", help="csv 파일 (smiles 컬럼 자동 인식)")
    ap.add_argument("--output", default=None, help="결과 csv 출력 경로")
    args = ap.parse_args()

    if args.smiles:
        out = predict_batch([args.smiles])
        print(out.to_dict(orient="records")[0])
    elif args.csv:
        inp = pd.read_csv(args.csv)
        smi_col = next((c for c in inp.columns if "smi" in c.lower()), inp.columns[0])
        print(f"입력 {len(inp)} SMILES (컬럼: {smi_col})")
        out = predict_batch(inp[smi_col].tolist())
        out_path = args.output or args.csv.replace(".csv", "_predicted.csv")
        out.to_csv(out_path, index=False)
        print(f"저장: {out_path}")
        print(f"source 분포: {out['source'].value_counts().to_dict()}")
        print(f"label 분포: {out['label'].value_counts().to_dict()}")
    else:
        # 데모
        demo = ["CC(=O)Nc1ccc(O)cc1", "CC(=O)Oc1ccccc1C(=O)O",
                 "NNC(=O)c1ccncc1", "CC(C)Cc1ccc(C(C)C(=O)O)cc1"]
        print(predict_batch(demo))


if __name__ == "__main__":
    main()
