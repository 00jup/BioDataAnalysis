"""DILI 추론 — SMILES 를 입력하면 간독성 1(위험)/0(안전) 을 예측한다.

모델: Chemprop v27 (D-MPNN ensemble 15, hidden 600, v1_rdkit_2d_normalized).
입력 SMILES 는 학습과 동일한 RDKit MolStandardize 체인으로 표준화한 뒤 예측한다.

사용:
    # 단일
    python src/predict.py "CC(=O)Nc1ccc(O)cc1"
    # 여러 개
    python src/predict.py "CC(=O)Nc1ccc(O)cc1" "CC(=O)Oc1ccccc1C(=O)O"
    # 파일 (한 줄에 SMILES 하나, 또는 "이름,SMILES")
    python src/predict.py --file drugs.txt
    # CSV (컬럼 지정)
    python src/predict.py --file in.csv --smiles-col canonical_smiles

옵션:
    --threshold 0.25      1 로 판정할 확률 임계값 (기본 DEFAULT_THRESHOLD)
    --model-dir <path>    모델 디렉터리 (기본 models/chemprop_v27)
    --out <path>          결과를 CSV 로 저장
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.standardize import standardize  # noqa: E402

DEFAULT_MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop_v27")
DEFAULT_FEATURIZER = "v1_rdkit_2d_normalized"
# v27 앙상블 자체 test 기준 양성-민감 운영점 (sanity 검증으로 확정).
DEFAULT_THRESHOLD = 0.25


def find_chemprop_bin() -> str:
    """chemprop 실행 파일 경로 탐색: 현재 venv → PATH → 프로젝트 .venv 폴백."""
    cand = os.path.join(os.path.dirname(sys.executable), "chemprop")
    if os.path.exists(cand):
        return cand
    on_path = shutil.which("chemprop")
    if on_path:
        return on_path
    for venv in (".venv", "../Bioinformatics/.venv"):
        c = os.path.join(PROJECT_ROOT, venv, "bin", "chemprop")
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        "chemprop 실행 파일을 못 찾았다. venv 를 활성화하거나 `pip install chemprop` 후 실행하라."
    )


def read_inputs(args) -> list[tuple[str, str]]:
    """(name, raw_smiles) 리스트 반환. name 없으면 빈 문자열."""
    rows: list[tuple[str, str]] = []
    if args.file:
        if args.file.lower().endswith(".csv"):
            df = pd.read_csv(args.file)
            col = args.smiles_col or next(
                (c for c in df.columns if "smiles" in c.lower()), df.columns[0]
            )
            name_col = next((c for c in df.columns if c.lower() in ("name", "drug")), None)
            for _, r in df.iterrows():
                rows.append((str(r[name_col]) if name_col else "", str(r[col])))
        else:
            with open(args.file) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    if "," in line:
                        name, smi = line.split(",", 1)
                        rows.append((name.strip(), smi.strip()))
                    else:
                        rows.append(("", line))
    rows += [("", s) for s in args.smiles]
    return rows


def chemprop_predict(canon_list: list[str], model_dir: str) -> dict[str, float]:
    """canonical SMILES 리스트 → {smiles: 양성확률}. 앙상블 평균은 chemprop 이 자동 처리."""
    if not canon_list:
        return {}
    uniq = sorted(set(canon_list))
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        pd.DataFrame({"canonical_smiles": uniq}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace(".csv", "_pred.csv")
    cmd = [
        find_chemprop_bin(),
        "predict",
        "--test-path", in_p,
        "-s", "canonical_smiles",
        "--model-paths", model_dir,
        "--preds-path", out_p,
        "--molecule-featurizers", DEFAULT_FEATURIZER,
        "--accelerator", "cpu",
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        sys.stderr.write(r.stderr.decode()[-1200:] + "\n")
        raise RuntimeError("chemprop predict 실패")
    df = pd.read_csv(out_p)
    pcol = [c for c in df.columns if c != "canonical_smiles"][0]
    return dict(zip(df["canonical_smiles"], df[pcol].astype(float)))


def main():
    ap = argparse.ArgumentParser(description="DILI(간독성) 예측 — SMILES → 1/0")
    ap.add_argument("smiles", nargs="*", help="SMILES 문자열 (여러 개 가능)")
    ap.add_argument("--file", help="SMILES 파일 (.txt 한 줄당 하나 / .csv)")
    ap.add_argument("--smiles-col", help="CSV 입력 시 SMILES 컬럼명")
    ap.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    ap.add_argument("--out", help="결과 CSV 저장 경로")
    args = ap.parse_args()

    rows = read_inputs(args)
    if not rows:
        ap.error("입력 SMILES 가 없다. 인자나 --file 로 전달하라.")

    # 표준화
    std = [(name, raw, standardize(raw)) for name, raw in rows]
    canon_list = [r[2][0] for r in std if r[2] is not None]
    probs = chemprop_predict(canon_list, args.model_dir)

    print(f"\n  모델: {os.path.basename(args.model_dir)}  |  threshold: {args.threshold}\n")
    print(f"  {'이름':<18}{'예측':>6}{'확률':>9}   {'판정':<12}{'SMILES(표준화)'}")
    print("  " + "-" * 78)
    out_records = []
    for name, raw, sr in std:
        if sr is None:
            print(f"  {name or '-':<18}{'?':>6}{'-':>9}   {'파싱 실패':<12}{raw}")
            out_records.append({"name": name, "input": raw, "canonical": "",
                                "prob": "", "pred": "", "verdict": "parse_error"})
            continue
        canon = sr[0]
        p = probs.get(canon, float("nan"))
        pred = int(p >= args.threshold) if not np.isnan(p) else -1
        verdict = "간독성 위험 ⚠" if pred == 1 else ("안전" if pred == 0 else "예측 실패")
        print(f"  {name or '-':<18}{pred:>6}{p:>9.3f}   {verdict:<12}{canon}")
        out_records.append({"name": name, "input": raw, "canonical": canon,
                            "prob": round(float(p), 4), "pred": pred, "verdict": verdict})
    print()

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(out_records[0].keys()))
            w.writeheader()
            w.writerows(out_records)
        print(f"  저장: {args.out}\n")


if __name__ == "__main__":
    main()
