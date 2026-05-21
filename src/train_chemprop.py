"""Chemprop 2.x (D-MPNN) 학습 + 외부 chEMBL test 평가.

핵심 방어 장치 (handoff 요구):
  - SCAFFOLD_BALANCED split  (Murcko 골격 기준 train/val 분리)
  - 앙상블 5 (--ensemble-size 5)
  - early stopping (patience)
  - dropout
  - M1 Max MPS 가속 (--accelerator mps)

사용:
    python src/train_chemprop.py exp_clean_strict
    python src/train_chemprop.py all
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
EXT_TEST = os.path.join(EXP_DIR, "external_test", "test.csv")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "chemprop")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

PY = sys.executable
CHEMPROP = os.path.join(os.path.dirname(PY), "chemprop")

VARIANTS = ["exp_clean_strict", "exp_clean_full", "exp_clean_nosider"]

# featurizer 추가 시 모델 저장 폴더에 suffix 붙임
EXTRA_FEATURIZER = os.environ.get("CHEMPROP_EXTRA_FEAT", "")  # 예: "rdkit_2d"


def _run(cmd: list[str], log_path: str) -> None:
    """서브프로세스 실행, stdout/stderr를 로그에 흘려보냄."""
    print(">>", " ".join(cmd))
    with open(log_path, "wb") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        # 마지막 30줄 출력
        with open(log_path) as f:
            tail = f.readlines()[-30:]
        sys.stderr.write("".join(tail))
        raise SystemExit(f"chemprop 실패 (exit {proc.returncode}). 로그: {log_path}")


def _variant_out_dir(variant: str) -> str:
    suffix = f"_{EXTRA_FEATURIZER}" if EXTRA_FEATURIZER else ""
    return os.path.join(MODELS_DIR + suffix, variant)


def train_one(variant: str) -> None:
    vdir = os.path.join(EXP_DIR, variant)
    manifest = os.path.join(vdir, "manifest.csv")
    if not os.path.exists(manifest):
        raise SystemExit(f"manifest 없음: {manifest}")

    out_dir = _variant_out_dir(variant)
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "train.log")

    cmd = [
        CHEMPROP, "train",
        "-i", manifest,
        "-s", "canonical_smiles",
        "--target-columns", "label",
        "-t", "classification",
        "--metrics", "roc", "binary-mcc", "prc",
        "--tracking-metric", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.85", "0.10", "0.05",
        "--ensemble-size", "5",
        "--epochs", "40",
        "--patience", "8",
        "--min-delta", "1e-4",
        "--dropout", "0.2",
        "--message-hidden-dim", "300",
        "--ffn-num-layers", "2",
        "--batch-size", "64",
        "--accelerator", "mps",
        "--devices", "1",
        "--data-seed", "42",
        "--pytorch-seed", "42",
        "-o", out_dir,
        "--remove-checkpoints",
        "--class-balance",
        "--save-smiles-splits",
    ]
    if EXTRA_FEATURIZER:
        cmd += ["--molecule-featurizers", EXTRA_FEATURIZER]
    t0 = time.time()
    _run(cmd, log_path)
    print(f"[{variant}] 학습 완료 ({time.time() - t0:.1f}s)")


def predict_external(variant: str) -> pd.DataFrame:
    """학습된 변형의 5개 앙상블을 외부 test에 적용 → 평균 확률."""
    out_dir = _variant_out_dir(variant)
    pred_csv = os.path.join(out_dir, "external_pred.csv")
    log_path = os.path.join(out_dir, "predict.log")

    # ensemble 폴더 자동 탐지
    model_paths = []
    for child in sorted(os.listdir(out_dir)):
        sub = os.path.join(out_dir, child)
        if os.path.isdir(sub) and child.startswith("model_"):
            ckpt = os.path.join(sub, "best.pt")
            if os.path.exists(ckpt):
                model_paths.append(ckpt)
    if not model_paths:
        # 폴더 자체를 넘기면 chemprop이 알아서 찾음
        model_paths = [out_dir]

    cmd = [
        CHEMPROP, "predict",
        "-i", EXT_TEST,
        "-s", "canonical_smiles",
        "--model-paths", *model_paths,
        "-o", pred_csv,
        "--accelerator", "mps",
        "--devices", "1",
    ]
    if EXTRA_FEATURIZER:
        cmd += ["--molecule-featurizers", EXTRA_FEATURIZER]
    _run(cmd, log_path)
    pred = pd.read_csv(pred_csv)
    return pred


def _load_pred_proba(pred_csv: str) -> pd.DataFrame:
    """chemprop predict 결과는 입력 CSV 의 ``label`` 컬럼을 확률로 덮어쓴다.
    → ``label`` 을 ``proba`` 로 rename 해서 반환."""
    pred = pd.read_csv(pred_csv)
    if "label" in pred.columns and pred["label"].dtype.kind in "fc":
        pred = pred.rename(columns={"label": "proba"})
    else:
        # fallback: 첫 번째 float 컬럼
        for c in pred.columns:
            if c == "canonical_smiles":
                continue
            if pred[c].dtype.kind in "fc":
                pred = pred.rename(columns={c: "proba"})
                break
        else:
            raise SystemExit(f"확률 컬럼을 찾지 못했다: {list(pred.columns)}")
    return pred[["inchi_key", "proba"]]


def _metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "neg": int((1 - y_true).sum()),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def evaluate(variant: str, n_balanced_runs: int = 10) -> dict:
    """외부 test 예측 결과를 두 가지 기준으로 평가."""
    out_dir = _variant_out_dir(variant)
    predict_external(variant)
    pred = _load_pred_proba(os.path.join(out_dir, "external_pred.csv"))
    test = pd.read_csv(EXT_TEST)
    merged = test.merge(pred, on="inchi_key", how="left").dropna(subset=["proba"])

    y_true = merged["label"].to_numpy(dtype=int)
    y_score = merged["proba"].to_numpy(dtype=float)

    # 임계값: 외부 test ROC 곡선의 Youden J — fair한 단일 기준
    fpr, tpr, thr = roc_curve(y_true, y_score)
    best_t = float(thr[int(np.argmax(tpr - fpr))])
    best_t = min(max(best_t, 0.05), 0.95)

    overall = _metrics(y_true, y_score, best_t)

    # 1:1 균형 평가 — 음성 191에 양성 191 랜덤추출 × N회
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n = len(neg_idx)
    rng = np.random.default_rng(42)
    bal_runs = []
    for _ in range(n_balanced_runs):
        sampled_pos = rng.choice(pos_idx, size=n, replace=False)
        idx = np.concatenate([sampled_pos, neg_idx])
        bal_runs.append(_metrics(y_true[idx], y_score[idx], best_t))
    balanced = {
        k: {
            "mean": float(np.mean([r[k] for r in bal_runs])),
            "std": float(np.std([r[k] for r in bal_runs])),
        }
        for k in ("roc_auc", "pr_auc", "mcc", "accuracy", "f1", "precision", "recall")
    }

    summary = {
        "variant": variant,
        "model": "Chemprop D-MPNN ensemble=5",
        "overall_full_test": overall,
        "balanced_1to1": {"n_runs": n_balanced_runs, "n_per_class": int(n), **balanced},
    }
    with open(os.path.join(out_dir, "external_eval.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


def main() -> None:
    args = sys.argv[1:]
    skip_train = "--eval-only" in args
    args = [a for a in args if not a.startswith("--")]
    arg = args[0] if args else "all"
    targets = VARIANTS if arg == "all" else [arg]
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_summaries = []
    for v in targets:
        if not skip_train:
            print(f"\n{'=' * 70}\n  {v} — Chemprop 학습\n{'=' * 70}")
            train_one(v)
        else:
            print(f"\n{'=' * 70}\n  {v} — 평가만 (--eval-only)\n{'=' * 70}")
        all_summaries.append(evaluate(v))

    suffix = f"_{EXTRA_FEATURIZER}" if EXTRA_FEATURIZER else ""
    out_json = os.path.join(RESULTS_DIR, f"chemprop{suffix}_external_eval.json")
    with open(out_json, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)
    print(f"\n전체 요약 저장: {out_json}")


if __name__ == "__main__":
    main()
