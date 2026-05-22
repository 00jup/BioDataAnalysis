"""기존 models/best_model.pkl을 v1.0.0/으로 마이그레이션 (1회성)."""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_CSV = PROJECT_ROOT / "data" / "train" / "dili_train.csv"


def _git_sha_for_path(path: Path) -> str:
    try:
        result = subprocess.run(
            ["git", "log", "--pretty=%H", "-1", "--", str(path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = result.stdout.strip()
        return sha if sha else "unknown"
    except Exception:
        return "unknown"


def _backfill_meta(legacy_meta: dict) -> dict:
    """기존 minimal model_meta.json에서 새 스키마로 backfill."""
    best_model = MODELS_DIR / "best_model.pkl"
    val_results_path = MODELS_DIR / "validation_results.json"

    git_sha = _git_sha_for_path(best_model)
    try:
        mtime = dt.datetime.fromtimestamp(best_model.stat().st_mtime).astimezone()
        trained_at = mtime.isoformat()
    except Exception:
        trained_at = "unknown"

    training_sources = []
    total_samples = 0
    if TRAIN_CSV.exists():
        df = pd.read_csv(TRAIN_CSV)
        pos = int((df["Label"] == 1).sum())
        neg = int((df["Label"] == 0).sum())
        total_samples = len(df)
        training_sources.append(
            {"name": "TDC DILI", "samples": total_samples, "pos": pos, "neg": neg}
        )

    val_auc = None
    if val_results_path.exists():
        try:
            val_auc = float(json.loads(val_results_path.read_text()).get("auc"))
        except Exception:
            pass

    return {
        "version": "1.0.0",
        "git_sha": git_sha,
        "trained_at": trained_at,
        "model": {
            "type": "Soft Voting Ensemble",
            "components": legacy_meta.get(
                "ensemble_components", ["Random Forest", "XGBoost", "LightGBM"]
            ),
            "feature_set": legacy_meta.get("feature_set", "B"),
            "feature_dims": 2058,
        },
        "training_data": {
            "sources": training_sources,
            "total_samples": total_samples,
            "class_balance_strategy": "balanced_class_weight",
            "final_train_samples": total_samples,
            "data_manifest_sha256": None,
        },
        "performance": {
            "cv_auc": float(legacy_meta.get("best_auc", 0.0)) or None,
            "validation_auc": val_auc,
            "test_auc": None,
            "comparison_to_previous": None,
        },
        "hyperparameters": {
            "RandomForestClassifier": {
                "n_estimators": 500,
                "class_weight": "balanced",
                "random_state": 42,
            },
            "XGBClassifier": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.1,
            },
            "LGBMClassifier": {
                "n_estimators": 500,
                "max_depth": 6,
                "learning_rate": 0.1,
                "is_unbalance": True,
            },
        },
        "previous_version": None,
    }


def migrate() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    target = MODELS_DIR / "v1.0.0"
    if target.exists():
        logger.info("이미 v1.0.0/ 존재 — skip")
        return 0

    target.mkdir(parents=True, exist_ok=True)

    src = MODELS_DIR / "best_model.pkl"
    if not src.exists():
        logger.error("best_model.pkl 없음")
        return 1
    shutil.copy2(src, target / "best_model.pkl")

    for fname in ("cv_results.json", "validation_results.json"):
        s = MODELS_DIR / fname
        if s.exists():
            shutil.copy2(s, target / fname)

    legacy_meta_path = MODELS_DIR / "model_meta.json"
    legacy_meta: dict = {}
    if legacy_meta_path.exists():
        legacy_meta = json.loads(legacy_meta_path.read_text())
        shutil.copy2(legacy_meta_path, MODELS_DIR / "model_meta.json.legacy")

    new_meta = _backfill_meta(legacy_meta)
    (target / "model_meta.json").write_text(json.dumps(new_meta, indent=2, ensure_ascii=False))

    (MODELS_DIR / "latest.txt").write_text("1.0.0")

    versions_md = MODELS_DIR / "VERSIONS.md"
    versions_md.write_text(
        "# Model Versions\n\n"
        "## v1.0.0 (마이그레이션됨)\n\n"
        f"- 학습 데이터: TDC DILI ({new_meta['training_data']['total_samples']}개)\n"
        f"- CV AUC: {new_meta['performance']['cv_auc']}\n"
        f"- Validation AUC: {new_meta['performance']['validation_auc']}\n"
        f"- 학습 시각: {new_meta['trained_at']}\n"
    )

    logger.info("✓ v1.0.0 마이그레이션 완료: %s", target)
    return 0


if __name__ == "__main__":
    sys.exit(migrate())
