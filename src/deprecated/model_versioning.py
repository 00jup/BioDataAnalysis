"""모델 버저닝 시스템 (semver, cross-platform)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def parse_version(version: str) -> tuple[int, int, int]:
    parts = version.split(".")
    if len(parts) != 3:
        raise ValueError(f"Invalid version: {version}")
    return tuple(int(p) for p in parts)  # type: ignore[return-value]


def get_current_version(models_dir: Path = DEFAULT_MODELS_DIR) -> str:
    """models/latest.txt 읽거나 fallback으로 1.0.0."""
    latest_file = Path(models_dir) / "latest.txt"
    if latest_file.exists():
        return latest_file.read_text().strip()
    return "1.0.0"


def get_next_version(bump: str = "minor", models_dir: Path = DEFAULT_MODELS_DIR) -> str:
    current = get_current_version(models_dir)
    major, minor, patch = parse_version(current)
    if bump == "major":
        return f"{major + 1}.0.0"
    if bump == "minor":
        return f"{major}.{minor + 1}.0"
    if bump == "patch":
        return f"{major}.{minor}.{patch + 1}"
    raise ValueError(f"Invalid bump: {bump}")


def save_versioned_model(
    model: Any,
    version: str,
    meta: dict,
    cv_results: dict,
    val_results: dict,
    manifest: pd.DataFrame | None = None,
    test_results: dict | None = None,
    models_dir: Path = DEFAULT_MODELS_DIR,
) -> Path:
    """models/v{version}/ 디렉토리에 모든 산출물 저장 + latest.txt 갱신."""
    models_dir = Path(models_dir)
    target = models_dir / f"v{version}"
    target.mkdir(parents=True, exist_ok=True)

    joblib.dump(model, target / "best_model.pkl")
    (target / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    (target / "cv_results.json").write_text(json.dumps(cv_results, indent=2))
    (target / "validation_results.json").write_text(json.dumps(val_results, indent=2))
    if test_results is not None:
        (target / "test_results.json").write_text(json.dumps(test_results, indent=2))
    if manifest is not None:
        manifest.to_csv(target / "training_data_manifest.csv", index=False)

    (models_dir / "latest.txt").write_text(version)
    logger.info("v%s 저장 완료: %s", version, target)
    return target


def load_versioned_model(version: str | None = None, models_dir: Path = DEFAULT_MODELS_DIR) -> Any:
    if version is None:
        version = get_current_version(models_dir)
    path = Path(models_dir) / f"v{version}" / "best_model.pkl"
    return joblib.load(path)


def load_versioned_meta(version: str | None = None, models_dir: Path = DEFAULT_MODELS_DIR) -> dict:
    if version is None:
        version = get_current_version(models_dir)
    path = Path(models_dir) / f"v{version}" / "model_meta.json"
    return json.loads(path.read_text())
