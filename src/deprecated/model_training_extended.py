"""TDC + 시판약 결합 학습 (v1.1.0+)."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import extract_features  # noqa: E402
from src.marketed_drugs.merge import smiles_to_inchikey  # noqa: E402
from src.model_training import (  # noqa: E402
    cross_validate_model,
    evaluate_on_set,
    get_models,
    load_split_data,
    tune_and_build_ensemble,
)
from src.model_versioning import (  # noqa: E402
    DEFAULT_MODELS_DIR,
    get_next_version,
    load_versioned_meta,
    load_versioned_model,
    save_versioned_model,
)

logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"


def _load_combined_dataset(strategy: str = "undersample", ratio: int = 5) -> pd.DataFrame:
    """TDC train + marketed clean + DILIrank/SIDER 통합 hepatotoxic_all 양성."""
    tdc_train = pd.read_csv(DATA_DIR / "train" / "dili_train.csv")
    marketed_clean_path = DATA_DIR / "marketed_drugs" / "non_hepatotoxic" / "marketed_clean.csv"
    hepatotoxic_all_path = DATA_DIR / "marketed_drugs" / "hepatotoxic" / "hepatotoxic_all.csv"

    if not marketed_clean_path.exists():
        raise FileNotFoundError(f"{marketed_clean_path} 없음 — `make collect-marketed` 먼저 실행")

    marketed_clean = pd.read_csv(marketed_clean_path)

    # TDC: 그대로 (양성/음성 둘 다)
    tdc_train = tdc_train.copy()
    tdc_train["inchi_key"] = tdc_train["SMILES"].apply(smiles_to_inchikey)
    tdc_rows = pd.DataFrame(
        {
            "smiles": tdc_train["SMILES"],
            "label": tdc_train["Label"].astype(int),
            "inchi_key": tdc_train["inchi_key"],
            "source_label": "tdc",
        }
    )
    tdc_keys = set(tdc_rows["inchi_key"]) - {""}

    # Hepatotoxic 통합 (DILIrank vMost+vLess + TDC + SIDER) — 모두 양성으로 사용
    if hepatotoxic_all_path.exists():
        hepa = pd.read_csv(hepatotoxic_all_path)
        hepa = hepa[~hepa["inchi_key"].isin(tdc_keys)]
        hepa_rows = pd.DataFrame(
            {
                "smiles": hepa["canonical_smiles"],
                "label": 1,
                "inchi_key": hepa["inchi_key"],
                "source_label": "hepatotoxic_all",
            }
        )
        logger.info("Hepatotoxic 통합 (DILIrank+SIDER+TDC매칭): %d (TDC 중복 제외)", len(hepa_rows))
    else:
        # 폴백: marketed_all.csv의 dilirank_category로만 (구버전)
        marketed_all_path = DATA_DIR / "marketed_drugs" / "all" / "marketed_all.csv"
        marketed_all = (
            pd.read_csv(marketed_all_path) if marketed_all_path.exists() else pd.DataFrame()
        )
        if not marketed_all.empty and "dilirank_category" in marketed_all.columns:
            dilirank_pos = marketed_all[marketed_all["dilirank_category"] == "vMost-DILI-Concern"]
            dilirank_pos = dilirank_pos[~dilirank_pos["inchi_key"].isin(tdc_keys)]
            hepa_rows = pd.DataFrame(
                {
                    "smiles": dilirank_pos["canonical_smiles"],
                    "label": 1,
                    "inchi_key": dilirank_pos["inchi_key"],
                    "source_label": "dilirank_vmost",
                }
            )
        else:
            hepa_rows = pd.DataFrame(columns=["smiles", "label", "inchi_key", "source_label"])

    pos_keys = tdc_keys | set(hepa_rows["inchi_key"]) - {""}

    clean_filtered = marketed_clean[~marketed_clean["inchi_key"].isin(pos_keys)]
    clean_rows = pd.DataFrame(
        {
            "smiles": clean_filtered["canonical_smiles"],
            "label": 0,
            "inchi_key": clean_filtered["inchi_key"],
            "source_label": "marketed_clean",
        }
    )

    combined = pd.concat([tdc_rows, hepa_rows, clean_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset="inchi_key", keep="first").reset_index(drop=True)

    pos_count = int((combined["label"] == 1).sum())
    neg_count = int((combined["label"] == 0).sum())
    logger.info("Combined: %d (pos=%d, neg=%d)", len(combined), pos_count, neg_count)

    if strategy == "undersample":
        target_neg = max(pos_count * ratio, pos_count + 50)
        if neg_count > target_neg:
            neg_sampled = combined[combined["label"] == 0].sample(n=target_neg, random_state=42)
            combined = (
                pd.concat([combined[combined["label"] == 1], neg_sampled], ignore_index=True)
                .sample(frac=1, random_state=42)
                .reset_index(drop=True)
            )
            logger.info(
                "Undersample 1:%d → %d (pos=%d, neg=%d)",
                ratio,
                len(combined),
                pos_count,
                target_neg,
            )
    elif strategy == "balanced":
        pass
    elif strategy == "smote":
        logger.info("SMOTE는 feature 추출 후 적용 (현재 단계에서는 데이터만 반환)")
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return combined


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _source_breakdown(combined: pd.DataFrame) -> list[dict]:
    out = []
    for src in combined["source_label"].unique():
        sub = combined[combined["source_label"] == src]
        out.append(
            {
                "name": src,
                "samples": int(len(sub)),
                "pos": int((sub["label"] == 1).sum()),
                "neg": int((sub["label"] == 0).sum()),
            }
        )
    return out


def _make_meta(
    version: str,
    cv_results: dict,
    val_results: dict,
    combined: pd.DataFrame,
    strategy: str,
    final_size: int,
    previous_version: str | None,
    previous_val_auc: float | None,
) -> dict:
    sha256 = hashlib.sha256(combined.to_csv(index=False).encode()).hexdigest()
    delta = None
    if previous_val_auc is not None:
        delta = round(float(val_results["auc"]) - previous_val_auc, 4)

    return {
        "version": version,
        "git_sha": _git_sha(),
        "trained_at": dt.datetime.now().astimezone().isoformat(),
        "model": {
            "type": "Soft Voting Ensemble",
            "components": ["Random Forest", "XGBoost", "LightGBM"],
            "feature_set": "B",
            "feature_dims": 2058,
        },
        "training_data": {
            "sources": _source_breakdown(combined),
            "total_samples": int(len(combined)),
            "class_balance_strategy": strategy,
            "final_train_samples": int(final_size),
            "data_manifest_sha256": sha256,
        },
        "performance": {
            "cv_auc": float(cv_results["auc"]["mean"]),
            "validation_auc": float(val_results["auc"]),
            "test_auc": None,
            "comparison_to_previous": {"auc_delta": delta} if delta is not None else None,
        },
        "previous_version": previous_version,
    }


def _generate_comparison_md(
    prev_version: str, prev: dict, new_version: str, new: dict, kind: str = "Validation"
) -> str:
    delta_auc = float(new["auc"]) - float(prev.get("auc", 0))
    delta_f1 = float(new["f1"]) - float(prev.get("f1", 0))
    delta_p = float(new["precision"]) - float(prev.get("precision", 0))
    delta_r = float(new["recall"]) - float(prev.get("recall", 0))
    return f"""# Model Comparison ({kind}): v{prev_version} vs v{new_version}

| Metric    | v{prev_version} | v{new_version} | Δ |
|-----------|-----|-----|---|
| AUC       | {float(prev.get("auc", 0)):.4f} | {float(new["auc"]):.4f} | {delta_auc:+.4f} |
| F1        | {float(prev.get("f1", 0)):.4f} | {float(new["f1"]):.4f} | {delta_f1:+.4f} |
| Precision | {float(prev.get("precision", 0)):.4f} | {float(new["precision"]):.4f} | {delta_p:+.4f} |
| Recall    | {float(prev.get("recall", 0)):.4f} | {float(new["recall"]):.4f} | {delta_r:+.4f} |

생성 시각: {dt.datetime.now().astimezone().isoformat()}
"""


def train(strategy: str = "undersample", version: str | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if version is None:
        version = get_next_version("minor")
        logger.info("자동 버전: v%s", version)

    combined = _load_combined_dataset(strategy=strategy)

    logger.info("Feature 추출 시작 (n=%d)...", len(combined))
    feature_df, valid_idx = extract_features(combined["smiles"].tolist(), feature_set="B")
    X = feature_df.values
    y_all = combined.iloc[valid_idx]["label"].values
    y = np.array(y_all, dtype=int)
    logger.info("Features: X=%s, y 분포=%s", X.shape, dict(pd.Series(y).value_counts()))

    models = get_models()
    # 큰 데이터셋(>5000)에서는 SVM/NN 스킵 (O(n²) — 너무 느림, 앙상블에 안 쓰임)
    if len(X) > 5000:
        slow = [n for n in list(models.keys()) if n in ("SVM (RBF)", "Neural Network")]
        for n in slow:
            logger.info("%s 스킵 (큰 데이터셋, n=%d)", n, len(X))
            models.pop(n, None)

    all_results: dict[str, dict] = {}
    for name, model in models.items():
        logger.info("%s 학습 중 (10-fold CV)...", name)
        results, _, _ = cross_validate_model(model, X, y)
        all_results[name] = results
        logger.info("  %s CV AUC: %.4f ± %.4f", name, results["auc"]["mean"], results["auc"]["std"])

    sorted_models = sorted(all_results.items(), key=lambda x: x[1]["auc"]["mean"], reverse=True)
    top_names = [name for name, _ in sorted_models[:3]]
    final_model, final_name = tune_and_build_ensemble(X, y, top_names, models)

    X_val, y_val = load_split_data("validation", feature_set="B")
    val_results = evaluate_on_set(final_model, X_val, y_val, set_name=f"Validation_v{version}")

    # F1 최대화 threshold 튜닝 (validation set 기준)
    from sklearn.metrics import f1_score

    y_val_prob = final_model.predict_proba(X_val)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        y_pred = (y_val_prob >= t).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        if f1 > best_f1:
            best_t, best_f1 = float(t), float(f1)
    logger.info("Threshold tuning: best=%.3f (F1=%.4f)", best_t, best_f1)

    cv_serializable = {
        name: {k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in res.items()}
        for name, res in all_results.items()
    }

    prev_version = "1.0.0"
    prev_val_auc = None
    prev_val_results = None
    try:
        prev_meta = load_versioned_meta(prev_version)
        prev_val_auc = (prev_meta.get("performance") or {}).get("validation_auc")
        prev_val_path = DEFAULT_MODELS_DIR / f"v{prev_version}" / "validation_results.json"
        if prev_val_path.exists():
            prev_val_results = json.loads(prev_val_path.read_text())
    except FileNotFoundError:
        logger.warning("v%s 메타 없음 — 비교 생략", prev_version)

    meta = _make_meta(
        version=version,
        cv_results=all_results[top_names[0]],
        val_results=val_results,
        combined=combined,
        strategy=strategy,
        final_size=int(X.shape[0]),
        previous_version=prev_version if prev_val_auc is not None else None,
        previous_val_auc=prev_val_auc,
    )
    meta["classification_threshold"] = round(best_t, 3)
    meta["classification_threshold_tuned_on"] = "validation_set_f1_max"
    meta["model"]["components"] = top_names  # 실제 ensemble 구성 반영

    manifest = combined[["smiles", "label", "inchi_key", "source_label"]].copy()
    save_versioned_model(
        model=final_model,
        version=version,
        meta=meta,
        cv_results=cv_serializable,
        val_results=val_results,
        manifest=manifest,
    )

    if prev_val_results is not None and prev_version != version:
        md = _generate_comparison_md(prev_version, prev_val_results, version, val_results)
        cmp_path = RESULTS_DIR / f"comparison_v{prev_version}_vs_v{version}.md"
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        cmp_path.write_text(md)
        logger.info("비교 리포트: %s", cmp_path)

    # VERSIONS.md 업데이트
    versions_md = DEFAULT_MODELS_DIR / "VERSIONS.md"
    new_entry = (
        f"\n## v{version}\n\n"
        f"- 학습 데이터: 결합 ({meta['training_data']['total_samples']}개, "
        f"strategy={strategy})\n"
        f"- CV AUC: {meta['performance']['cv_auc']:.4f}\n"
        f"- Validation AUC: {meta['performance']['validation_auc']:.4f}\n"
        f"- 학습 시각: {meta['trained_at']}\n"
    )
    if versions_md.exists():
        versions_md.write_text(versions_md.read_text() + new_entry)
    else:
        versions_md.write_text("# Model Versions\n" + new_entry)

    return 0


def test(version: str | None = None) -> int:
    if version is None:
        from src.model_versioning import get_current_version

        version = get_current_version()

    model = load_versioned_model(version)
    X_test, y_test = load_split_data("test", feature_set="B")
    test_results = evaluate_on_set(model, X_test, y_test, set_name=f"Test_v{version}")

    target = DEFAULT_MODELS_DIR / f"v{version}" / "test_results.json"
    target.write_text(json.dumps(test_results, indent=2))
    logger.info("Test 결과 저장: %s", target)

    prev_version = "1.0.0"
    prev_test_path = DEFAULT_MODELS_DIR / f"v{prev_version}" / "test_results.json"
    if prev_test_path.exists() and prev_version != version:
        prev = json.loads(prev_test_path.read_text())
        md = _generate_comparison_md(prev_version, prev, version, test_results, kind="Test")
        cmp_path = RESULTS_DIR / f"comparison_v{prev_version}_vs_v{version}_test.md"
        cmp_path.parent.mkdir(parents=True, exist_ok=True)
        cmp_path.write_text(md)
        logger.info("Test 비교 리포트: %s", cmp_path)

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="확장 학습 (v1.1.0+)")
    parser.add_argument(
        "--strategy",
        choices=["undersample", "balanced", "smote"],
        default="undersample",
    )
    parser.add_argument("--version", default=None, help="명시적 버전 (예: 1.2.0)")
    parser.add_argument("--test-only", action="store_true", help="Test set만 평가")
    args = parser.parse_args()

    if args.test_only:
        return test(args.version)
    return train(strategy=args.strategy, version=args.version)


if __name__ == "__main__":
    sys.exit(main())
