"""Clean Codex experiment for DILI model tuning.

Outputs are isolated under data/codex/ so existing model artifacts are untouched.
The experiment uses the original TDC train/validation split, feature set C, and a
small set of stronger fixed baselines before building a soft-voting ensemble.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_engineering import extract_features  # noqa: E402

warnings.filterwarnings("ignore")

DATA_DIR = PROJECT_ROOT / "data"
OUT_DIR = DATA_DIR / "codex"
FEATURE_SET = "C"
RANDOM_STATE = 42


def _load_split(split: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    short_name = "val" if split == "validation" else split
    path = DATA_DIR / split / f"dili_{short_name}.csv"
    df = pd.read_csv(path)
    features, valid_idx = extract_features(df["SMILES"].tolist(), feature_set=FEATURE_SET)
    y = df["Label"].iloc[valid_idx].to_numpy(dtype=int)
    return df.iloc[valid_idx].reset_index(drop=True), features.to_numpy(), y


def _pipe(model) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", model)])


def _scale_pos_weight(y: np.ndarray) -> float:
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / max(pos, 1)


def _candidate_models(y_train: np.ndarray) -> dict[str, Pipeline]:
    scale_pos_weight = _scale_pos_weight(y_train)
    models: dict[str, Pipeline] = {
        "RandomForest_C": _pipe(
            RandomForestClassifier(
                n_estimators=1000,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "ExtraTrees_C": _pipe(
            ExtraTreesClassifier(
                n_estimators=1000,
                max_features="sqrt",
                min_samples_leaf=1,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "SVM_RBF_C": _pipe(
            SVC(
                C=2.0,
                kernel="rbf",
                gamma="scale",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )
        ),
        "Logistic_C": _pipe(
            LogisticRegression(
                C=0.2,
                penalty="l2",
                class_weight="balanced",
                solver="liblinear",
                random_state=RANDOM_STATE,
                max_iter=2000,
            )
        ),
    }

    if HAS_XGB:
        models["XGBoost_C"] = _pipe(
            XGBClassifier(
                n_estimators=800,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=2,
                reg_lambda=2.0,
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        )

    if HAS_LGBM:
        models["LightGBM_C"] = _pipe(
            LGBMClassifier(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=4,
                num_leaves=15,
                min_child_samples=8,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbose=-1,
                n_jobs=-1,
            )
        )

    return models


def _threshold_by_youden(y_true: np.ndarray, prob: np.ndarray) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, prob)
    threshold = float(thresholds[int(np.argmax(tpr - fpr))])
    return min(max(threshold, 0.05), 0.95)


def _metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    pred = (prob >= threshold).astype(int)
    return {
        "auc": float(roc_auc_score(y_true, prob)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }


def _save_predictions(
    path: Path,
    df: pd.DataFrame,
    y: np.ndarray,
    prob: np.ndarray,
    threshold: float,
) -> None:
    out = df[["Name", "SMILES", "Label"]].copy()
    out["y_true"] = y
    out["probability"] = prob
    out["prediction"] = (prob >= threshold).astype(int)
    out.to_csv(path, index=False)


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    train_df, X_train, y_train = _load_split("train")
    val_df, X_val, y_val = _load_split("validation")

    print(f"Codex experiment output: {OUT_DIR}")
    print(f"Feature set: {FEATURE_SET}")
    print(f"Train: X={X_train.shape}, labels={dict(pd.Series(y_train).value_counts())}")
    print(f"Validation: X={X_val.shape}, labels={dict(pd.Series(y_val).value_counts())}")

    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    candidates = _candidate_models(y_train)
    fitted: dict[str, Pipeline] = {}
    results: dict[str, dict] = {}

    for name, model in candidates.items():
        print(f"\n▶ {name}")
        oof_prob = cross_val_predict(
            clone(model),
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        cv_threshold = _threshold_by_youden(y_train, oof_prob)
        cv_metrics = _metrics(y_train, oof_prob, cv_threshold)
        print(f"  CV AUC={cv_metrics['auc']:.4f}, threshold={cv_threshold:.3f}")

        model.fit(X_train, y_train)
        fitted[name] = model
        val_prob = model.predict_proba(X_val)[:, 1]
        val_threshold = _threshold_by_youden(y_val, val_prob)
        val_metrics = _metrics(y_val, val_prob, val_threshold)
        print(
            "  Val AUC={auc:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, "
            "threshold={threshold:.3f}".format(**val_metrics)
        )

        results[name] = {
            "cv": cv_metrics,
            "validation": val_metrics,
        }
        _save_predictions(
            OUT_DIR / f"{name}_validation_predictions.csv",
            val_df,
            y_val,
            val_prob,
            val_threshold,
        )

    top_names = sorted(results, key=lambda n: results[n]["cv"]["auc"], reverse=True)[:3]
    ensemble = VotingClassifier(
        estimators=[(name, clone(fitted[name])) for name in top_names],
        voting="soft",
        n_jobs=1,
    )
    print(f"\n▶ Ensemble_C top3 by CV AUC: {top_names}")
    ens_oof_prob = cross_val_predict(
        clone(ensemble),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]
    ens_cv_threshold = _threshold_by_youden(y_train, ens_oof_prob)
    ens_cv_metrics = _metrics(y_train, ens_oof_prob, ens_cv_threshold)

    ensemble.fit(X_train, y_train)
    ens_val_prob = ensemble.predict_proba(X_val)[:, 1]
    ens_val_threshold = _threshold_by_youden(y_val, ens_val_prob)
    ens_val_metrics = _metrics(y_val, ens_val_prob, ens_val_threshold)
    print(
        "  Val AUC={auc:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}, "
        "threshold={threshold:.3f}".format(**ens_val_metrics)
    )
    results["Ensemble_C"] = {
        "components": top_names,
        "cv": ens_cv_metrics,
        "validation": ens_val_metrics,
    }
    fitted["Ensemble_C"] = ensemble
    _save_predictions(
        OUT_DIR / "Ensemble_C_validation_predictions.csv",
        val_df,
        y_val,
        ens_val_prob,
        ens_val_threshold,
    )

    best_name = max(results, key=lambda n: results[n]["validation"]["auc"])
    joblib.dump(fitted[best_name], OUT_DIR / "best_model.pkl")

    summary = {
        "feature_set": FEATURE_SET,
        "train_shape": list(X_train.shape),
        "validation_shape": list(X_val.shape),
        "best_model": best_name,
        "best_validation_auc": results[best_name]["validation"]["auc"],
        "results": results,
    }
    (OUT_DIR / "results.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = [
        "# Codex DILI Experiment",
        "",
        f"- Feature set: `{FEATURE_SET}`",
        f"- Train shape: `{X_train.shape}`",
        f"- Validation shape: `{X_val.shape}`",
        f"- Best model by validation AUC: `{best_name}`",
        f"- Best validation AUC: `{results[best_name]['validation']['auc']:.4f}`",
        "",
        "| Model | CV AUC | Val AUC | Val F1 | Val Precision | Val Recall | Threshold |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in sorted(
        results.items(), key=lambda kv: kv[1]["validation"]["auc"], reverse=True
    ):
        val = item["validation"]
        lines.append(
            f"| {name} | {item['cv']['auc']:.4f} | {val['auc']:.4f} | {val['f1']:.4f} "
            f"| {val['precision']:.4f} | {val['recall']:.4f} | {val['threshold']:.3f} |"
        )
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n")

    print(f"\nBest: {best_name} (validation AUC={results[best_name]['validation']['auc']:.4f})")
    print(f"Saved: {OUT_DIR / 'results.json'}")
    print(f"Saved: {OUT_DIR / 'best_model.pkl'}")


if __name__ == "__main__":
    run()
