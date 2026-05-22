"""실험 변형(exp1/exp2/exp3) 학습 — Mordred 개선판.

- Feature: Morgan + MACCS + Mordred 2D (feature_cache.parquet)
- Feature selection: train 기준 무분산·고상관 제거
- RF/XGB/LGBM RandomizedSearchCV 튜닝 → Soft Voting 앙상블
- 임계값: Youden's J

사용법:
    python src/train_experiment.py all
"""

from __future__ import annotations

import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.base import clone  # noqa: E402
from sklearn.ensemble import RandomForestClassifier, VotingClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (  # noqa: E402
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src.exp_features import load_xy, select_features  # noqa: E402

EXP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
# EXP_MODELS_DIR 환경변수로 출력 경로 교체 가능 (RDKit ↔ Mordred 버전)
MODELS_DIR = os.environ.get("EXP_MODELS_DIR", os.path.join(PROJECT_ROOT, "models", "experiments"))
RANDOM_STATE = 42
SEARCH_ITER = 15
SEARCH_CV = 3


def _pipe(clf):
    return Pipeline([("scaler", StandardScaler()), ("clf", clf)])


def _search_spaces(spw: float):
    rf = _pipe(
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    )
    rf_p = {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [None, 12, 20, 30],
        "clf__max_features": ["sqrt", "log2", 0.3],
        "clf__min_samples_leaf": [1, 2, 4],
    }
    xgb = _pipe(
        XGBClassifier(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="logloss",
            use_label_encoder=False,
            scale_pos_weight=spw,
        )
    )
    xgb_p = {
        "clf__n_estimators": [300, 500, 800],
        "clf__max_depth": [3, 5, 7, 9],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
    }
    lgbm = _pipe(
        LGBMClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1, verbose=-1)
    )
    lgbm_p = {
        "clf__n_estimators": [300, 500, 800],
        "clf__num_leaves": [31, 63, 127],
        "clf__max_depth": [-1, 7, 12],
        "clf__learning_rate": [0.03, 0.05, 0.1],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
    }
    return [("Random Forest", rf, rf_p), ("XGBoost", xgb, xgb_p), ("LightGBM", lgbm, lgbm_p)]


def train_one(exp: str) -> None:
    print(f"\n{'=' * 60}\n  {exp} 학습 (Morgan+MACCS+Mordred, 튜닝)\n{'=' * 60}")
    vdir = os.path.join(EXP_DIR, exp)
    X_train_full, y_train = load_xy(os.path.join(vdir, "train.csv"))
    X_val_full, y_val = load_xy(os.path.join(vdir, "val.csv"))

    kept = select_features(X_train_full)
    X_train = X_train_full[kept].to_numpy(dtype=np.float64)
    X_val = X_val_full[kept].to_numpy(dtype=np.float64)
    print(f"feature: {X_train_full.shape[1]} → selection 후 {len(kept)}")
    print(f"train X={X_train.shape}, val X={X_val.shape}")

    spw = float((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    cv = StratifiedKFold(n_splits=SEARCH_CV, shuffle=True, random_state=RANDOM_STATE)

    tuned, cv_scores = [], {}
    for name, pipe, params in _search_spaces(spw):
        print(f"▶ {name} RandomizedSearchCV ({SEARCH_ITER} iter)...")
        search = RandomizedSearchCV(
            pipe,
            params,
            n_iter=SEARCH_ITER,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        search.fit(X_train, y_train)
        cv_scores[name] = float(search.best_score_)
        print(f"  best CV AUC={search.best_score_:.4f}")
        tuned.append((name, clone(search.best_estimator_)))

    ensemble = VotingClassifier(estimators=tuned, voting="soft", n_jobs=-1)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
    ens_prob = cross_val_predict(
        ensemble, X_train, y_train, cv=skf, method="predict_proba", n_jobs=-1
    )[:, 1]
    ens_cv_auc = float(roc_auc_score(y_train, ens_prob))
    print(f"앙상블 10-fold CV AUC: {ens_cv_auc:.4f}")

    ensemble.fit(X_train, y_train)

    val_prob = ensemble.predict_proba(X_val)[:, 1]
    fpr, tpr, thr = roc_curve(y_val, val_prob)
    best_t = float(thr[int(np.argmax(tpr - fpr))])
    best_t = min(max(best_t, 0.05), 0.95)
    val_pred = (val_prob >= best_t).astype(int)
    val_results = {
        "auc": float(roc_auc_score(y_val, val_prob)),
        "accuracy": float(accuracy_score(y_val, val_pred)),
        "f1": float(f1_score(y_val, val_pred, zero_division=0)),
        "precision": float(precision_score(y_val, val_pred, zero_division=0)),
        "recall": float(recall_score(y_val, val_pred, zero_division=0)),
    }
    print(f"Val AUC={val_results['auc']:.4f}, Youden J 임계값={best_t:.3f}")

    odir = os.path.join(MODELS_DIR, exp)
    os.makedirs(odir, exist_ok=True)
    joblib.dump(ensemble, os.path.join(odir, "best_model.pkl"))
    with open(os.path.join(odir, "selected_features.json"), "w") as f:
        json.dump(kept, f)

    manifest = pd.read_csv(os.path.join(vdir, "manifest.csv"))
    meta = {
        "experiment": exp,
        "model": {
            "type": "Soft Voting Ensemble (tuned)",
            "components": ["Random Forest", "XGBoost", "LightGBM"],
            "feature": "Morgan+MACCS+Mordred",
            "n_features_selected": len(kept),
        },
        "training_data": {
            "total": int(len(manifest)),
            "positives": int((manifest["label"] == 1).sum()),
            "negatives": int((manifest["label"] == 0).sum()),
        },
        "per_model_cv_auc": cv_scores,
        "cv_auc": ens_cv_auc,
        "validation": val_results,
        "classification_threshold": round(best_t, 3),
        "threshold_method": "youden_j",
    }
    with open(os.path.join(odir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"저장: {odir}/")


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    exps = ["exp1", "exp2", "exp3"] if arg == "all" else [arg]
    for exp in exps:
        train_one(exp)


if __name__ == "__main__":
    main()
