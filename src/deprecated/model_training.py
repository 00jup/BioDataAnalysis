"""
모델 학습 및 평가 모듈
- 10-fold Stratified Cross Validation
- 모델 비교: RF, SVM, MLP, XGBoost, LightGBM
- Confusion Matrix, ROC Curve
- Hyperparameter Tuning
- 앙상블 (VotingClassifier)
"""

import json
import os
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
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

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

RANDOM_STATE = 42


TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "train")
TEST_DIR = os.path.join(PROJECT_ROOT, "data", "test")


def load_data(feature_set="B"):
    """Feature와 Label 로드."""
    feature_path = os.path.join(PROCESSED_DIR, f"features_{feature_set}.csv")
    label_path = os.path.join(PROCESSED_DIR, "labels.csv")
    X = pd.read_csv(feature_path)
    y = pd.read_csv(label_path).squeeze()
    print(f"데이터 로드: X={X.shape}, y 분포={dict(y.value_counts())}")
    return X.values, y.values


def get_models():
    """평가할 모델 딕셔너리 반환."""
    models = {
        "Random Forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "SVM (RBF)": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    SVC(
                        kernel="rbf",
                        class_weight="balanced",
                        probability=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Neural Network": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    MLPClassifier(
                        hidden_layer_sizes=(256, 128),
                        max_iter=500,
                        early_stopping=True,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    XGBClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.1,
                        scale_pos_weight=1,  # 나중에 자동 계산
                        random_state=RANDOM_STATE,
                        use_label_encoder=False,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    if HAS_LGBM:
        models["LightGBM"] = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LGBMClassifier(
                        n_estimators=500,
                        max_depth=6,
                        learning_rate=0.1,
                        is_unbalance=True,
                        random_state=RANDOM_STATE,
                        verbose=-1,
                        n_jobs=-1,
                    ),
                ),
            ]
        )

    return models


def cross_validate_model(model, X, y, n_folds=10):
    """10-fold Stratified CV로 모델 평가."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)

    metrics = {
        "auc": [],
        "accuracy": [],
        "f1": [],
        "precision": [],
        "recall": [],
    }
    all_y_true = []
    all_y_prob = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        metrics["auc"].append(roc_auc_score(y_val, y_prob))
        metrics["accuracy"].append(accuracy_score(y_val, y_pred))
        metrics["f1"].append(f1_score(y_val, y_pred))
        metrics["precision"].append(precision_score(y_val, y_pred, zero_division=0))
        metrics["recall"].append(recall_score(y_val, y_pred))

        all_y_true.extend(y_val)
        all_y_prob.extend(y_prob)

    results = {}
    for metric_name, values in metrics.items():
        results[metric_name] = {
            "mean": np.mean(values),
            "std": np.std(values),
        }

    return results, np.array(all_y_true), np.array(all_y_prob)


def evaluate_all_models(X, y, n_folds=10):
    """모든 모델을 평가하고 결과를 비교."""
    models = get_models()
    all_results = {}
    roc_data = {}

    print(f"\n{'=' * 60}")
    print("10-fold Stratified Cross Validation 시작")
    print(f"{'=' * 60}")

    for name, model in models.items():
        print(f"\n▶ {name} 학습 중...")
        results, y_true, y_prob = cross_validate_model(model, X, y, n_folds)
        all_results[name] = results
        roc_data[name] = (y_true, y_prob)

        print(f"  AUC:       {results['auc']['mean']:.4f} ± {results['auc']['std']:.4f}")
        print(f"  Accuracy:  {results['accuracy']['mean']:.4f} ± {results['accuracy']['std']:.4f}")
        print(f"  F1:        {results['f1']['mean']:.4f} ± {results['f1']['std']:.4f}")
        print(
            f"  Precision: {results['precision']['mean']:.4f} ± {results['precision']['std']:.4f}"
        )
        print(f"  Recall:    {results['recall']['mean']:.4f} ± {results['recall']['std']:.4f}")

    return all_results, roc_data, models


def print_comparison_table(all_results):
    """모델 비교 테이블 출력."""
    print(f"\n{'=' * 80}")
    print(f"{'모델 비교':^80}")
    print(f"{'=' * 80}")
    header = (
        f"{'Model':<20} {'AUC':>12} {'Accuracy':>12} {'F1':>12} {'Precision':>12} {'Recall':>12}"
    )
    print(header)
    print("-" * 80)

    rows = []
    for name, results in all_results.items():
        row = (
            f"{name:<20} "
            f"{results['auc']['mean']:.4f}±{results['auc']['std']:.3f} "
            f"{results['accuracy']['mean']:.4f}±{results['accuracy']['std']:.3f} "
            f"{results['f1']['mean']:.4f}±{results['f1']['std']:.3f} "
            f"{results['precision']['mean']:.4f}±{results['precision']['std']:.3f} "
            f"{results['recall']['mean']:.4f}±{results['recall']['std']:.3f}"
        )
        rows.append(row)
        print(row)

    # 최고 AUC 모델
    best_model = max(all_results.items(), key=lambda x: x[1]["auc"]["mean"])
    print(f"\n🏆 최고 AUC 모델: {best_model[0]} (AUC={best_model[1]['auc']['mean']:.4f})")

    return best_model[0]


def plot_roc_curves(roc_data, save=True):
    """모든 모델의 ROC curve를 하나의 그래프에 그림."""
    plt.figure(figsize=(8, 6))

    for name, (y_true, y_prob) in roc_data.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})", linewidth=2)

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curves - Model Comparison", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "roc_curves.png")
        plt.savefig(path, dpi=150)
        print(f"ROC curve 저장: {path}")
    plt.close()


def plot_confusion_matrices(roc_data, save=True):
    """각 모델의 Confusion Matrix를 그림."""
    n_models = len(roc_data)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (name, (y_true, y_prob)) in zip(axes, roc_data.items()):
        y_pred = (y_prob >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax,
            xticklabels=["Non-toxic", "Toxic"],
            yticklabels=["Non-toxic", "Toxic"],
        )
        ax.set_title(name, fontsize=11)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices", fontsize=14, y=1.02)
    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix 저장: {path}")
    plt.close()


def plot_feature_importance(model, feature_names, top_n=20, save=True):
    """Random Forest의 Feature Importance 상위 N개를 시각화."""
    if hasattr(model, "named_steps"):
        clf = model.named_steps["clf"]
    else:
        clf = model

    if not hasattr(clf, "feature_importances_"):
        print("이 모델은 feature_importances_를 지원하지 않습니다.")
        return

    importances = clf.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(top_n),
        importances[indices],
        color="#3498db",
        alpha=0.8,
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Top {top_n} Most Important Features", fontsize=14)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "feature_importance.png")
        plt.savefig(path, dpi=150)
        print(f"Feature importance 저장: {path}")
    plt.close()

    return indices, importances[indices]


def plot_pr_curves(roc_data, save=True):
    """모든 모델의 Precision-Recall curve를 하나의 그래프에 그림."""
    plt.figure(figsize=(8, 6))

    for name, (y_true, y_prob) in roc_data.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})", linewidth=2)

    baseline = np.mean(list(roc_data.values())[0][0])
    plt.axhline(
        y=baseline, color="k", linestyle="--", alpha=0.5, label=f"Baseline ({baseline:.2f})"
    )
    plt.xlabel("Recall", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.title("Precision-Recall Curves", fontsize=14)
    plt.legend(loc="best", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.tight_layout()

    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        path = os.path.join(FIGURES_DIR, "pr_curves.png")
        plt.savefig(path, dpi=150)
        print(f"PR curve 저장: {path}")
    plt.close()


def tune_and_build_ensemble(X, y, top_model_names, models):
    """상위 모델들로 앙상블을 구성하고 전체 데이터로 학습."""
    print(f"\n앙상블 구성: {top_model_names}")

    estimators = []
    for name in top_model_names:
        if name in models:
            estimators.append((name, models[name]))

    if len(estimators) < 2:
        # 앙상블 구성 불가 → 단일 최적 모델 사용
        best_name = top_model_names[0]
        best_model = models[best_name]
        best_model.fit(X, y)
        return best_model, best_name

    ensemble = VotingClassifier(
        estimators=estimators,
        voting="soft",
        n_jobs=-1,
    )

    # 앙상블 평가
    print("앙상블 10-fold CV 평가 중...")
    ens_results, _, _ = cross_validate_model(ensemble, X, y)
    print(f"  앙상블 AUC: {ens_results['auc']['mean']:.4f} ± {ens_results['auc']['std']:.4f}")

    # 전체 데이터로 학습
    ensemble.fit(X, y)
    return ensemble, "Ensemble"


def load_split_data(split, feature_set="B"):
    """Train/Validation/Test 분할 데이터를 로드하고 feature 추출."""
    import sys

    sys.path.insert(0, PROJECT_ROOT)
    from src.feature_engineering import extract_features

    short_names = {"validation": "val"}
    split_dir = os.path.join(PROJECT_ROOT, "data", split)
    file_name = short_names.get(split, split)
    csv_path = os.path.join(split_dir, f"dili_{file_name}.csv")
    if not os.path.exists(csv_path):
        return None, None

    df = pd.read_csv(csv_path)
    feature_df, valid_indices = extract_features(df["SMILES"].tolist(), feature_set=feature_set)
    labels = df["Label"].iloc[valid_indices].values
    return feature_df.values, labels


def evaluate_on_set(model, X, y, set_name="Validation"):
    """주어진 데이터셋에서 모델 성능을 평가."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    results = {
        "auc": roc_auc_score(y, y_prob),
        "accuracy": accuracy_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
    }

    print(f"\n{'=' * 50}")
    print(f"  {set_name} Set 평가 결과")
    print(f"{'=' * 50}")
    for metric, value in results.items():
        print(f"  {metric:>12}: {value:.4f}")

    # 결과 저장
    os.makedirs(MODELS_DIR, exist_ok=True)
    result_path = os.path.join(MODELS_DIR, f"{set_name.lower()}_results.json")
    with open(result_path, "w") as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f"  저장: {result_path}")

    return results


def train_final_model(X, y, feature_set="B"):
    """전체 파이프라인: 평가 → 시각화 → 최종 모델 저장."""
    # 1. 모든 모델 평가 (Train set으로 CV)
    all_results, roc_data, models = evaluate_all_models(X, y)

    # 2. 비교 테이블
    best_name = print_comparison_table(all_results)

    # 3. 시각화
    plot_roc_curves(roc_data)
    plot_confusion_matrices(roc_data)

    # 4. 상위 3개 모델로 앙상블
    sorted_models = sorted(
        all_results.items(),
        key=lambda x: x[1]["auc"]["mean"],
        reverse=True,
    )
    top_names = [name for name, _ in sorted_models[:3]]

    final_model, final_name = tune_and_build_ensemble(X, y, top_names, models)

    # 5. 모델 저장
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    joblib.dump(final_model, model_path)
    print(f"\n최종 모델 저장: {model_path} ({final_name})")

    # 6. 결과 저장
    results_path = os.path.join(MODELS_DIR, "cv_results.json")
    serializable_results = {}
    for name, res in all_results.items():
        serializable_results[name] = {
            k: {"mean": float(v["mean"]), "std": float(v["std"])} for k, v in res.items()
        }
    with open(results_path, "w") as f:
        json.dump(serializable_results, f, indent=2)
    print(f"CV 결과 저장: {results_path}")

    # 7. 메타데이터 저장
    meta = {
        "feature_set": feature_set,
        "best_single_model": best_name,
        "final_model": final_name,
        "ensemble_components": top_names,
        "best_auc": float(all_results[best_name]["auc"]["mean"]),
    }
    meta_path = os.path.join(MODELS_DIR, "model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return final_model, all_results


if __name__ == "__main__":
    # Train 데이터로 학습
    X_train, y_train = load_split_data("train", feature_set="B")
    if X_train is None:
        # 분할 데이터 없으면 전체 데이터 사용 (하위 호환)
        X_train, y_train = load_data(feature_set="B")

    model, results = train_final_model(X_train, y_train, feature_set="B")

    # Validation 평가
    X_val, y_val = load_split_data("validation", feature_set="B")
    if X_val is not None:
        evaluate_on_set(model, X_val, y_val, "Validation")
