"""
DILI 예측 스크립트 (SHAP 설명 포함)
- SMILES를 직접 입력하면 독성 예측 + 이유를 출력
- 파일 입력도 지원

사용법:
    python src/predict.py "CCO" "CC(=O)Oc1ccccc1C(=O)O"
    python src/predict.py -f input.csv -o output.txt
"""

import argparse
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.feature_engineering import extract_features  # noqa: E402

# 사람이 읽을 수 있는 descriptor 설명
DESCRIPTOR_INFO = {
    "MolWt": ("분자량", "Da", "높을수록 간 대사 부담 증가"),
    "MolLogP": ("지용성 (LogP)", "", "높으면 간 축적 위험 증가"),
    "TPSA": ("극성 표면적", "Å²", "낮으면 막 투과성 높아 간 노출 증가"),
    "NumHDonors": ("수소결합 공여체 수", "개", "약물 대사 경로에 영향"),
    "NumHAcceptors": ("수소결합 수용체 수", "개", "약물 용해성/대사에 영향"),
    "NumRotatableBonds": ("회전 가능 결합 수", "개", "분자 유연성 지표"),
    "NumAromaticRings": ("방향족 고리 수", "개", "반응성 대사체 생성 가능성"),
    "FractionCSP3": ("sp3 탄소 비율", "", "높을수록 3D 복잡성 증가"),
    "HeavyAtomCount": ("무거운 원자 수", "개", "분자 크기 지표"),
    "RingCount": ("고리 수", "개", "구조적 복잡성 지표"),
}


def load_model(model_path=None, version=None):
    """저장된 모델 로드.

    우선순위:
      1) model_path 명시 → 해당 경로
      2) version 명시 → models/v{version}/best_model.pkl
      3) models/latest.txt 있음 → 해당 버전
      4) fallback: models/best_model.pkl (deprecated)
    """
    from src.model_versioning import (
        DEFAULT_MODELS_DIR,
        get_current_version,
        load_versioned_model,
    )

    if model_path is not None:
        if not os.path.exists(model_path):
            print(f"오류: 모델 파일이 없습니다: {model_path}")
            sys.exit(1)
        return joblib.load(model_path)

    if version is None:
        latest_file = DEFAULT_MODELS_DIR / "latest.txt"
        if latest_file.exists():
            version = get_current_version()

    if version is not None:
        try:
            return load_versioned_model(version)
        except FileNotFoundError:
            print(f"오류: v{version} 모델 없음 ({DEFAULT_MODELS_DIR}/v{version}/)")
            sys.exit(1)

    legacy = os.path.join(PROJECT_ROOT, "models", "best_model.pkl")
    if os.path.exists(legacy):
        warnings.warn(
            "models/best_model.pkl is deprecated. Run `make migrate-v1`.",
            DeprecationWarning,
            stacklevel=2,
        )
        return joblib.load(legacy)

    print("오류: 모델 파일이 없습니다.")
    print("먼저 `make train` 또는 `make migrate-v1` 실행.")
    sys.exit(1)


def read_smiles_from_file(input_path, column=None):
    """파일에서 SMILES + 이름 추출. (names, smiles) 튜플 리스트 반환."""
    ext = os.path.splitext(input_path)[1].lower()

    if ext == ".sdf":
        suppl = Chem.SDMolSupplier(input_path)
        smiles = [Chem.MolToSmiles(mol) if mol else None for mol in suppl]
        return [None] * len(smiles), smiles

    elif ext == ".csv":
        df = pd.read_csv(input_path)

        # SMILES 컬럼 찾기
        smiles_col = None
        for col in ([column] if column else []) + ["SMILES", "smiles", "Drug"]:
            if col and col in df.columns:
                smiles_col = col
                break
        if smiles_col is None:
            smiles_col = df.columns[0]

        # 이름 컬럼 찾기
        name_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if (
                col_lower in ("name", "drug_name", "drug_id")
                or "유발" in str(col)
                or "억제" in str(col)
            ):
                name_col = col
                break
        # 이름 컬럼 없으면 SMILES가 아닌 문자열 컬럼 사용
        if name_col is None:
            for col in df.columns:
                if col != smiles_col and df[col].dtype == object:
                    name_col = col
                    break

        # SMILES 공백 제거 + 섹션 헤더 필터링
        names = []
        smiles = []
        for _, row in df.iterrows():
            smi = str(row[smiles_col]).strip()
            # 섹션 헤더 행 건너뛰기 (SMILES 값이 "SMILES"인 경우)
            if smi.upper() == "SMILES" or not smi or smi == "nan":
                continue
            smiles.append(smi)
            if name_col is not None:
                names.append(str(row[name_col]).strip())
            else:
                names.append(None)

        return names, smiles

    else:
        with open(input_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]
        smiles = [line.split("\t")[0].split(",")[0] for line in lines]
        return [None] * len(smiles), smiles


def explain_with_shap(model, feature_df, feature_names):
    """SHAP으로 각 샘플의 예측 이유를 계산."""
    import shap

    # 모델에서 tree-based estimator 추출
    explanations = []
    tree_model = None
    scaler = None

    if hasattr(model, "estimators_"):
        # VotingClassifier — estimators_는 fitted Pipeline 리스트
        for est in model.estimators_:
            if hasattr(est, "named_steps"):
                clf = est.named_steps["clf"]
                sc = est.named_steps.get("scaler")
                if hasattr(clf, "feature_importances_"):
                    tree_model = clf
                    scaler = sc
                    break
    elif hasattr(model, "named_steps"):
        # 단일 Pipeline
        tree_model = model.named_steps["clf"]
        scaler = model.named_steps.get("scaler")

    if tree_model is None:
        return None

    # 스케일링된 데이터로 SHAP 계산
    X = feature_df.values
    if scaler is not None:
        X_scaled = scaler.transform(X)
    else:
        X_scaled = X

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_scaled)

    # binary classification: 독성(1) 클래스의 SHAP values
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]  # class 1 (toxic)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[:, :, 1]  # (samples, features, classes) -> class 1
    else:
        shap_vals = shap_values

    for i in range(len(feature_df)):
        sample_shap = shap_vals[i]
        # SHAP 절대값 기준 상위 feature 선택
        top_indices = np.argsort(np.abs(sample_shap))[::-1]

        reasons = []
        for idx in top_indices:
            idx = int(idx)
            name = feature_names[idx]
            shap_val = float(sample_shap[idx])
            raw_val = feature_df.iloc[i, idx]

            # descriptor인 경우 사람이 읽을 수 있게
            if name in DESCRIPTOR_INFO:
                desc_name, unit, desc = DESCRIPTOR_INFO[name]
                unit_str = f" {unit}" if unit else ""
                direction = "독성 증가" if shap_val > 0 else "독성 감소"
                reasons.append(
                    {
                        "feature": name,
                        "display_name": desc_name,
                        "value": raw_val,
                        "unit": unit_str,
                        "shap": shap_val,
                        "direction": direction,
                        "description": desc,
                    }
                )
            elif name.startswith("Morgan_") and raw_val == 1:
                # 활성화된 fingerprint bit만 표시
                direction = "독성 증가" if shap_val > 0 else "독성 감소"
                reasons.append(
                    {
                        "feature": name,
                        "display_name": f"구조 특성 ({name})",
                        "value": int(raw_val),
                        "unit": "",
                        "shap": shap_val,
                        "direction": direction,
                        "description": "분자 내 특정 하위구조 존재",
                    }
                )

            if len(reasons) >= 5:
                break

        explanations.append(reasons)

    return explanations


def predict_and_explain(smiles_list, model, names=None, feature_set="B", threshold=0.5):
    """SMILES 리스트에 대해 예측 + 설명 수행.

    threshold: 양성 분류 임계값. 0.5는 sklearn 기본값.
    학습된 모델의 model_meta.json에 'classification_threshold' 있으면
    main()에서 이를 자동 로드해서 전달.
    """
    if names is None:
        names = [None] * len(smiles_list)

    # Feature 추출
    feature_df, valid_indices = extract_features(smiles_list, feature_set=feature_set)
    feature_names = feature_df.columns.tolist()

    X = feature_df.values
    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    # SHAP 설명 — 일부 모델 조합(SVM 포함 앙상블 등)에서 실패할 수 있음
    try:
        explanations = explain_with_shap(model, feature_df, feature_names)
    except Exception as e:
        import warnings

        warnings.warn(f"SHAP 계산 실패 — 근거 설명 생략: {e}")
        explanations = [[] for _ in range(len(feature_df))]

    # 결과 조합
    results = []
    valid_set = set(valid_indices)

    for i, smi in enumerate(smiles_list):
        if i not in valid_set:
            results.append(
                {
                    "name": names[i],
                    "smiles": smi,
                    "prediction": -1,
                    "probability": None,
                    "label": "변환 실패",
                    "reasons": [],
                }
            )
            continue

        vi = valid_indices.index(i)
        pred = int(predictions[vi])
        prob = float(probabilities[vi])

        results.append(
            {
                "name": names[i],
                "smiles": smi,
                "prediction": pred,
                "probability": prob,
                "label": "독성 (DILI)" if pred == 1 else "비독성",
                "reasons": explanations[vi] if explanations else [],
            }
        )

    return results


def format_result(result, index=None):
    """결과를 보기 좋게 포맷팅."""
    lines = []
    prefix = f"[{index}] " if index is not None else ""

    name = result.get("name")
    name_str = f" ({name})" if name and name != "nan" else ""

    if result["prediction"] == -1:
        lines.append(f"{prefix}{result['smiles']}{name_str}")
        lines.append("  -> 유효하지 않은 SMILES (변환 실패)")
        return "\n".join(lines)

    prob = result["probability"]
    label = result["label"]

    lines.append(f"{prefix}{name_str.strip() or result['smiles']}")
    if name_str:
        lines.append(f"  SMILES: {result['smiles']}")
    lines.append(f"  예측: {label} (독성 확률: {prob:.1%})")

    if result["reasons"]:
        lines.append("  근거:")
        for r in result["reasons"]:
            arrow = "+" if r["shap"] > 0 else "-"
            if r["feature"] in DESCRIPTOR_INFO:
                val_str = f"{r['value']:.2f}" if isinstance(r["value"], float) else str(r["value"])
                lines.append(
                    f"    [{arrow}] {r['display_name']} = {val_str}{r['unit']} "
                    f"({r['direction']}) — {r['description']}"
                )
            else:
                lines.append(f"    [{arrow}] {r['display_name']} ({r['direction']})")

    return "\n".join(lines)


def format_markdown_table(results):
    """Notion/마크다운 붙여넣기용 테이블 생성."""
    lines = []
    lines.append("| 약물명 | SMILES | 예측 | 독성 확률 | 주요 근거 |")
    lines.append("|--------|--------|------|-----------|-----------|")

    for r in results:
        name = r.get("name") or ""
        if name == "nan":
            name = ""
        smiles = f"`{r['smiles']}`"

        if r["prediction"] == -1:
            lines.append(f"| {name} | {smiles} | SMILES 오류 | - | - |")
            continue

        label = r["label"]
        prob = f"{r['probability']:.1%}"

        # 근거 요약 (상위 3개)
        reason_parts = []
        for reason in r["reasons"][:3]:
            arrow = "+" if reason["shap"] > 0 else "-"
            if reason["feature"] in DESCRIPTOR_INFO:
                val = reason["value"]
                val_str = f"{val:.1f}" if isinstance(val, float) else str(val)
                reason_parts.append(f"[{arrow}] {reason['display_name']}={val_str}")
            else:
                reason_parts.append(f"[{arrow}] {reason['display_name']}")
        reasons_str = ", ".join(reason_parts) if reason_parts else "-"

        lines.append(f"| {name} | {smiles} | {label} | {prob} | {reasons_str} |")

    return "\n".join(lines)


def format_csv(results):
    """예측 결과를 CSV 문자열로 생성.

    컬럼: Name, SMILES, prediction(0/1/-1), label, probability, top_reasons
    변환 실패한 SMILES도 prediction=-1 행으로 기록 → 입력 N개 = 출력 N행.
    """
    import csv
    import io

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Name", "SMILES", "prediction", "label", "probability", "top_reasons"])

    for r in results:
        name = r.get("name") or ""
        if name == "nan":
            name = ""

        if r["prediction"] == -1:
            writer.writerow([name, r["smiles"], -1, "SMILES 오류", "", ""])
            continue

        prob = f"{r['probability']:.4f}"
        reason_parts = []
        for reason in r["reasons"][:3]:
            arrow = "+" if reason["shap"] > 0 else "-"
            reason_parts.append(f"[{arrow}] {reason['display_name']}")
        reasons_str = "; ".join(reason_parts)

        writer.writerow([name, r["smiles"], r["prediction"], r["label"], prob, reasons_str])

    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(
        description="DILI 독성 예측 — SMILES 입력 → 예측 + 이유",
        usage="%(prog)s SMILES [SMILES ...]\n       %(prog)s -f input_file [-o output.txt]",
    )
    parser.add_argument("smiles", nargs="*", help="예측할 SMILES 문자열 (여러 개 가능)")
    parser.add_argument("-f", "--file", default=None, help="SMILES가 담긴 입력 파일")
    parser.add_argument("-o", "--output", default=None, help="결과 저장 파일")
    parser.add_argument(
        "--format",
        choices=["text", "md", "csv"],
        default="text",
        help="출력 형식 (text/md/csv). csv는 콘솔 요약 + CSV 파일 저장",
    )
    parser.add_argument("--model", default=None, help="모델 파일 경로 (직접 지정)")
    parser.add_argument(
        "--model-version", default=None, help="버전 명시 (예: 1.0.0). 없으면 latest 사용"
    )
    parser.add_argument(
        "--compare-versions",
        default=None,
        help="쉼표로 구분된 두 버전 (예: 1.0.0,1.1.0) — 각 버전 결과 비교",
    )
    parser.add_argument("--column", default=None, help="CSV의 SMILES 컬럼명")
    parser.add_argument("--feature-set", default="B", help="Feature set (A/B/C/D)")
    args = parser.parse_args()

    # SMILES 수집
    names = None
    smiles_list = []
    if args.smiles:
        smiles_list = args.smiles
    elif args.file:
        names, smiles_list = read_smiles_from_file(args.file, column=args.column)
    else:
        parser.print_help()
        print("\nSMILES를 직접 입력하거나 -f 옵션으로 파일을 지정하세요.")
        sys.exit(1)

    # 유효성 체크
    if names:
        filtered = [(n, s) for n, s in zip(names, smiles_list) if s and str(s).strip()]
        if filtered:
            names, smiles_list = zip(*filtered)
            names, smiles_list = list(names), list(smiles_list)
    else:
        smiles_list = [s for s in smiles_list if s and str(s).strip()]

    if not smiles_list:
        print("오류: 입력된 SMILES가 없습니다.")
        sys.exit(1)

    # 메타에서 threshold 로드 헬퍼
    def _load_threshold(version: str | None) -> float:
        try:
            from src.model_versioning import load_versioned_meta

            meta = load_versioned_meta(version)
            return float(meta.get("classification_threshold", 0.5))
        except Exception:
            return 0.5

    # --compare-versions: 두 버전을 동시에 예측하고 비교 출력
    if args.compare_versions:
        versions = [v.strip() for v in args.compare_versions.split(",") if v.strip()]
        if len(versions) < 2:
            print("오류: --compare-versions 에는 최소 2개 버전이 필요합니다 (예: 1.0.0,1.1.0)")
            sys.exit(1)

        per_version: dict[str, list] = {}
        per_threshold: dict[str, float] = {}
        for v in versions:
            m = load_model(version=v)
            t = _load_threshold(v)
            per_threshold[v] = t
            per_version[v] = predict_and_explain(
                smiles_list, m, names=names, feature_set=args.feature_set, threshold=t
            )

        # 비교 테이블 출력
        print(f"\n{'=' * 80}")
        print(f"  버전 비교: {' vs '.join('v' + v for v in versions)}")
        print(f"{'=' * 80}")

        # 헤더에 threshold 정보 표시
        def version_label(v: str) -> str:
            return f"v{v} (t={per_threshold[v]:.2f})"

        header = f"{'#':<4}{'Name':<25}{'SMILES (앞 30자)':<35}" + "".join(
            version_label(v).ljust(20) for v in versions
        )
        print(header)
        print("-" * len(header))
        for i, smi in enumerate(smiles_list):
            n = names[i] if names else ""
            n_short = (n or "")[:24]
            s_short = smi[:32] + ("…" if len(smi) > 32 else "")
            row = f"{i + 1:<4}{n_short:<25}{s_short:<35}"
            for v in versions:
                r = per_version[v][i]
                if r["prediction"] == -1:
                    row += "ERR".ljust(20)
                else:
                    label = "독성" if r["prediction"] == 1 else "비독성"
                    row += f"{label} {r['probability']:.1%}".ljust(20)
            print(row)
        return

    # 모델 로드 + threshold (메타에서)
    model = load_model(args.model, version=args.model_version)
    threshold = _load_threshold(args.model_version)
    if threshold != 0.5:
        print(f"  (모델 메타에서 임계값 {threshold:.2f} 적용 — F1 최적화)")

    # 예측 + 설명
    results = predict_and_explain(
        smiles_list, model, names=names, feature_set=args.feature_set, threshold=threshold
    )

    # 마크다운 테이블 출력 (Notion 붙여넣기용)
    if args.format == "md":
        md = format_markdown_table(results)
        print(md)
        if args.output:
            with open(args.output, "w") as f:
                f.write(md + "\n")
            print(f"\n결과 저장: {args.output}")
        return

    # 텍스트 콘솔 출력 (text / csv 형식 공통)
    print(f"\n{'=' * 60}")
    print(f"  DILI 독성 예측 결과 ({len(results)}개)")
    print(f"{'=' * 60}")

    output_lines = []
    for i, result in enumerate(results, 1):
        text = format_result(result, index=i)
        print(f"\n{text}")
        output_lines.append(text)

    # 요약
    n_toxic = sum(1 for r in results if r["prediction"] == 1)
    n_safe = sum(1 for r in results if r["prediction"] == 0)
    n_fail = sum(1 for r in results if r["prediction"] == -1)
    print(f"\n{'=' * 60}")
    print(f"  요약: 독성 {n_toxic}개 / 비독성 {n_safe}개", end="")
    if n_fail:
        print(f" / 실패 {n_fail}개", end="")
    print(f"\n{'=' * 60}")

    # 파일 저장
    if args.format == "csv":
        out_path = args.output or "results/predictions.csv"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        # utf-8-sig: Excel에서 한글이 깨지지 않도록 BOM 포함
        with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
            f.write(format_csv(results))
        print(f"\nCSV 저장: {out_path}")
    elif args.output:
        with open(args.output, "w") as f:
            f.write("\n\n".join(output_lines) + "\n")
        print(f"\n결과 저장: {args.output}")


if __name__ == "__main__":
    main()
