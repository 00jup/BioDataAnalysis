"""대화형 DILI 예측 — 실행해두고 SMILES 만 붙여넣으면 바로 1/0.

실행:
    cd /Users/parkjeong-uk/CODING/2026/school/BioDataAnalysis
    .venv/bin/python src/predict_app.py

그러면 'SMILES> ' 프롬프트가 뜬다. SMILES 붙여넣고 Enter → 1(간독성)/0(안전).
종료: q 입력 후 Enter.

모델: stacked (0.8*chemprop v31 + 0.2*ChemBERTa-zinc), threshold 0.55.
"""

from __future__ import annotations

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from src.predict_stack import (  # noqa: E402
    DEFAULT_THRESHOLD,
    W_CHEMBERTA,
    W_CHEMPROP,
    chemberta_predict,
    chemprop_predict,
)
from src.standardize import standardize  # noqa: E402


def predict_one(smi):
    r = standardize(smi)
    if r is None:
        return None
    c = r[0]
    a = chemprop_predict([c]).get(c, float("nan"))
    b = chemberta_predict([c]).get(c, float("nan"))
    return c, a, b, W_CHEMPROP * a + W_CHEMBERTA * b


def main():
    print("=" * 48)
    print(" DILI 간독성 예측 (stacked: chemprop + ChemBERTa)")
    print(f" 1 = 간독성 위험,  0 = 안전   (threshold {DEFAULT_THRESHOLD})")
    print(" SMILES 붙여넣고 Enter.  종료: q")
    print("=" * 48)
    # ChemBERTa 미리 1회 로드 (첫 예측 지연 줄이기)
    print("모델 준비 중…", flush=True)
    try:
        chemberta_predict(["CCO"])
    except Exception:
        pass
    print("준비 완료.\n", flush=True)

    while True:
        try:
            s = input("SMILES> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if s.lower() in ("q", "quit", "exit", ""):
            print("종료.")
            break
        res = predict_one(s)
        if res is None:
            print("  ❌ SMILES 파싱 실패 — 다시 입력\n")
            continue
        c, a, b, P = res
        pred = int(P >= DEFAULT_THRESHOLD)
        verdict = "간독성 위험 ⚠" if pred else "안전 ✅"
        print(f"  → 예측 {pred}  ({verdict})   확률 {P:.3f}  [chemprop {a:.2f} / ChemBERTa {b:.2f}]\n")


if __name__ == "__main__":
    main()
