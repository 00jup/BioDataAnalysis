# DILI Prediction — SMILES 기반 약물성 간독성 예측

약물의 SMILES를 입력하면 간독성(Drug-Induced Liver Injury, DILI) 여부를 1(있음) / 0(없음)으로 예측한다.

## 최종 모델 (제출본)

표현 계열이 서로 다른 세 모델을 honest stacking으로 결합한다.

| 구성 모델 | 표현 | 가중치 |
|---|---|---:|
| Chemprop D-MPNN ensemble | 분자 그래프 | 0.8 |
| ChemBERTa-zinc (fine-tuned) | SMILES 트랜스포머 (전이학습) | 0.2 |
| RF / CatBoost (5 fingerprint) | fingerprint | 0.0 (ablation) |

- 결합: `P = 0.8 × Chemprop + 0.2 × ChemBERTa`, threshold 0.45
- scaffold-OOD test(2,086분자): AUC 0.797 / MCC 0.42 / Specificity 0.96

## 데이터

9개 출처(DILIrank, LiverTox, DailyMed, PubMed, CTD, FAERS, ChEMBL, Marketed pool, Class Expansion)를 InChIKey 기준 통합 → vivo 라벨 13,902분자(양성 3,381 / 음성 10,521, 양성률 24%). Bemis-Murcko scaffold-balanced split 70/15/15 (train 9,731 / val 2,085 / test 2,086).

음성의 약 94%(9,919)는 "독성 신호가 없는 시판약"을 음성 anchor로 사용한 것이다.

## 예측 사용법

```bash
# 단일 / 다중 SMILES → 1/0
.venv/bin/python src/predict_stack.py "CC(=O)Nc1ccc(O)cc1"

# 파일 입력 (한 줄에 SMILES 하나)
.venv/bin/python src/predict_stack.py --file drugs.txt --out out.csv

# 대화형 (실행해두고 SMILES 붙여넣기)
.venv/bin/python src/predict_app.py

# 모델별 비교 (chemprop / RFCB / ChemBERTa / stacked)
.venv/bin/python src/predict_compare.py "SMILES"
```

전부 로컬 CPU에서 동작한다(GPU·인터넷 불필요). 입력 SMILES는 학습과 동일한 RDKit MolStandardize 체인으로 표준화 후 예측한다.

## ChemBERTa Fine-tuning (Colab GPU)

`notebooks/chemberta_finetune.ipynb` — ZINC 사전학습 ChemBERTa에 분류 헤드를 붙여 DILI train(9,731)으로 fine-tuning. 클래스 가중 손실, 10 epoch, val MCC 최고 epoch 채택. 산출된 val/test 확률을 stacking에 사용한다.

## 주요 스크립트

```
src/
  predict_stack.py        최종 stacked 추론 (chemprop v31 + ChemBERTa)
  predict_app.py          대화형 추론
  predict_compare.py      전 모델 비교 추론
  predict.py              단일 chemprop(v27) 추론
  gen_stack_preds.py      chemprop/RFCB의 val/test 예측 생성
  stack_final.py          honest stacking (val 가중치/threshold → test)
  build_strict_labels.py  엄격 라벨(DILIrank vMost) 재구성 (실험)
notebooks/
  chemberta_finetune.ipynb   ChemBERTa 파인튜닝 (Colab)
```

## 평가 결과 요약

| 평가셋 | AUC | MCC | 비고 |
|---|---:|---:|---|
| scaffold-OOD test (2,086) | 0.797 | 0.42 | 동일 라벨 정의 |
| 외부 263 신약 (구성 모델) | 0.51 | 0.01 | 학습 DB와 0 중첩 |

## 한계 (핵심 발견)

성능은 모델보다 라벨 정의에 종속된다. 학습 라벨은 "독성 신호 없으면 안전"(양성 24%) 기준이라 보수적이며, specificity(0.96)는 높으나 sensitivity(0.38)는 낮다(독성 약을 일부 놓침). 평가 기준의 양성률이 다르면 정확도가 크게 달라진다. 외부 신약 평가에서 random 수준으로 떨어지는 것은 본 모델만의 문제가 아니라 DILI prediction 분야 전반의 한계로, MoLFormer·ChemBERTa·GROVER·KPGT 등 최신 모델도 동일한 경향을 보인다.

## References

- Yang et al. (2019), Heid et al. (2024) — Chemprop D-MPNN
- Chithrananda et al. (2020) — ChemBERTa
- Bemis & Murcko (1996) — Scaffold split
- FDA DILIrank, NIH LiverTox, CTD, FAERS, ChEMBL 등 데이터 출처
