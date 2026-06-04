# DILI Prediction

SMILES를 입력하면 간독성(Drug-Induced Liver Injury) 여부를 1(있음) / 0(없음)으로 예측한다.

## 실행 방법

```bash
# 단일 / 다중 SMILES → 1/0
.venv/bin/python src/predict_stack.py "CC(=O)Nc1ccc(O)cc1"
.venv/bin/python src/predict_stack.py "SMILES1" "SMILES2"

# 파일 입력 (한 줄에 SMILES 하나) + 결과 CSV 저장
.venv/bin/python src/predict_stack.py --file drugs.txt --out out.csv

# 대화형 (실행해두고 SMILES 붙여넣기)
.venv/bin/python src/predict_app.py

# 모델별 비교 (chemprop / RFCB / ChemBERTa / stacked)
.venv/bin/python src/predict_compare.py "SMILES"
```

전부 로컬 CPU에서 동작한다 (GPU·인터넷 불필요).

## 모델 학습 방법

학습 데이터(`data/chemprop_scaffold_v3/vivo/`: `all.csv`, `splits.json`)는 이미 빌드돼 있다. 최종 모델(stacking)은 아래 순서로 만든다.

```bash
# 1) Chemprop D-MPNN ensemble (로컬 CPU, 약 45분)
.venv/bin/python src/train_chemprop_v31.py

# 2) RF / CatBoost fingerprint ensemble
.venv/bin/python src/train_rfcb_v31.py

# 3) ChemBERTa fine-tuning  →  Colab GPU에서 실행
#    notebooks/chemberta_finetune.ipynb 업로드 → GPU 켜고 셀 실행
#    all.csv, splits.json 업로드 → val/test 확률 CSV + 가중치 다운로드

# 4) Stacking — 각 모델의 val/test 예측 생성 후 가중치/threshold 결정
.venv/bin/python src/gen_stack_preds.py    # chemprop / RFCB 의 val·test 예측
.venv/bin/python src/stack_final.py         # honest stacking (val→test)
```

데이터 DB(라벨)부터 새로 빌드하려면:

```bash
make labels     # 9개 출처 통합 라벨 DB
make splits     # Bemis-Murcko scaffold split (70/15/15)
```

## Make 명령어

| 명령어 | 설명 |
|---|---|
| `make init` | 환경 설정 (conda 또는 venv 선택) |
| `make init-venv` | venv 환경 생성 및 패키지 설치 (python 3.10~3.13) |
| `make init-conda` | conda 환경 생성 및 패키지 설치 |
| `make init-py` | Windows: py launcher로 패키지 직접 설치 (venv 없음) |
| `make init-update` | 환경 업데이트 (현재 활성 환경 기준) |
| `make labels` | 통합 라벨 DB 빌드 (data/raw → data/labels_db/full.parquet) |
| `make curate` | 충돌 1,210건 큐레이션 (conflicts_curated.csv) |
| `make splits` | 도메인별 train/val/test 분할 (vivo, vitro) |
| `make train-rfcb` | RF/CB 5 fingerprint × scaffold 학습 |
| `make train-chemprop` | Chemprop ensemble (size 15, hidden 600) 학습 |
| `make stack` | Honest stacking (val에서 α/threshold 결정 → test 평가) |
| `make sanity` | 잘 알려진 약(Acetaminophen 등) sanity check |
| `make all` | 전체 파이프라인 (labels → curate → splits → train → stack) |
| `make format` | ruff 코드 포맷팅 |
| `make lint` | ruff 린트 검사 |
| `make check` | 포맷 + 린트 검사 (pre-push에서 사용) |
| `make clean` | 생성된 파일 정리 |
| `make clean-env` | 가상환경 삭제 |
