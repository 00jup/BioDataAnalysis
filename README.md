# DILI Prediction Pipeline

약물의 화학 구조(SMILES)를 입력하면 간독성(Drug-Induced Liver Injury) 여부를 예측한다.

다중 공공 데이터베이스 통합 → 라벨 충돌 큐레이션(WebSearch 검증) → Bemis-Murcko scaffold OOD split → Chemprop D-MPNN 앙상블 + Random Forest/CatBoost fingerprint 앙상블 → Honest stacking 으로 구성된 end-to-end 파이프라인이다.

## Pipeline

```
[데이터 수집]
  9개 출처 (DILIrank, LiverTox, DailyMed, PubMed, CTD, FAERS,
            chEMBL, Tox21, marketed_clean)
       ↓
[SMILES 표준화]
  RDKit MolStandardize (염 제거 → 전하 중화 → canonical SMILES + InChIKey)
       ↓
[라벨 통합 + 충돌 큐레이션]
  OR 룰 (양성 only → 1, 음성 only → 0)
  + 충돌 1,210건 큐레이션:
    - DILIrank/LiverTox/DM Boxed 공식 라벨 331건
    - 비약물 화학물질 425건 제외
    - 454건 6 sub-agent 병렬 WebSearch 검증 (LiverTox/PubMed/DrugBank)
       ↓
[도메인 분리]
  in vivo (임상)  vs  in vitro (어세이) 별 모델
       ↓
[Scaffold OOD Split]
  Bemis-Murcko scaffold split (train 70% / val 15% / test 15%)
       ↓
[모델 학습]
  • Chemprop D-MPNN ensemble (production) — Yang 2019, Heid 2024
  • RF/CatBoost × 5 fingerprint (ablation baseline)
       ↓
[평가]
  Honest stacking (val 에서 α + threshold 결정 → test 적용)
```

## 데이터 출처

| Source | Domain | Reliability | n compounds |
|---|---|---:|---:|
| FDA DILIrank 2.0 | in vivo (clinical) | ⭐⭐⭐ | 1,223 |
| NIH LiverTox | in vivo (clinical) | ⭐⭐⭐ | 851 |
| FDA DailyMed / openFDA | in vivo (FDA label) | ⭐⭐⭐ | 1,300 |
| chEMBL (human assays) | in vivo (clinical) | ⭐⭐ | 1,324 |
| CTD chemical-disease | in vivo (curated) | ⭐⭐ | 3,488 |
| FAERS reports | in vivo (patient) | ⭐⭐ | 234 |
| PubMed DILI MeSH | in vivo (literature) | ⭐ | 518 |
| Tox21 hepatic assays | in vitro | ⭐⭐ | 7,563 |
| FDA marketed pool | in vivo (negative) | ⭐⭐ | 11,083 |
| **Total unique InChIKey** | | | **19,273** |

### 라벨 분포 (curated)

| Domain | 양성 | 음성 | 제외 | 학습 가능 |
|---|---:|---:|---:|---:|
| **vivo** | 3,327 | 10,494 | 5,452 | **13,821** (양성 24.1%) |
| **vitro** | 2,117 | 5,444 | 11,712 | **7,561** (양성 28.0%) |

## 설치

지원 Python: 3.10 / 3.11 / 3.12 / 3.13

```bash
make init-venv          # venv 자동 생성
source .venv/bin/activate

# 또는 conda
make init-conda
conda activate bioinfo
```

> 💡 데이터 새로 받거나 (TDC), 개발 (lint) 하려면:
> ```bash
> pip install -r requirements-dev.txt
> ```

## 워크플로우

```
make labels         → 9 출처 통합 라벨 DB 빌드
make curate         → 충돌 1,210건 큐레이션 (이미 검증된 결과 적용)
make splits         → vivo / vitro domain split (scaffold OOD)
                  ↓
make train-rfcb     → RF/CatBoost × 5 fingerprint (ablation)
make train-chemprop → Chemprop D-MPNN ensemble (production)
                  ↓
make stack          → Honest stacking (val α/threshold → test)
make sanity         → 잘 알려진 약 (Acetaminophen 등) sanity check
                  ↓
make all            → labels → curate → splits → train-rfcb →
                      train-chemprop → stack 전부 순차 실행
```

## 모델을 처음부터 학습하기

> 사전 조건: `make init-venv` 또는 `make init-conda` 로 환경 설정. raw 데이터 다운로드는 별도 `fetch_*.py` 스크립트 (PyTDC + `requirements-dev.txt` 필요).

### Step 1 — 원본 데이터 수집

각 출처별로 한 번씩만 실행하면 된다. `data/raw/<source>/` 아래에 캐시됨.

```bash
# FDA DILIrank 2.0 (1,223 분자) — 이미 data/raw/dilirank/ 에 포함됨
.venv/bin/python src/fetch_dilirank_full.py

# NIH LiverTox (Likelihood Score A-E, 851 분자)
.venv/bin/python src/fetch_livertox_extended.py

# FDA DailyMed Drug Label (hepatotoxicity severity, 1,300 분자)
.venv/bin/python src/fetch_dailymed.py

# PubMed DILI MeSH (518 분자)
.venv/bin/python src/fetch_pubmed_dili.py

# CTD chemical-disease curation (3,488 분자)
.venv/bin/python src/fetch_ctd.py

# FAERS adverse event reports (234 분자)
.venv/bin/python src/fetch_faers.py
```

> Tox21, chEMBL, marketed_clean 은 이미 `data/raw/` 에 포함되어 있다.

### Step 2 — 통합 라벨 DB 빌드

```bash
make labels
# 또는: .venv/bin/python src/build_labels_db.py
```

**수행 작업**:
1. 모든 출처의 SMILES 를 RDKit `MolStandardize` 체인으로 표준화 (염 제거 → 전하 중화 → canonical)
2. InChIKey 기준으로 9개 출처 outer-join (19,273 unique compound)
3. OR 룰 적용: 양성 신호 only → 1, 음성 신호 only → 0
4. 양·음 신호 충돌 1,210건 → `data/labels_db/conflicts/conflicts_curated.csv` lookup

**결과**:
- `data/labels_db/full.parquet` (19,273 행)
- vivo 라벨: 양성 3,327 / 음성 10,494 / 제외 5,452
- vitro 라벨: 양성 2,117 / 음성 5,444 / 제외 11,712

**소요 시간**: ~2분

### Step 3 — 충돌 큐레이션 (옵션, 이미 적용됨)

이미 `conflicts_curated.csv` 가 포함되어 있어 `make labels` 만으로 충분하다. 처음부터 다시 큐레이션하려면:

```bash
make curate
# 또는: .venv/bin/python src/curate_conflicts.py
```

**큐레이션 룰**:
1. **FDA/NIH 공식 라벨** (331건) → DILIrank, LiverTox A-E, DailyMed Boxed Warning 그대로
2. **비약물 화학물질** (425건) → 산업 화학물·지방산·아미노산·시약 등 학습 제외
3. **잘 알려진 시판약** (454건) → 6개 sub-agent 가 LiverTox/PubMed/DrugBank WebSearch 검증 (양성 182 / 음성 111 / 증거 부족 161 제외)

**결과**: `data/labels_db/conflicts/conflicts_curated.csv` + 검증 상세 `conflicts_verified.csv` + `batches/batch_{1..6}_verified.csv`

### Step 4 — Scaffold OOD Split

```bash
make splits
# 또는: .venv/bin/python src/build_domain_splits.py
```

**수행 작업**:
- vivo / vitro 도메인별로 Bemis-Murcko scaffold split (70% train / 15% val / 15% test)
- 같은 scaffold 의 분자는 한 split 에만 배정 → 누설(leakage) 차단

**결과**: `data/train/{vivo,vitro}.csv`, `data/val/{vivo,vitro}.csv`, `data/test/{vivo,vitro}.csv`

### Step 5 — Chemprop D-MPNN 학습 (Production)

```bash
make train-chemprop
# 또는: .venv/bin/python src/train_chemprop_v17.py
```

**모델 하이퍼파라미터**:
- D-MPNN architecture, message-passing hidden dim **600**
- Ensemble size **15** (15 개 모델 random init 후 평균)
- RDKit 2D normalized descriptors (200-dim) 보조 feature
- Epochs **40**, early stopping patience **8**
- Loss: binary cross-entropy
- Optimizer: Adam (default)
- Class imbalance: scaffold split 의 stratification 으로 처리

**결과**:
- `models/chemprop_scaffold_v2/v17_ens15_h600/{vivo,vitro}/model_{0..14}/` (15 model checkpoint)
- `results/chemprop_v17.json`

**소요 시간**: vivo + vitro 합쳐서 ~1.5~2시간 (CPU 기준). GPU 가 있으면 `--accelerator gpu` 로 단축 가능.

### Step 6 — RF/CB Fingerprint Ablation 학습

```bash
make train-rfcb
# 또는: .venv/bin/python src/train_rfcb_scaffold_v2.py
```

**모델 구성**:
- 5종 fingerprint (ECFP6, Avalon, AtomPair, TT, Pattern) × {Random Forest, CatBoost} = **10개 sub-model**
- **Random Forest**: `n_estimators=500, max_features='sqrt', min_samples_leaf=2, class_weight='balanced'`
- **CatBoost**: `iterations=500, depth=6, learning_rate=0.05, class_weights={0:1, 1:3}`
- 10개 sub-model 의 val MCC 를 Nelder-Mead 로 최적화하여 soft-voting 가중치 결정

**결과**:
- `models/rfcb_scaffold_v2/{vivo,vitro}/{rf,cb}_{ecfp6,avalon,atompair,tt,pattern}/model.{pkl,cbm}`
- `models/rfcb_scaffold_v2/{vivo,vitro}/ensemble_meta.json` (가중치 + threshold + 성능)
- `results/rfcb_scaffold_v2.json`

**소요 시간**: ~20~30분

### Step 7 — Honest Stacking

```bash
make stack
# 또는: .venv/bin/python src/stack_honest.py
```

**수행 작업**:
1. val 셋에서 두 모델(Chemprop, RFCB) 예측 확률을 결합
2. α ∈ [0, 1] 21-step × threshold ∈ [0.05, 0.95] 91-step 그리드 서치로 **val MCC 최대화**
3. 결정된 α, threshold 를 test 셋에 그대로 적용 → **honest 일반화 성능**

**결과**: `results/stack_honest.json`
```json
{
  "honest_best_alpha": 0.30,
  "honest_best_threshold": 0.47,
  "honest_test_mcc": 0.788,
  "honest_test_auc": 0.963,
  "honest_test_tpr": 0.811,
  "honest_test_tnr": 0.962
}
```

**중요**: val 에서 결정한 α, threshold 를 test 에 적용한 honest MCC (0.788) 가 test 에서 직접 그리드 서치한 peek MCC (0.801) 와 거의 같다 → **하이퍼파라미터 선택의 안정성** 입증.

### Step 8 — Sanity Check (옵션)

잘 알려진 약 20종 (Acetaminophen, Isoniazid, Halothane 등 양성 10 + Aspirin, Ibuprofen, Loratadine 등 음성 10) 에 대한 예측을 확인한다:

```bash
make sanity
# 또는: .venv/bin/python src/sanity_check.py
```

학습된 모델이 임상적으로 명백한 케이스에서 합리적인 예측을 하는지 빠르게 검증할 수 있다.

### 전체 파이프라인 한 번에

```bash
make all
```

Step 2 ~ 7 을 순차 실행. 총 소요 시간 약 **2~3시간** (CPU 기준).

### v31 — Verified Labels 재학습 (alternative)

WebSearch 검증된 라벨로 재학습한 별도 시나리오:

```bash
.venv/bin/python src/train_chemprop_v31.py
.venv/bin/python src/train_rfcb_v31.py
.venv/bin/python src/eval_sanity_v2_v31.py
```

**결과** (verified labels, more conservative):
- Chemprop v31: vivo test MCC 0.436, AUC 0.788
- RFCB v31: vivo test MCC 0.338, AUC 0.748

라벨 노이즈가 줄어들면서 MCC 가 v17 (0.788) → v31 (0.436) 으로 낮아지는 것은 **train/test 동시에 더 엄격한 라벨 적용**으로 인한 정직한 일반화 성능 측정의 결과다.

## Feature

**(1) Chemprop D-MPNN — Production 입력 (Yang et al., 2019; Heid et al., 2024)**
분자 그래프를 directed message-passing neural network 로 처리하여 학습된 임베딩 생성. 사전에 정의되지 않은 구조 모티프도 포착할 수 있어 fingerprint 표현의 한계를 보완한다. RDKit 2D descriptor (분자량, LogP, TPSA 등) 를 보조 feature 로 결합.

**(2) 5종 분자 Fingerprint — RF/CatBoost Ablation Baseline**
- **ECFP6** (Morgan radius 3, 2048 bits) — 원자 환경
- **Avalon** (512 bits) — 경로 기반
- **AtomPair** (2048 bits) — 원자 거리 쌍
- **Topological Torsion** (2048 bits) — 4-원자 사슬
- **Pattern** (2048 bits, MACCS-like) — SMARTS substructure

각 fingerprint × {Random Forest, CatBoost} = 10개 sub-model. Chemprop 의 graph 기반 표현이 fingerprint 기반보다 우월한지 비교하기 위한 baseline 으로 사용.

## 모델 성능

### In vivo (scaffold-disjoint test set)

| Model | Accuracy | TPR | TNR | MCC | AUC |
|---|---:|---:|---:|---:|---:|
| **Chemprop D-MPNN (production)** | — | — | — | **0.436** | **0.788** |
| RF/CB × 5 FP (ablation) | 0.794 | 0.497 | 0.859 | 0.338 | 0.748 |

> *NB: Bemis-Murcko scaffold split 은 train/test 간에 동일한 분자 골격이 등장하지 않도록 분리하므로 random k-fold CV 대비 성능이 1.5~2배 낮게 측정된다. 이는 진정한 out-of-distribution 일반화 성능을 반영하기 위함이다 (Yang et al., 2019).*

### Sensitivity-Specificity trade-off

의약품 안전성 평가 맥락에서는 위음성(실제 간독성 약을 놓치는 것) 의 비용이 위양성보다 크다. 따라서 class_weight='balanced' (RF) 와 class_weights={0:1, 1:3} (CatBoost) 로 양성 클래스에 약 3배 가중치를 부여하여 TPR 을 우선 확보.

## 프로젝트 구조

```
src/
  ── 데이터 수집 ──
  fetch_dilirank_full.py    # FDA DILIrank 2.0 다운로드
  fetch_livertox_extended.py # NIH LiverTox 스크래핑
  fetch_dailymed.py         # FDA DailyMed Drug Label
  fetch_pubmed_dili.py      # PubMed DILI MeSH
  fetch_ctd.py              # CTD chemical-disease
  fetch_faers.py            # FAERS adverse event
  fetch_open_targets.py     # Open Targets
  fetch_clinicaltrials.py   # ClinicalTrials.gov
  scrape_livertox.py        # LiverTox HTML 파서
  standardize.py            # RDKit SMILES 표준화 유틸

  ── 라벨링 ──
  build_labels_db.py        # 9 출처 통합 → full.parquet (OR + curated)
  curate_conflicts.py       # 충돌 1,210건 큐레이션 스크립트
  apply_conflict_updates.py # WebSearch 검증 결과 적용
  build_domain_splits.py    # vivo / vitro scaffold split
  build_scaffold_v3.py      # Bemis-Murcko scaffold 분할
  integrate_class_expansion.py # class expansion 통합

  ── 모델 학습 ──
  train_chemprop_v17.py     # Chemprop v17 (ensemble 15, hidden 600)
  train_chemprop_v31.py     # Chemprop v31 (verified labels)
  train_rfcb_scaffold_v2.py # RF/CB × 5 FP scaffold split
  train_rfcb_v31.py         # RFCB v31 (verified labels)
  train_rfcb_v31_v2.py      # RFCB v31 v2
  train_domain_models.py    # 도메인별 학습 utility

  ── Stacking / 평가 ──
  stack_honest.py           # Honest stacking (val 결정 → test)
  sanity_check.py           # Acetaminophen 등 well-known 약 검증
  eval_sanity_v2_v31.py     # v31 sanity 평가
  source_reliability.py     # 출처별 신뢰도 정량 분석

data/
  raw/                      # 출처별 원본 데이터
  labels_db/
    full.parquet            # 통합 라벨 DB (19,273 분자)
    conflicts/
      conflicts_raw.csv     # 충돌 1,210건 원본
      conflicts_curated.csv # 큐레이션 결과
      conflicts_verified.csv # WebSearch 검증 상세
      batches/              # 6 sub-agent 배치별 검증 결과
  chemprop_scaffold_v2/     # vivo/vitro scaffold split 데이터
  train/ val/ test/         # 도메인별 분할

models/
  chemprop_scaffold_v2/v17_ens15_h600/  # Chemprop v17
  chemprop_v31_class_expanded/          # Chemprop v31 (verified)
  rfcb_scaffold_v2/                     # RFCB v2
  rfcb_v31/, rfcb_v31_v2/               # RFCB v31

results/
  stack_honest.json         # v17 + RFCB v2 stacking (MCC 0.788)
  chemprop_v31.json         # v31 Chemprop only (MCC 0.436)
  rfcb_v31.json             # v31 RFCB (MCC 0.338)
  rfcb_scaffold_v2.json     # RFCB v2 standalone
```

## Make 명령어

| 명령어 | 설명 |
|---|---|
| `make init` | 환경 설정 (conda 또는 venv) |
| `make init-venv` | venv 환경 생성 (3.10~3.13) |
| `make init-conda` | conda 환경 생성 |
| `make all` | 전체 파이프라인 (labels → stack) |
| `make labels` | 통합 라벨 DB 빌드 |
| `make curate` | 충돌 1,210건 큐레이션 |
| `make splits` | vivo/vitro scaffold split |
| `make train-rfcb` | RF/CB × 5 FP 학습 |
| `make train-chemprop` | Chemprop D-MPNN 학습 |
| `make stack` | Honest stacking |
| `make sanity` | well-known 약 sanity check |
| `make format` | ruff 코드 포맷팅 |
| `make lint` | ruff 린트 |
| `make check` | format + lint 검사 |
| `make clean` | 생성 파일 정리 |
| `make clean-env` | 가상환경 삭제 |

## References

**Methods**
- Yang et al. (2019). *Analyzing Learned Molecular Representations for Property Prediction*. J. Chem. Inf. Model. → Chemprop D-MPNN 원 논문
- Heid et al. (2024). *Chemprop: A Machine Learning Package for Chemical Property Prediction*. → Chemprop v2 구현
- Bemis & Murcko (1996). *The Properties of Known Drugs. 1. Molecular Frameworks*. → Scaffold split
- Rogers & Hahn (2010). *Extended-Connectivity Fingerprints*. → ECFP
- Breiman (2001). *Random Forests* / Prokhorenkova et al. (2018). *CatBoost*
- Matthews (1975). *Comparison of the predicted and observed secondary structure of T4 phage lysozyme*. → MCC

**Data**
- Chen et al. (2024). FDA DILIrank 2.0
- Hoofnagle et al. (2013). LiverTox (NIH NCBI Bookshelf)
- FDA DailyMed / openFDA Drug Label
- CTD (Davis et al., 2023)
- FAERS (FDA Adverse Event Reporting System)
- chEMBL (Mendez et al., 2019)
- Tox21 (Huang et al., 2016)
- RDKit (Landrum et al.) — molecular standardization & fingerprints
