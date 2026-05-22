# Hepatotoxic 약물 데이터셋

학습용 양성(간독성) 샘플의 출처별 분류. 나중에 라벨 신뢰도 판단할 때 참고.

## 폴더 구조

```
hepatotoxic/
├── README.md                      ← 이 파일 (출처/신뢰도 설명)
├── hepatotoxic_all.csv            ← 통합 (학습에 실제 사용, 1080개)
├── breakdown.csv                  ← 출처별 카운트 요약
├── marketed_hepatotoxic.csv       ← marketed_all.csv ∩ hepatotoxic (494개, 시판 풀에 있는 것만)
├── sider_hepatotoxic.csv          ← SIDER에서 추출 (815개, 시판 풀과 무관)
└── by_source/                     ← 1차 출처별 분리
    ├── dilirank_vmost.csv         (151) ★ 신뢰도 최고
    ├── dilirank_vless.csv         (318) ★ 신뢰도 높음
    ├── tdc_dili.csv               (25)  ★ 신뢰도 높음
    └── sider_liver.csv            (586) ⚠ 신뢰도 중간
```

## 출처별 신뢰도 가이드

### ⭐⭐⭐⭐⭐ DILIrank vMost-DILI-Concern (151개)

**출처**: FDA Drug-Induced Liver Injury Rank Dataset 2.0 (2024)
**다운로드**: https://www.fda.gov/media/113052/download?attachment (Excel)
**기준**:
- 시판 후 회수된 약물
- FDA 박스 경고 (Box Warning) 포함
- 라벨에 심각한/중등도 간손상 명시
- 인과관계 (causality) 검증 완료

**판단**: FDA가 공식 인과관계 분석을 거쳐 분류. **인용 가능한 가장 강력한 근거**.
**컬럼**: `dilirank_category == "vMost-DILI-Concern"`

---

### ⭐⭐⭐⭐ DILIrank vLess-DILI-Concern (318개)

**출처**: 위와 동일 (FDA DILIrank 2.0)
**기준**: 약물 라벨 기반 저위험으로 평가, 인과관계 검증 거침.

**판단**: vMost보다는 약하지만 FDA 공식 분류. 학습용으로 양성에 포함하는 게 일반적 (논문 표준 관행 — vMost+vLess = DILI Positive).
**컬럼**: `dilirank_category == "vLess-DILI-Concern"`

---

### ⭐⭐⭐⭐ TDC DILI (25개, 다른 출처와 중복 제외 후)

**출처**: Therapeutics Data Commons DILI 데이터셋
**기준**: 기존 본 프로젝트의 학습 데이터셋. ~475 entries 중 Y=1인 약물 ~164개.

**판단**: 학술적으로 정제된 데이터셋. v1.0.0부터 사용해온 baseline.
**컬럼**: `in_tdc_dili_pos == 1`

---

### ⭐⭐⭐ SIDER (586개)

**출처**: SIDER (Side Effect Resource), http://sideeffects.embl.de/
**다운로드**: `meddra_all_se.tsv.gz` + `drug_names.tsv`
**기준**: 약물 라벨/임상시험 보고서에서 **간 관련 부작용** 보고된 약물.

매칭에 사용한 MedDRA Preferred Term 키워드:
```
hepatic, hepatotox, hepatitis, hepatomegaly, hepatocellular,
cholestasis, jaundice, liver, bilirubin, transaminase,
ALT, AST, GGT increased, alkaline phosphatase, hepatorenal
```

**91가지 고유 부작용** 중 하나라도 보고되면 양성으로 간주.

**판단**:
- ⚠ FDA DILIrank만큼 엄격하지 않음 (단순 보고만으로 hepatotoxic 라벨)
- ⚠ 인과관계 검증 안 됨 (다른 약 동시 복용 등 confounder 가능)
- 그러나 약물-부작용 연관성은 의미 있는 신호
- 양성 풀 다양성 확보 목적으로 사용
- 각 row의 `sider_terms` 컬럼에 **구체적 부작용** 기록 → 검토 시 활용

**컬럼**: `sources`에 "sider" 포함 + `sider_terms`에 부작용명

---

## 컬럼 설명

| 컬럼 | 의미 |
|---|---|
| `name` | 약물명 (preferred) |
| `canonical_smiles` | RDKit canonical SMILES (dedup 키) |
| `smiles` | 원본 SMILES (소스에서 받은 그대로) |
| `inchi_key` | 매칭/대조용 |
| `sources` | 어떤 소스에서 왔는지 (`;`로 구분, 예: "chembl;sider") |
| `source_ids` | 각 소스의 원본 ID |
| `dilirank_category` | DILIrank 분류 (vMost/vLess/vNo/Ambiguous/unknown) |
| `in_dilirank` | 0/1 |
| `in_tdc_dili_pos` | TDC DILI Y=1 매칭 여부 |
| `sider_terms` | SIDER에서 보고된 구체적 간 관련 부작용 (`;`로 구분) |
| `known_hepatotoxic` | 최종 학습용 라벨 (1 = positive) |

## 통계 (현재)

| 출처 | 개수 | 비율 |
|---|---|---|
| DILIrank vMost | 151 | 14.0% |
| DILIrank vLess | 318 | 29.4% |
| TDC DILI | 25 | 2.3% |
| SIDER | 586 | 54.3% |
| **전체 unique** | **1080** | 100% |

## 라벨 품질 검토 시 참고

라벨 의심스러운 케이스 발견 시:
1. `by_source/sider_liver.csv` 의 `sider_terms` 컬럼 확인 — 단순 ALT 상승 1건만 있다면 약함
2. DILIrank vNo-DILI-Concern으로 명시된 약물(414개)과 충돌 없는지 확인 (현재는 vMost/vLess만 양성으로 사용)
3. 의심스러운 entry는 `hepatotoxic_all.csv`에서 제외 후 재학습 가능

## 전체 hepatotoxic을 1만개로 늘리려면 (옵션)

현재 1080개. 1만개 도달을 위한 추가 출처 (라벨 품질↓ 트레이드오프 있음):
- **OpenFDA FAERS**: FDA 부작용 신고 시스템 (수십만 건, 매우 노이즈)
- **OffSides/TwoSides** (Tatonetti): FAERS에서 통계적으로 정제 (~3-5k)
- **TG-GATEs**: 동물 in-vivo 독성 (~200, 신뢰도 높지만 작음)
- **ChEMBL bioassay**: liver target IC50 데이터 (수천 개)
- **LiverTox NIH**: ~1,300 약물 narrative DB (텍스트 파싱 필요)

각 출처는 별도 모듈로 추가 가능 (현재 sider.py처럼).
