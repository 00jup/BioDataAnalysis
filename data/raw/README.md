# data/raw — 출처별 원본 라벨 데이터

각 출처에서 직접 받은 라벨 데이터를 출처별 폴더로 정리한다. `build_labels_db.py` 가 여기서 읽어 통합 DB 를 만든다.

## 출처별 설명

### 🏛️ dilirank — FDA Drug-Induced Liver Injury Rank Dataset 2.0 (2024)
- **도메인**: in vivo (임상)
- **신뢰도**: ⭐⭐⭐⭐⭐
- `dilirank_vmost.csv` (152) — Most-DILI-Concern: FDA 회수 + 박스 경고
- `dilirank_vless.csv` (319) — Less-DILI-Concern: FDA 라벨 + 인과 검증
- 출처: https://www.fda.gov/media/113052/download

### 📚 dilist — DILIst Consensus (학술)
- **도메인**: in vivo (임상)
- **신뢰도**: ⭐⭐⭐⭐
- `dilist_positives.csv` (707) — 다기관 학술 합의 양성

### 🥇 gold_standard
- **도메인**: in vivo (임상)
- **신뢰도**: ⭐⭐⭐⭐
- `gold_positives.csv` (90) — 학술 큐레이션 골드

### 💊 sider — Side Effect Resource
- **도메인**: in vivo (시판 후 부작용)
- **신뢰도**: ⭐⭐ (인과 미검증, 보고 기반)
- `sider_liver_strict.csv` (587) — 간 직접 키워드 매칭
- `sider_hepatotoxic_lenient.csv` (816) — 더 넓은 hepatotoxic 매칭
- `sider_strict.csv` (613) — strict 키워드 매칭

### 🧪 tdc_dili — Therapeutics Data Commons
- **도메인**: in vivo (혼합, 학술 정제)
- **신뢰도**: ⭐⭐⭐
- `tdc_dili.csv` (26) — TDC DILI 양성 (다른 출처와 중복 제외 후)

### 🌐 chembl — ChEMBL Hepatotoxicity
- **도메인**: in vitro (어세이)
- **신뢰도**: ⭐⭐
- `chembl_hepatotoxicity_compounds.csv` (28,063) — 원본 어세이 활동 주석
- `chembl_hepatotoxicity_cleaned.csv` (7,881) — 정제본
- `chembl_toxic_positive_set.csv` (1,508) — 양성 셋
- 출처: 팀원이 ChEMBL 에서 추출

### 💻 clintox — MoleculeNet ClinTox
- **도메인**: in vivo (임상시험)
- **신뢰도**: ⭐⭐⭐
- `clintox.csv` (1,478) — 임상시험 toxicity 실패 vs FDA 승인 (Drug, Drug_ID, Y)
- 출처: PyTDC (MoleculeNet)

### 🔬 tox21 — EPA Tox21 High-Throughput Screening
- **도메인**: in vitro (HTS)
- **신뢰도**: ⭐ (낮은 임상 상관, 어세이 단위)
- 5개 stress response 어세이 (간독성 관련):
  - `tox21_sr_mmp.csv` (5,810) — 미토콘드리아 막전위
  - `tox21_sr_p53.csv` (6,774) — p53 스트레스 반응
  - `tox21_sr_are.csv` (5,832) — 항산화 반응
  - `tox21_sr_hse.csv` (6,467) — 열충격 반응
  - `tox21_nr_ahr.csv` (6,549) — 약물 대사 수용체
- `tox21_combined_5tasks.csv` (7,808) — 통합 (어느 어세이라도 양성 = 양성)
- 출처: PyTDC (EPA)

### 🏥 marketed — 시판약 풀
- **도메인**: in vivo (시판 여부 메타데이터)
- `marketed_clean.csv` (11,227) — 시판약 음성 풀 (DILIrank 카테고리 포함, hepatotoxic 풀과 비충돌)
- `marketed_all.csv` (11,721) — 시판약 전체 (hepatotoxic 포함)

## 보류 (다음 라운드)

### 📖 livertox — NIH LiverTox (NLP 필요)
- 출처: https://www.ncbi.nlm.nih.gov/books/NBK547852/
- ~1,300 약물 narrative + Likelihood A~E 라벨
- 스크래핑 + NLP 추출 필요 (3-5시간 작업)

### 🧬 toxcast — EPA ToxCast (~700 어세이)
- 간 관련 어세이 선별 + HepG2 cell health 등 추출 필요
