CONDA_ENV := bioinfo
VENV_DIR := .venv

# Windows에서 사용할 py launcher 버전 (override 가능: make data PY_VERSION=3.10)
PY_VERSION ?= 3.14

# ─────────────────────────────────────────────
# OS 감지 (Windows vs Unix)
# ─────────────────────────────────────────────
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    VENV_BIN := $(VENV_DIR)/Scripts
    EXE := .exe
else
    DETECTED_OS := $(shell uname -s 2>/dev/null || echo Unknown)
    VENV_BIN := $(VENV_DIR)/bin
    EXE :=
endif

# ─────────────────────────────────────────────
# 환경 자동 감지: venv → py(Windows) → conda
# ─────────────────────────────────────────────
ifneq ($(wildcard $(VENV_BIN)/python$(EXE)),)
    PYTHON := $(VENV_BIN)/python$(EXE)
    RUFF := $(VENV_BIN)/ruff$(EXE)
    ENV_NAME := venv
else ifeq ($(DETECTED_OS),Windows)
    PYTHON := py -$(PY_VERSION)
    RUFF := py -$(PY_VERSION) -m ruff
    ENV_NAME := py-$(PY_VERSION)
else
    PYTHON := conda run -n $(CONDA_ENV) python
    RUFF := conda run -n $(CONDA_ENV) ruff
    ENV_NAME := conda
endif

.PHONY: init init-conda init-venv init-py init-update format lint check labels curate splits train-rfcb train-chemprop stack sanity all clean clean-env

## ──────────────────────────────────────────────
## 환경 설정
## ──────────────────────────────────────────────

ifeq ($(DETECTED_OS),Windows)
init:
	@echo ========================================
	@echo  Windows 사용자: 다음 중 하나 직접 실행
	@echo ========================================
	@echo   make init-py        - py launcher로 직접 설치 (venv 없음, 가장 간단)
	@echo   make init-venv      - venv 격리 환경 생성
	@echo   make init-conda     - conda 사용
	@echo   .\setup.ps1         - PowerShell 직접 실행
else
init: ## 환경 설정 (conda 또는 venv 선택)
	@echo "========================================"
	@echo " 환경 선택"
	@echo "========================================"
	@echo "  1) conda  (anaconda/miniconda + environment.yml)"
	@echo "  2) venv   (python -m venv + requirements.txt)"
	@echo ""
	@read -p "선택 [1/2]: " choice; \
	case $$choice in \
		1) $(MAKE) init-conda ;; \
		2) $(MAKE) init-venv ;; \
		*) echo "❌ 잘못된 선택입니다."; exit 1 ;; \
	esac
endif

init-conda: ## conda 환경 생성 및 패키지 설치
	conda env create -f environment.yml
	conda run -n $(CONDA_ENV) python scripts/install_hooks.py
	@echo ========================================
	@echo  conda 환경 설정 완료!
	@echo  활성화: conda activate $(CONDA_ENV)
	@echo ========================================

# init-py: Windows 전용 - venv 없이 py launcher로 직접 설치
init-py: ## Windows: py launcher로 패키지 직접 설치 (venv 없음)
	py -$(PY_VERSION) -m pip install --upgrade pip
	py -$(PY_VERSION) -m pip install -r requirements.txt
	py -$(PY_VERSION) scripts/install_hooks.py
	@echo ========================================
	@echo  py -$(PY_VERSION) 환경 설정 완료!
	@echo  모든 make 명령이 'py -$(PY_VERSION)'로 실행됩니다
	@echo ========================================

ifeq ($(DETECTED_OS),Windows)
init-venv:
	@echo Detecting Python via py launcher (3.11 -^> 3.10 -^> 3.12 -^> 3.13)...
	py -3.11 -m venv $(VENV_DIR) || py -3.10 -m venv $(VENV_DIR) || py -3.12 -m venv $(VENV_DIR) || py -3.13 -m venv $(VENV_DIR)
	$(VENV_BIN)/python$(EXE) -m pip install --upgrade pip
	$(VENV_BIN)/python$(EXE) -m pip install -r requirements.txt
	$(VENV_BIN)/python$(EXE) scripts/install_hooks.py
	@echo ========================================
	@echo  venv 환경 설정 완료!
	@echo  활성화 (cmd):        $(VENV_BIN)\activate.bat
	@echo  활성화 (PowerShell): $(VENV_BIN)\Activate.ps1
	@echo ========================================
else
init-venv: ## venv 환경 생성 및 패키지 설치 (python 3.10/3.11/3.12/3.13 지원)
	@VENV_PY="$$(command -v python3.11 2>/dev/null || command -v python3.12 2>/dev/null || command -v python3.10 2>/dev/null || command -v python3.13 2>/dev/null)"; \
	if [ -z "$$VENV_PY" ]; then \
		echo "❌ python3.10 / 3.11 / 3.12 / 3.13 중 하나가 필요합니다."; \
		echo "   설치 예시 (macOS):  brew install python@3.12"; \
		echo "   설치 예시 (Windows): winget install Python.Python.3.12"; \
		echo "   설치 예시 (Ubuntu):  sudo apt install python3.12 python3.12-venv"; \
		exit 1; \
	fi; \
	echo "✓ 사용할 Python: $$VENV_PY"; \
	"$$VENV_PY" -m venv $(VENV_DIR)
	$(VENV_BIN)/python$(EXE) -m pip install --upgrade pip
	$(VENV_BIN)/python$(EXE) -m pip install -r requirements.txt
	$(VENV_BIN)/python$(EXE) scripts/install_hooks.py
	@echo ""
	@echo "========================================"
	@echo " ✅ venv 환경 설정 완료!"
	@echo " 활성화: source $(VENV_DIR)/bin/activate"
	@echo "========================================"
endif

init-update: ## 환경 업데이트 (현재 활성 환경 기준)
ifeq ($(ENV_NAME),venv)
	$(VENV_BIN)/python$(EXE) -m pip install -r requirements.txt --upgrade
else ifeq ($(DETECTED_OS),Windows)
	py -$(PY_VERSION) -m pip install -r requirements.txt --upgrade
else
	conda env update -n $(CONDA_ENV) -f environment.yml --prune
endif

## ──────────────────────────────────────────────
## 코드 품질
## ──────────────────────────────────────────────

format: ## ruff로 코드 포맷팅
	$(RUFF) format src/ notebooks/

lint: ## ruff로 린트 검사
	$(RUFF) check src/

check: lint ## 포맷 + 린트 검사 (pre-push에서 사용)
	$(RUFF) format --check src/

## ──────────────────────────────────────────────
## 파이프라인
## ──────────────────────────────────────────────

labels: ## 통합 라벨 DB 빌드 (data/raw → data/labels_db/full.parquet)
	$(PYTHON) src/build_labels_db.py

curate: ## 충돌 1,210건 curate (data/labels_db/conflicts/conflicts_curated.csv)
	$(PYTHON) src/curate_conflicts.py

splits: ## 도메인별 train/val/test 분할 (vivo, vitro)
	$(PYTHON) src/build_domain_splits.py

train-rfcb: ## RF/CB 5 fingerprint × scaffold-v2 학습
	$(PYTHON) src/train_rfcb_scaffold_v2.py

train-chemprop: ## Chemprop v17 (ensemble 15 + hidden 600)
	$(PYTHON) src/train_chemprop_v17.py

stack: ## Honest stacking (val 에서 α/threshold 결정 → test 평가)
	$(PYTHON) src/stack_honest.py

sanity: ## 잘 알려진 약 (Acetaminophen 등) sanity check
	$(PYTHON) src/sanity_check.py

all: labels curate splits train-rfcb train-chemprop stack ## 전체 파이프라인

## ──────────────────────────────────────────────
## 유틸리티
## ──────────────────────────────────────────────

clean: ## 생성된 파일 정리
	rm -rf data/legacy/ models/*.pkl models/*.json results/figures/*.png
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

clean-env: ## 환경 삭제 (현재 활성 환경 기준)
ifeq ($(ENV_NAME),venv)
	$(PYTHON) -c "import shutil; shutil.rmtree('$(VENV_DIR)', ignore_errors=True)"
else ifneq ($(filter py-%,$(ENV_NAME)),)
	@echo "py 환경은 패키지 제거가 필요합니다: py -$(PY_VERSION) -m pip uninstall -r requirements.txt"
else
	conda env remove -n $(CONDA_ENV)
endif
