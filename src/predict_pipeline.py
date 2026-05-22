"""통합 예측 파이프라인 — SMILES → 라벨.

흐름:
  1. SMILES → standardize (염 제거·canonical·InChIKey)
  2. DB lookup (data/labels_db/full.parquet)
       - HIT → 룰 선택 (vivo_priority / weighted / consensus) → 즉답
       - MISS → 두 모델 fallback
  3. 두 모델 (vivo + vitro) 호출
       - 각 모델은 RF + CatBoost × 5 FP = 10-way 앙상블
       - 도메인별 가중평균으로 p_vivo, p_vitro
  4. 룰별 결합:
       - vivo_priority: vivo 답을 우선 (p_vivo 기반)
       - weighted: confidence-style 평균
       - consensus: 둘 다 동의해야 결정, 아니면 보수적 음성
  5. (선택) Applicability Domain 검사 → threshold 조정

사용:
    from src.predict_pipeline import HepatotoxPipeline
    pipe = HepatotoxPipeline(rule="weighted")
    res = pipe.predict("CC(=O)Nc1ccc(O)cc1")
    # → {"label": 1, "source": "db_lookup", "rule": "weighted", ...}
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from rdkit import Chem, RDLogger
from rdkit.Avalon import pyAvalonTools
from rdkit.Chem.rdFingerprintGenerator import (
    GetAtomPairGenerator,
    GetMorganGenerator,
    GetTopologicalTorsionGenerator,
)

from src.lookup import LabelDB
from src.standardize import standardize

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
FP_DIR = os.path.join(PROJECT_ROOT, "data", "fp_cache")

FPS = {"ecfp6": 2048, "avalon": 512, "atompair": 2048, "tt": 2048, "pattern": 2048}


def _fp(name: str, mol) -> np.ndarray:
    nb = FPS[name]
    arr = np.zeros(nb, dtype=np.uint8)
    if name == "ecfp6":
        bits = GetMorganGenerator(radius=3, fpSize=nb).GetFingerprint(mol).GetOnBits()
    elif name == "avalon":
        bits = pyAvalonTools.GetAvalonFP(mol, nBits=nb).GetOnBits()
    elif name == "atompair":
        bits = GetAtomPairGenerator(fpSize=nb).GetFingerprint(mol).GetOnBits()
    elif name == "tt":
        bits = GetTopologicalTorsionGenerator(fpSize=nb).GetFingerprint(mol).GetOnBits()
    elif name == "pattern":
        bits = Chem.PatternFingerprint(mol, fpSize=nb).GetOnBits()
    else:
        raise ValueError(name)
    arr[list(bits)] = 1
    return arr


@dataclass
class Prediction:
    label: int | None              # 0/1 or None (uncertain)
    score: float | None            # 0~1 확률
    source: str                    # "db_lookup" | "model" | "db_no_label"
    rule: str                      # 사용한 룰
    threshold: float | None
    smiles_input: str
    canonical_smiles: str | None
    inchi_key: str | None
    # DB lookup 결과 (있을 때)
    db_hit: bool
    db_vivo_label: int | None
    db_vitro_label: int | None
    db_final_source: str | None
    # 모델 결과 (호출됐을 때)
    model_p_vivo: float | None
    model_p_vitro: float | None
    explanation: str = ""


class DomainModel:
    """한 도메인의 10-way 앙상블 (RF+CB × 5FP) + meta(가중치, threshold)."""

    def __init__(self, domain: str):
        self.domain = domain
        self.dir = os.path.join(MODELS_DIR, domain)
        meta_path = os.path.join(self.dir, "ensemble_meta.json")
        with open(meta_path) as f:
            self.meta = json.load(f)
        # sub-model 로드
        self.models = {}
        for name in self.meta["members"]:
            sub = os.path.join(self.dir, name)
            if name.startswith("rf_"):
                self.models[name] = joblib.load(os.path.join(sub, "model.pkl"))
            else:
                m = CatBoostClassifier()
                m.load_model(os.path.join(sub, "model.cbm"))
                self.models[name] = m
        self.weights = np.array(self.meta["weights"])

    def predict(self, mol) -> float:
        """분자 1개 → 앙상블 확률 (가중평균)."""
        probas = []
        for name in self.meta["members"]:
            fp_name = name.split("_", 1)[1]
            arr = _fp(fp_name, mol).reshape(1, -1)
            p = float(self.models[name].predict_proba(arr)[0, 1])
            probas.append(p)
        return float(np.dot(self.weights, np.array(probas)))


class HepatotoxPipeline:
    def __init__(self, rule: str = "weighted", use_lookup: bool = True, load_db_fps: bool = False):
        assert rule in ("vivo_priority", "weighted", "consensus"), f"unknown rule: {rule}"
        self.rule = rule
        self.use_lookup = use_lookup
        self.db = LabelDB(load_fps=load_db_fps) if use_lookup else None
        self.vivo_model = DomainModel("vivo")
        self.vitro_model = DomainModel("vitro")
        mode = "lookup+model" if use_lookup else "ML-only (lookup off)"
        print(f"  HepatotoxPipeline 로드 (rule={rule}, mode={mode})")

    # ---- 룰별 모델 출력 결합 ----
    def _combine(self, p_vivo: float, p_vitro: float) -> tuple[int | None, float]:
        """두 모델 출력 → 라벨 + 점수. 룰에 따라 결합."""
        # 각 도메인의 threshold 적용
        t_v = self.vivo_model.meta["threshold"]
        t_vi = self.vitro_model.meta["threshold"]
        label_v = 1 if p_vivo >= t_v else 0
        label_vi = 1 if p_vitro >= t_vi else 0

        if self.rule == "vivo_priority":
            # vivo 답 우선
            return label_v, p_vivo

        if self.rule == "weighted":
            # 두 점수 평균 + 0.5 cutoff (각 도메인의 정규화 점수 평균)
            # p_vivo, p_vitro 가 각자의 threshold 기준으로 의미 다름 → score 정규화
            score_v = (p_vivo - t_v) / max(1.0 - t_v, t_v, 1e-9)   # ±1 범위
            score_vi = (p_vitro - t_vi) / max(1.0 - t_vi, t_vi, 1e-9)
            combined = (score_v + score_vi) / 2.0
            label = 1 if combined > 0 else 0
            # 0~1 점수로 변환
            score = (combined + 1) / 2
            return label, float(score)

        if self.rule == "consensus":
            if label_v == 1 and label_vi == 1:
                return 1, (p_vivo + p_vitro) / 2
            if label_v == 0 and label_vi == 0:
                return 0, (p_vivo + p_vitro) / 2
            return None, (p_vivo + p_vitro) / 2  # 충돌 → 불확실

    def predict(self, smiles: str) -> Prediction:
        """단일 SMILES 예측 — lookup 먼저, miss 시 모델."""
        std = standardize(smiles)
        if std is None:
            return Prediction(
                label=None, score=None, source="invalid_smiles", rule=self.rule,
                threshold=None, smiles_input=smiles, canonical_smiles=None, inchi_key=None,
                db_hit=False, db_vivo_label=None, db_vitro_label=None, db_final_source=None,
                model_p_vivo=None, model_p_vitro=None,
                explanation="SMILES parsing/표준화 실패",
            )
        canon, ikey = std

        # 1) DB lookup (use_lookup=False면 건너뜀)
        r = self.db.lookup_exact(canon) if self.use_lookup else None
        if r is not None and r.hit:
            db_label = r.get_label(self.rule)
            if db_label is not None:
                return Prediction(
                    label=int(db_label), score=float(db_label), source="db_lookup",
                    rule=self.rule, threshold=None,
                    smiles_input=smiles, canonical_smiles=canon, inchi_key=ikey,
                    db_hit=True, db_vivo_label=r.vivo_label, db_vitro_label=r.vitro_label,
                    db_final_source=r.final_source,
                    model_p_vivo=None, model_p_vitro=None,
                    explanation=f"DB hit ({r.final_source}); rule={self.rule}",
                )
            # 라벨 없음 (consensus 등에서 abstain) → 모델로
            db_hit_no_label = True
        else:
            db_hit_no_label = False

        # 2) 모델 fallback
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            return Prediction(
                label=None, score=None, source="invalid_mol", rule=self.rule,
                threshold=None, smiles_input=smiles, canonical_smiles=canon, inchi_key=ikey,
                db_hit=db_hit_no_label, db_vivo_label=None, db_vitro_label=None,
                db_final_source=None, model_p_vivo=None, model_p_vitro=None,
                explanation="canonical SMILES 재파싱 실패",
            )
        p_vivo = self.vivo_model.predict(mol)
        p_vitro = self.vitro_model.predict(mol)
        label, score = self._combine(p_vivo, p_vitro)
        src = "model" if not db_hit_no_label else "db_abstain→model"
        return Prediction(
            label=label, score=float(score), source=src, rule=self.rule,
            threshold=self.vivo_model.meta["threshold"],
            smiles_input=smiles, canonical_smiles=canon, inchi_key=ikey,
            db_hit=db_hit_no_label,
            db_vivo_label=(r.vivo_label if r is not None else None),
            db_vitro_label=(r.vitro_label if r is not None else None),
            db_final_source=(r.final_source if r is not None else None),
            model_p_vivo=float(p_vivo), model_p_vitro=float(p_vitro),
            explanation=f"model {self.rule}: p_vivo={p_vivo:.3f}, p_vitro={p_vitro:.3f}",
        )

    def predict_batch(self, smiles_list: list[str]) -> list[Prediction]:
        return [self.predict(s) for s in smiles_list]


def _demo():
    pipe = HepatotoxPipeline(rule="weighted")
    samples = [
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("Troglitazone", "Cc1c(C)c2c(c(C)c1O)CCC(C)(COc1ccc(CC3SC(=O)NC3=O)cc1)O2"),
        ("새 분자", "CCN(CC)CCNC(=O)c1cc(Cl)c(N)c(Cl)c1"),
    ]
    for name, smi in samples:
        r = pipe.predict(smi)
        s = f"{r.score:.3f}" if r.score is not None else "None"
        print(f"{name:18s} label={r.label}  score={s}  src={r.source}")
        print(f"  └─ {r.explanation}")


if __name__ == "__main__":
    _demo()
