"""Rule-based Override — 통합 DB 에서 SMILES → 라벨 검색.

흐름:
  1. 입력 SMILES → standardize → InChIKey
  2. DB 에서 InChIKey 매치
  3. 매치 found → 그 라벨 반환 (모델 우회)
  4. 매치 없음 → None (호출자가 모델 fallback)

추가 기능:
  - 유사도 매치 (Tanimoto): 정확 매치 안 되도 ≥0.95 유사 분자 있으면 그 라벨 후보
  - 신뢰도 등급 (vivo confidence + final_source 활용)
  - 도메인별 라벨 선택 (vivo / vitro / final)

사용:
    from src.lookup import LabelDB
    db = LabelDB("data/labels_db/full.parquet")
    res = db.lookup("CC(=O)Nc1ccc(O)cc1")
    # → {"hit": True, "vivo_label": 1, "vitro_label": None,
    #    "final_label": 1, "final_source": "vivo_only",
    #    "vivo_confidence": "high", "n_sources_total": 4, ...}
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator

from src.standardize import standardize

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")


@dataclass
class LookupResult:
    hit: bool                       # DB 매치 여부
    match_type: str                 # "exact" | "similar" | "none"
    inchi_key: str | None
    canonical_smiles: str | None
    similarity: float | None        # similar 매치 시 Tanimoto
    final_source: str | None        # "vivo_only"|"vitro_only"|"both_agree"|"conflict"
    # 3 가지 룰 라벨 (None = 모델 fallback 필요)
    label_vivo_priority: int | None
    label_weighted: int | None
    label_consensus: int | None
    # 원시 도메인별 라벨
    vivo_label: int | None
    vivo_confidence: str | None
    vivo_n_sources: int | None
    vitro_label: int | None
    vitro_confidence: str | None
    vitro_n_sources: int | None
    raw: dict                       # 전체 row dict (디버깅용)

    def get_label(self, rule: str = "vivo_priority") -> int | None:
        """rule = 'vivo_priority' | 'weighted' | 'consensus'."""
        if rule == "vivo_priority": return self.label_vivo_priority
        if rule == "weighted":      return self.label_weighted
        if rule == "consensus":     return self.label_consensus
        raise ValueError(f"unknown rule: {rule}")


class LabelDB:
    """통합 라벨 DB 메모리 로드 + 검색."""

    def __init__(self, path: str = DB_PATH, load_fps: bool = True):
        self.df = pd.read_parquet(path)
        # 인덱스 만들기
        self.by_ikey = self.df.set_index("inchi_key")
        print(f"LabelDB 로드: {len(self.df)} 분자")
        self._fp_gen = GetMorganGenerator(radius=2, fpSize=2048)
        self._fps = None  # 지연 로딩
        if load_fps:
            self._build_fp_cache()

    def _build_fp_cache(self):
        """모든 DB 분자 Morgan FP 사전 계산 (유사도 검색용)."""
        print("  Morgan FP 캐시 빌드 중 ...")
        fps = []
        valid_idx = []
        for i, smi in enumerate(self.df["canonical_smiles"].tolist()):
            mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            if mol is None:
                fps.append(None)
            else:
                fps.append(self._fp_gen.GetFingerprint(mol))
                valid_idx.append(i)
        self._fps = fps
        self._valid_fp_idx = valid_idx
        print(f"  FP 캐시 완료 ({len(valid_idx)}/{len(self.df)})")

    def _row_to_result(self, row: pd.Series, match_type: str, similarity: float | None = None) -> LookupResult:
        def safe_int(v):
            return int(v) if pd.notna(v) else None
        def safe_str(v):
            return str(v) if pd.notna(v) else None
        return LookupResult(
            hit=True,
            match_type=match_type,
            inchi_key=safe_str(row.get("inchi_key")),
            canonical_smiles=safe_str(row.get("canonical_smiles")),
            similarity=similarity,
            final_source=safe_str(row.get("final_source")),
            label_vivo_priority=safe_int(row.get("label_vivo_priority")),
            label_weighted=safe_int(row.get("label_weighted")),
            label_consensus=safe_int(row.get("label_consensus")),
            vivo_label=safe_int(row.get("vivo_label")),
            vivo_confidence=safe_str(row.get("vivo_confidence")),
            vivo_n_sources=safe_int(row.get("vivo_n_sources")),
            vitro_label=safe_int(row.get("vitro_label")),
            vitro_confidence=safe_str(row.get("vitro_confidence")),
            vitro_n_sources=safe_int(row.get("vitro_n_sources")),
            raw=row.to_dict(),
        )

    def lookup_exact(self, smiles: str) -> LookupResult | None:
        """정확 매치 (InChIKey 동일)."""
        std = standardize(smiles)
        if std is None:
            return None
        _, ikey = std
        if ikey not in self.by_ikey.index:
            return None
        row = self.by_ikey.loc[ikey]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        # raw 에 inchi_key 가 index 라서 컬럼엔 없음 — 수동 추가
        row = row.copy()
        row["inchi_key"] = ikey
        return self._row_to_result(row, "exact")

    def lookup_similar(self, smiles: str, threshold: float = 0.95) -> LookupResult | None:
        """Tanimoto 유사도 매치. 가장 유사한 분자 1개 반환."""
        if self._fps is None:
            raise RuntimeError("FP 캐시 미빌드 — LabelDB(load_fps=True) 사용")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        q_fp = self._fp_gen.GetFingerprint(mol)

        best_sim = 0.0
        best_idx = -1
        for i in self._valid_fp_idx:
            sim = DataStructs.TanimotoSimilarity(q_fp, self._fps[i])
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        if best_sim < threshold or best_idx < 0:
            return None
        row = self.df.iloc[best_idx].copy()
        return self._row_to_result(row, "similar", similarity=float(best_sim))

    def lookup(self, smiles: str, similarity_threshold: float = 0.95) -> LookupResult:
        """exact → similar fallback. 둘 다 실패 시 hit=False 결과."""
        r = self.lookup_exact(smiles)
        if r is not None:
            return r
        if self._fps is not None:
            r = self.lookup_similar(smiles, similarity_threshold)
            if r is not None:
                return r
        std = standardize(smiles)
        ikey = std[1] if std else None
        canon = std[0] if std else None
        return LookupResult(
            hit=False, match_type="none", inchi_key=ikey, canonical_smiles=canon,
            similarity=None, final_source=None,
            label_vivo_priority=None, label_weighted=None, label_consensus=None,
            vivo_label=None, vivo_confidence=None, vivo_n_sources=None,
            vitro_label=None, vitro_confidence=None, vitro_n_sources=None,
            raw={},
        )


def _demo():
    """간단 동작 테스트."""
    db = LabelDB(load_fps=False)
    samples = [
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1"),
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C"),
        ("Troglitazone", "CC1=C(C(=C(C(=C1OC2(CCC(=O)N2)C)CC)C)C)O.OC(=O)[C@@H](N)CCC(=O)NCC(=O)O"),
        ("Unknown novel", "CCN(CC)CCNC(=O)c1cc(Cl)c(N)c(Cl)c1"),
    ]
    for name, smi in samples:
        r = db.lookup_exact(smi)
        if r is not None and r.hit:
            print(f"{name:25s} HIT  source={r.final_source}  vivo={r.vivo_label}({r.vivo_confidence})  vitro={r.vitro_label}({r.vitro_confidence})")
            print(f"  └─ rule labels: vivo_priority={r.label_vivo_priority}  weighted={r.label_weighted}  consensus={r.label_consensus}")
        else:
            std = standardize(smi)
            ikey = std[1] if std else "?"
            print(f"{name:25s} miss (ikey={ikey})")


if __name__ == "__main__":
    _demo()
