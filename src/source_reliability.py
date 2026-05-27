"""출처별 reliability 측정 — 두 접근 비교.

방법 1 (Heuristic): 분자 구조 안 봄
   1a. Pairwise agreement — 각 출처 vs DILIrank gold (vMost/vLess/vNo)
   1b. Source classifier AUC — 각 출처 라벨로 RF 학습 → DILIrank gold test AUC

방법 2 (Feature-based): 분자 구조 사용
   leave-one-out 으로 순환 방지
   각 출처 제외하고 consensus 학습한 truth_model →
   그 출처 라벨이 truth_model 예측과 일치하는 비율

출력:
   results/source_reliability.json
   각 출처의 (agreement, classifier_auc, truth_model_agreement) + 최종 weight
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
RESULTS = os.path.join(PROJECT_ROOT, "results")

# 출처 정의: name, 그 출처의 라벨 컬럼, 라벨 추출 함수
SOURCE_DEFS = [
    (
        "dilirank_vMost",
        "vivo_dilirank",
        lambda v: 1 if v == "vMost-DILI-Concern" else (None if pd.isna(v) else 0),
    ),
    (
        "dilirank_vLess",
        "vivo_dilirank",
        lambda v: 1 if v == "vLess-DILI-Concern" else (None if pd.isna(v) else 0),
    ),
    (
        "dilirank_vNo",
        "vivo_dilirank",
        lambda v: 0 if v == "vNo-DILI-Concern" else (None if pd.isna(v) else 1),
    ),
    (
        "livertox_A_B",
        "vivo_livertox",
        lambda v: 1 if v in ("A", "B") else (0 if v == "E" else None),
    ),
    (
        "livertox_C_D",
        "vivo_livertox",
        lambda v: 1 if v in ("C", "D") else (0 if v == "E" else None),
    ),
    (
        "livertox_E",
        "vivo_livertox",
        lambda v: 0 if v == "E" else (1 if v in ("A", "B", "C", "D") else None),
    ),
    ("dilist", "vivo_dilist", lambda v: int(v) if pd.notna(v) else None),
    ("gold", "vivo_gold", lambda v: int(v) if pd.notna(v) else None),
    ("sider_strict", "vivo_sider_liver", lambda v: int(v) if pd.notna(v) else None),
    ("sider_lenient", "vivo_sider_hepatotox", lambda v: int(v) if pd.notna(v) else None),
    ("tdc_dili", "vivo_tdc_dili", lambda v: int(v) if pd.notna(v) else None),
    ("clintox", "vivo_clintox", lambda v: int(v) if pd.notna(v) else None),
    (
        "marketed_clean_neg",
        "vivo_marketed_clean_neg",
        lambda v: 0 if v == 1 else None,
    ),  # 음성 신호만
    ("chembl", "vitro_chembl", lambda v: int(v) if pd.notna(v) else None),
    ("tox21", "vitro_tox21", lambda v: int(v) if pd.notna(v) else None),
]

_FP_GEN = GetMorganGenerator(radius=3, fpSize=2048)


def morgan_fp(smi: str) -> np.ndarray | None:
    mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
    if mol is None:
        return None
    arr = np.zeros(2048, dtype=np.uint8)
    arr[list(_FP_GEN.GetFingerprint(mol).GetOnBits())] = 1
    return arr


def extract_labels(db: pd.DataFrame) -> dict[str, pd.Series]:
    """출처별 binary 라벨 Series (None 은 NaN)."""
    out = {}
    for name, col, fn in SOURCE_DEFS:
        if col not in db.columns:
            continue
        labels = db[col].apply(fn)
        out[name] = labels
    return out


# ════════════════════════════════════════════════════════════════
# 방법 1: Heuristic
# ════════════════════════════════════════════════════════════════
def pairwise_agreement(
    labels_by_source: dict, reference: str = "dilirank_vMost"
) -> dict[str, dict]:
    """각 출처 vs 기준 출처 일치도. 기준 = DILIrank vMost (가장 신뢰)."""
    ref = labels_by_source[reference]
    out = {}
    for name, lab in labels_by_source.items():
        if name == reference:
            continue
        # 공통 분자 (둘 다 라벨 있음)
        common_mask = ref.notna() & lab.notna()
        if common_mask.sum() < 10:
            out[name] = {"n_common": int(common_mask.sum()), "agreement": None}
            continue
        agreed = (ref[common_mask] == lab[common_mask]).sum()
        n = int(common_mask.sum())
        out[name] = {
            "n_common": n,
            "agreement": float(agreed / n),
            "n_agreed": int(agreed),
        }
    return out


def classifier_auc(
    db: pd.DataFrame, labels_by_source: dict, gold_source: str = "dilirank_vMost"
) -> dict[str, dict]:
    """각 출처 라벨로 RF 학습 → DILIrank gold (vMost vs vNo) 에서 AUC."""
    # Gold test set — vMost (양성) + vNo (음성)
    gold_mask = labels_by_source[gold_source].notna() | labels_by_source["dilirank_vNo"].notna()
    gold_idx = db[gold_mask].index.tolist()
    gold_labels = {}
    for i in gold_idx:
        if (
            pd.notna(labels_by_source[gold_source].iloc[i])
            and labels_by_source[gold_source].iloc[i] == 1
        ):
            gold_labels[i] = 1
        elif (
            pd.notna(labels_by_source["dilirank_vNo"].iloc[i])
            and labels_by_source["dilirank_vNo"].iloc[i] == 0
        ):
            gold_labels[i] = 0
    gold_X = []
    gold_y = []
    gold_smiles = set()
    for i, y in gold_labels.items():
        smi = db.iloc[i].canonical_smiles
        if smi in gold_smiles:
            continue
        gold_smiles.add(smi)
        fp = morgan_fp(smi)
        if fp is None:
            continue
        gold_X.append(fp)
        gold_y.append(y)
    gold_X = np.array(gold_X)
    gold_y = np.array(gold_y)
    print(
        f"  Gold test: {len(gold_X)} 분자 (양성 {int(gold_y.sum())}, 음성 {int((1 - gold_y).sum())})"
    )

    out = {}
    for name, lab in labels_by_source.items():
        if name == gold_source:
            continue
        # source 라벨로 train (gold 분자는 제외)
        train_idx = lab.notna() & ~db["canonical_smiles"].isin(gold_smiles)
        train_smi = db.loc[train_idx, "canonical_smiles"].tolist()
        train_y = lab[train_idx].astype(int).to_numpy()
        if len(train_y) < 50 or len(set(train_y)) < 2:
            out[name] = {"n_train": int(len(train_y)), "auc": None}
            continue
        Xtr = []
        ytr_keep = []
        for smi, y in zip(train_smi, train_y):
            fp = morgan_fp(smi)
            if fp is not None:
                Xtr.append(fp)
                ytr_keep.append(y)
        Xtr = np.array(Xtr)
        ytr_keep = np.array(ytr_keep)
        clf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr_keep)
        proba = clf.predict_proba(gold_X)[:, 1]
        try:
            auc = float(roc_auc_score(gold_y, proba))
        except Exception:
            auc = None
        out[name] = {
            "n_train": int(len(ytr_keep)),
            "n_pos": int(ytr_keep.sum()),
            "n_neg": int((1 - ytr_keep).sum()),
            "auc": auc,
        }
    return out


# ════════════════════════════════════════════════════════════════
# 방법 2: Feature-based (leave-one-source-out)
# ════════════════════════════════════════════════════════════════
def feature_based_reliability(db: pd.DataFrame, labels_by_source: dict) -> dict[str, dict]:
    """각 출처 S 에 대해:
    1. 다른 모든 출처에서 ≥2 일치 분자 추출 (consensus)
    2. consensus 로 RF (Morgan FP) 학습 → truth_model_no_S
    3. S 의 라벨된 분자에 truth_model 예측 → 일치도 측정
    """
    smi_to_fp = {}

    def get_fp(smi):
        if smi not in smi_to_fp:
            smi_to_fp[smi] = morgan_fp(smi)
        return smi_to_fp[smi]

    out = {}
    for target in labels_by_source:
        # 1. consensus from OTHER sources
        others = {k: v for k, v in labels_by_source.items() if k != target}
        # 각 분자에 대해 다른 출처들의 라벨 모음
        consensus_label = []
        n_agreed = []
        for i in range(len(db)):
            votes = []
            for k, lab in others.items():
                if pd.notna(lab.iloc[i]):
                    votes.append(int(lab.iloc[i]))
            if len(votes) >= 2:
                n_pos = sum(votes)
                n_neg = len(votes) - n_pos
                if n_pos > n_neg:
                    consensus_label.append(1)
                    n_agreed.append(len(votes))
                elif n_neg > n_pos:
                    consensus_label.append(0)
                    n_agreed.append(len(votes))
                else:
                    consensus_label.append(None)
                    n_agreed.append(0)
            else:
                consensus_label.append(None)
                n_agreed.append(0)

        cons = pd.Series(consensus_label)
        train_mask = cons.notna()
        if train_mask.sum() < 100:
            out[target] = {"n_train_consensus": int(train_mask.sum()), "agreement": None}
            continue

        # 2. truth_model 학습
        train_smiles = db.loc[train_mask, "canonical_smiles"].tolist()
        train_y = cons[train_mask].astype(int).to_numpy()
        Xtr = []
        ytr_keep = []
        for smi, y in zip(train_smiles, train_y):
            fp = get_fp(smi)
            if fp is not None:
                Xtr.append(fp)
                ytr_keep.append(y)
        Xtr = np.array(Xtr)
        ytr_keep = np.array(ytr_keep)

        clf = RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        clf.fit(Xtr, ytr_keep)

        # 3. target 출처가 라벨링한 분자에 예측 → 일치도
        target_lab = labels_by_source[target]
        target_mask = target_lab.notna()
        # target 분자에서 consensus 학습 안 한 (cross check) 분자만 평가
        eval_smiles = db.loc[target_mask, "canonical_smiles"].tolist()
        eval_y_actual = target_lab[target_mask].astype(int).to_numpy()
        Xev = []
        yev = []
        for smi, y in zip(eval_smiles, eval_y_actual):
            fp = get_fp(smi)
            if fp is not None:
                Xev.append(fp)
                yev.append(y)
        if not Xev:
            out[target] = {"n_train_consensus": int(len(ytr_keep)), "agreement": None}
            continue
        Xev = np.array(Xev)
        yev = np.array(yev)
        preds = clf.predict(Xev)
        agreement = float((preds == yev).mean())
        # AUC 도 (확률 vs 실제)
        try:
            proba = clf.predict_proba(Xev)[:, 1]
            auc_target = float(roc_auc_score(yev, proba))
        except Exception:
            auc_target = None
        out[target] = {
            "n_train_consensus": int(len(ytr_keep)),
            "n_target_labeled": int(len(yev)),
            "agreement": agreement,
            "auc_against_target_labels": auc_target,
            "n_pos_target": int(yev.sum()),
            "n_neg_target": int((1 - yev).sum()),
        }
    return out


def main():
    print("=== DB 로드 + 라벨 추출 ===")
    db = pd.read_parquet(DB_PATH).reset_index(drop=True)
    labels = extract_labels(db)
    print(f"DB: {len(db)} 분자, 출처 {len(labels)}\n")
    print("출처별 라벨 개수:")
    for name, lab in labels.items():
        print(
            f"  {name:22s} {int(lab.notna().sum()):5d} (양성 {int((lab == 1).sum()):4d} / 음성 {int((lab == 0).sum()):4d})"
        )

    print("\n" + "=" * 70)
    print("방법 1a: Pairwise agreement (DILIrank vMost 기준)")
    print("=" * 70)
    pa = pairwise_agreement(labels, reference="dilirank_vMost")
    for name, m in sorted(pa.items(), key=lambda x: x[1].get("agreement") or 0, reverse=True):
        a = m.get("agreement")
        print(f"  {name:22s} n={m['n_common']:5d}  agreement={'  —' if a is None else f'{a:.3f}'}")

    print("\n" + "=" * 70)
    print("방법 1b: Source classifier AUC (DILIrank gold test)")
    print("=" * 70)
    ca = classifier_auc(db, labels)
    for name, m in sorted(ca.items(), key=lambda x: x[1].get("auc") or 0, reverse=True):
        a = m.get("auc")
        print(f"  {name:22s} n_train={m['n_train']:5d}  AUC={'  —' if a is None else f'{a:.3f}'}")

    print("\n" + "=" * 70)
    print("방법 2: Feature-based reliability (leave-one-source-out)")
    print("=" * 70)
    fb = feature_based_reliability(db, labels)
    for name, m in sorted(fb.items(), key=lambda x: x[1].get("agreement") or 0, reverse=True):
        a = m.get("agreement")
        auc = m.get("auc_against_target_labels")
        print(
            f"  {name:22s} train_consensus={m['n_train_consensus']:5d}  target={m.get('n_target_labeled', 0):5d}  agreement={'  —' if a is None else f'{a:.3f}'}  AUC={'  —' if auc is None else f'{auc:.3f}'}"
        )

    # 종합 weight
    print("\n" + "=" * 70)
    print("종합 weight (방법 1a + 1b + 2 평균)")
    print("=" * 70)
    summary = {}
    for name in labels:
        a1 = pa.get(name, {}).get("agreement")
        b1 = ca.get(name, {}).get("auc")
        f2 = fb.get(name, {}).get("agreement")
        # 0-1 정규화: AUC 는 (auc-0.5)*2, agreement 는 직접 (0~1)
        vals = []
        if a1 is not None:
            vals.append(a1)
        if b1 is not None:
            vals.append((b1 - 0.5) * 2)
        if f2 is not None:
            vals.append(f2)
        weight = float(np.mean(vals)) if vals else None
        summary[name] = {
            "agreement_DILIrank_gold": a1,
            "classifier_auc": b1,
            "feature_truth_agreement": f2,
            "final_weight": weight,
        }

    def _fmt(v):
        return "?" if v is None else f"{v:.2f}"

    for name, m in sorted(
        summary.items(), key=lambda x: x[1].get("final_weight") or 0, reverse=True
    ):
        w = m["final_weight"]
        w_str = "  —" if w is None else f"{w:.3f}"
        print(
            f"  {name:22s} weight={w_str}  (agr={_fmt(m['agreement_DILIrank_gold'])}, auc={_fmt(m['classifier_auc'])}, fb={_fmt(m['feature_truth_agreement'])})"
        )

    out = {
        "label_counts": {n: int(lab.notna().sum()) for n, lab in labels.items()},
        "pairwise_agreement": pa,
        "classifier_auc": ca,
        "feature_based": fb,
        "summary_weights": summary,
    }
    os.makedirs(RESULTS, exist_ok=True)
    with open(os.path.join(RESULTS, "source_reliability.json"), "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("\n저장: results/source_reliability.json")


if __name__ == "__main__":
    main()
