"""exp_clean 3변형 데이터셋 생성.

기존 exp1~exp4가 chEMBL 외부 데이터(jeje 노트북 경로)에 의존하던 것과 달리,
이 빌더는 **저장소 안 데이터**만으로 동작한다. 외부 test 는
`data/experiments/external_test/test.csv` 를 그대로 사용해 누수만 차단한다.

세 변형 (모두 동일 음성 풀 공유):
  exp_clean_strict   : repo·DILIst·GoldStandard 중 2곳+ 합의 ∪ DILIrank vMost
                       (가장 깨끗, 양성 ~424)
  exp_clean_full     : 위 3원 union (현행 exp2 와 같은 크기 ~1473)
                       — 출처 일치도(agreement) 컬럼을 같이 저장해 학습 시
                         sample_weight 로 활용 가능
  exp_clean_nosider  : union 에서 SIDER **단독** 출처(인과관계 미검증) 제거

음성: marketed_clean − Ambiguous-DILI-Concern(292) 에서 5,000 stratified 샘플.

외부 test 와 InChIKey 누수 차단, train/val 0.85/0.15 stratified.
"""

from __future__ import annotations

import os

import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.model_selection import train_test_split

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MD = os.path.join(PROJECT_ROOT, "data", "marketed_drugs")
EXP_DIR = os.path.join(PROJECT_ROOT, "data", "experiments")
EXT_TEST = os.path.join(EXP_DIR, "external_test", "test.csv")

RANDOM_STATE = 42
VAL_RATIO = 0.15
NEG_SAMPLE = 5000


def canonicalize(smiles: str) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def inchikey_from_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    return Chem.MolToInchiKey(mol) if mol is not None else ""


def load_positives() -> pd.DataFrame:
    """3원 출처(repo, dilist, gold) + DILIrank vMost 표시를 분자별로 모은 DF."""
    lenient = pd.read_csv(os.path.join(MD, "hepatotoxic", "hepatotoxic_all_lenient.csv"))
    ext = pd.read_csv(os.path.join(MD, "hepatotoxic", "external", "external_positives.csv"))

    lenient = lenient.dropna(subset=["inchi_key", "canonical_smiles"]).drop_duplicates("inchi_key")
    ext = ext.dropna(subset=["inchi_key", "canonical_smiles"]).drop_duplicates("inchi_key")

    repo_ik = set(lenient["inchi_key"])
    dilist_ik = set(ext.loc[ext["source_label"] == "dilist", "inchi_key"])
    gold_ik = set(ext.loc[ext["source_label"] == "goldstandard", "inchi_key"])
    vmost_ik = set(lenient.loc[lenient["dilirank_category"] == "vMost-DILI-Concern", "inchi_key"])

    # SIDER 단독 출처 분자: lenient.sources == "sider" (다른 출처와 결합 안 됨)
    sider_only_ik = set(lenient.loc[lenient["sources"].fillna("") == "sider", "inchi_key"])

    # union 마스터 표
    smiles_map: dict[str, str] = {}
    for _, r in lenient[["inchi_key", "canonical_smiles"]].iterrows():
        smiles_map.setdefault(r["inchi_key"], r["canonical_smiles"])
    for _, r in ext[["inchi_key", "canonical_smiles"]].iterrows():
        smiles_map.setdefault(r["inchi_key"], r["canonical_smiles"])

    rows = []
    for ik in repo_ik | dilist_ik | gold_ik:
        in_repo = int(ik in repo_ik)
        in_dilist = int(ik in dilist_ik)
        in_gold = int(ik in gold_ik)
        rows.append(
            {
                "inchi_key": ik,
                "canonical_smiles": smiles_map.get(ik, ""),
                "in_repo": in_repo,
                "in_dilist": in_dilist,
                "in_gold": in_gold,
                "in_vmost": int(ik in vmost_ik),
                "sider_only": int(ik in sider_only_ik),
                "agreement": in_repo + in_dilist + in_gold,
            }
        )
    pos = pd.DataFrame(rows)
    pos = pos[pos["canonical_smiles"].astype(bool)].reset_index(drop=True)
    return pos


def load_negatives() -> pd.DataFrame:
    neg = pd.read_csv(os.path.join(MD, "non_hepatotoxic", "marketed_clean.csv"))
    neg = neg.dropna(subset=["inchi_key", "canonical_smiles"]).drop_duplicates("inchi_key")
    before = len(neg)
    neg = neg[neg["dilirank_category"] != "Ambiguous-DILI-Concern"].copy()
    print(f"음성: marketed_clean {before} → Ambiguous 제거 {len(neg)}")
    neg["source"] = "marketed_clean"
    return neg[["inchi_key", "canonical_smiles", "source"]]


def save_variant(name: str, pos: pd.DataFrame, neg: pd.DataFrame) -> dict:
    """양성/음성 합쳐 stratified 분할 + 저장."""
    p = pos.assign(label=1, source="positive")
    n = neg.assign(label=0)
    cols = ["canonical_smiles", "inchi_key", "label", "source"]
    if "agreement" in pos.columns:
        # full 변형은 weight 컬럼 추가 (agreement 1=0.5, 2=1.0, 3=1.5; vMost +0.5)
        p["weight"] = p["agreement"].map({1: 0.5, 2: 1.0, 3: 1.5}).fillna(1.0)
        p.loc[p["in_vmost"] == 1, "weight"] += 0.5
        cols = cols + ["weight"]
        n["weight"] = 1.0

    combined = pd.concat([p[cols], n[cols]], ignore_index=True).drop_duplicates("inchi_key")
    combined = combined.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    train, val = train_test_split(
        combined,
        test_size=VAL_RATIO,
        random_state=RANDOM_STATE,
        stratify=combined["label"],
    )

    vdir = os.path.join(EXP_DIR, name)
    os.makedirs(vdir, exist_ok=True)
    train.to_csv(os.path.join(vdir, "train.csv"), index=False)
    val.to_csv(os.path.join(vdir, "val.csv"), index=False)
    combined.to_csv(os.path.join(vdir, "manifest.csv"), index=False)

    summary = {
        "variant": name,
        "total": len(combined),
        "train": len(train),
        "val": len(val),
        "positives": int((combined["label"] == 1).sum()),
        "negatives": int((combined["label"] == 0).sum()),
        "neg_ratio": round(int((combined["label"] == 0).sum()) / max(1, int((combined["label"] == 1).sum())), 2),
    }
    print(f"  {name}: 양성 {summary['positives']} / 음성 {summary['negatives']} "
          f"(train {summary['train']}, val {summary['val']})")
    return summary


def main() -> None:
    pos_all = load_positives()
    neg_all = load_negatives()

    # 외부 test 누수 차단
    test_ik = set(pd.read_csv(EXT_TEST)["inchi_key"].dropna())
    pos_all = pos_all[~pos_all["inchi_key"].isin(test_ik)].reset_index(drop=True)
    neg_all = neg_all[~neg_all["inchi_key"].isin(test_ik)].reset_index(drop=True)
    print(f"외부 test 제외 후 — 양성 풀 {len(pos_all)}, 음성 풀 {len(neg_all)}")

    # 양성-음성 InChIKey 충돌 시 음성에서 제거
    conflict = set(pos_all["inchi_key"]) & set(neg_all["inchi_key"])
    if conflict:
        neg_all = neg_all[~neg_all["inchi_key"].isin(conflict)].reset_index(drop=True)
        print(f"양·음 충돌 {len(conflict)} 음성에서 제거")

    # 음성 5000 샘플 (변형 공유)
    neg_sample = neg_all.sample(n=min(NEG_SAMPLE, len(neg_all)), random_state=RANDOM_STATE)

    # 변형 정의
    strict = pos_all[(pos_all["agreement"] >= 2) | (pos_all["in_vmost"] == 1)]
    full = pos_all  # 그대로
    nosider = pos_all[~((pos_all["agreement"] == 1) & (pos_all["in_repo"] == 1) & (pos_all["sider_only"] == 1))]

    print("\n변형 생성:")
    summaries = []
    summaries.append(save_variant("exp_clean_strict", strict.drop(columns=["agreement"]), neg_sample))
    summaries.append(save_variant("exp_clean_full", full, neg_sample))
    summaries.append(save_variant("exp_clean_nosider", nosider.drop(columns=["agreement"]), neg_sample))

    print("\n요약")
    for s in summaries:
        print(f"  {s['variant']:22s} 양성 {s['positives']:>5d}  음성 {s['negatives']:>5d}  비율 1:{s['neg_ratio']}")


if __name__ == "__main__":
    main()
