"""엄격(gold-standard) vivo 라벨 빌더 — 데모용 재학습 데이터.

기존 OR 룰 라벨은 9개 DB 중 한 곳이라도 간 신호가 있으면 1로 찍어 흔한 약
(Aspirin, Atorvastatin 등)까지 양성이 된다. 데모에서 임상 상식과 어긋난다.

여기서는 FDA/NIH 최고신뢰 출처만으로 라벨을 재정의한다:
  양성(1): DILIrank vMost-DILI-Concern  OR  DailyMed boxed_hepatotox
  음성(0): DILIrank vNo-DILI-Concern     OR  LiverTox E (간독성 근거 없음)
  충돌 → 양성 우선,  그 외(vLess/Ambiguous/unknown 등) → 학습 제외

출력: data/strict/vivo/all.csv  (canonical_smiles, label)
"""

from __future__ import annotations

import os

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FULL = os.path.join(PROJECT_ROOT, "data", "labels_db", "full.parquet")
OUT_DIR = os.path.join(PROJECT_ROOT, "data", "strict", "vivo")


def build() -> pd.DataFrame:
    df = pd.read_parquet(FULL)
    dr = df["vivo_dilirank"].astype(str)
    dm = df["vivo_dailymed"].astype(str)
    lt = df["vivo_livertox"].astype(str)

    is_pos = dr.eq("vMost-DILI-Concern") | dm.eq("boxed_hepatotox")
    is_neg = dr.eq("vNo-DILI-Concern") | lt.eq("E")

    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label[is_neg] = 0
    label[is_pos] = 1  # 양성 우선

    out = df.loc[label.notna(), ["canonical_smiles"]].copy()
    out["label"] = label[label.notna()].astype(int).to_numpy()
    out = out.dropna(subset=["canonical_smiles"]).drop_duplicates("canonical_smiles")
    return out


def main():
    out = build()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "all.csv")
    out.to_csv(path, index=False)
    npos = int((out.label == 1).sum())
    nneg = int((out.label == 0).sum())
    print(f"strict vivo: n={len(out)}  pos={npos} ({npos / len(out):.1%})  neg={nneg}")
    print(f"저장: {path}")


if __name__ == "__main__":
    main()
