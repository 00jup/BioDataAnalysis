"""hepatotoxic flag로 all/non_hepatotoxic/hepatotoxic 분리."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def split_by_hepatotoxic(df: pd.DataFrame, output_root: Path) -> dict[str, Path]:
    """3개 폴더로 분리 저장:
    - all/marketed_all.csv: 전체 (플래그 포함)
    - non_hepatotoxic/marketed_clean.csv: hepatotoxic=0 (학습 negative pool)
    - hepatotoxic/marketed_hepatotoxic.csv: hepatotoxic=1 (양성 샘플 보강용)
    """
    all_dir = output_root / "all"
    clean_dir = output_root / "non_hepatotoxic"
    hepa_dir = output_root / "hepatotoxic"
    all_dir.mkdir(parents=True, exist_ok=True)
    clean_dir.mkdir(parents=True, exist_ok=True)
    hepa_dir.mkdir(parents=True, exist_ok=True)

    all_path = all_dir / "marketed_all.csv"
    clean_path = clean_dir / "marketed_clean.csv"
    hepa_path = hepa_dir / "marketed_hepatotoxic.csv"

    df.to_csv(all_path, index=False)

    clean = df[df["known_hepatotoxic"] == 0].reset_index(drop=True)
    clean.to_csv(clean_path, index=False)

    hepa = df[df["known_hepatotoxic"] == 1].reset_index(drop=True)
    hepa.to_csv(hepa_path, index=False)

    # hepatotoxic 폴더에 출처별 breakdown CSV도 추가 (분석 편의)
    if not hepa.empty:
        breakdown = pd.DataFrame(
            {
                "category": [
                    "DILIrank vMost-DILI-Concern",
                    "DILIrank vLess-DILI-Concern",
                    "TDC DILI Y=1 매칭",
                    "전체 hepatotoxic (중복 제외)",
                ],
                "count": [
                    int((hepa["dilirank_category"] == "vMost-DILI-Concern").sum()),
                    int((hepa["dilirank_category"] == "vLess-DILI-Concern").sum()),
                    int((hepa["in_tdc_dili_pos"] == 1).sum()),
                    len(hepa),
                ],
            }
        )
        breakdown.to_csv(hepa_dir / "breakdown.csv", index=False)

    logger.info(
        "Split: all=%d, non_hepatotoxic=%d, hepatotoxic=%d",
        len(df),
        len(clean),
        len(hepa),
    )
    return {"all": all_path, "clean": clean_path, "hepatotoxic": hepa_path}
