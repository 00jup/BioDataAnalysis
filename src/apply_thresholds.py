"""bAcc-optimal threshold 적용 — vivo, vitro 둘 다.

사용자 의도: TPR/TNR 균형 (양성은 양성, 음성은 음성)
threshold_sweep 결과:
  - vivo: 0.315 → 0.235 (MCC 동일, TPR 0.360 → 0.513)
  - vitro: 0.520 → 0.490 (MCC 0.423 → 0.443, TPR 0.568 → 0.649)
"""

from __future__ import annotations

import json
import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# threshold_sweep 결과로부터
NEW_THR = {"vivo": 0.235, "vitro": 0.490}


def main():
    for domain, thr in NEW_THR.items():
        meta_path = os.path.join(MODELS_DIR, domain, "ensemble_meta.json")
        backup = meta_path + ".v1_thr"
        meta = json.load(open(meta_path))
        old_thr = meta["threshold"]

        if not os.path.exists(backup):
            shutil.copy(meta_path, backup)
            print(f"[{domain}] v1 threshold 백업 → {backup}")

        meta["threshold"] = float(thr)
        meta["threshold_method"] = "bacc_max_on_val"
        meta["v1_threshold"] = float(old_thr)
        json.dump(meta, open(meta_path, "w"), indent=2)
        print(f"[{domain}] threshold {old_thr:.3f} → {thr:.3f}")
    print("\n각 모델의 ensemble_meta.json 업데이트 완료. predict_pipeline 자동 반영.")


if __name__ == "__main__":
    main()
