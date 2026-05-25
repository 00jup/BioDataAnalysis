"""v8 — v1 과 완전 동일 설정 재학습. 모델 출력만 models/v8/ 로 분리.

v1 과 같음:
  - 같은 데이터 (data/train|val|test/{vivo,vitro}.csv)
  - 같은 도메인 분리 (vivo, vitro)
  - 같은 모델 (5 FP × RF/CB)
  - 같은 sample weight (data-driven, 출처 강도)
  - 같은 random_state=42

목적: v1 결과 재현성 확인. 다른 seed 없이 그대로 → 비트 단위 동일 결과 기대.
"""

from __future__ import annotations

import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
V8_DIR = os.path.join(PROJECT_ROOT, "models", "v8")
DEFAULT_DIR = os.path.join(PROJECT_ROOT, "models")


def main():
    from src.train_domain_models import train_domain

    os.makedirs(V8_DIR, exist_ok=True)
    summary = {}

    for domain in ("vivo", "vitro"):
        # train_domain 은 models/{domain}/ 에 저장하므로
        # v1 결과를 보존하려면 임시로 백업 후 학습 → v8 로 이동 → 복원.
        orig_dir = os.path.join(DEFAULT_DIR, domain)
        backup_dir = orig_dir + ".v1_backup"

        if os.path.isdir(orig_dir):
            if os.path.isdir(backup_dir): shutil.rmtree(backup_dir)
            shutil.move(orig_dir, backup_dir)
            print(f"[{domain}] v1 백업 → {backup_dir}")

        try:
            meta = train_domain(domain)
            summary[domain] = meta["test_metrics"]
            # 학습 산출물 → v8 으로 이동
            v8_target = os.path.join(V8_DIR, domain)
            if os.path.isdir(v8_target): shutil.rmtree(v8_target)
            shutil.move(orig_dir, v8_target)
            print(f"[{domain}] v8 저장 → {v8_target}")
        finally:
            # v1 복원
            if os.path.isdir(backup_dir):
                if os.path.isdir(orig_dir): shutil.rmtree(orig_dir)
                shutil.move(backup_dir, orig_dir)
                print(f"[{domain}] v1 복원 ← {orig_dir}")

    # v8 결과 저장 + v1 과 비교
    v1_v = json.load(open(os.path.join(DEFAULT_DIR, "vivo",  "ensemble_meta.json")))
    v1_vi = json.load(open(os.path.join(DEFAULT_DIR, "vitro", "ensemble_meta.json")))
    cmp = {
        "vivo": {
            "v1": {k: v1_v["test_metrics"][k] for k in ("auc", "mcc", "f1", "tpr", "tnr")},
            "v8": {k: summary["vivo"][k]      for k in ("auc", "mcc", "f1", "tpr", "tnr")},
        },
        "vitro": {
            "v1": {k: v1_vi["test_metrics"][k] for k in ("auc", "mcc", "f1", "tpr", "tnr")},
            "v8": {k: summary["vitro"][k]      for k in ("auc", "mcc", "f1", "tpr", "tnr")},
        },
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "results"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results", "v1_vs_v8.json"), "w") as f:
        json.dump(cmp, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  v1 vs v8 비교 (재현성 확인)")
    print(f"{'='*70}")
    for domain in ("vivo", "vitro"):
        v1m, v8m = cmp[domain]["v1"], cmp[domain]["v8"]
        print(f"\n[{domain}]")
        print(f"  {'metric':<6s} {'v1':>7s} {'v8':>7s} {'Δ':>7s}")
        for k in ("auc", "mcc", "f1", "tpr", "tnr"):
            d = v8m[k] - v1m[k]
            print(f"  {k:<6s} {v1m[k]:>7.3f} {v8m[k]:>7.3f} {d:>+7.4f}")
    print(f"\n저장: results/v1_vs_v8.json")


if __name__ == "__main__":
    main()
