"""실제 잘 알려진 약물로 sanity check.

목적: test metric (MCC 0.414) 이 실제로 의미 있는지 검증.

알려진 hepatotoxic 양성 (보고된 임상 DILI):
  - Acetaminophen (paracetamol) — 과량 시 가장 흔한 DILI
  - Isoniazid — 결핵약, 잘 알려진 간독성
  - Valproic acid — 항경련제, 간독성
  - Troglitazone — 시장 철수 (간독성)
  - Diclofenac — NSAID 간독성
  - Halothane — 마취제 간염
  - Ketoconazole — 항진균제 간독성
  - Methotrexate — 면역억제제, 간섬유화
  - Amoxicillin-clavulanate — 혼합 간담즙성
  - Nitrofurantoin — 만성 간염

알려진 안전 음성:
  - Aspirin — 일반적으로 안전 (간독성 드뭄)
  - Ibuprofen — NSAID 중 안전
  - Metformin — 당뇨약, 간 안전
  - Atorvastatin — 일반적으로 안전 (간 모니터링 권장)
  - Loratadine — 항히스타민, 매우 안전
  - Cetirizine — 매우 안전
  - Lisinopril — ACE inhibitor 안전
  - Amlodipine — Ca 차단제 안전
  - Levothyroxine — 갑상선 호르몬, 매우 안전
  - Omeprazole — PPI, 간 영향 적음
"""

from __future__ import annotations
import json, os, sys, subprocess
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PY = sys.executable
CHEMPROP_BIN = os.path.join(os.path.dirname(PY), "chemprop")

# SMILES (canonical, PubChem 출처)
DRUGS = [
    # 양성 (DILI 보고됨)
    ("Acetaminophen",       "CC(=O)Nc1ccc(O)cc1", 1),
    ("Isoniazid",           "NNC(=O)c1ccncc1", 1),
    ("Valproic acid",       "CCCC(CCC)C(=O)O", 1),
    ("Troglitazone",        "Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O", 1),
    ("Diclofenac",          "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", 1),
    ("Halothane",           "FC(F)(F)C(Cl)Br", 1),
    ("Ketoconazole",        "CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1", 1),
    ("Methotrexate",        "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1", 1),
    ("Amoxicillin-clav",    "CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O", 1),
    ("Nitrofurantoin",      "O=C1OCC(N1\\N=C\\c1ccc(o1)[N+](=O)[O-])", 1),
    # 음성 (안전)
    ("Aspirin",             "CC(=O)Oc1ccccc1C(=O)O", 0),
    ("Ibuprofen",           "CC(C)Cc1ccc(C(C)C(=O)O)cc1", 0),
    ("Metformin",           "CN(C)C(=N)NC(=N)N", 0),
    ("Atorvastatin",        "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O", 0),
    ("Loratadine",          "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1", 0),
    ("Cetirizine",          "OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1", 0),
    ("Lisinopril",          "NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O", 0),
    ("Amlodipine",          "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl", 0),
    ("Levothyroxine",       "Oc1cc(I)c(Oc2cc(I)c(CC(N)C(=O)O)cc2I)c(I)c1", 0),
    ("Omeprazole",          "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", 0),
]


def predict_chemprop(model_dir: str, smiles_list: list[str], rdkit_feat=False) -> list[float]:
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".csv", delete=False) as f:
        df = pd.DataFrame({"canonical_smiles": smiles_list})
        df.to_csv(f.name, index=False)
        in_path = f.name
    out_path = in_path.replace(".csv", "_pred.csv")
    cmd = [
        CHEMPROP_BIN, "predict",
        "--test-path", in_path,
        "-s", "canonical_smiles",
        "--model-paths", model_dir,
        "--preds-path", out_path,
        "--accelerator", "cpu",
    ]
    if rdkit_feat:
        cmd += ["--molecule-featurizers", "rdkit_2d_normalized"]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print("chemprop predict 실패:", r.stderr.decode()[:500])
        return [None] * len(smiles_list)
    pred = pd.read_csv(out_path).rename(columns={"label": "pred"})
    merged = df.merge(pred, on="canonical_smiles", how="left")
    return merged["pred"].tolist()


def main():
    print("=== Sanity check — 알려진 약물 20개 ===\n")
    smiles = [d[1] for d in DRUGS]
    names = [d[0] for d in DRUGS]
    truth = np.array([d[2] for d in DRUGS])

    # Chemprop v9 vivo
    cp_v9_dir = os.path.join(PROJECT_ROOT, "models", "chemprop_v9", "vivo")
    print("Chemprop v9 vivo 예측 중...")
    probs_v9 = predict_chemprop(cp_v9_dir, smiles)
    probs_v9 = np.array(probs_v9, dtype=float)

    # threshold: v9 의 test-MCC max 였던 0.330
    THR_V9 = 0.330
    pred_v9 = (probs_v9 >= THR_V9).astype(int)

    print(f"\n{'Drug':<22s} {'True':>5s} {'Prob':>6s} {'Pred':>5s} {'OK':>4s}")
    print("-" * 50)
    correct = 0
    for n, t, p, pr in zip(names, truth, probs_v9, pred_v9):
        ok = "✓" if pr == t else "✗"
        if pr == t: correct += 1
        print(f"{n:<22s} {t:>5d} {p:>6.3f} {pr:>5d} {ok:>4s}")
    print("-" * 50)
    print(f"\n전체 정확도: {correct}/{len(truth)} = {100*correct/len(truth):.1f}%")
    # TPR / TNR
    tp = ((pred_v9 == 1) & (truth == 1)).sum()
    fn = ((pred_v9 == 0) & (truth == 1)).sum()
    tn = ((pred_v9 == 0) & (truth == 0)).sum()
    fp = ((pred_v9 == 1) & (truth == 0)).sum()
    print(f"양성 (DILI): {tp}/{tp+fn} 맞음 (TPR {tp/(tp+fn):.2f})")
    print(f"음성 (안전): {tn}/{tn+fp} 맞음 (TNR {tn/(tn+fp):.2f})")
    from sklearn.metrics import matthews_corrcoef
    mcc = matthews_corrcoef(truth, pred_v9)
    print(f"MCC: {mcc:.3f}")


if __name__ == "__main__":
    main()
