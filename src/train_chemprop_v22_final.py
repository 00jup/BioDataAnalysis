"""v22 final — train+val+test 합쳐서 v17 같은 hp 로 재학습.

목적: 채점 시 사용할 강한 final model.
모든 13,777 vivo 분자로 학습 → generalization 강화.

split: 95% train + 5% val (val 은 early stopping 용 작게)
       → test 분리 안 함 (final model 이라 외부 sanity 만 측정)
hp: v17 동일 (ensemble 15 + hidden 600 + epochs 40 + featurizer)
"""
from __future__ import annotations
import json, os, sys, subprocess, time
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(PROJECT_ROOT, "data", "chemprop_scaffold_v2")
SAVE = os.path.join(PROJECT_ROOT, "models", "chemprop_v22_final")
RESULTS = os.path.join(PROJECT_ROOT, "results")
CHEMPROP_BIN = os.path.join(os.path.dirname(sys.executable), "chemprop")


def train():
    print(f"\n=== Chemprop v22 final — train+val+test 모두 사용 ===")
    save_dir = SAVE; os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(DATA, "vivo", "all.csv")
    df = pd.read_csv(csv_path)
    print(f"  전체 데이터: {len(df)} (양성 {(df.label==1).sum()} / 음성 {(df.label==0).sum()})")

    cmd = [
        CHEMPROP_BIN, "train",
        "-i", csv_path, "-s", "canonical_smiles",
        "--target-columns", "label", "-t", "classification", "-l", "bce",
        "--metrics", "binary-mcc", "roc",
        "--split", "SCAFFOLD_BALANCED",
        "--split-sizes", "0.95", "0.05", "0.00",   # ← 거의 다 train, val 작게, test 없음
        "--ensemble-size", "15",
        "--message-hidden-dim", "600",
        "--epochs", "40", "--patience", "8",
        "--molecule-featurizers", "v1_rdkit_2d_normalized",
        "--accelerator", "cpu",
        "-o", save_dir,
    ]
    log_path = os.path.join(save_dir, "train.log")
    t0 = time.time()
    print(f"  v17 동일 hp + 95% train")
    with open(log_path, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    print(f"  학습 끝 ({(time.time()-t0)/60:.1f}분, exit={r.returncode})")
    if r.returncode != 0:
        with open(log_path) as f:
            print("\n".join(f.readlines()[-30:]))
        return None
    return save_dir


def sanity_check(model_dir):
    """외부 의학 약물 20 으로 sanity check."""
    DRUGS = [
        # 양성
        ('Acetaminophen', 'CC(=O)Nc1ccc(O)cc1', 1),
        ('Isoniazid', 'NNC(=O)c1ccncc1', 1),
        ('Valproic acid', 'CCCC(CCC)C(=O)O', 1),
        ('Troglitazone', 'Cc1c(C)c2OC(C)(COc3ccc(CC4SC(=O)NC4=O)cc3)CCc2c(C)c1O', 1),
        ('Diclofenac', 'OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl', 1),
        ('Halothane', 'FC(F)(F)C(Cl)Br', 1),
        ('Ketoconazole', 'CC(=O)N1CCN(c2ccc(OCC3COC(Cn4ccnc4)(c4ccc(Cl)cc4Cl)O3)cc2)CC1', 1),
        ('Methotrexate', 'CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)NC(CCC(=O)O)C(=O)O)cc1', 1),
        ('Amoxicillin-clav', 'CC1(C)SC2C(NC(=O)C(N)c3ccc(O)cc3)C(=O)N2C1C(=O)O', 1),
        ('Nitrofurantoin', 'O=C1OCC(N1\\N=C\\c1ccc(o1)[N+](=O)[O-])', 1),
        ('Aspirin', 'CC(=O)Oc1ccccc1C(=O)O', 1),
        ('Ibuprofen', 'CC(C)Cc1ccc(C(C)C(=O)O)cc1', 1),
        ('Atorvastatin', 'CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CCC(O)CC(O)CC(=O)O', 1),
        ('Metformin', 'CN(C)C(=N)NC(=N)N', 1),
        ('Lisinopril', 'NCCCCC(NC(CCc1ccccc1)C(=O)O)C(=O)N1CCCC1C(=O)O', 1),
        ('Amlodipine', 'CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl', 1),
        ('Omeprazole', 'COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1', 1),
        # 음성
        ('Loratadine', 'CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1', 0),
        ('Cetirizine', 'OC(=O)COCCN1CCN(C(c2ccccc2)c2ccc(Cl)cc2)CC1', 0),
        ('Levothyroxine', 'Oc1cc(I)c(Oc2cc(I)c(CC(N)C(=O)O)cc2I)c(I)c1', 0),
    ]
    import tempfile
    names = [d[0] for d in DRUGS]
    smiles = [d[1] for d in DRUGS]
    truth = np.array([d[2] for d in DRUGS])

    with tempfile.NamedTemporaryFile('w', suffix='.csv', delete=False) as f:
        pd.DataFrame({'canonical_smiles': smiles}).to_csv(f.name, index=False)
        in_p = f.name
    out_p = in_p.replace('.csv', '_pred.csv')
    cmd = [CHEMPROP_BIN, 'predict', '--test-path', in_p,
           '-s', 'canonical_smiles', '--model-paths', model_dir, '--preds-path', out_p,
           '--molecule-featurizers', 'v1_rdkit_2d_normalized', '--accelerator', 'cpu']
    subprocess.run(cmd, capture_output=True, check=True)
    pred_df = pd.read_csv(out_p)
    pcol = [c for c in pred_df.columns if c != 'canonical_smiles'][0]
    by_smi = dict(zip(pred_df['canonical_smiles'], pred_df[pcol]))
    probs = np.array([by_smi.get(s, np.nan) for s in smiles])

    print(f"\n=== Sanity check (외부 20 약물) ===")
    print(f"{'Drug':<22s} {'True':>5s} {'Prob':>6s}")
    for n, t, p in zip(names, truth, probs):
        print(f"{n:<22s} {t:>5d} {p:>6.3f}")

    from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_auc_score
    print()
    print(f"AUC: {roc_auc_score(truth, probs):.3f}")
    print(f"{'thr':>5s} {'TPR':>5s} {'TNR':>5s} {'MCC':>6s}")
    out = {}
    for thr in (0.30, 0.35, 0.40, 0.45, 0.50):
        pred = (probs >= thr).astype(int)
        tp = ((pred==1)&(truth==1)).sum(); fn = ((pred==0)&(truth==1)).sum()
        tn = ((pred==0)&(truth==0)).sum(); fp = ((pred==1)&(truth==0)).sum()
        tpr = tp/max(tp+fn,1); tnr = tn/max(fp+tn,1)
        mcc = matthews_corrcoef(truth, pred)
        print(f"{thr:>5.2f} {tpr:>5.2f} {tnr:>5.2f} {mcc:>+6.3f}")
        out[thr] = {"tpr": float(tpr), "tnr": float(tnr), "mcc": float(mcc)}
    return out


def main():
    os.makedirs(SAVE, exist_ok=True); os.makedirs(RESULTS, exist_ok=True)
    model_dir = train()
    if not model_dir: return
    out = sanity_check(model_dir)
    with open(os.path.join(RESULTS, "chemprop_v22_final.json"), "w") as f:
        json.dump({"sanity": out}, f, indent=2)
    print(f"\n저장: results/chemprop_v22_final.json")


if __name__ == "__main__":
    main()
