"""
Feature Engineering 모듈
- Morgan Fingerprint (ECFP4)
- MACCS Keys
- Physicochemical Descriptors
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, MACCSkeys

RDLogger.logger().setLevel(RDLogger.ERROR)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")


def smiles_to_mol(smiles):
    """SMILES → RDKit Mol 객체 변환."""
    return Chem.MolFromSmiles(smiles)


def compute_morgan_fp(mol, radius=2, n_bits=2048):
    """Morgan Fingerprint (ECFP4) 계산."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)


def compute_maccs_keys(mol):
    """MACCS Keys (167 bits) 계산."""
    fp = MACCSkeys.GenMACCSKeys(mol)
    return np.array(fp)


def compute_descriptors(mol):
    """Physicochemical descriptors 계산."""
    return {
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "NumAromaticRings": Descriptors.NumAromaticRings(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": Descriptors.RingCount(mol),
    }


def extract_features(smiles_list, feature_set="B"):
    """
    SMILES 리스트에서 feature를 추출.

    feature_set:
        "A" - Morgan FP only (2048)
        "B" - Morgan FP + Physicochemical Descriptors (~2058) [기본값]
        "C" - Morgan + MACCS + Physicochemical (~2225)
        "D" - Physicochemical only (~10)
    """
    print(f"Feature 추출 중 (feature_set={feature_set})...")

    morgan_list = []
    maccs_list = []
    desc_list = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        mol = smiles_to_mol(smi)
        if mol is None:
            print(f"  경고: 인덱스 {i} SMILES 변환 실패, 건너뜀: {smi}")
            continue

        valid_indices.append(i)

        if feature_set in ("A", "B", "C"):
            morgan_list.append(compute_morgan_fp(mol))
        if feature_set == "C":
            maccs_list.append(compute_maccs_keys(mol))
        if feature_set in ("B", "C", "D"):
            desc_list.append(compute_descriptors(mol))

    # Feature 조합
    parts = []
    col_names = []

    if morgan_list:
        morgan_arr = np.array(morgan_list)
        parts.append(morgan_arr)
        col_names.extend([f"Morgan_{i}" for i in range(morgan_arr.shape[1])])

    if maccs_list:
        maccs_arr = np.array(maccs_list)
        parts.append(maccs_arr)
        col_names.extend([f"MACCS_{i}" for i in range(maccs_arr.shape[1])])

    if desc_list:
        desc_df = pd.DataFrame(desc_list)
        parts.append(desc_df.values)
        col_names.extend(desc_df.columns.tolist())

    X = np.hstack(parts)
    feature_df = pd.DataFrame(X, columns=col_names)

    # NaN 처리 (descriptor 계산 실패 시)
    n_nan = feature_df.isna().sum().sum()
    if n_nan > 0:
        print(f"  NaN 발견: {n_nan}개 → 0으로 대체")
        feature_df = feature_df.fillna(0)

    print(f"  완료: {feature_df.shape[0]}개 샘플, {feature_df.shape[1]}개 feature")
    return feature_df, valid_indices


def prepare_features(data_path=None, feature_set="B"):
    """전처리된 데이터에서 feature를 추출하고 저장."""
    if data_path is None:
        data_path = os.path.join(PROCESSED_DIR, "dili_dataset.csv")

    df = pd.read_csv(data_path)
    print(f"데이터 로드: {data_path} ({len(df)}개)")

    feature_df, valid_indices = extract_features(df["SMILES"].tolist(), feature_set=feature_set)

    # 라벨 매칭
    labels = df["Label"].iloc[valid_indices].reset_index(drop=True)

    # 저장
    feature_path = os.path.join(PROCESSED_DIR, f"features_{feature_set}.csv")
    label_path = os.path.join(PROCESSED_DIR, "labels.csv")

    feature_df.to_csv(feature_path, index=False)
    labels.to_csv(label_path, index=False)

    print(f"Feature 저장: {feature_path}")
    print(f"Label 저장: {label_path}")

    return feature_df, labels


if __name__ == "__main__":
    # 기본 feature set B로 추출
    X, y = prepare_features(feature_set="B")
    print(f"\nFeature shape: {X.shape}")
    print(f"Label 분포:\n{y.value_counts()}")
