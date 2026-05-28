"""Diagnose which SMILES failed parsing."""

import sys

sys.path.insert(0, "/Users/parkjeong-uk/CODING/2026/school/Bioinformatics/scripts")
from build_sanity_v2 import DRUGS
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

fails = []
for name, smi, label, ref, reason in DRUGS:
    if ref.startswith("EXCLUDED") or ref == "DUP":
        continue
    m = Chem.MolFromSmiles(smi)
    if m is None:
        fails.append((name, smi))

print(f"Total failed: {len(fails)}")
for n, s in fails:
    print(f"- {n}: {s[:80]}")
