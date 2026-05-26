"""
Post-merge cleanup: convert elements and generic chemical class names to NaN.
"""
import pandas as pd
import os

OUT_DIR = "/Users/parkjeong-uk/CODING/2026/school/Bioinformatics/data/labels_db/conflicts/verify_v28"
path = os.path.join(OUT_DIR, "weak_positives_verified.csv")
df = pd.read_csv(path)

# Elements / pure inorganics
ELEMENTS_INORGANIC = {
    "bromine", "tungsten", "lanthanum", "ruthenium", "cerium",
    "beryllium", "zirconium", "carbon", "hydrogen", "propane",
    "radon", "uranium", "talc",
}
# Generic chemical class names
CLASSES = {
    "imidazolidines", "sulfones", "isoxazoles", "steroids", "triterpenes",
    "hydantoins", "acetates", "oxazolidinones", "indazoles", "benzofurans",
    "benzoxazines", "nitroimidazoles", "oxadiazoles", "pyrroles",
    "pyrrolidines", "chalcone", "1,4-dihydropyridine",
}

def fix(row):
    n = str(row["name"]).lower().strip()
    if n in ELEMENTS_INORGANIC:
        return (pd.NA, "non_drug", "element / inorganic substance")
    if n in CLASSES:
        return (pd.NA, "non_drug", "generic chemical class")
    return (row["manual_label"], row["source"], row["reason"])

results = df.apply(fix, axis=1)
df["manual_label"] = results.apply(lambda x: x[0])
df["source"] = results.apply(lambda x: x[1])
df["reason"] = results.apply(lambda x: x[2])

print("=== After cleanup ===")
print(df["manual_label"].value_counts(dropna=False))
print(df["source"].value_counts())

df.to_csv(path, index=False)
print(f"Saved {path}")

# Re-export batches
batch_dir = os.path.join(OUT_DIR, "batches")
import shutil
shutil.rmtree(batch_dir, ignore_errors=True)
os.makedirs(batch_dir, exist_ok=True)
for i in range(0, len(df), 100):
    chunk = df.iloc[i:i+100]
    chunk.to_csv(os.path.join(batch_dir, f"batch_{i//100:04d}.csv"), index=False)
print(f"Re-saved {(len(df)+99)//100} batch files")
