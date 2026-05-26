"""
Merge all classification outputs into weak_positives_verified.csv

Sources:
- auto_nan.csv (ChEMBL IDs + IUPAC chemistry names + obvious non-drugs)
- expert_classified.csv (V1 pass)
- expert_classified_v2.csv (V2 pass)
- expert_classified_v3.csv (V3 final 46)
"""
import pandas as pd
import os

OUT_DIR = "/Users/parkjeong-uk/CODING/2026/school/Bioinformatics/data/labels_db/conflicts/verify_v28"
SRC = os.path.join(OUT_DIR, "weak_positives_to_verify.csv")

src_df = pd.read_csv(SRC)
print(f"Original input: {len(src_df)} rows")

frames = []
for f in ["auto_nan.csv", "expert_classified.csv", "expert_classified_v2.csv", "expert_classified_v3.csv"]:
    p = os.path.join(OUT_DIR, f)
    d = pd.read_csv(p)
    print(f"  {f}: {len(d)}")
    frames.append(d[["inchi_key", "canonical_smiles", "name", "manual_label", "source", "reason"]])

merged = pd.concat(frames, ignore_index=True)
print(f"Merged: {len(merged)}")

# Ensure 1:1 with source. Dedup by inchi_key, keeping first
before = len(merged)
merged = merged.drop_duplicates(subset=["inchi_key"], keep="first")
after = len(merged)
print(f"Dedup: {before} -> {after}")

# Validate coverage
src_keys = set(src_df["inchi_key"])
merged_keys = set(merged["inchi_key"])
missing = src_keys - merged_keys
extra = merged_keys - src_keys
print(f"Missing from output: {len(missing)}")
print(f"Extra in output: {len(extra)}")
if missing:
    print("First 10 missing:", list(missing)[:10])

# Reorder to follow source order
merged = merged.set_index("inchi_key").reindex(src_df["inchi_key"]).reset_index()

# Summary
print(f"\n=== FINAL SUMMARY ===")
print(f"Total: {len(merged)}")
print(f"  manual_label=1 (hepatotoxic): {(merged['manual_label']==1).sum()}")
print(f"  manual_label=0 (safe): {(merged['manual_label']==0).sum()}")
print(f"  manual_label=NaN (non-drug/no evidence): {merged['manual_label'].isna().sum()}")
print(f"\nSource distribution:")
print(merged["source"].value_counts(dropna=False))

# Save
out_path = os.path.join(OUT_DIR, "weak_positives_verified.csv")
merged.to_csv(out_path, index=False)
print(f"\nSaved {out_path}")

# Save batch checkpoints (100-row chunks) for safety
batch_dir = os.path.join(OUT_DIR, "batches")
os.makedirs(batch_dir, exist_ok=True)
for i in range(0, len(merged), 100):
    chunk = merged.iloc[i:i+100]
    chunk.to_csv(os.path.join(batch_dir, f"batch_{i//100:04d}.csv"), index=False)
print(f"Saved {(len(merged)+99)//100} batch files")
