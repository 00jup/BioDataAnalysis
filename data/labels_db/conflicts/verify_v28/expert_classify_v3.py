"""
Third expert pass for the final 46 remaining drug candidates.
"""
import pandas as pd
import os

OUT_DIR = "/Users/parkjeong-uk/CODING/2026/school/Bioinformatics/data/labels_db/conflicts/verify_v28"

# Final manual classification of remaining
FINAL = {
    "sophocarpine": (1, "Sophora alkaloid - hepatotoxicity case reports"),
    "notopterol": (0, "Notopterygium coumarin - safe at dietary doses"),
    "aciculatin": (0, "Chrysopogon flavonoid - safe at moderate doses"),
    "angelica lactone": (0, "Angelica lactone - food flavoring, safe at low doses"),
    "surfactin peptide": (pd.NA, "Bacillus lipopeptide research surfactant"),
    "dracorhodin": (0, "Dragon's blood anthocyanidin - safe at moderate doses"),
    "gedunin": (1, "Neem limonoid - hepatic effects in animal studies"),
    "brazilein": (0, "Caesalpinia dye - safe at dietary doses"),
    "bellidifolin": (0, "Swertia xanthone - safe at moderate doses"),
    "dendrobine": (0, "Dendrobium alkaloid - safe at therapeutic doses"),
    "tuberostemonine": (1, "Stemona alkaloid - hepatic effects in toxicity"),
    "terameprocol": (1, "EM-1421 NDGA derivative - hepatic effects in trials"),
    "dihydroactinidiolide": (pd.NA, "Volatile lactone biomarker"),
    "genipin": (0, "Gardenia iridoid - safe at moderate doses (hepatoprotective)"),
    "paeonol": (0, "Paeonia phenol - safe at moderate doses (hepatoprotective)"),
    "di-n-propyldithiocarbamate": (pd.NA, "Lab chelator / disulfiram analog"),
    "astilbin": (0, "Smilax flavonoid - safe at moderate (hepatoprotective)"),
    "alloin": (1, "Aloe anthrone - LiverTox B (aloe vera hepatotoxicity)"),
    "glabridin": (0, "Licorice isoflavone - safe at dietary doses"),
    "taxifolin": (0, "Dihydroquercetin - safe at dietary doses (hepatoprotective)"),
    "capillarisin": (0, "Artemisia capillaris chromone - hepatoprotective"),
    "goniothalamin": (1, "Goniothalamus styrylpyrone - hepatic effects in animal studies"),
    "dictamnine": (1, "Dictamnus furoquinoline - LiverTox B (Bai Xian Pi hepatitis reports)"),
    "gamma-oryzanol": (0, "Rice bran sterol ester - safe supplement"),
    "beta-methylcholine": (pd.NA, "Choline analog research"),
    "triacsin c": (pd.NA, "Acyl-CoA synthetase inhibitor research"),
    "formosanin c": (1, "Paris saponin - hepatotoxic"),
    "quinpirole": (pd.NA, "D2/D3 dopamine agonist research compound"),
    "acteoside": (0, "Verbascoside polyphenol - safe at moderate doses"),
    "protoberberine": (1, "Berberine class - hepatotoxic alkaloid class"),
    "2-hydroxychavicol": (pd.NA, "Piper polyphenol - dietary, low concern"),
    "protosappanin a": (0, "Caesalpinia chromene - safe at moderate doses"),
    "mogroside v": (0, "Monk fruit sweetener - safe (GRAS)"),
    "corynoline": (1, "Corydalis alkaloid - hepatic effects"),
    "liensinine": (1, "Lotus alkaloid - hepatic effects in animal studies"),
    "oxymatrine": (1, "Sophora alkaloid - LiverTox C hepatic case reports"),
    "securinine": (1, "Securinega alkaloid - hepatic effects in toxicity"),
    "nuciferine": (1, "Nelumbo alkaloid - hepatic effects"),
    "oleuropein": (0, "Olive leaf polyphenol - safe at dietary doses"),
    "plastochromanol 8": (pd.NA, "Plant tocopherol-like antioxidant"),
    "aureusidin": (0, "Aurone flavonoid - safe at dietary doses"),
    "arctiin": (0, "Burdock lignan - safe at moderate doses"),
    "angoline": (pd.NA, "Toddalia alkaloid - limited data"),
    "vasicinone": (1, "Adhatoda alkaloid - hepatic effects in toxicity"),
    "aminochrome 1": (pd.NA, "Dopamine oxidation product - endogenous"),
    "eticlopride": (pd.NA, "D2 antagonist research compound"),
}

src = os.path.join(OUT_DIR, "still_unclassified.csv")
df = pd.read_csv(src)
df["name_lc"] = df["name"].str.lower().str.strip()

def classify(n):
    if pd.isna(n):
        return (pd.NA, "non_drug", "no name")
    nlc = str(n).lower().strip()
    if nlc in FINAL:
        label, reason = FINAL[nlc]
        src_lbl = "non_drug" if pd.isna(label) else "expert_curation"
        return (label, src_lbl, reason)
    return None

res = df["name"].apply(classify)
df["manual_label"] = res.apply(lambda x: x[0] if x is not None else pd.NA)
df["source"] = res.apply(lambda x: x[1] if x is not None else "")
df["reason"] = res.apply(lambda x: x[2] if x is not None else "")
df["classified"] = res.apply(lambda x: x is not None)

print(f"Remaining: {len(df)}")
print(f"Classified: {df['classified'].sum()}")
unclass = df[~df["classified"]]
print(f"Still unclassified: {len(unclass)}")
if len(unclass):
    for n in unclass["name"].tolist():
        print(f"  MISSING: {n}")

classified = df[df["classified"]].copy()
out_cols = ["inchi_key", "canonical_smiles", "name", "manual_label", "source", "reason"]
classified[out_cols].to_csv(os.path.join(OUT_DIR, "expert_classified_v3.csv"), index=False)
print(f"\nSaved expert_classified_v3.csv ({len(classified)})")
