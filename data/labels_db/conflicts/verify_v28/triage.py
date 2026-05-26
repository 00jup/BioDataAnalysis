"""
Triage weak_positives_to_verify.csv:
- Auto-classify ChEMBL IDs, IUPAC chemistry names, and obvious non-drugs as NaN
- Output drug_candidates.csv with real drug names for WebSearch verification
"""
import pandas as pd
import re
import os

SRC = "/Users/parkjeong-uk/CODING/2026/school/Bioinformatics/data/labels_db/conflicts/verify_v28/weak_positives_to_verify.csv"
OUT_DIR = "/Users/parkjeong-uk/CODING/2026/school/Bioinformatics/data/labels_db/conflicts/verify_v28"

df = pd.read_csv(SRC)
print(f"Total rows: {len(df)}")

# 1) ChEMBL IDs (e.g. "CHEMBL1443491")
def is_chembl_id(name):
    if pd.isna(name):
        return False
    return bool(re.fullmatch(r"CHEMBL\d+", str(name).strip()))

# 2) IUPAC / chemistry names (long with brackets, commas, numbers)
def is_iupac_chemistry(name):
    if pd.isna(name):
        return False
    n = str(name).strip()
    # Contains parens with substituents AND numbers AND hyphens, > 25 chars
    has_brackets = "(" in n or "[" in n
    has_numbers = bool(re.search(r"\d", n))
    has_chem_separators = ("," in n and "-" in n)
    if len(n) > 30 and has_brackets and has_numbers and has_chem_separators:
        return True
    # Pure IUPAC patterns (typical chemistry naming)
    iupac_patterns = [
        r"^\d+[,'\-]",                                  # starts "1,2-..."
        r"\b(alpha|beta|gamma|delta|epsilon|omega)-",   # greek-prefix chem
        r"^[NCO]-\(",                                   # N-(...), C-(...), O-(...)
        r"^[NCO]\(\d+\)",                               # N(6)-...
        r"^trans-|^cis-",                              # trans-/cis- prefix
        r"-\d+[,-]\d+",                                # -3,4- numeric
        r"hexa(deca|hydro|ene)|tetra(hydro|chloro|hexa)|penta(hydro|chloro)",  # alkane chains
        r"benzo\(", r"benzo\[",                        # fused arenes
        r"isoquinoline|imidazo|pyrazino|pyrimidino|pyrazolo|chromenone|furanone|naphthalen",
        r"-\d-yl-|-\d+H-|\d+H-\)",                     # ring designators
        r"phenyl|piperazin|piperidin|pyrrolidin|morpholin",  # chemistry ring keywords (with numbers nearby)
    ]
    if len(n) > 20:
        for pat in iupac_patterns:
            if re.search(pat, n, re.IGNORECASE):
                # Count chemistry tokens to confirm
                chem_score = (
                    int(bool(re.search(r"\d", n))) +
                    int("-" in n) +
                    int("(" in n or "[" in n) +
                    int("," in n)
                )
                if chem_score >= 3:
                    return True
    return False

# 3) Obvious non-drug exact-match keywords
NON_DRUG_KEYWORDS = [
    # Organometallic / industrial reagents
    "trimethyltin", "tributyltin", "dibutyltin", "triethyltin", "tetraethyl",
    "organotin", "organomercury", "methylmercury", "phenylmercury",
    "diphenyl phosphate", "tricresyl phosphate", "tributyl phosphate",
    "phosphate ester", "phosphorothioate", "phosphonates",
    # Lab chemicals / reagents
    "bromoacetamide", "trinitrobenzene", "nitrobenzene",
    "phenanthroline", "benzodioxaphosphorin",
    # Biochemicals/metabolites (not drugs)
    "glycogen", "guanosine triphosphate", "cyclic amp", "5'-methylthioadenosine",
    "organophosphonates", "stigmast-7-enol", "lithospermate", "procyanidin",
    "norwogonin", "glycitein", "tocotrienol", "tocopherol",
    "hydroxyeicosatetraenoic", "hexamethylene bisacetamide",
    "methylarginine", "phosphorothioate",
    # Tool/research compounds
    "phorbol", "myristate acetate",
    # Industrial chemicals
    "talc", "lignin", "cellulose", "starch", "dextran",
    "asbestos", "benzopyrene", "benzo(a)pyrene", "benzo[a]pyrene",
    "dichloro-", "tetrachloro", "pentachloro", "hexachloro",
    "dioxin", "dibenzofuran", "polychlorinated",
    "nitrosamine", "nitrosoamino",
    # Nerve agents / chemical warfare
    "soman", "sarin", "tabun", "vx ",
    "radon",
    # Bile acids / lipids / hormones often listed as metabolites
    "cholate", "deoxycholate", "taurocholate", "glycocholate",
    "prosta-", "prostaglandin",  # often endogenous
    # Natural products typically not drugs
    "anthocyanin", "catechin", "flavone", "flavanone", "flavonol",
    "stilbene", "saponin", "tannin", "chlorophyll",
    # Specific compound types
    "ethyl ester", "methyl ester", "palmitoyl", "stearoyl",
    "tetrahydrocannabinol",  # THC compound metabolites only
    "carbamylphosphatidylcholine", "phosphatidyl",
]

def is_non_drug_keyword(name):
    if pd.isna(name):
        return True
    n = str(name).lower().strip()
    return any(k in n for k in NON_DRUG_KEYWORDS)

# Apply
df["chembl_id_only"] = df["name"].apply(is_chembl_id)
df["likely_iupac"] = df["name"].apply(is_iupac_chemistry)
df["non_drug_kw"] = df["name"].apply(is_non_drug_keyword)
df["auto_nan"] = df["chembl_id_only"] | df["likely_iupac"] | df["non_drug_kw"]

print(f"ChEMBL-only: {df['chembl_id_only'].sum()}")
print(f"IUPAC chemistry names: {df['likely_iupac'].sum()}")
print(f"Non-drug keywords: {df['non_drug_kw'].sum()}")
print(f"Auto-NaN total (union): {df['auto_nan'].sum()}")
print(f"To verify via WebSearch: {(~df['auto_nan']).sum()}")

auto_nan_df = df[df["auto_nan"]].copy()
to_verify_df = df[~df["auto_nan"]].copy()

# Save auto-NaN
auto_nan_out = auto_nan_df[["inchi_key", "canonical_smiles", "name"]].copy()
auto_nan_out["manual_label"] = pd.NA
auto_nan_out["source"] = "non_drug"
def reason(r):
    if r["chembl_id_only"]:
        return "ChEMBL ID only"
    if r["likely_iupac"]:
        return "IUPAC chemistry name"
    return "industrial/lab/non-drug chemical"
auto_nan_out["reason"] = auto_nan_df.apply(reason, axis=1)
auto_nan_out.to_csv(os.path.join(OUT_DIR, "auto_nan.csv"), index=False)

# Save drug candidates
to_verify_out = to_verify_df[["inchi_key", "canonical_smiles", "name"]].copy()
to_verify_out.to_csv(os.path.join(OUT_DIR, "drug_candidates.csv"), index=False)

print(f"\nSaved auto_nan.csv ({len(auto_nan_out)} rows)")
print(f"Saved drug_candidates.csv ({len(to_verify_out)} rows)")

print("\n--- Sample DRUG CANDIDATES (will WebSearch) ---")
for n in to_verify_out["name"].head(40).tolist():
    print(f"  {n}")
print("\n--- Sample AUTO-NAN (IUPAC) ---")
for n in auto_nan_df[auto_nan_df["likely_iupac"]]["name"].head(20).tolist():
    print(f"  {n}")
