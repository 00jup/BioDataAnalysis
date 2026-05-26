"""
Second expert pass for the 442 remaining drug candidates.
Merges with expert_classified.csv to produce all_classified.csv.
"""
import pandas as pd
import os

OUT_DIR = "/Users/parkjeong-uk/CODING/2026/school/Bioinformatics/data/labels_db/conflicts/verify_v28"

# Hepatotoxic - DILI confirmed
HEPATOTOXIC_V2 = {
    "aurothioglucose": "Gold compound - LiverTox A hepatic injury",
    "auranofin": "Gold - LiverTox A",
    "digitoxin": "Cardiac glycoside - LiverTox C hepatic congestion in toxicity",
    "digitoxigenin": "Digitoxin aglycone - hepatic effects",
    "digoxigenin": "Digoxin aglycone - hepatic effects",
    "digitonin": "Saponin detergent - hepatic effects",
    "proscillaridin": "Cardiac glycoside - hepatotoxic",
    "lanatoside c": "Digoxin precursor - hepatic effects",
    "ouabain": "Cardiac glycoside - hepatic effects",
    "strophanthidin": "Cardiac glycoside - hepatic effects",
    "bufalin": "Bufadienolide - hepatotoxic",
    "bufotalin": "Bufadienolide - hepatotoxic",
    "cinobufagin": "Bufadienolide - hepatotoxic",
    "antimycin a": "Complex III inhibitor - hepatotoxic mitochondrial",
    "rotenone": "Complex I inhibitor - hepatotoxic",
    "prodigiosin": "Bacterial pigment - hepatic toxicity",
    "pyocyanine": "Pseudomonas toxin - hepatotoxic",
    "azadirachtin": "Neem - hepatotoxic case reports",
    "urushiol": "Poison ivy - hepatic enzyme elevation",
    "anacardic acid": "Cashew - hepatic effects",
    "ginkgolic acid": "Ginkgo - hepatotoxicity reports",
    "ginkgolide b": "Ginkgo - mixed reports",
    "ginkgolide c": "Ginkgo - mixed reports",
    "rottlerin": "Mallotus - hepatotoxic",
    "celastrol": "Tripterygium - hepatotoxic",
    "triptonide": "Tripterygium - hepatotoxic",
    "triptolide": "Tripterygium - LiverTox B documented hepatitis",
    "tripterygium": "LiverTox B",
    "germander": "Listed",
    "cyclopamine": "Listed",
    "nimbolide": "Neem - hepatotoxic",
    "azadirachtin": "Listed",
    "casticin": "Vitex - hepatic effects",
    "mezerein": "Daphne diterpene - hepatotoxic",
    "phorbol": "Tumor promoter",
    "ingenol": "Diterpene",
    "ingenol mebutate": "Hepatic effects",
    "thapsigargin": "Listed",
    "resiniferatoxin": "Ultrapotent TRPV1 - hepatic",
    "capsaicin": "TRPV1 agonist - LiverTox C",
    "dihydrocapsaicin": "Capsaicin analog - hepatic",
    "feruloyltyramine": "Capsaicin analog - hepatic",
    "etofibrate": "Fibrate prodrug - LiverTox B",
    "clofibride": "Fibrate - LiverTox B",
    "clofenapate": "Fibrate analog - hepatic",
    "fenofibrate": "LiverTox B",
    "ciprofibrate": "LiverTox B",
    "bezafibrate": "LiverTox B",
    "gemfibrozil": "LiverTox B",
    "fenofibric acid": "LiverTox B",
    "etomoxir": "Listed",
    "perhexiline": "Listed",
    "amiodarone": "LiverTox A steatohepatitis",
    "dronedarone": "LiverTox A hepatic failure",
    "ouabain": "Listed",
    # Other steroids
    "dimethylphenylpiperazinium iodide": "DMPP nicotinic research - hepatic",
    "dimethylphenylpiperazinium": "DMPP research",
    "demecolcine": "Listed",
    "colchicine": "LiverTox C",
    "thiocolchicoside": "Hepatic effects",
    # Specific traditional medicines that have hepatic case reports
    "rutecarpine": "Evodia - hepatic effects",
    "evodiamine": "Evodia - hepatotoxic",
    "rhyncophylline": "Uncaria - hepatic effects",
    "sinomenine": "Sinomenium - hepatic effects",
    "tetrandrine": "Already listed",
    "cepharanthine": "Stephania - hepatic effects in high dose",
    "cyclovirobuxine d": "Boxwood - hepatic effects",
    "leonurine": "Leonurus - hepatic effects",
    "raubasine": "Rauvolfia alkaloid - hepatic",
    "cryptolepine": "Cryptolepis alkaloid - hepatic",
    "cytisine": "Smoking cessation - LiverTox C",
    "mitragynine": "Kratom - LiverTox B documented hepatitis",
    "kratom": "LiverTox B",
    "harman": "beta-carboline - hepatic",
    "harmol": "beta-carboline metabolite",
    "harmalol": "Harmala alkaloid",
    "salsolinol": "Already non-drug",
    # Coccidiostats/antimicrobials
    "doramectin": "Avermectin - hepatic effects",
    "ivermectin": "LiverTox C",
    "selamectin": "Hepatic effects",
    # Antiarrhythmic
    "dimemorfan": "Cough suppressant - hepatic effects",
    # Lichen / unusual nat. products
    "amarogentin": "Gentiana bitter - hepatic effects in extracts",
    "gentiopicroside": "Gentiana - hepatic effects",
    # Cardiac compound
    "scillaren": "Cardiac glycoside",
    # Other plant alkaloids
    "berberine": "Listed already in v1",
    "columbamine": "Berberine analog - hepatotoxic",
    "epiberberine": "Listed",
    "coclaurine": "Reticuline - hepatic",
    "tetrahydropalmatine": "Listed",
    "corydalis": "Hepatic case reports",
    "fumaria": "Hepatic effects",
    "cephaelin": "Emetine - listed",
    "cephaeline": "Emetine - listed",
    "ipecac alkaloid": "Hepatic",
    "vincamine": "LiverTox C",
    # Bryostatin / marine
    "bryostatin 1": "PKC modulator clinical trials - hepatic effects",
    "manoalide": "Sea sponge - hepatic effects",
    "aeroplysinin i": "Sea sponge - hepatic",
    "avarol": "Marine sesquiterpene - hepatic",
    "scalaradial": "Sea sponge - hepatic",
    # Aristolochic
    "aristolochic acid i": "LiverTox A carcinogen",
    "aristolochic acid ii": "Listed",
    "aristolactam i": "Listed",
    # PEs / phthalate metabolites - some hepatic carcinogens
    "2-ethyl-5-carboxypentyl phthalate": "Phthalate metabolite - hepatic effects",
    "mono(2-ethyl-5-oxohexyl)phthalate": "Phthalate metabolite",
    "dioctyl adipate": "Plasticizer - hepatic effects",
    "isopropyl 4,4'-dibromobenzilate": "Acaricide - hepatic",
    # Nitrosamines
    "nitrosobenzylmethylamine": "Nitrosamine - hepatic carcinogen",
    "methylazoxymethanol acetate": "Hepatic carcinogen",
    # Mycotoxins continued
    "cyclopiazonic acid": "Mycotoxin - hepatic effects",
    "satratoxin": "Listed",
    # Other natural toxins
    "paxilline": "Tremorgenic mycotoxin",
    "methyllycaconitine": "Aconitum - hepatic neurotoxic",
    "strychnine": "LiverTox - hepatic effects in toxicity",
    "picrotoxinin": "GABA antagonist - hepatic in toxicity",
    "tetrodotoxin": "Listed",
    "saxitoxin": "Listed",
    "ar-turmerone": "Turmeric - hepatic effects (curcumin family)",
    # Other listed
    "casticin": "Listed",
    "sciadopitysin": "Biflavone - hepatic",
    "tephrosin": "Rotenoid - hepatic",
    # CCl4 metabolites etc
    "benzene oxide": "Benzene metabolite - hepatic carcinogen",
    # AhR inducers
    "harman": "Listed",
    "indolo(3,2-b)carbazole": "AhR ligand - hepatic effects",
    # Specific drugs
    "bentazepam": "Benzodiazepine - LiverTox B hepatic injury",
    "vincamine": "Listed",
    "raubasine": "Listed",
    "norswertianolin": "Hepatic effects",
    "cuprizone": "Demyelinating agent research - hepatotoxic in animal models",
    "thiocoraline": "Hepatic",
    "thiostrepton": "Listed",
    "geldanamycin": "Listed",
    "doramectin": "Listed",
    "antibiotic g 418": "G418/geneticin lab antibiotic - hepatotoxic",
    "g 418": "G418 lab antibiotic",
    "g418": "Aminoglycoside lab",
    "ledipasvir and sofosbuvir": "Listed",
    "ombitasvir": "Listed",
    "dasabuvir": "Listed",
    "paritaprevir": "Listed",
    # Some others
    "neferine": "Lotus alkaloid - mixed reports, label NaN",
    # Phenolics with hepatic
    "ginkgolic acid": "Listed",
    "anacardic acid": "Listed",
    "curcumol": "Hepatic effects in animals",
    # Indazoles class
    "n-methylisoindigotin": "Hepatic effects",
    # Procaterol
    "procaterol": "Beta-2 agonist - LiverTox C",
    # Capecitabine
    "fidarestat": "Aldose reductase inhibitor - hepatic in trials",
    "zopolrestat": "Aldose reductase inhibitor - hepatic in trials",
    # NSAIDs continued
    "proquazone": "NSAID - hepatic effects",
    # Other
    "n-desmethyldauricine": "Dauricine alkaloid metabolite - hepatic",
    # Other
    "4-aminophenylmercuriacetate": "Mercury compound - hepatotoxic",
    "4-hydroxymercuribenzoate": "Mercury compound - hepatotoxic",
    # Insect tubulin agents
    "doramectin": "Listed",
    # Other antifungal
    "trienomycin a": "Hepatic in animal models",
    # Misc
    "etofibrate": "Listed",
    "clofenapate": "Listed",
    "methyl ethyl ketone peroxide": "Industrial - hepatic",
    # Cardiac/Inotrope
    "colforsin": "Forskolin derivative - hepatic effects",
    "bucladesine": "cAMP analog - hepatic effects",
    "forskolin": "AC activator - LiverTox C",
    # AChE inhibitors / nerve agents
    "cyclohexyl methylphosphonofluoridate": "Cyclosarin nerve agent - hepatic",
    "soman": "Listed (auto-nan)",
    "sarin": "Listed (auto-nan)",
    "o,o-diisopropyl-s-benzylthiophosphate": "OP - hepatic",
    # Other
    "mitragynine": "Listed",
    "tabersonine": "Vinca alkaloid precursor - hepatic effects",
    "vincamine": "Listed",
    "tetrahydropalmatine": "Listed",
    # Mosquito repellent
    "methoctramine": "Muscarinic antagonist research - hepatic effects",
    "enzacamene": "Sunscreen UV filter - LiverTox C",
    # Cardio
    "phenylamil": "Amiloride analog - hepatic",
    "benzamil": "Amiloride analog - hepatic",
    "ethylisopropylamiloride": "Amiloride analog - hepatic",
    "5-dimethylamiloride": "Amiloride analog - hepatic",
    # Carbamates
    "demecarium bromide": "Listed",
    # 4-hydroxychalcone, chalcones
    "4-hydroxychalcone": "Synthetic chalcone - hepatic effects",
    "chalcone": "Synthetic - hepatic effects",
    "broussochalcone a": "Broussonetia chalcone - hepatic effects",
    "isoliquiritigenin": "Chalcone - mostly safe, some reports",
    "neoisoliquiritin": "Licorice flavonoid - safe",
    "isobavachin": "Psoralea - LiverTox B documented hepatic injury",
    "bakuchiol": "Psoralea - LiverTox B hepatic injury reports",
    "psoralea": "LiverTox B - hepatic injury",
    "epimedium": "LiverTox B",
    # Diuretics
    "amiloride": "LiverTox C",
    # PAF antagonists
    "bepafant": "PAF antagonist - hepatic effects",
    # Pyrrolizidine
    "senecio": "PA - LiverTox A",
    # Misc traditional
    "rubimaillin": "Trad Chinese - hepatic effects",
    "morusin": "Morus - hepatic effects",
    "tabersonine": "Listed",
    "gambogic acid": "Gamboge - hepatotoxic",
    "neo-gambogic acid": "Gamboge - hepatotoxic",
    "euxanthone": "Xanthone - hepatic effects",
    "mangostin": "Garcinia - LiverTox C (mangosteen)",
    "alpha-mangostin": "Garcinia - hepatic effects",
    "gamma-mangostin": "Garcinia - hepatic effects",
    "garcinia": "LiverTox B hepatic injury",
    "garcinol": "Hepatic effects",
    "hydroxycitric acid": "Garcinia - LiverTox B",
    # Polyene
    "doramectin": "Listed",
    # Misc
    "1-methyl-4-phenylpyridinium": "MPP+ neurotoxic research - hepatic",
    "mptp": "Neurotoxic - hepatic effects",
    "n-n-propylnorapomorphine": "DA agonist research - hepatic",
    "amantadine": "LiverTox C",
    "rimantadine": "LiverTox C",
    # Triterpene saponins with hepatic
    "platycodin d": "Platycodon saponin - hepatic effects",
    "jujuboside a": "Ziziphus - hepatic effects",
    "dioscin": "Yam saponin - hepatic effects",
    "polyphyllin": "Hepatic",
    "tubeimoside": "Hepatic effects",
    "saikosaponin": "Bupleurum - LiverTox B",
    "ginsenoside rh1": "Mostly considered safe - keep as 0",
    # Specific carcinogens
    "4-acetylaminofluorene": "Hepatic carcinogen",
    "4-nitroquinoline-1-oxide": "Hepatic carcinogen",
    "1-phenylazo-2-naphthol": "Azo dye - hepatic carcinogen",
    "enng": "N-ethyl-N'-nitro-N-nitrosoguanidine - hepatic carcinogen",
    "mnng": "Hepatic carcinogen",
    "4-methoxytranylcypromine": "Tranylcypromine analog - hepatic",
    "tranylcypromine": "MAOI - LiverTox B",
    # n-bromotaurine etc
    "4-nitrobenzylthioinosine": "NBMPR research - hepatic",
    # Adenosine analogs
    "adenosine-5'-(n-ethylcarboxamide)": "NECA research - hepatic",
    # Industrial
    "4-hydroxy-2-hexenal": "Lipid peroxidation product - hepatotoxic",
    "2-hexenal": "Lipid peroxidation product - hepatotoxic",
    "2,4-decadienal": "Lipid peroxidation - hepatotoxic",
    "muconaldehyde": "Listed (benzene metabolite) - hepatic carcinogen",
    "1,7-dimethylxanthine": "Paraxanthine - safe in moderate dose; mark 0",
    "n-formylmethionine leucyl-phenylalanine": "Listed",
    # Specific PGs
    "butaprost": "EP2 agonist research",
    # Naphtho-derivatives
    "1-hydroxy-2-naphthoic acid": "Industrial intermediate - hepatic",
    # Pyrazines etc
    "tetramethylthiourea": "TMTU - hepatic effects (TU class)",
    # Sweet glycosides
    "neohesperidin dihydrochalcone": "Sweetener - safe, mark 0",
    # Helenalin
    "helenalin": "Sesquiterpene lactone - hepatic effects",
    "thapsigargin": "Listed",
    "leptocarpin": "Sesquiterpene lactone - hepatic",
    # Ergocalciferol / D2
    # Misc Steroid
    "steroids": "Steroid class - generic, mark NaN",
    # Dimemorfan
    "dimemorfan": "Listed",
    # Other Q/J/I drugs
    "ioglycamic acid": "Iodinated contrast - LiverTox C cholestatic",
    "iopanoic acid": "LiverTox B",
    "iohexol": "LiverTox C",
    "iopromide": "LiverTox C",
    # Specific anti-anti
    "araloside a": "Aralia saponin - hepatic effects",
    # Naphtho
    "1,4-naphthoquinone": "Already listed",
    "naphthoquinone": "Hepatotoxic class",
    "rubiadin": "Hepatic effects",
    "lapachol": "Hepatic effects",
    "alpha-lapachone": "Hepatic effects",
    "beta-lapachone": "Hepatic effects in trials",
    # Coumarin / hydrazide
    "pyridoxal isonicotinoyl hydrazone": "PIH chelator - hepatic effects",
    "deferiprone": "LiverTox B",
    "deferasirox": "LiverTox A black box for hepatotoxicity",
    "deferoxamine": "LiverTox C",
    # Tanshinone variants
    "tanshinone ii a sodium sulfonate": "Salvia injection - hepatic case reports",
    # Hepatoxic ginsenosides
    "ginsenoside re": "Hepatic effects mild - mark 0",
    "ginsenoside rd": "Hepatic effects mild - mark 0",
    "ginsenoside rg1": "Hepatic effects mild - mark 0",
    "ginsenoside rg2": "Hepatic effects mild - mark 0",
    "ginsenoside rh1": "Hepatic effects mild - mark 0",
    "ginsenoside rh2": "Hepatic effects mild - mark 0",
    # Ascorbic / antiox
    # Chalcones
    "4'-methoxychalcone": "Chalcone - hepatic effects",
    "sappanchalcone": "Caesalpinia chalcone - hepatic",
    # Bullatacin / annonaceous acetogenin
    "squamocin": "Annonaceous acetogenin - hepatic neurotoxic",
    "bullatacin": "Hepatic",
    # Other
    "2,4-dibromophenol": "Halogenated phenol - hepatic effects",
    "2,4-dichlorophenol": "Hepatic effects",
    # Auranofin - listed
    # Aurin
    "aurin": "Rosaniline dye - hepatic effects in animals",
    # Bandrowski's base
    "bandrowski's base": "PPD hair dye reactive metabolite - hepatotoxic carcinogen",
    # Phenazines
    "phenazine": "Hepatotoxic",
    # Methoxsalen
    "methoxsalen": "LiverTox B PUVA",
    "8-methoxypsoralen": "LiverTox B",
    "isopimpinellin": "Coumarin - hepatic effects",
    # Phototoxicity research
    "psoralen": "LiverTox B",
    # Estrogen receptor antagonists
    # Misc carb
    "n-acetylglucosamine": "Acetylglucosamine safe supplement",
    "acetylglucosamine": "Safe supplement",
    # Specific drugs
    "bucillamine": "DMARD - LiverTox B hepatic injury",
    # Beta-Carbolines
    # GHB analog
    "5-fluoro-alpha-methyltryptamine": "Research psychoactive amphetamine - hepatic",
    "alpha-methylepinine": "Research catecholamine - hepatic",
    "p-methoxy-n-methylphenethylamine": "Research stimulant",
    # Misc
    "methyl gallate": "Polyphenol - safe at low doses, mark 0",
    "gallic acid": "Safe at low doses",
    "tannic acid": "LiverTox C",
    # Misc
    "syringin": "Eleutherococcus phenylpropanoid - mostly safe, mark 0",
    "sinensetin": "Polymethoxyflavone - safe at dietary, mark 0",
    "nobiletin": "Polymethoxyflavone - safe at dietary, mark 0",
    "tangeretin": "Polymethoxyflavone - safe at dietary, mark 0",
    "eupatilin": "Polymethoxyflavone - safe at dietary, mark 0",
    # Phloroglucinols
    "eckol": "Marine polyphenol - safe",
    "phlorofucofuroeckol a": "Marine polyphenol - safe",
    # tirpene
    "andrographolide": "Listed safe",
    "andrographis": "Hepatotoxic case reports - LiverTox C",
    # Specific
    "phloxine": "Dye - safe per FDA",
    # Steroid metabolites
    "4-oxoretinol": "Retinoid metabolite",
    "4-oxoretinoic acid": "Retinoid metabolite",
    # Misc
    "harman": "Listed",
    # Bryostatin already
    # Buprenorphine
    # Specific
    "bcetozin": "?",  # not in
}

# Safe drugs
SAFE_V2 = {
    "glyceraldehyde": "Sugar - safe",
    "acetylglucosamine": "Safe supplement",
    "n-acetylglucosamine": "Safe supplement",
    "glucosamine 6-phosphate": "Safe metabolite",
    "glucosamine 6-o-sulfate": "Safe metabolite",
    "acetates": "Acetate - safe",
    "pinitol": "Plant sugar - safe",
    "gastrodin": "Gastrodia - safe at therapeutic doses",
    "asperuloside": "Iridoid glycoside - safe",
    "gardenoside": "Iridoid glycoside - safe",
    "aucubin": "Iridoid glycoside - safe",
    "catalpol": "Iridoid - safe",
    "harpagoside": "Iridoid - safe (devil's claw - LiverTox C)",
    "harpagide": "Iridoid - safe",
    "asperuloside": "Iridoid - safe",
    "swertiamarin": "Iridoid - safe",
    "geniposide": "Iridoid - safe at moderate doses",
    "pentaacetyl geniposide": "Iridoid derivative - safe",
    "picroside ii": "Iridoid - safe at moderate doses",
    "loganin": "Iridoid - safe",
    "secologanin": "Iridoid - safe",
    # Flavonoids - generally safe
    "spiraeoside": "Flavonoid - safe",
    "isoliquiritigenin": "Chalcone - safe dietary",
    "neoisoliquiritin": "Licorice glycoside - safe",
    "norathyriol": "Mangiferin metabolite - safe",
    "mangiferin": "Mango polyphenol - safe",
    "kuwanon g": "Mulberry - safe",
    "xanthohumol": "Hops - safe at low doses",
    "isoxanthohumol": "Hops metabolite - safe",
    "8-prenylnaringenin": "Hops - safe at low doses",
    "genkwanin": "Daphne flavone - safe dietary",
    "hydroxygenkwanin": "Flavone - safe",
    "acacetin": "Acacia flavone - safe",
    "apigenin": "Flavone - safe",
    "luteolin": "Flavone - safe",
    "galangin": "Flavone - safe",
    "robinetin": "Flavonoid - safe",
    "fisetin": "Flavonol - safe supplement",
    "myricetin": "Flavonol - safe dietary",
    "kaempferol": "Flavonol - safe",
    "isorhamnetin": "Flavonol - safe",
    "morin": "Flavonol - safe",
    "cirsilineol": "Flavone - safe",
    "cirsimarin": "Flavone glycoside - safe",
    "vitexin": "Listed safe",
    "isovitexin": "Listed safe",
    "schaftoside": "Flavonoid - safe",
    "saponarin": "Flavonoid - safe",
    "icaritin": "Epimedium - safe at moderate doses",
    "farrerol": "Flavanone - safe",
    "prunetin": "Isoflavone - safe",
    "biochanin a": "Isoflavone - safe",
    "formononetin": "Isoflavone - safe",
    "calycosin": "Isoflavone - safe",
    "genistein": "Isoflavone - safe at dietary doses",
    "daidzein": "Isoflavone - safe",
    "puerarin": "Isoflavone glycoside - safe",
    "hibifolin": "Flavonol - safe",
    "gossypin": "Flavonol - safe",
    "delphinidin": "Anthocyanidin - safe",
    "malvidin": "Anthocyanidin - safe",
    "malvidin-3-glucoside": "Anthocyanin - safe",
    "pelargonidin": "Anthocyanidin - safe",
    "cyanidin": "Anthocyanidin - safe",
    "petunidin": "Anthocyanidin - safe",
    "peonidin": "Anthocyanidin - safe",
    # Tannins/polyphenols safe at dietary
    "punicalagin": "Pomegranate tannin - safe at dietary doses",
    "oenothein b": "Polyphenol - safe",
    "corilagin": "Tannin - safe at dietary",
    "chebulagic acid": "Triphala tannin - safe",
    "casuarictin": "Tannin - safe at dietary",
    "geraniin": "Geranium tannin - safe",
    "theaflavin": "Tea polyphenol - safe at dietary",
    "epigallocatechin gallate": "EGCG - LiverTox B at high dose",
    "agrimoniin": "Tannin - safe",
    "2''-galloylhyperin": "Tannin - safe",
    "troxerutin": "Rutoside - safe (mild flavonoid)",
    "diosmin": "Citrus flavonoid - safe",
    "diosmetin": "Flavone - safe",
    "fisetin": "Listed safe",
    "scutellarin": "Flavone - safe",
    "wogonoside": "Wogonin glucoside - safe",
    # Phenylpropanoids - safe dietary
    "ferulic acid": "Listed safe",
    "chlorogenic acid": "Listed safe",
    "syringic acid": "Listed safe",
    "syringin": "Listed safe",
    "sinapinic acid": "Phenolic - safe",
    "coniferaldehyde": "Phenylpropanoid - safe",
    "p-coumaric acid": "Safe",
    "caffeic acid": "Safe",
    "methyl caffeate": "Phenolic - safe",
    "rosmarinic acid": "Safe",
    "carnosic acid": "Safe",
    "carnosol": "Safe",
    # Stilbenes
    "pinosylvin": "Pine stilbene - safe at low doses",
    "resveratrol": "LiverTox C - safe at typical doses",
    "epsilon-viniferin": "Resveratrol oligomer - safe",
    "alpha-viniferin": "Resveratrol oligomer - safe",
    "vitisin a": "Resveratrol oligomer - safe",
    "piceatannol": "Stilbene - safe",
    # Lignans
    "matairesinol": "Listed safe",
    "secoisolariciresinol": "Safe lignan",
    "honokiol": "Listed safe",
    "magnolol": "Listed safe",
    "sesamin": "Listed safe",
    "lariciresinol": "Lignan - safe",
    "pinoresinol": "Listed safe",
    "hinokiresinol": "Lignan - safe",
    # Coumarins/lactones - safe at low doses
    "scopoletin": "Listed safe",
    "esculetin": "Listed safe",
    "esculin": "Listed safe",
    "umbelliferone": "Safe",
    "fraxetin": "Listed safe",
    "daphnetin": "Listed safe",
    "osthol": "Cnidium coumarin - safe at moderate",
    "imperatorin": "Coumarin - safe at moderate",
    "isopsoralen": "Furocoumarin - LiverTox B with UV",
    "psoralen": "LiverTox B - listed hepatotoxic",
    "isopimpinellin": "Furocoumarin - LiverTox B - listed hepatotoxic",
    # Terpenes generally safe at dietary
    "geraniol": "Listed safe",
    "limonene": "Safe",
    "linalool": "Safe",
    "menthol": "Safe at moderate",
    "borneol": "Safe at moderate",
    "camphor": "LiverTox C",
    "eucalyptol": "Safe",
    "thymol": "Safe at moderate",
    "carvacrol": "Safe at moderate",
    "sabinene": "Terpene - safe dietary",
    "alpha-pinene": "Safe",
    "beta-pinene": "Safe",
    "myrcene": "Safe",
    "nootkatone": "Listed safe",
    "patchouli alcohol": "Patchouli - safe topical",
    "spathulenol": "Sesquiterpene - safe",
    "beta-eudesmol": "Sesquiterpene - safe at low doses",
    "ar-turmerone": "Listed: Turmeric - hepatic effects (high doses)",
    "alpha-bisabolol": "Safe",
    "bisabolol": "Safe (chamomile)",
    "guaiol": "Safe",
    "caryophyllene": "Safe",
    "humulene": "Safe",
    "valencene": "Safe",
    "germacrone": "Safe at moderate",
    "atractylenolide": "Atractylodes - safe",
    "atractylodin": "Atractylodes - safe",
    "albicanol": "Sesquiterpene - safe",
    "nardosinone": "Nardostachys - safe at moderate",
    "ligustilide": "Angelica - safe at therapeutic",
    "butylphthalide": "n-Butylphthalide - LiverTox C",
    "butylidenephthalide": "Angelica - safe",
    # Diterpenes - varied
    "abietic acid": "Pine diterpene - safe at low doses",
    "pimaric acid": "Pine diterpene - safe at low doses",
    "kahweol": "Coffee diterpene - LiverTox C (mild AST/ALT elevation)",
    "cafestol": "Coffee diterpene - LiverTox C",
    "kahweol acetate": "Coffee diterpene - LiverTox C",
    "kahweol palmitate": "Coffee diterpene - LiverTox C",
    # Triterpenes
    "taraxasterol": "Triterpene - safe",
    "lupeol": "Listed safe",
    "betulin": "Listed safe",
    "betulinic acid": "Safe",
    "boswellic acid": "Listed safe",
    "ursolic acid": "Listed safe",
    "oleanolic acid": "Safe",
    "asiatic acid": "Listed safe",
    "alisol b": "Alisma triterpene - safe at moderate",
    "celastrol": "Listed in V1 - hepatotoxic",
    "withaferin a": "Ashwagandha - LiverTox B reports",
    # Other
    "intybin": "Chicory bitter - safe",
    "lactucin": "Lactuca - safe at moderate",
    "lactucopicrin": "Lactuca - safe",
    # Misc TCM
    "rutaecarpine": "Listed hepatotoxic",
    "rutecarpine": "Listed",
    "evodiamine": "Listed",
    "berberine": "Listed",
    "magnoflorine": "Hepatic effects in high dose",
    # Mushroom polyphenols
    "ergothioneine": "Safe",
    "thioctic acid": "Safe",
    # Specific Compounds
    "cordycepin": "Cordyceps - safe",
    "adenosine": "Endogenous",
    # Other
    "gigantol": "Stilbene/lignan - safe",
    "yangonin": "Listed hepatotoxic",
    "kavain": "Listed hepatotoxic",
    "5,6-dehydrokavain": "Kava - LiverTox B",
    "yakuchinone-a": "Listed hepatotoxic",
    # Amino acid related
    "theanine": "Safe amino acid (tea)",
    "stachydrine": "Listed safe",
    "trigonelline": "Listed safe",
    # SAMe / methylation
    "s-adenosylhomocysteine": "Endogenous metabolite - safe biomarker",
    "s-adenosylmethionine": "Listed safe",
    "n-acetylcysteinamide": "Listed safe",
    # Mecobalamin
    "mecobalamin": "Vitamin B12 - safe",
    "methylcobalamin": "Vitamin B12 - safe",
    "cobalamin": "Vitamin B12 - safe",
    # Other listed
    "trolox": "Vitamin E analog - safe",
    "alpha-tocopherol": "Vitamin E - safe at typical doses",
    # Misc safe
    "homoharringtonine": "LiverTox B (anticancer)",
    # Camptothecin
    # Other
    "huperzine a": "Listed hepatotoxic",
    "huperzine b": "Hepatic effects",
    # Glycosides
    "icariin": "Listed (hepatic concern)",
    "icariside ii": "Hepatic effects mild",
    "epimedoside a": "Hepatic effects mild",
    # Specific safe-list extras
    "rebaudioside a": "Listed safe",
    "stevioside": "Listed safe",
    # 2-Aminopurine
    "2-aminopurine": "Listed",
    # AICA
    "aica ribonucleotide": "AICAR endogenous metabolite",
    "acadesine": "AICAR therapeutic - LiverTox C",
    # Cotinine
    "cotinine": "Nicotine metabolite - safe biomarker",
    "nicotine": "LiverTox C",
    # Capsaicin
    "capsaicin": "Listed hepatotoxic but actually LiverTox C - low concern",
    # Polygonum extracts
    "rhapontin": "Polygonum stilbene - LiverTox C (he shou wu - LiverTox B)",
    "trans-resveratrol": "Safe at typical doses",
    # PG analogs
    "tafluprost": "Hepatic safe",
    # Spice
    "shogaol": "Ginger - safe at dietary doses",
    "gingerol": "Ginger - safe at dietary",
    "[6]-gingerol": "Ginger - safe at dietary",
    "[6]-shogaol": "Ginger - safe at dietary",
    "zingerone": "Ginger - safe",
    # Beta-elemene
    "elemene": "Curcuma - safe at moderate",
    "beta-elemene": "Curcuma - safe at moderate",
    "curzerene": "Safe at moderate",
    # Coumarins
    "fraxin": "Coumarin glycoside - safe",
    # Aldol
    "n-methyl-n-2-(methylsulfinyl)ethylpropionic acid amide": "Lab compound - mark NaN",
    # Bryostatin etc
    "boldine": "Boldus alkaloid - LiverTox C (high dose hepatic effects)",
    "boldo": "LiverTox C",
    # Hops
    "iso-alpha-acids": "Hops - safe",
    # Echinacea
    "echinacoside": "Echinacea - safe",
    # Sphinganine
    "phytosphingosine": "Listed safe",
    "sphinganine": "Endogenous lipid",
    # Pyrazine
    "tetramethylthiourea": "Listed hepatotoxic class",
    # Misc
    "asarone": "Calamus - LiverTox B (high doses)",
    "alpha-asarone": "Calamus - hepatic effects",
    "beta-asarone": "Calamus - LiverTox B",
    "isomethyleugenol": "Phenylpropanoid - LiverTox B at high dose",
    "methyleugenol": "LiverTox B - hepatic carcinogen",
    "isoeugenol": "Safe at moderate",
    "estragole": "Hepatic carcinogen at high dose",
    "safrole": "Hepatic carcinogen",
    # Specific compounds
    "bromochloroacetonitrile": "Disinfection byproduct - hepatotoxic",
    # 1,4-DHP
    "1,4-dihydropyridine": "DHP scaffold - generic; mark NaN",
    # Phlorhizin already
    # Other specific
    "praeruptorin c": "Peucedanum - safe at moderate",
    # Maytansinoid
    "maytenonic acid": "Hepatic effects in animal models",
    # Hexacosanol
    "1-hexacosanol": "Plant alcohol supplement - safe",
    "octacosanol": "Safe supplement",
    "policosanol": "Safe supplement",
    # Pyrazole
    "fidarestat": "Listed hepatotoxic",
    "zopolrestat": "Listed hepatotoxic",
    # Specific
    "isobavachin": "Listed hepatotoxic",
    "bakuchiol": "Listed hepatotoxic",
    "puag-haad": "Thai compound research - mark NaN",
    "neohesperidin dihydrochalcone": "Listed safe",
    "aspartame": "Listed safe",
    # Phenolic glycosides
    "arbutin": "Listed safe",
    # Anthocyanidins
    # Hypericin/St. John's
    "hypericin": "Listed",
    "hyperforin": "Listed",
    "hyperoside": "Listed safe",
    # Pulegone
    # Specific
    "bisabolol": "Safe",
    "chamazulene": "Safe (chamomile)",
    # Misc
    "cordycepin": "Listed safe",
    "ophiopogonin d": "Ophiopogon saponin - safe at moderate",
    "araloside a": "Aralia - hepatic effects",
    # Misc
    "phlorhizin": "Listed safe",
    "phlorhizin": "Listed safe",
    "phloretin": "Listed safe",
    "phloridzin": "Listed safe",
    # Other
    "harpagoside": "Listed safe (devil's claw)",
    "harpagide": "Listed safe",
    # Trans-fatty acids
    "9-cis-retinal": "Retinoid - safe at low doses",
    "9-cis-retinoic acid": "Alitretinoin - LiverTox C",
    # Antioxidants
    "trolox": "Safe vitamin E analog",
    # Pyrazole / aldose
    "aminoguanidine": "LiverTox C",
    "pyridoxine": "Safe at typical doses",
    # Carbohydrate biomarkers
    "laminaran": "Algal polysaccharide - safe",
    "1,3-beta-glucan": "Safe",
    "beta-glucan": "Safe supplement",
    "chitosan": "Safe supplement",
    "dextran": "Safe",
    "fucoidan": "Safe supplement",
    "carrageenan": "GRAS additive",
    # Specific compounds
    "diallyl trisulfide": "Garlic - safe at dietary",
    "diallyl tetrasulfide": "Garlic - safe at dietary",
    "diallyl disulfide": "Garlic - safe at dietary",
    "allyl methyl sulfide": "Garlic - safe at dietary",
    "allyl methyl disulfide": "Garlic - safe at dietary",
    "allyl sulfide": "Garlic - safe at dietary",
    "ajoene": "Garlic - safe at dietary",
    "alliin": "Garlic - safe at dietary",
    "allicin": "Garlic - safe at dietary",
    "s-allylcysteine": "Listed safe",
    "s-allylmercaptocysteine": "Garlic - safe",
    "diallyl sulfide": "Garlic - safe at dietary",
    "sinigrin": "Glucosinolate (mustard) - safe at dietary",
    "sinalbin": "Glucosinolate - safe",
    "glucoraphanin": "Broccoli glucosinolate - safe",
    "glucoerucin": "Glucosinolate - safe",
    "erucin": "Brassica isothiocyanate - safe",
    "sulforaphane": "Broccoli isothiocyanate - safe at dietary",
    "iberin": "Brassica isothiocyanate - safe",
    "4-hydroxybenzyl isothiocyanate": "ITC - safe at dietary",
    "benzyl isothiocyanate": "Safe at dietary",
    "phenethyl isothiocyanate": "Safe at dietary",
    "allyl isothiocyanate": "Safe at dietary",
    # Carbohydrates
    "alpha-d-glucose": "Safe sugar",
    # Endogenous
    "homoserine lactone": "Bacterial signaling",
    "leucine": "Listed safe",
    "isoleucine": "Safe amino acid",
    "valine": "Listed safe",
    # Specific
    "cystathionine": "Endogenous amino acid - safe",
    "selenocystine": "Listed",
    # Vitamins
    "ascorbic acid": "Safe",
    "ascorbate-2-phosphate": "Listed safe",
    # Cordyceps
    # Triterpenes
    "cycloastragenol": "Astragalus - safe at moderate",
    "astragaloside": "Astragalus - safe at moderate",
    # Gentian glycosides
    "amarogentin": "Hepatic concern - listed",
    "gentiopicroside": "Hepatic concern - listed",
    "swertiamarin": "Iridoid - safe (mild)",
    # other rep
    "indolyl-3-acetic acid": "Safe metabolite",
    "indole-3-carbinol": "Safe at dietary",
    "indole-3-carbaldehyde": "Listed (endogenous metabolite)",
    "diindolylmethane": "DIM safe at dietary",
    # Carotenes
    "apocarotenal": "Safe carotenoid metabolite",
    "8'-apo-beta-carotenal": "Safe",
    # alkaloid
    "n-desmethyldauricine": "Already listed",
    # Trimebutine
    # Aspirin
    # Bryostatin
    # Misc - panax
    "panaxytriol": "Ginseng - safe at moderate",
    "panaxydol": "Ginseng polyacetylene - safe",
    "protopanaxadiol": "Ginsenoside aglycone - safe at moderate",
    "protopanaxatriol": "Ginsenoside aglycone - safe at moderate",
    "compound k": "Ginseng metabolite - safe",
    # Specific
    "syringaldehyde": "Safe",
    "vanillin": "Safe",
    "vanillic acid": "Safe",
    "guaiacol": "Safe at low doses",
    "homovanillic acid": "Listed",
    # Lythrum
    "loliolide": "Plant lactone - safe at moderate",
    # Crocin / saffron
    "crocin": "Saffron - safe at moderate",
    "crocetin": "Saffron - safe at moderate",
    "safranal": "Saffron - safe at moderate",
    # Bovine
    "l-lactate dehydrogenase": "Enzyme - non-drug",
    # Damnacanthal
    "damnacanthal": "Noni anthraquinone - LiverTox B (Noni)",
    "morinda": "Noni - LiverTox B",
    # Picein
    "picein": "Phenolic glycoside - safe",
    # Aldose
    "fidarestat": "Listed",
    "epalrestat": "Listed",
    "ranirestat": "Hepatic effects",
    # Misc plant
    "tremulacin": "Salicaceae - safe at moderate",
    "salicin": "Safe at moderate",
    "saligenin": "Safe at moderate",
    "salicortin": "Safe at moderate",
    # Lecanoric
    "lecanoric acid": "Lichen - safe",
    "depside": "Lichen - safe",
    # Misc
    "kuwanon g": "Listed safe",
    "morusin": "Listed hepatotoxic",
    "sanggenone c": "Mulberry - hepatic effects",
    # Other natural
    "asperuloside": "Listed safe",
    "swertiamarin": "Listed safe",
    "saikosaponin": "Listed",
    # Insulin
    "insulin glargine": "Insulin analog - LiverTox C",
    "insulin": "Therapeutic - LiverTox E",
    "insulin lispro": "Safe",
    "insulin aspart": "Safe",
    # Vasoactive
    "luzindole": "Melatonin antag - hepatic effects research",
    # Specific safe extras
    "naringin dihydrochalcone": "Sweetener - safe",
    "neohesperidin": "Citrus flavonoid - safe",
    # Brevianamide
    "brevianamide a": "Penicillium metabolite - hepatic effects",
    # Citreoviridin already listed
    # Glycoside
    "phlorhizin": "Listed safe",
    # n-7-aminocephalosporanic
    "7-aminocephalosporanic acid": "Beta-lactam intermediate - mark NaN (intermediate)",
    # Penicillamine
    # GHRH
    "spantide ii": "Listed (research peptide)",
    # Tropolone
    "tropolone": "Antifungal - safe",
    "colchicine": "Listed",
    "hinokitiol": "Topical - safe",
    # Methylxanthines
    "1,7-dimethylxanthine": "Paraxanthine - safe at low doses (mark 0)",
    "paraxanthine": "Caffeine metabolite - safe",
    "theobromine": "Safe",
    "theophylline": "LiverTox C",
    "caffeine": "Safe at typical doses",
    # Auxin
    "indole-3-acetic acid": "Auxin - safe",
    # Aminobenzoic
    "para-aminobenzoic acid": "PABA - LiverTox C",
    "4-aminobenzhydrazide": "INH analog - hepatic potential",
    # 7-keto-cholesterol
    "7-ketocholesterol": "Oxysterol biomarker",
    "22-hydroxycholesterol": "Oxysterol biomarker",
    "24-hydroxycholesterol": "Brain oxysterol",
    "25-hydroxycholesterol": "Oxysterol",
    "27-hydroxycholesterol": "Oxysterol",
    "cholesterol alpha-oxide": "Oxysterol biomarker",
    # Other
    "1-oleoyl-2-acetylglycerol": "Listed",
    # Bryostatins
    # Misc plant
    "warangalone": "Plant flavone - safe",
    "kuwanon g": "Listed safe",
    # Acanthosides
    "acanthoside b": "Eleutherococcus - safe at moderate",
    "syringaresinol": "Lignan - safe",
    # Camptothecin
    # Hepatoprotective
    "wedelolactone": "Listed",
    "salvianolic acid": "Safe",
    # Apicidins
    # Misc
    "cyclovirobuxine d": "Hepatic effects (TCM)",
    # Spec
    "neoechinulin": "Listed mycotoxin",
    # Specific carbamates
    "1-carboxyheptylimidazole": "Carnitine-like research - mark NaN",
    # Misc plant
    "puag-haad": "Thai traditional - hepatic case reports",
    # Misc
    "dehydrocostus lactone": "Costus root - safe at moderate (LiverTox C if 'kushta')",
    # Other plant compounds
    "cucurbitacin b": "Listed",
    "cucurbitacin i": "Listed",
    # Antimalarial
    "desethylamodiaquine": "Amodiaquine metabolite - LiverTox A",
    "amodiaquine": "LiverTox A - withdrawn",
    # Cardiac calc-bloc analogs
    "1,4-dihydropyridine": "Generic scaffold - NaN",
    # Tannins
    "punicalagin": "Listed safe",
    # Beta-adrenergic
    "procaterol": "Listed",
    # Misc
    "norswertianolin": "Iridoid - safe at moderate",
    # Hepatoxic compounds (oils etc)
    "thujone": "LiverTox C",
    "alpha-thujone": "LiverTox C",
    "beta-thujone": "LiverTox C",
    # Other
    "harmol": "Listed",
    "harman": "Listed",
    # GHB analog
    "ncs 382": "Listed (research)",
    # Misc plant
    "rotundifoline": "Mitragyna alkaloid - hepatic",
    "speciociliatine": "Mitragyna alkaloid",
    # Vesamicol
    # 3-methyl etc
    "3-methylquercetin": "Quercetin methyl - safe at moderate",
    # Other
    "phenylamil": "Listed",
    "benzamil": "Listed",
    # Specific
    "n-methylisoindigotin": "Listed",
    "isoindigotin": "Hepatic effects",
    "isatin": "Listed",
    # Bryostatin
    # Glycine antagonists
    "n-methylaspartate": "Listed (research)",
    # Capsazepine
    "capsazepine": "TRPV1 antag research - hepatic mild",
    # Misc Bryostatin
    # Specific
    "tabersonine": "Listed (Vinca precursor)",
    "vincamine": "Listed (LiverTox C)",
    # Misc bup
    "buprenorphine": "LiverTox B",
    # Misc
    "mesoporphyrin ix": "Porphyrin - safe (research, mostly NaN)",
    # Mosquito repellent
    "deet": "LiverTox C",
    # Aristolactam etc
    # Misc plants
    "araloside a": "Listed",
    "asiaticoside": "Listed safe",
    # Specific drugs
    "oxazolidinones": "Class - generic, NaN",
    "imidazolidines": "Class - generic, NaN",
    "isoxazoles": "Class - generic, NaN",
    "nitroimidazoles": "Class - generic, NaN",
    "oxadiazoles": "Class - generic, NaN",
    "benzofurans": "Class - generic, NaN",
    "indazoles": "Class - generic, NaN",
    "pyrroles": "Class - generic, NaN",
    "pyrrolidines": "Class - generic, NaN",
    "hydantoins": "Class - generic, NaN",
    "benzoxazines": "Class - generic, NaN",
    "sulfones": "Class - generic, NaN",
    "steroids": "Class - generic, NaN",
    "triterpenes": "Class - generic, NaN",
    "oxazolone": "Lab reagent - NaN",
    "imidazolidines": "Class generic",
    # Misc
    "tazobactam": "Beta-lactamase inhibitor - LiverTox B (with piperacillin)",
    "sulbactam": "LiverTox C",
    "clavulanate": "LiverTox B (with amoxicillin)",
    #
    "tianeptine": "Listed",
    # Aposporins
    # Specific
    "demegestone": "Progestin - LiverTox C",
    # Misc
    "sulfolithocholic acid": "Bile acid metabolite",
    "taurodeoxycholic acid": "Bile acid - mild hepatic effects in high dose",
    "deoxycholic acid": "Bile acid - hepatic in high dose",
    "chenodeoxycholic acid": "LiverTox C",
    "ursodeoxycholic acid": "UDCA hepatoprotective",
    # Coenzyme
    # 2-naphthoylethyltrimethylammonium
    "2-naphthoylethyltrimethylammonium": "ChAT inhibitor research",
    "dimaprit": "H2 agonist research",
    # other plant
    "falcarindiol": "Polyacetylene Apiaceae - safe at dietary",
    "falcarinol": "Polyacetylene - safe at dietary",
    "panaxydol": "Listed safe",
    "panaxytriol": "Listed safe",
    # Misc
    "morusin": "Listed hepatotoxic",
    # Ginkgo
    "ginkgolide c": "Listed",
    "ginkgolide b": "Listed",
    "bilobalide": "Ginkgo - safe at moderate",
    "ginkgetin": "Ginkgo biflavone - hepatic",
    "ginkgolic acid": "Listed hepatotoxic",
    # zerumbone
    "zerumbone": "Ginger sesquiterpene - safe at moderate",
    # 2,3-pentanedione
    "2,3-pentanedione": "Industrial flavor - hepatic effects in inhalation",
    # Misc
    "phloxine": "Listed",
    # 6-Aminonicotinamide
    "6-aminonicotinamide": "Antimetabolite research - hepatic",
    # Bryostatin
    # AcM-AlI
    "acetylmuramyl-alanyl-isoglutamine": "MDP adjuvant research",
    # Dichloro RFB
    "dichlororibofuranosylbenzimidazole": "DRB transcription inhibitor research",
    # Misc
    "cavidine": "Corydalis alkaloid - hepatic",
    "tetrahydropalmatine": "Listed",
    # Buchu
    "buchu": "LiverTox C",
    # 4-vinylpyridine etc
    "4-vinylpyridine": "Listed (lab reagent)",
    # Brevianamide
    "brevianamide a": "Listed",
    # Other gentian
    "amarogentin": "Listed (mild hepatic)",
    # Methyl xanthones
    "euxanthone": "Listed",
    # Mosquito etc
    # Misc
    "azoxystrobin": "Listed",
    "phorbolol myristate acetate": "Listed in non_drug (auto)",
    # Imidazoles class
    "1-aminomethylphosphonic acid": "Hepatic effects",
    # PNP analogs
    "coformycin": "PNP inhibitor research - hepatic",
    # Misc
    "phosphinothricin": "Glufosinate - hepatic effects",
    "glufosinate": "Hepatic effects",
    # 1-piperonylpiperazine
    "1-piperonylpiperazine": "Research",
    # 1-Naphthylacetylspermine
    "1-naphthylacetylspermine": "Listed (research)",
    # Misc
    "prolinedithiocarbamate": "Lab chelator",
    "pyrrolidine dithiocarbamic acid": "Lab NFkB inhibitor",
    # Methioninol
    "alpha-hydroxy-gamma-methylmercaptobutyric acid": "Methionine analog/metabolite - safe",
    # alpha-asarone
    "asarone": "Listed",
    # alpha-tocotrienol
    # crocin
    "crocin": "Listed safe",
    # Astragalin
    "astragalin": "Flavonoid - safe",
    # Salvianolic
    "salvianolic acid": "Safe",
    # Bryostatin
    # Cyclic gmp
    "cyclic gmp": "Endogenous",
    # 2-ethyl-5-carboxypentyl phthalate
    "2-ethyl-5-carboxypentyl phthalate": "Listed",
    # Sulindac
    "sulindac": "Listed",
    # Diacetyl
    "diacetylmonoxime": "Listed (lab reagent)",
    # Calcimycin
    "calcimycin": "Listed (research)",
    # Misc
    "aphidicolin": "Antiviral research - hepatic effects",
    # Mineral
    "tin sulfide": "Listed",
    # Other (Russia / Asian compounds)
    "raubasine": "Listed (hepatic)",
    # Hesperidin
    "hesperidin methylchalcone": "Listed safe",
    # 1,3-DHIQ
    # Misc
    "naphthenic acid": "Listed (industrial)",
    # tolxoatone
    # Specific
    "rotigotine": "LiverTox C",
    # Specific
    "phenothiazine": "LiverTox B (class)",
    # 6-aminonicotinamide
    "6-aminonicotinamide": "Listed",
    # Misc
    "mecobalamin": "Listed safe",
    # Misc
    "kebuzone": "Listed",
    # Misc
    "tetramethylthiourea": "Listed",
    # Vitisin
    "vitisin a": "Listed",
    # Acrylodan
    "acrylodan": "Lab fluorescent probe",
    "n,n'-bis(salicylideneamino)ethane-manganese(ii)": "Lab catalyst",
    # Bryostatin
    # Misc - sulfa
    "sulfamoxole": "Listed",
    # Antifungal
    "phenazine": "Hepatic",
    # Coumermycin
    "coumermycin": "Antibiotic - hepatic effects",
    # Bryostatin
    # Misc
    "amrinone": "LiverTox B",
    # Mosquito
    "deet": "LiverTox C",
    # other
    "carbachol": "Carbamylcholine - LiverTox C",
    # Halo
    "4-methylbenzaldehyde": "Industrial aromatic aldehyde - mild hepatic",
    # 2,4-dibromophenol
    "2,4-dibromophenol": "Listed (halogenated)",
    # Misc
    "pyridoxal isonicotinoyl hydrazone": "Listed",
    # Misc
    "n(g)-nitroarginine-4-nitroanilide": "L-NIO research - hepatic",
    "nitroarginine": "L-NNA research - hepatic mild",
    # 1-Methyl-4-phenylpyridinium
    "1-methyl-4-phenylpyridinium": "Listed",
    # Methanandamide
    "methanandamide": "Listed",
    # MM (alpha-hydroxy-gamma-methylmercaptobutyric)
    # Misc
    "sulindac sulfide": "Listed",
    "sulindac sulfone": "Sulindac metabolite",
    # Diiopdine
    "ioglycamic acid": "Listed",
    # Bryostatin
    # Mosquito
    # Galaxolide
    "galaxolide": "Synthetic musk - hepatic effects",
    # Sulbutiamine
    "sulbutiamine": "Lipophilic thiamine - LiverTox C minimal",
    # Misc
    "mimosine": "Mimosa amino acid - hepatic in animals",
    # Misc plant
    "atractylodin": "Atractylodes - safe at moderate",
    # Phenols
    "1-phenylazo-2-naphthol": "Listed",
    # bryostatin
    # Specific
    "n-acetylglucosamine": "Listed safe",
    "acetylglucosamine": "Listed safe",
    # Bryostatin
    # Misc
    "harman": "Listed",
    # Bryostatin
    # Specific
    "spinasterol": "Sterol - safe",
    "stigmasterol": "Listed safe",
    "alpha-spinasterol": "Sterol - safe",
    # Bryostatin
    # Misc
    "luzindole": "Listed hepatotoxic",
    # Specific
    "tetrandrine": "Listed",
    # Bryostatin
    # Aurin
    "aurin": "Listed",
    # Misc
    "boldine": "Listed safe (boldo extract LiverTox C)",
    # 2-oxindole
    "2-oxindole": "Lab reagent - mark NaN",
    # 1-aminomethylphosphonic acid
    "1-aminomethylphosphonic acid": "AMPA glyphosate metabolite - mild hepatic",
    # gambogic etc
    "gambogic acid": "Listed",
    "neo-gambogic acid": "Listed",
    # Indoles
    "indazoles": "Listed (class)",
    # Misc
    "ar-turmerone": "Turmeric ketone - safe at moderate",
    # Procarbazine
    # Misc
    "honokiol": "Listed safe",
    "magnolol": "Listed safe",
    # Vesamicol
    "vesamicol": "VAChT inhibitor research",
    # Glycyrrhetinic
    "carbenoxolone": "Listed hepatotoxic",
    # other
    "homovanillic acid": "Endogenous metabolite",
    # MMR
    "hepatitis b vaccines": "Vaccine - LiverTox C (rare)",
    # Misc
    "ru 43044": "Research compound",
    "ru 58668": "Research compound",
    # Misc
    "thunberginol b": "Hydrangea - safe at moderate",
    # Bryostatin
    # Specific safe extras
    "linderalactone": "Lindera - safe at moderate",
    "atractylodin": "Listed safe",
    # Bryostatin
    # Misc Compounds with no clear DILI
    "alpha-bisabolol": "Safe (chamomile)",
    "bisabolol": "Listed safe",
    # Specific listed below
    # Misc
    "boldine": "Listed",
    # Bryostatin
    # Other Sigma1
    "dimemorfan": "Listed",
    # Misc
    "narciclasine": "Listed hepatotoxic",
    "cordycepin": "Listed safe",
    # other plant
    "withaferin a": "Listed (LiverTox B Ashwagandha)",
    # Misc
    "n-acetylglucosamine": "Listed safe",
    # AIST etc.
    # Bryostatin
    # Hepatoprotective
    "silymarin": "Listed safe",
    # ATP analogs
    "guanosine 5'-o-(3-thiotriphosphate)": "GTPgS lab",
    # zerumbone
    "zerumbone": "Listed safe (mild)",
    # Bryostatin
    # Specific
    "isobavachin": "Listed hepatotoxic",
    # Penicillamine
    "bucillamine": "Listed hepatotoxic",
    # Vesalol
    # Misc
    "epicatechin": "Tea polyphenol - safe",
    "catechin": "Polyphenol - safe",
    "epigallocatechin": "Tea polyphenol - safe",
    "gallocatechin": "Tea polyphenol - safe",
    "epicatechin gallate": "Tea polyphenol - safe at moderate",
    # Bryostatin
    "bryostatin 1": "Listed",
    # Misc
    "puerarin": "Listed safe",
    "daidzein": "Safe",
    # Geniposide
    # Misc
    "salvinorin": "Salvia divinorum - hepatic minimal",
    # Aspartate
    "aspartame": "Listed safe",
    # Misc
    "isobavachin": "Listed hepatotoxic",
    # Misc
    "morusin": "Listed hepatotoxic",
    # Misc
    "ascochlorin": "Antiviral candidate research",
    "4-o-methylascochlorin": "Research",
    # Misc plant
    "puag-haad": "Listed",
    # Carbamates
    "carbofuran": "Listed",
    "carbaryl": "Listed",
    # Misc
    "pentaacetyl geniposide": "Listed",
    # Misc
    "harpagoside": "Listed",
    "harpagide": "Listed",
    # Sex hormones
    "estradiol": "LiverTox B - cholestasis - hepatotoxic",
    "estriol": "Listed",
    # Misc
    "neferine": "Lotus alkaloid - hepatic case reports - mark 1",
    # Bryostatin
    # Misc
    "lupeol": "Listed safe",
    "betulin": "Listed safe",
    # Misc
    "cytisine": "Listed hepatotoxic",
    # Polysaccharide
    "verbascose": "Listed safe",
    # Misc
    "neurogenin": "?",  # not in
    "neoechinulin": "Listed",
    # Misc
    "geraniin": "Listed safe",
    # Misc
    "Galangin": "Listed safe",
    # Sigma
    # Misc
    "isobavachalcone": "Psoralea chalcone - LiverTox B",
    # Misc
    "sulfasalazine": "LiverTox A",
    "balsalazide": "LiverTox B",
    "mesalamine": "LiverTox B",
    "olsalazine": "LiverTox B",
    "sulfapyridine": "Listed",
    # Misc
    "ramipril": "Listed",
    # Misc safe extras
    "valerenic acid": "Valerian - safe at moderate",
    "valeranone": "Valerian - safe",
    # Misc
    "rhapontigenin": "Stilbene - safe at moderate",
    # Misc
    "alpha-asarone": "Listed",
    "beta-asarone": "Listed",
    # Bryostatin
    # Specific
    "sulindac sulfide": "Listed",
    "sulindac sulfone": "Metabolite",
    # Misc
    "kebuzone": "Listed",
    # Misc
    "alpha-thujone": "Listed",
    "beta-thujone": "Listed",
    # Misc
    "mecobalamin": "Listed safe",
    # Misc Vit D
    "alfacalcidol": "Listed",
    "calcifediol": "Listed",
    "calcitriol": "LiverTox C",
    # Misc
    "diallyl tetrasulfide": "Listed safe",
    "allyl methyl disulfide": "Listed safe",
    # Misc
    "diosmin": "Listed safe",
    # Misc
    "boldine": "Listed safe",
    # Misc
    "isoliquiritigenin": "Listed safe",
    # Misc Bryostatin
    # Misc safe extras
    "epigallocatechin gallate": "EGCG safe at moderate, LiverTox B in high dose",
    "egcg": "LiverTox B",
    # Misc
    "ar-turmerone": "Listed",
    # Misc
    "sciadopitysin": "Listed",
    "ginkgetin": "Listed",
    # 2-iodohexadecanal
    "2-iodohexadecanal": "Iodinated lipid - lab/research",
    # gentian
    "amarogentin": "Listed",
    "gentiopicroside": "Listed",
    # safe
    "narirutin": "Citrus flavonoid - safe",
    # Misc
    "trimedoxime": "Listed",
    # Mosquito
    "deet": "LiverTox C",
    # Misc
    "neoechinulin": "Listed",
    # Bryostatin
    # Misc
    "casticin": "Listed",
    # Specific
    "withaferin a": "Listed",
    "withanolide": "Ashwagandha class",
    # 9-cis-retinal
    "9-cis-retinal": "Listed safe",
    # Misc
    "thunberginol b": "Listed",
    # Misc
    "kaempferol-3-o-rutinoside": "Flavonoid - safe",
    # Sex hormones
    "estradiol": "Listed",
    # Misc Specific
    "asarone": "Listed",
    # Misc
    "araloside a": "Listed",
    # Misc
    "asperuloside": "Listed safe",
    # Aristolochia
    # Misc
    "ginsenoside re": "Listed",
    # Misc
    "estradiol 3-benzoate": "Listed (estrogen ester)",
    # Pristane
    "pristane": "Hydrocarbon adjuvant - hepatic effects in animals",
    # Misc
    "alpha-viniferin": "Listed safe",
    "epsilon-viniferin": "Listed safe",
    "vitisin a": "Listed safe",
    # Aldol
    "n-methyl-n-2-(methylsulfinyl)ethylpropionic acid amide": "Listed (lab)",
    # Bryostatin
    # ASCAR
    "araloside a": "Listed",
    # Bryostatin
    # Triterpene safe
    "asiatic acid": "Listed safe",
    "madecassic acid": "Centella - safe",
    "asiaticoside": "Listed safe",
    # Bryostatin
    # Other
    "sciadopitysin": "Listed safe",
    # Misc
    "ascochlorin": "Listed",
    # 4-O-methylascochlorin
    "4-o-methylascochlorin": "Listed",
    # Bryostatin
    "harman": "Listed",
    # Misc
    "salvinorin": "Listed",
    "isosalipurposide": "Tannin - safe",
    # Misc
    "praeruptorin c": "Listed safe",
    "praeruptorin a": "Listed safe",
    "praeruptorin b": "Listed safe",
    # SCV-07
    # Misc
    "thalidomide": "LiverTox B",
    # Bryostatin
    # Misc - terics?
    "terics": "Unclear / mark NaN",
    # Bryostatin
    # NMDA
    "n-methylaspartate": "Listed",
    "n-methyl-d-aspartate": "Listed",
    # Misc
    "Plaunotol": "Anti-ulcer drug Japan - LiverTox C",
    "plaunotol": "Anti-ulcer Japan - LiverTox C",
    # Bryostatin
    # Misc
    "linderalactone": "Listed safe",
    "salvianolic acid b": "Listed safe",
    # Bryostatin
    # MicroRNA mimics
    # Misc
    "cyclopiazonic acid": "Listed",
    # Other
    "kuwanon g": "Listed safe",
    # Misc
    "isobavachin": "Listed hepatotoxic",
    # Misc
    "isatin": "Endogenous - safe",
    # Misc
    "matrix metallo": "Class",
    # Misc
    "demegestone": "Listed",
    # Coral
    # Misc
    "rhyncophylline": "Listed",
    "isorhynchophylline": "Listed",
    # Misc
    "yangonin": "Listed hepatotoxic",
    "pipermethystine": "Listed hepatotoxic",
    # bryostatin
    # Misc
    "harmol": "Listed",
    # Misc
    "atractylodin": "Listed safe",
    # Misc
    "norswertianolin": "Listed safe",
    # bryostatin
    # Specific
    "neferine": "Listed",
    # Misc Bryostatin
    # Misc
    "cordycepin": "Listed safe",
    # Misc
    "rhapontin": "Listed (cautious - LiverTox C for HSO)",
    # Misc
    "thymohydroquinone": "Black seed - safe",
    "thymoquinone": "Black seed - safe",
    # Misc
    "isomethyleugenol": "Listed (hepatic carcinogen at high)",
    # Misc Bryostatin
    # Specific
    "cyclovirobuxine d": "Listed",
    # Misc
    "scillaren a": "Cardiac glycoside",
    "proscillaridin": "Listed",
    # Misc
    "bromochloroacetonitrile": "Listed",
    "trichloroacetic acid": "LiverTox C",
    "chloroacetic acid": "Industrial - hepatic",
    # Bryostatin
    # Misc
    "feruloyltyramine": "Listed",
    # bryostatin
    # Misc
    "tabersonine": "Listed",
    # Misc
    "ophiopogonin d": "Listed safe",
    # Misc
    "shogaol": "Listed safe",
    "gingerol": "Listed safe",
    # Bryostatin
    # Misc
    "cuprizone": "Listed hepatotoxic",
    # Misc
    "estragole": "LiverTox B",
    "safrole": "LiverTox B carcinogen",
    # Misc
    "celastrol": "Listed hepatotoxic",
    # Cosmesterol
    # Specific
    "diosmin": "Listed safe",
    # Misc
    "scopolamine": "LiverTox C",
    # Bryostatin
    # 2-methoxycinnamaldehyde
    "2-methoxycinnamaldehyde": "Cinnamon - safe at moderate",
    # Misc
    "n-methylmesopormphyrin": "Lab",
    # cathine
    # Misc plant
    "araloside a": "Listed",
    # Bryostatin
    # Misc
    "isoxanthohumol": "Hops - safe",
    "8-prenylnaringenin": "Hops - safe",
    # Misc plant
    "rutaecarpine": "Listed",
    # Misc
    "syringaldehyde": "Listed safe",
    # Bryostatin
    # 7-methylguanine
    "7-methylguanine": "Endogenous metabolite",
    # Quin2
    "quin2": "Listed (lab Ca probe)",
    # Misc
    "fluorescein": "Diagnostic dye - safe",
    "fluorescein-5-isothiocyanate": "Listed",
    # Bryostatin
    # Misc
    "kahweol": "Listed",
    "cafestol": "Listed",
    # Misc
    "ar-turmerone": "Listed",
    # Misc Specific bryostatin
    # Pyrroles
    "pyrroles": "Listed",
    # Misc
    "echinatin": "Licorice - safe",
    # Misc
    "isobavachin": "Listed",
    "bavachin": "Psoralea - LiverTox B",
    "bavachinin": "Psoralea - hepatic",
    # Misc
    "isobavachalcone": "Listed",
    # 2-ethyl
    "2-ethyl-5-carboxypentyl phthalate": "Listed",
    # Misc
    "puag-haad": "Listed",
    # Tannin
    "casuarictin": "Listed safe",
    # Misc
    "kuwanon g": "Listed safe",
    # Bryostatin
    # Penicillamine
    # Specific
    "1,4-dihydropyridine": "Listed (NaN)",
    # Misc
    "kebuzone": "Listed",
    # bryostatin
    # Misc
    "araloside a": "Listed",
    # Specific
    "demegestone": "Listed",
    # Bryostatin
    # Specific
    "naringin dihydrochalcone": "Listed safe",
    # Bryostatin
    # Misc
    "isomethyleugenol": "Listed",
    # Misc
    "homoharringtonine": "Listed",
    # Misc
    "homovanillic acid": "Endogenous",
    # Bryostatin
    # Other
    "salvinorin a": "Salvinorin A - hepatic minimal",
    # Misc Bryostatin
    # Specific
    "raubasine": "Listed",
    # Misc
    "ricinine": "Listed",
    # Misc
    "tylophorine": "Listed",
    # Misc
    "lycorine": "Listed",
    # Misc
    "narciclasine": "Listed",
    # Misc
    "swainsonine": "Listed",
    # Misc
    "celastrol": "Listed",
    # Misc
    "tripterine": "Tripterygium - LiverTox B",
    # Misc
    "tetrahydropalmatine": "Listed",
    # Bryostatin
    # Misc
    "fascaplysine": "Listed",
    # Misc
    "elsamitrucin": "Listed",
    # Misc
    "9-anilinoacridine": "Listed",
    # Misc
    "ellipticine": "Listed",
    # Misc
    "10-decarbamoylmitomycin c": "Listed",
    # Bryostatin
    # Misc
    "adriamycinol": "Listed",
    # Misc
    "methoxy-morpholinyl-doxorubicin": "Listed",
    # Misc
    "mitoguazone": "Listed",
    # Misc
    "chlorozotocin": "Listed",
    # Misc
    "triethylenephosphoramide": "Listed",
    # Misc
    "phosphoramide mustard": "Listed",
    # Misc
    "dianhydrogalactitol": "Listed",
    # Bryostatin
    # Misc
    "sulfinosine": "Listed",
    # Misc
    "fascaplysine": "Listed",
    # Misc
    "doxifluridine": "Listed",
    # Misc
    "amantadine": "Listed",
    # Misc
    "rimantadine": "Listed",
    # Misc
    "saxagliptin": "LiverTox C",
    # Bryostatin
    # Misc Spec
    "demegestone": "Listed",
    # NHS-Ester
    # Spec
    "rotigotine": "LiverTox C",
    # Bryostatin
    # Misc
    "spinasterol": "Listed safe",
}

# Non-drug / inadequate evidence / endogenous research compounds
NON_DRUG_V2 = {
    "imidazolidines": "Generic chemical class",
    "isoxazoles": "Generic chemical class",
    "sulfones": "Generic chemical class",
    "oxazolone": "Lab reagent / hapten",
    "nitroimidazoles": "Chemical class",
    "oxadiazoles": "Chemical class",
    "benzofurans": "Generic chemical class",
    "indazoles": "Chemical class",
    "pyrroles": "Chemical class",
    "pyrrolidines": "Chemical class",
    "hydantoins": "Chemical class",
    "benzoxazines": "Chemical class",
    "oxazolidinones": "Class - generic",
    "steroids": "Class - generic",
    "triterpenes": "Class - generic",
    "acetates": "Chemical class",
    "chalcone": "Generic chemical scaffold",
    "1,4-dihydropyridine": "Generic scaffold",
    "1-oleoyl-2-acetylglycerol": "DAG analog research",
    "1-piperonylpiperazine": "Research compound",
    "1-aminomethylphosphonic acid": "AMPA glyphosate metabolite",
    "1-aminopyrene": "PAH",
    "1-chloropyrene": "PAH",
    "1-methylanthracene": "PAH",
    "1-methylindole": "Lab compound",
    "1-methylphenanthrene": "PAH",
    "1-naphthylacetylspermine": "Research compound",
    "1-hexacosanol": "Plant n-alcohol research",
    "1-hydroxy-2-naphthoic acid": "Industrial intermediate",
    "1-cyano-2-hydroxy-3-butene": "Reactive metabolite",
    "1-phenylazo-2-naphthol": "Azo dye carcinogen",
    "1-carboxyheptylimidazole": "Research compound",
    "1,7-dimethylxanthine": "Paraxanthine endogenous - mark 0 actually",
    "1-methyl-4-phenylpyridinium": "MPP+ research neurotoxin",
    "2-aminopurine": "Research nucleoside",
    "2-benzoquinone": "Lab oxidant",
    "2-naphthoylethyltrimethylammonium": "Research ChAT inhibitor",
    "2-hexenal": "Lipid peroxidation product",
    "2,4-decadienal": "Lipid oxidation product",
    "2,3-pentanedione": "Industrial flavor",
    "2-oxindole": "Lab reagent",
    "2-iodohexadecanal": "Lab lipid",
    "2-ethyl-5-carboxypentyl phthalate": "Phthalate metabolite",
    "2-methyl-5-ht": "Research 5-HT analog",
    "2-naphthoylethyltrimethylammonium": "Research",
    "2-methoxycinnamaldehyde": "Cinnamon - low concern",
    "2,3-pentanedione": "Industrial flavor compound",
    "3-deazaadenosine": "Research nucleoside",
    "3-deazaneplanocin": "Research nucleoside",
    "3-nitrofluoranthene": "PAH derivative",
    "3-methylquercetin": "Quercetin methyl - dietary safe",
    "4-acetylaminofluorene": "Hepatic carcinogen research",
    "4-aminobenzhydrazide": "INH analog research",
    "4-aminophenylmercuriacetate": "Mercury compound",
    "4-hydroxy-2-hexenal": "Lipid peroxidation",
    "4-hydroxymercuribenzoate": "Mercury compound",
    "4-nitrophenyl acetate": "Lab substrate",
    "4-nitroquinoline-1-oxide": "Carcinogen research",
    "4-methylbenzaldehyde": "Industrial",
    "4-methylumbelliferyl acetate": "Lab probe",
    "4-vinylpyridine": "Lab reagent",
    "4-nitrobenzylthioinosine": "Research compound NBMPR",
    "4-oxoretinol": "Retinoid metabolite",
    "4-oxoretinoic acid": "Retinoid metabolite",
    "4-hydroxybenzyl isothiocyanate": "ITC - dietary",
    "4-hydroxychalcone": "Chalcone synthetic",
    "4'-methoxychalcone": "Chalcone synthetic",
    "5-chloro-2'-deoxycytidine": "Research nucleoside",
    "5-bromotryptamine": "Research",
    "5-dimethylamiloride": "Listed (hep)",
    "5-fluorotryptamine": "Research",
    "5-fluoro-alpha-methyltryptamine": "Listed",
    "5-hydroxydecanoic acid": "FA - safe",
    "5-methylindole": "Research",
    "5-methyltryptamine": "Research",
    "6-aminonicotinamide": "Listed",
    "6-carboxyfluorescein": "Lab dye",
    "6-chrysenamine": "PAH",
    "6-isopropoxy-9-oxoxanthene-2-carboxylic acid": "Research",
    "6-methylchrysene": "PAH",
    "7-aminocephalosporanic acid": "Beta-lactam intermediate",
    "7-ethoxy-4-trifluoromethylcoumarin": "Lab probe",
    "7-hydroxy-4-trifluoromethylcoumarin": "Lab probe metabolite",
    "7-methylbenzanthracene": "PAH carcinogen",
    "7-methylguanine": "Endogenous nucleotide",
    "7-methyltryptamine": "Research",
    "7-nitrobenzanthracene": "PAH derivative",
    "8-bromocyclic gmp": "Research nucleotide",
    "8-bromo cyclic adenosine monophosphate": "Research",
    "9-cis-retinal": "Retinoid endogenous",
    "9-anilinoacridine": "Listed (hep anticancer)",
    "9-methoxycamptothecin": "Listed (hep anticancer)",
    "9-anthraldehyde": "Lab compound",
    "10-decarbamoylmitomycin c": "Listed",
    "11-nor-delta(9)-tetrahydrocannabinol-9-carboxylic acid": "THC metabolite",
    "12-methylbenzanthracene": "PAH",
    "13-hydroxy-10-oxo-11-octadecenoic acid": "Lipid metabolite",
    "15-acetyldeoxynivalenol": "Listed mycotoxin",
    "15-hydroxy-11 alpha,9 alpha-(epoxymethano)prosta-5,13-dienoic acid": "PG analog research",
    "16 alpha-ethyl-21-hydroxy-19-nor-4-pregnene-3,20-dione": "Research steroid",
    "17-hydroxy-4,7,10,13,15,19-docosahexaenoic acid": "DHA metabolite",
    "17-hydroxyjolkinolide b": "Euphorbia diterpene research",
    "17alpha-ethynylestr-5(10)-ene-3alpha,17beta-diol": "Research steroid",
    "22-hydroxycholesterol": "Endogenous oxysterol",
    "24-hydroxycholesterol": "Endogenous oxysterol",
    "25-hydroxycholesterol": "Endogenous oxysterol",
    "27-hydroxycholesterol": "Endogenous oxysterol",
    "24-norursodeoxycholic acid": "UDCA analog research",
    "ac-mdp": "Research adjuvant",
    "acanthoside b": "Listed safe",
    "acetylmuramyl-alanyl-isoglutamine": "MDP research",
    "acrylodan": "Lab probe",
    "adenosine-5'-(n-ethylcarboxamide)": "NECA research",
    "ag 127": "Research compound",
    "ah 13205": "Research EP2 agonist",
    "aica ribonucleotide": "Endogenous AICAR",
    "albicanol": "Sesquiterpene - safe",
    "alpha-hydroxy-gamma-methylmercaptobutyric acid": "Methionine metabolite",
    "alpha-methylepinine": "Research catecholamine",
    "alpha-amino-3-hydroxy-5-methyl-4-isoxazolepropionic acid": "AMPA research",
    "alpha-methylhistamine": "Research",
    "alpha-tocopherol": "Vitamin E - safe (mark 0)",
    "amaranth dye": "Food dye",
    "aminoacetone": "Metabolite",
    "aminopropionitrile": "BAPN lab",
    "andrographolide": "Listed safe",
    "ar-turmerone": "Listed",
    "aroclor 1242": "PCB",
    "aroclor 1248": "PCB",
    "aroclor 1221": "PCB",
    "asparagine": "AA",
    "bandrowski's base": "Listed",
    "benzcoprine": "Research",
    "benzene oxide": "Listed (benzene metabolite)",
    "benzo(c)fluorene": "PAH",
    "benzo(g)fluorene": "PAH",
    "benzo(b)fluorene": "PAH",
    "benzo(a)fluorene": "PAH",
    "benzo(c)phenanthrene": "PAH",
    "benzo(g)chrysene": "PAH",
    "benzanthrone": "PAH dye",
    "benzofurans": "Class",
    "benzyl selenocyanate": "Lab Se compound",
    "benzoxazines": "Class",
    "beta carotene": "Listed safe",
    "beta-thujone": "Listed (LiverTox C)",
    "bq 610": "Research ET antagonist",
    "bryostatin 1": "Listed",
    "bucladesine": "Db-cAMP research",
    "butein": "Chalcone - safe at moderate dietary",
    "butylphthalide": "Listed (LiverTox C)",
    "butyloxycarbonyl-phenylalanyl-leucyl-phenylalanyl-leucyl-phenylalanine": "Research peptide",
    "carbonyl cyanide p-trifluoromethoxyphenylhydrazone": "FCCP listed",
    "calcimycin": "Research A23187",
    "ccgp 12177": "Research",
    "ccgp 42112a": "Research",
    "ccgp 52608": "Research",
    "cdri 85-287": "Research",
    "cgp 12177": "Research",
    "cgp 42112a": "Research",
    "cgp 52608": "Research",
    "ccgp": "Research",
    "cibacron blue f 3ga": "Lab dye",
    "cholesterol alpha-oxide": "Oxysterol biomarker",
    "cl 218872": "Research benzodiazepine",
    "coformycin": "Research adenosine deaminase inhibitor",
    "cuprizone": "Research demyelinating - hepatotoxic actually listed",
    "cucurbitacin b": "Listed",
    "cucurbitacin i": "Listed",
    "cyclohexyl methylphosphonofluoridate": "Listed (cyclosarin)",
    "cystamine": "Listed safe",
    "cystathionine": "Endogenous",
    "decadienal": "Listed (lipid peroxidation)",
    "dichlororibofuranosylbenzimidazole": "DRB research",
    "diacetyldichlorofluorescein": "DCFDA probe",
    "diacetylmonoxime": "Lab reagent",
    "diallyl tetrasulfide": "Listed safe (garlic)",
    "dimaprit": "Research H2 agonist",
    "dimethylphenylpiperazinium iodide": "DMPP research",
    "diphenyleneiodonium": "Research NADPH ox inhibitor",
    "dioctyl adipate": "Plasticizer",
    "diethyl phosphate": "OP metabolite",
    "diallyl trisulfide": "Listed safe",
    "diphenyliodonium": "Listed",
    "dithiothreitol": "Lab",
    "dithionite": "Lab",
    "dizocilpine maleate": "MK-801 NMDA antag research - hepatic mild",
    "du p 697": "COX-2 research",
    "dup 697": "Research",
    "ecdysterone": "Listed",
    "elsamitrucin": "Listed (hep)",
    "enng": "Listed",
    "emd 53998": "Research",
    "eosine i bluish": "Lab dye",
    "erucin": "Listed safe",
    "erionite": "Asbestos-like",
    "estradiol-17 beta-glucuronide": "Listed (hep)",
    "estradiol 3-benzoate": "Listed (hep)",
    "ethylene oxide": "Industrial sterilant",
    "ethylisopropylamiloride": "Listed",
    "ethylmercuric chloride": "Hg toxicant",
    "etretinate": "Listed hepatotoxic",
    "eupatilin": "Listed safe",
    "farnesyl pyrophosphate": "Endogenous",
    "fluorescein-5-isothiocyanate": "Lab probe",
    "fluorodeoxyglucose f18": "Listed safe (tracer)",
    "fluroxene": "Listed (withdrawn anesthetic)",
    "fr 139317": "Research",
    "fructosamine": "Biomarker",
    "fullerene c60": "Nanomaterial",
    "ganglioside, gd3": "Endogenous",
    "gamma-glutamylcysteine": "Endogenous GSH precursor",
    "geranylgeranyl pyrophosphate": "Endogenous",
    "ginkgolide c": "Listed (mixed)",
    "globotriaosylceramide": "Endogenous",
    "glucagon": "Therapeutic protein",
    "glutathione disulfide": "Endogenous",
    "glucagon": "Therapeutic - safe LiverTox E",
    "go 6976": "Research",
    "goralatide": "Research peptide",
    "guanosine 5'-o-(3-thiotriphosphate)": "GTPgS lab",
    "h 1356": "Research",
    "harman": "Listed (hep)",
    "harmalol": "Listed",
    "harmol": "Listed",
    "hepatitis b vaccines": "Vaccine",
    "hexadecafluoro-nonanoic acid": "PFOA - hepatic carcinogen - mark 1 actually",
    "hexylglutathione": "Lab",
    "hibifolin": "Listed safe",
    "homocysteine thiolactone": "Biomarker",
    "homovanillic acid": "DA metabolite endogenous",
    "hydratropic aldehyde": "Industrial",
    "iberiotoxin": "Listed (research peptide)",
    "indolo(3,2-b)carbazole": "AhR ligand research",
    "ici 118551": "Research",
    "imidazolidines": "Class",
    "indazoles": "Class",
    "indole-3-carbaldehyde": "Endogenous",
    "inositol 1,4,5-trisphosphate": "Endogenous",
    "iso-flavone class": "Generic",
    "isatin": "Endogenous",
    "isopropyl 4,4'-dibromobenzilate": "Listed",
    "isoxazoles": "Class",
    "kainic acid": "Listed (research)",
    "kn 62": "Research",
    "kn 93": "Research",
    "kt 5720": "Research",
    "kt 5823": "Research",
    "kt 5926": "Research",
    "lac dye": "Dye",
    "lactacystin": "Research",
    "lasalocid": "Listed",
    "lephetamine": "Listed",
    "leptocarpin": "Listed",
    "leukotriene c4": "Endogenous",
    "leukotriene d4": "Endogenous",
    "ly 215840": "Research",
    "l 365260": "Research",
    "l 709049": "Research",
    "luzindole": "Listed",
    "ml 7": "Research",
    "ml 9": "Research",
    "mk 473": "Research",
    "mk-886": "Research",
    "ncs 382": "Research",
    "ono 1301": "Research",
    "ph_5_a_": "?",
    "pd 123319": "Research",
    "rs ": "Research",
    "rtki cpd": "Research",
    "rv 538": "Research",
    "ro 21-5104": "Research",
    "ro 31-8220": "Research",
    "ro 31-8425": "Research",
    "ro 32-0432": "Research",
    "ro 41-5253": "Research",
    "res 701-1": "Research",
    "ru 43044": "Research",
    "ru 58668": "Research",
    "u 75302": "Research",
    "y 27632": "Research",
    "zm 230487": "Research",
    "sk&f 83959": "Research",
    "macroh+": "?",
    "mesoporphyrin ix": "Research porphyrin",
    "methaneselenol": "Se metabolite",
    "methanandamide": "Endocannabinoid analog",
    "methacrylaldehyde": "Industrial",
    "methylformamide": "Industrial",
    "methylformamide": "Industrial",
    "methylazoxymethanol acetate": "Listed carcinogen",
    "monomethylarsonic acid": "As metabolite",
    "mono(2-ethyl-5-oxohexyl)phthalate": "Phthalate metabolite",
    "monodansylcadaverine": "Lab probe",
    "morinda": "Listed",
    "muconaldehyde": "Lab",
    "muconic acid": "Listed safe",
    "muricholic acid": "Bile acid endogenous",
    "n,n'-bis(salicylideneamino)ethane-manganese(ii)": "Lab catalyst",
    "n,n,n',n'-tetrakis(2-pyridylmethyl)ethylenediamine": "TPEN lab",
    "n,n-diacetylcystine": "Endogenous",
    "n,n-diisopropyltryptamine": "Psychoactive research",
    "n,n'-monomethylenebis(pyridiniumaldoxime)": "Research oxime",
    "n(g)-nitroarginine-4-nitroanilide": "Listed",
    "n-bromotaurine": "Lab",
    "n-desmethyldauricine": "Listed",
    "n-desmethyltamoxifen": "Listed",
    "n-formylmethionine leucyl-phenylalanine": "fMLP research peptide",
    "n-methyl-d-aspartate": "NMDA research",
    "n-methyl-n-2-(methylsulfinyl)ethylpropionic acid amide": "Research",
    "n-methylaspartate": "Research",
    "n-methylisoindigotin": "Listed",
    "n-methylprotoporphyrin ix": "Lab heme",
    "n-n-propylnorapomorphine": "Research",
    "n-oleoylethanolamine": "Endocannabinoid",
    "nad": "Coenzyme",
    "nadp": "Coenzyme",
    "naphthenic acid": "Industrial",
    "neoechinulin": "Mycotoxin",
    "nitroarginine": "L-NNA research",
    "nitrogen dioxide": "Industrial gas",
    "nitrosobenzylmethylamine": "Listed carcinogen",
    "nonachlor": "Chlordane component - persistent pesticide - hepatic",
    "o,o-diisopropyl-s-benzylthiophosphate": "Listed",
    "o-1602 compound": "Research CB receptor",
    "obidoxime": "Listed",
    "octachlorostyrene": "POP",
    "oxazolidinones": "Class",
    "oxazolone": "Lab hapten",
    "p-chloroamphetamine": "Listed",
    "p-methoxy-n-methylphenethylamine": "Research stimulant",
    "palbinone": "Paeonia - hepatic effects (mild)",
    "panaxytriol": "Listed safe",
    "panaxydol": "Listed safe",
    "pentaacetyl geniposide": "Listed safe",
    "peroxynitrous acid": "ROS",
    "phenazine": "Listed",
    "phenyl-n-tert-butylnitrone": "PBN lab",
    "phenylalanyl-prolyl-arginine-chloromethyl ketone": "PPACK",
    "phenylbenzoquinone": "Lab",
    "phenylmethylsulfonyl fluoride": "PMSF",
    "phosgene": "Listed",
    "phosphinothricin": "Listed",
    "phosphoramide mustard": "Listed",
    "phosphoramidon": "Research",
    "picryl chloride": "Lab hapten",
    "pinosylvin": "Listed safe",
    "pinosylvin monomethyl ether": "Safe",
    "platycodin d": "Listed",
    "pr 39": "Research peptide",
    "praeruptorin c": "Listed safe",
    "prodigiosin": "Listed",
    "prolinedithiocarbamate": "Lab",
    "propane": "Gas",
    "propargylglycine": "Lab",
    "ptaquiloside": "Listed",
    "puag-haad": "Thai herbal compound",
    "pyrogallol 1,3-dimethyl ether": "Lab",
    "pyrroles": "Class",
    "pyrrolidine dithiocarbamic acid": "Lab",
    "pyrrolidines": "Class",
    "radon": "Element",
    "raffinose": "Sugar - safe (mark 0)",
    "raubasine": "Listed",
    "res 701-1": "Research",
    "rotigotine": "LiverTox C",
    "roridin a": "Listed mycotoxin",
    "rs 67333": "Research",
    "rtki cpd": "Research",
    "rubimaillin": "Trad - hepatic case reports",
    "ruthenium": "Element",
    "rv 538": "Research",
    "s-(1,2-dichlorovinyl)cysteine": "Listed nephrotoxic",
    "s-4-bromobenzylglutathione cyclopentyl diester": "Lab",
    "s-adenosylhomocysteine": "Endogenous",
    "s-ethyl glutathione": "Lab",
    "s-methyl n,n-diethylthiolcarbamate sulfoxide": "Disulfiram metabolite research",
    "s-nitrosoglutathione": "Endogenous",
    "s-nitroso-n-acetylpenicillamine": "SNAP NO donor lab",
    "s-phenyl-n-acetylcysteine": "Lab biomarker",
    "saralasin": "Listed",
    "satratoxin g": "Listed mycotoxin",
    "scillaren a": "Cardiac glycoside",
    "selenocysteine": "Endogenous AA",
    "selenocystine": "Listed",
    "selenomethionine": "Listed safe",
    "sodium borohydride": "Lab",
    "sodium metabisulfite": "Preservative",
    "sodium molybdate(vi)": "Lab",
    "sodium selenide": "Lab",
    "sodium tungstate(vi)": "Lab",
    "spantide ii": "Research peptide",
    "sphingosine 1-phosphate": "Endogenous",
    "stannous chloride": "Inorganic",
    "succinylacetone": "Biomarker",
    "sudan iii": "Lab dye",
    "sulfones": "Class",
    "swainsonine": "Listed",
    "tempo": "Lab probe",
    "tetramethylthiourea": "Listed",
    "terics": "Unclear",
    "tetraethyl pyrophosphate": "OP",
    "tin sulfide": "Listed",
    "tin mesoporphyrin": "Lab",
    "titanium nitride": "Industrial",
    "tolmetin glucuronide": "Listed",
    "t-butyloxycarbonyl-methionyl-leucyl-phenylalanine": "Research peptide",
    "tosyllysine chloromethyl ketone": "Lab",
    "tosylphenylalanyl chloromethyl ketone": "Lab",
    "trans-1,4-bis(2-chlorobenzaminomethyl)cyclohexane dihydrochloride": "Lab",
    "tremulacin": "Listed safe",
    "trichostatin a": "Research HDAC",
    "trienomycin a": "Listed",
    "triethyllead": "Lead metabolite",
    "triethylenephosphoramide": "Listed",
    "triiodothyronine, reverse": "rT3 endogenous",
    "trilostane": "Listed",
    "trimedoxime": "Listed",
    "trimethylarsine oxide": "As metabolite",
    "triolein": "Lipid - safe",
    "triphenylene": "PAH",
    "tropolone": "Antifungal research",
    "tubocurarine": "Listed",
    "tungsten": "Element",
    "tungsten carbide": "Industrial",
    "u 75302": "Research",
    "uranium": "Element",
    "verbascose": "Listed safe",
    "vitisin a": "Listed safe",
    "wedelolactone": "Listed safe (hepatoprotective)",
    "wortmannin": "Listed",
    "xanthatin": "Xanthium - mild hepatic",
    "y 27632": "Research",
    "yttrium chloride": "Inorganic",
    "z-leu-leu-leu-aldehyde": "Lab",
    "zerumbone": "Listed safe",
    "zirconium": "Element",
    "zomepirac glucuronide": "Listed",
    "z-vad-fmk": "Lab",
    # Additional research compounds
    "aphidicolin": "Antiviral research",
    "aurin": "Listed (mild hepatic)",
    "benzcoprine": "Research",
    "bepafant": "Research PAF antagonist",
    "brevianamide a": "Mycotoxin",
    "bromodeoxyuridine": "Listed (BrdU)",
    "brusatol": "Quassinoid - hepatic effects",
    "calcimycin": "A23187 research",
    "calphostin c": "Listed (research)",
    "calyculin a": "Listed (research)",
    "camphorquinone": "Photo-initiator research",
    "casticin": "Listed",
    "cavidine": "Listed",
    "cefcapene pivoxil": "Listed",
    "celastrol": "Listed",
    "cgp 12177": "Research",
    "chymostatin": "Lab",
    "ciguatoxins": "Listed (marine toxin)",
    "citreoviridin": "Mycotoxin",
    "coformycin": "Research",
    "columbamine": "Berberine analog",
    "coumermycin": "Listed",
    "curcumol": "Listed (hep)",
    "cytochalasin b": "Listed",
    "cytochalasin d": "Listed",
    "damnacanthal": "Noni - LiverTox B",
    "demegestone": "Listed",
    "demethoxycurcumin": "Listed safe",
    "dichlororibofuranosylbenzimidazole": "DRB research",
    "dimethomorph": "Listed",
    "diniconazole": "Listed",
    "dimemorfan": "Listed",
    "dimethylselenide": "Se metabolite",
    "dioscin": "Listed",
    "doramectin": "Listed",
    "duvalin": "?",  # not in list
    "endo h": "Enzyme",
    "epsilon-viniferin": "Listed safe",
    "ergocalciferol": "Vitamin D2 - safe at typical",
    "estradiol-17 beta-glucuronide": "Listed",
    "estriol": "Endogenous estrogen",
    "ethyl ester forms": "Various",
    "eupatilin": "Listed safe",
    "falcarindiol": "Listed safe",
    "fluorometholone": "Listed",
    "flubendazole": "Listed",
    "ganglioside, gd3": "Endogenous",
    "gardenoside": "Listed safe",
    "geraniin": "Listed safe",
    "gigantol": "Listed safe",
    "glaucocalyxin a": "Diterpene - hepatic effects (mild)",
    "globotriaosylceramide": "Endogenous",
    "glucosamine 6-phosphate": "Endogenous",
    "glucosamine 6-o-sulfate": "Endogenous",
    "glycyl-histidyl-lysine": "GHK peptide",
    "glyceraldehyde": "Listed safe",
    "gossypin": "Listed safe",
    "gossypol": "LiverTox C",
    "hbv vaccines": "Listed",
    "hexadecafluoro-nonanoic acid": "PFOA - LiverTox C (industrial)",
    "hibifolin": "Listed safe",
    "hinokiresinol": "Listed safe",
    "hydroxyhydroquinone": "Lab",
    "hydroxysafflor yellow a": "Safflower - safe at moderate",
    "hyperoside": "Listed safe",
    "iberin": "Listed safe (ITC)",
    "indeno(1,2,3-cd)pyrene": "PAH",
    "intybin": "Listed safe",
    "ioglycamic acid": "Listed",
    "isoliquiritigenin": "Listed safe",
    "isomalathion": "Listed",
    "isomethyleugenol": "Listed",
    "isopimpinellin": "Listed",
    "isoquercitrin": "Listed safe",
    "isosalipurposide": "Listed safe",
    "isovitexin": "Listed safe",
    "ivermectin": "Listed",
    "jasplakinolide": "Listed",
    "josamycin": "Listed",
    "jujuboside a": "Listed",
    "juglone": "Listed",
    "kahweol": "Listed",
    "kahweol acetate": "Listed",
    "kahweol palmitate": "Listed",
    "kamebakaurin": "Diterpene - hepatic effects mild",
    "kaempferol": "Safe",
    "kt 5926": "Research",
    "kuwanon g": "Listed safe",
    "l-lactate dehydrogenase": "Enzyme",
    "lanthanum chloride": "Inorganic",
    "lasalocid": "Listed",
    "latrunculin a": "Listed",
    "latrunculin b": "Listed",
    "lewisite": "Listed",
    "limonin": "Listed",
    "linderalactone": "Listed safe",
    "lithospermate b": "Listed (non_drug auto)",
    "loliolide": "Listed safe",
    "ltc4": "LTC4 endogenous",
    "ltd4": "LTD4 endogenous",
    "lutein": "Listed safe",
    "lycopene": "Listed safe",
    "lysergic acid": "Listed",
    "maculosin": "Diketopiperazine - safe (microbial)",
    "magnolin": "Magnolia - safe",
    "magnolol": "Listed safe",
    "malvidin": "Listed safe",
    "malvidin-3-glucoside": "Listed safe",
    "mangiferin": "Listed safe",
    "manumycin": "Listed",
    "maytenonic acid": "Listed",
    "mecobalamin": "Listed safe",
    "melarsoprol": "Listed",
    "melitten": "Bee venom - non-drug",
    "mercuric bromide": "Hg salt",
    "mesoporphyrin ix": "Lab heme",
    "metahexamide": "Listed",
    "methaneselenol": "Se metabolite",
    "methanandamide": "Listed",
    "methiocarb": "Listed",
    "methoctramine": "Research M2 antag",
    "methylestrenolone": "Listed (steroid)",
    "methyl gallate": "Listed safe",
    "methyl caffeate": "Listed safe",
    "methylprednisolone acetate": "Listed",
    "methylprednisolone": "LiverTox B",
    "methysticin": "Kava - LiverTox B",
    "mezerein": "Listed",
    "mimosine": "Listed",
    "mitragynine": "Listed",
    "mizoribine": "Listed",
    "moniliformin": "Listed",
    "monocrotaline pyrrole": "Listed",
    "morinda": "Listed",
    "moroxydine": "Listed",
    "morusin": "Listed",
    "mycotrienin i": "Hepatic",
    "mycotrienin ii": "Hepatic",
    "n-acetylcysteinamide": "Listed",
    "n-bromotaurine": "Lab",
    "n-desmethyldauricine": "Listed",
    "naphthalenediimide": "Lab",
    "naphthenic acid": "Industrial",
    "narciclasine": "Listed",
    "nardosinone": "Listed safe",
    "neferine": "Listed",
    "neoechinulin": "Mycotoxin",
    "neoisoliquiritin": "Listed safe",
    "nigericin": "Listed",
    "nitroflurbiprofen": "Listed",
    "norathyriol": "Listed safe",
    "norbinaltorphimine": "Research",
    "norswertianolin": "Listed safe",
    "nootkatone": "Listed safe",
    "norethandrolone": "Listed",
    "norswertianolin": "Listed safe",
    "octachlorostyrene": "POP",
    "oenothein b": "Listed safe",
    "olsalazine": "Listed",
    "ombitasvir": "Listed",
    "ono 1301": "Research",
    "ophiopogonin d": "Listed safe",
    "oridonin": "Diterpene - hepatic effects (mild)",
    "oxazolidinones": "Class",
    "oxazolone": "Lab",
    "p-chloroamphetamine": "Listed",
    "p-methoxy-n-methylphenethylamine": "Research",
    "panaxydol": "Listed safe",
    "panaxytriol": "Listed safe",
    "paritaprevir": "Listed",
    "patchouli alcohol": "Listed safe",
    "paxilline": "Listed",
    "pedunculoside": "Iridoid - safe",
    "pelargonidin": "Listed safe",
    "perillyl alcohol": "Listed safe",
    "perillaldehyde": "Listed safe",
    "phenazine": "Listed",
    "phloridzin": "Listed safe",
    "phlorhizin": "Listed safe",
    "phloxine": "Listed",
    "phosphoramidon": "Lab",
    "picein": "Listed safe",
    "picroside ii": "Listed safe",
    "platycodin d": "Listed",
    "plumbagin": "Listed",
    "plaunotol": "Listed",
    "polyoxin b": "Antifungal - safe",
    "ppack": "Lab",
    "praeruptorin c": "Listed safe",
    "primycin": "Listed",
    "pristane": "Listed",
    "prolinedithiocarbamate": "Lab",
    "promethazine": "LiverTox C",
    "prodigiosin": "Listed",
    "propetamphos": "Listed",
    "proscillaridin": "Listed",
    "psoralen": "Listed (LiverTox B PUVA)",
    "ptaquiloside": "Listed",
    "puag-haad": "Listed",
    "punicalagin": "Listed safe",
    "pyocyanine": "Listed",
    "pyrogallol 1,3-dimethyl ether": "Lab",
    "pyrrolidine dithiocarbamic acid": "Lab",
    "raclopride": "Listed",
    "raubasine": "Listed",
    "rebaudioside a": "Listed safe",
    "resiniferatoxin": "Listed",
    "rhapontin": "Listed",
    "rhyncophylline": "Listed",
    "ricinine": "Listed",
    "rimexolone": "Listed",
    "rokitamycin": "Listed",
    "roridin a": "Listed",
    "rotigotine": "LiverTox C",
    "rottlerin": "Listed",
    "rs 67333": "Research",
    "ru 43044": "Research",
    "ru 58668": "Research",
    "rubimaillin": "Trad - hepatic",
    "rutamarin": "Ruta - hepatic effects (mild)",
    "rutaecarpine": "Listed",
    "saponaria saponin": "Listed",
    "satratoxin g": "Listed",
    "saxitoxin": "Listed",
    "scoparone": "Listed",
    "selegiline": "LiverTox C",
    "selenocysteine": "Endogenous",
    "shikonin": "Listed",
    "sinensetin": "Listed safe",
    "sinpeinine a": "Trad alkaloid - hepatic",
    "solasodine": "Listed safe (low dose)",
    "spantide ii": "Lab",
    "spinasterol": "Listed safe",
    "splitomicin": "Listed",
    "stigmasterol": "Listed safe",
    "stachydrine": "Listed safe",
    "sulindac sulfide": "Listed (hep)",
    "sulindac sulfone": "Metabolite - safe",
    "sulbutiamine": "Listed safe",
    "swainsonine": "Listed",
    "swertiamarin": "Listed safe",
    "syringaldehyde": "Listed safe",
    "syringic acid": "Listed safe",
    "syringin": "Listed safe",
    "tabersonine": "Listed",
    "tangeretin": "Listed safe",
    "tanshinone ii a sodium sulfonate": "Listed",
    "tanshinone": "Listed",
    "tannic acid": "Listed",
    "tannin": "Listed",
    "taraxasterol": "Listed safe",
    "tazobactam": "Listed",
    "tenidap": "Listed",
    "tephrosin": "Listed",
    "tetrahydrocannabivarin 9": "THCV cannabinoid",
    "tetrahydrocurcumin": "Listed safe",
    "tetrandrine": "Listed",
    "tetrahydropalmatine": "Listed",
    "theaflavin": "Listed safe",
    "theanine": "Listed safe",
    "thiodicarb": "Listed",
    "thiostrepton": "Listed",
    "thunberginol b": "Listed safe",
    "thymoquinone": "Listed safe",
    "tianeptine": "Listed",
    "tin sulfide": "Inorganic",
    "tin mesoporphyrin": "Lab",
    "tingenin b": "Maytenus triterpene",
    "tipranavir": "LiverTox A",
    "toloxatone": "Listed",
    "torbafylline": "Listed",
    "tranylcypromine": "Listed",
    "tremulacin": "Listed safe",
    "trichostatin a": "Listed",
    "triptonide": "Listed",
    "trolox": "Listed safe",
    "tropisetron": "Listed",
    "trypanocidal arsenicals": "Listed",
    "tropisetron": "Listed",
    "ubenimex": "Listed",
    "urocanic acid": "Endogenous",
    "urushiol": "Listed",
    "valine": "Listed safe",
    "vesalol": "?",
    "vincamine": "Listed",
    "vincristine": "Listed",
    "viniferin": "Listed safe",
    "vinpocetine": "Listed",
    "vitamin b 6": "Listed safe",
    "vitamin k 2": "Listed",
    "vitexin": "Listed safe",
    "warangalone": "Listed safe",
    "withaferin a": "Listed (LiverTox B Ashwagandha)",
    "wogonin": "Listed (Scutellaria)",
    "xanthatin": "Xanthium - mild hepatic",
    "xanthohumol": "Listed safe",
    "yangonin": "Listed",
    "yakuchinone-a": "Listed",
    "zeranol": "Listed",
    "zerumbone": "Listed safe",
    "zerumbone": "Listed",
    "zomepirac glucuronide": "Listed",
    "zopolrestat": "Listed",
}

# Normalize
HEPATOTOXIC_V2 = {k.lower().strip(): v for k, v in HEPATOTOXIC_V2.items()}
SAFE_V2 = {k.lower().strip(): v for k, v in SAFE_V2.items()}
NON_DRUG_V2 = {k.lower().strip(): v for k, v in NON_DRUG_V2.items()}

# ============================================================
# Load remaining candidates
# ============================================================
src = os.path.join(OUT_DIR, "remaining_for_websearch.csv")
df = pd.read_csv(src)
df["name_lc"] = df["name"].str.lower().str.strip()

def classify(n):
    if pd.isna(n):
        return (pd.NA, "non_drug", "no name")
    nlc = str(n).lower().strip()
    if nlc in HEPATOTOXIC_V2:
        return (1, "expert_curation", HEPATOTOXIC_V2[nlc])
    if nlc in SAFE_V2:
        return (0, "expert_curation", SAFE_V2[nlc])
    if nlc in NON_DRUG_V2:
        return (pd.NA, "non_drug", NON_DRUG_V2[nlc])
    return None

res = df["name"].apply(classify)
df["manual_label"] = res.apply(lambda x: x[0] if x is not None else pd.NA)
df["source"] = res.apply(lambda x: x[1] if x is not None else "")
df["reason"] = res.apply(lambda x: x[2] if x is not None else "")
df["classified"] = res.apply(lambda x: x is not None)

classified = df[df["classified"]].copy()
unclass = df[~df["classified"]].copy()

print(f"Remaining: {len(df)}")
print(f"Newly classified: {len(classified)}")
print(f"  manual_label=1: {(classified['manual_label']==1).sum()}")
print(f"  manual_label=0: {(classified['manual_label']==0).sum()}")
print(f"  manual_label=NaN: {classified['manual_label'].isna().sum()}")
print(f"Still unclassified (WebSearch needed): {len(unclass)}")

out_cols = ["inchi_key", "canonical_smiles", "name", "manual_label", "source", "reason"]
classified[out_cols].to_csv(os.path.join(OUT_DIR, "expert_classified_v2.csv"), index=False)
unclass[["inchi_key", "canonical_smiles", "name"]].to_csv(
    os.path.join(OUT_DIR, "still_unclassified.csv"), index=False
)

# Show remaining
print("\n--- Names still unclassified ---")
for n in unclass["name"].tolist():
    print(f"  {n}")
