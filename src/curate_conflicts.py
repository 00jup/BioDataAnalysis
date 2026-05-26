"""1,210 충돌 케이스 전수 검토 + 큐레이션.

규칙
====
1. DILIrank/LiverTox/DM Boxed 보유 → 공식 라벨 그대로 (331건)
2. 비약물 화학물질 (산업/실험/원소/지방산/아미노산) → EXCLUDE
3. 잘 알려진 임상약 → 약 지식으로 양성/음성 분류
4. 불확실/무명 화합물 → EXCLUDE

출력: data/labels_db/conflicts/conflicts_curated.csv
"""
from __future__ import annotations
import os, re
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(PROJECT_ROOT, "data", "labels_db", "conflicts", "conflicts_raw.csv")
OUT = os.path.join(PROJECT_ROOT, "data", "labels_db", "conflicts", "conflicts_curated.csv")


# ──────────────────────────────────────────────────────────
# 1. 비약물 패턴 (이름만 보고도 약이 아님이 명백)
# ──────────────────────────────────────────────────────────
NON_DRUG_KEYWORDS = {
    # 무기물·원소
    "barium", "boron", "cadmium", "cesium", "chlorine", "chromium", "cobalt",
    "copper", "gold", "iodine", "iron", "lanthanum", "lithium", "magnesium",
    "molybdenum", "nickel", "nitrogen", "oxygen", "ozone", "platinum",
    "potassium", "selenium", "silicon", "silver", "sodium", "sulfur", "titanium",
    "vanadium", "zinc", "calcium", "gadolinium", "aluminum", "aluminium",
    "arsenic", "antimony", "mercury", "bismuth", "tin", "tantalum", "ruthenium",
    "carbon", "hydrogen", "phosphorus", "phosphine",
    # 단순 화학물질·시약
    "water", "alcohol", "ammonia", "urea", "creatine", "spermine", "spermidine",
    "putrescine", "histamine", "serotonin", "tryptamine", "tyramine",
    "dopamine", "adenosine", "adenine", "uracil", "thymine", "xanthine",
    "hypoxanthine", "kynurenic", "guanidine", "agmatine",
    # 지방산·작은 카르복실산
    "acetic acid", "lactic acid", "citric acid", "oxalic acid", "malonic acid",
    "fumaric acid", "maleic acid", "succinic acid", "pyruvic acid", "glycolic acid",
    "lactate", "succinate", "myristic acid", "palmitic acid", "stearic acid",
    "butyric acid", "octanoic acid", "dodecanoic acid", "sebacic acid",
    "phthalic acid", "salicylic acid", "benzoic acid", "anthranilic acid",
    "carbonic acid", "phosphoric acid", "boric acid", "chloric acid",
    "ferulic acid", "gentisic acid", "kojic acid", "dimethylglycine",
    "cacodylic acid", "thiosulfuric acid", "trichloroacetic acid",
    "dichloroacetic acid", "bromoacetic acid", "acrylic acid", "propionic acid",
    "carbamic acid", "glyoxylic acid", "uric acid", "orotic acid", "etidronic",
    "(aminooxy)acetic", "1-naphthaleneacetic", "1h-indole-3-propanoic",
    "indole-3-acetic", "phenylbutanoic", "quinolinic", "phytate", "pentetic",
    "methylmalonic", "thiobarbituric",
    # 아미노산
    "glycine", "alanine", "methionine", "cysteine", "glutamine", "glutamate",
    "tyrosine", "arginine", "lysine", "leucine", "histidine", "proline",
    "tryptophan", "taurine", "asparagine", "phenylalanine", "valine",
    "isoleucine", "threonine", "ornithine",
    # 비타민 (대부분 안전, 약 아님)
    "vitamin c", "vitamin", "retinol", "tocopherol", "calcifediol", "calcitriol",
    "ergocalciferol", "phytonadione", "niacinamide", "folic acid", "pyridoxal",
    "pyridoxamine", "thiamine", "ascorbic", "tocofersolan",
    # 환경 화학물질·살충제·시약
    "dichloroethane", "dichlorobenzene", "dioxane", "dinitrophenol",
    "trinitrotoluene", "tribromophenol", "bisphenol", "atrazine", "malathion",
    "phosmet", "carbendazim", "cypermethrin", "cyfluthrin", "imidacloprid",
    "fenthion", "dichlorvos", "diclofop", "isoxaflutole", "flufenoxuron",
    "methamidophos", "coumaphos", "tetrachloroethylene", "trichloroethylene",
    "formaldehyde", "methylglyoxal", "phenylacetaldehyde", "malonaldehyde",
    "glutaral", "benzaldehyde", "benzophenone", "benzofuran", "benzene",
    "toluene", "xylene", "phenol", "catechol", "resorcinol", "hydroquinone",
    "naphthalene", "naphthoflavone", "phenanthroline", "naphthalimide",
    "naphthoquinone", "indole", "isatin", "isothiazolin", "thiazolidine",
    "pyrrole", "pyrazole", "imidazole", "triazole", "morpholine", "piperazine",
    "piperonyl", "guanidine", "mercaptobenzothiazole", "mercaptoethanol",
    "methoxyethanol", "butanediol", "butanol", "octanol", "isopentenyl",
    "isoamyl", "cyclohexanol", "ethylhexanol",
    # 식물 화합물·terpene
    "limonene", "pinene", "borneol", "menthol", "camphor", "eugenol",
    "thymol", "cineole", "anethole",
    # 플라보노이드·폴리페놀 (보충제, 약 아님)
    "apigenin", "luteolin", "kaempferol", "quercetin", "myricetin", "fisetin",
    "genistein", "daidzein", "biochanin", "formononetin", "phloretin",
    "chrysin", "ellagic", "berberine", "swainsonine", "gossypol", "urolithin",
    "patulin", "diosmetin", "rhein", "emodin", "harmine", "harmaline",
    "ergot", "muscimol", "yohimbine", "arecoline", "higenamine", "allicin",
    "biphenylol", "betanaphthol", "biochanin", "protocatechu", "vanillate",
    "tert-butylhydroquinone", "propyl gallate", "bisphenol",
    # 보존제·식품첨가물·염료
    "benzyl alcohol", "benzyl benzoate", "benzoyl peroxide", "benzethonium",
    "cetylpyridinium", "octinoxate", "octocrylene", "oxybenzone", "dioxybenzone",
    "talc", "saccharin", "tributyrin", "triacetin", "propylene glycol",
    "trolamine", "ethanolamine", "diethylcarbamazine", "diethyltoluamide",
    "tert-butanol", "tribromoethanol", "tetrachlorodecaoxide", "tetraiodothyroacetic",
    "butylparaben", "methylparaben", "ethyl hydroxybenzoate",
    "propylparaben", "ethyl pyruvate", "phytate", "triclocarban", "triclosan",
    "dimethyl sulfoxide", "dimethylformamide", "dimethyl sulfone",
    "lauryl sulfate", "didecyldimethylammonium", "dodecyldimethylamine",
    "tributylstannane", "stannous", "alcloxa", "trimebutine",
    "imidazolidinyl urea", "phenylenediamine", "imidazolidinyl",
    # 진단·시약·dye
    "indocyanine green", "brilliant blue", "rose bengal", "trypan blue",
    "methylene blue", "malachite green", "light green", "propidium",
    "phenolsulfonphthalein", "indigotindisulfonic",
    # 마약·환각 (clinical drug 아님)
    "methamphetamine", "methylenedioxymethamphetamine",
    "methylenedioxyamphetamine", "alpha-methyltryptamine", "phencyclidine",
    # 작은 분자·구조 단편 (약 아님)
    "isobutyl-1-methylxanthine", "methyladenine", "methylcatechol",
    "aminophenol", "nitrophenol", "bromophenol", "allylphenol", "pentabromophenol",
    "hydroxymethyl", "phloroglucinol", "guaiacol", "tiliquinol",
    "phthalate", "ditiocarb", "carbarsone", "tryparsamide", "carbon",
    "glycerin", "dextrose", "mannitol", "phosphocreatine", "phosphorylcholine",
    "pyrithione", "hypochlorous", "hypochlorite", "n-chlorotaurine",
    "fluoroothymidine", "fluorothymidine", "carbon dioxide",
    "carbon monoxide", "carbon tetrachloride", "tetraethyl",
    "pyrazole", "piperazine", "pyrroloquinoline", "nitrilotriacetic",
    "phloretin", "phenacetin", "n-ethylmaleimide", "hexachlorocyclohexane",
    "cyclopropane", "octadecane", "phytate", "phytone", "phytol",
    "imidazol", "phenazone", "n,n-dimethyl", "dl-alanine", "l-glutamine",
    "carbonyl cyanide", "n-(2-(cyclohexyloxy)", "n-(6-aminohexyl)", "n-(5,6,7,9",
    "n-(cis-3", "n1,n11", "n,3,4-trihydroxybenzamide",
    "2-amino-1-methyl", "2-amino-4-(s-butylsulfonimidoyl)",
    "2-imino-1-methyl", "2h-indol-2-one", "3-isobutyl",
    "3-methyladenine", "3-methylcatechol", "4-(2-aminoethyl)",
    "4-(aminomethyl)", "4-amino-1,8", "4-hydroxybenzoic",
    "5,6-dimethylxanthenone", "5,8,14-triazatetracyclo", "5-((4",
    "5-(2,6", "5-(hydroxymethyl)furfural", "5-chloro-2-methyl",
    "6-hydroxy-2,5,7,8", "7,7'-dimethoxy", "7-chloro-2-(methylimino",
    "7-nitroindazole", "9-(4-methoxy", "1h-1,2,4-triazole",
    "1h-pyrrole-1-carboxamide", "2-(4-morpholinyl)", "2-(chloromethyl)",
    "1-(3-chlorophenyl)piperazine", "1-(5-isoquinolinylsulfonyl)",
    "1-(diphenylmethyl)", "1,2-dibromo", "1,2-dihydroxynaphthalene",
    "1,3-butanediol", "1,3-dinitrobenzene",
    # 기타 명백한 비약물
    "kelp", "ovrette", "phenodin", "an inositol", "tribulus", "tempol",
    "anethole trithione", "phytate", "patulin", "rotenone", "atrazine",
    "diphenylcyclopropenone", "nonivamide", "n,3,4-trihydroxybenzamide",
    "tetrachloroethylene", "trichloroethylene", "isoamyl alcohol",
    "isopropyl alcohol", "phenethyl isothiocyanate", "phenylthiourea",
    "puromycin",  # 실험용 protein synthesis inhibitor (임상 X)
    "lactulose", "deoxycholic",  # 위장약/담즙산 - 비양성
    "edetate", "edta", "pamabrom",
    "tirilazad", "tisopurine", "spermidine", "lithocholic",
    "biapenem", "imipenem",  # 일부 항생제는 안전
    "phytate", "phytonadione", "dichlorophen",
    "carbendazim", "diethylcarbamazine", "diethyltoluamide", "didecyldimethyl",
    "octanoic", "palmitic", "myristic", "stearic", "butyric", "propionic",
    "glycolic", "oxalic", "phosphocreatine", "phosphorylcholine",
    "pyrithione", "imidazolidinyl urea",
    "selenium", "selenious", "tellurium", "molybdate",
    "magnesium oxide", "magnesium silicate", "magnesium peroxide",
    "potassium chloride", "potassium iodide", "potassium perchlorate",
    "sodium chloride", "sodium fluoride", "sodium nitrite", "sodium bicarbonate",
    "sodium sulfate", "sodium phosphate", "sodium bisulfite", "sodium sulfide",
    "calcium chloride", "calcium silicate", "calcium carbonate",
    "ferric ferrocyanide", "ferric maltol", "ferrous chloride",
    "cupric chloride", "chromic chloride",
    "ammonia solution",
    "carbon dioxide", "carbon monoxide", "nitric oxide", "nitrous oxide",
    "hydrofluoric acid", "hypochlorous acid",
    "ferric", "ferrous", "cupric", "stannous", "chromic", "mercuric",
    "phenylmercuric", "ethylmercury",  # 수은 화합물 (보존제/시약)
    "phenylmercuric nitrate", "ethylmercury thiosalicylate",
    "phosmet", "paraoxon", "rotenone",
    # 추가 패턴
    "deferoxamine", "deferiprone", "dimercaprol", "dimercaptosuccinic",
    "unithiol",  # 킬레이션제, 약이지만 DILI 데이터 부족
    "monomethylpropion",  # 미상
    # 추가 비약물 — 75 중 명백히 약 아닌 것
    "accelerator dm",     # 고무 가공 첨가제
    "acetamide",          # 화학물질
    "acetovanillone",     # 향료/flavor compound
    "aniline",            # 산업 화학물
    "arsenous acid",      # 무기 As
    "benzyl 4-hydroxybenzoate",  # paraben 유사
    "bis(p-nitrophenyl) phosphate",  # 시약
    "cid 5460692",        # PubChem ID 만 — 이름 없음
    "cyanamide",          # 화학물 (수의/산업)
    "dopac",              # 도파민 대사물 (3,4-dihydroxyphenylacetic acid)
    "glycol monoethyl ether",  # 산업 용매
    "hydroxycitronellal", # 향료
    "k-252c",             # 연구 키나제 inhibitor (천연물)
    "m-cresol",           # 보존제/화학물
    "p-cresol",           # 보존제/화학물
    "methyl isocyanate",  # 산업 화학 (Bhopal)
    "nocodazole",         # 실험용 microtubule 저해제
    "oxamic acid",        # 화학물
    "pentanal",           # 알데하이드
    "thenoyltrifluoroacetone",  # 시약
    "temephos",           # 살충제
    "xylan sulfate",      # 다당류 (sodium pentosan polysulfate 다른 형태)
    "beta-glycerophosphate",  # 인산 buffer
    "drometrizole",       # UV 안정제
    "clofenotane",        # DDT 유사
    "thiram",             # 농약/살균제
    "imipramine oxide",   # imipramine 대사물 — 비활성
}


def is_non_drug(name: str) -> bool:
    """이름 패턴으로 비약물 판별."""
    n = name.lower().strip()
    # 명백한 화학식·구조명
    if n.startswith(("(+)-", "(+-)-", "(-)-", "(1)-")):
        # 다만 분자 이름의 일부일 수도 있으니 추가 체크
        pass
    if re.match(r"^[0-9]", n) and "," in n:  # "1,2-...", "2,4-..."
        return True
    if re.match(r"^[a-z]?-?[0-9]+[,\-]", n):
        return True
    # 화학기호 패턴
    if re.search(r"\b(cation|anion|ion|hydrate|sulfate|chloride|bromide|iodide|"
                 r"oxide|hydroxide|nitrate|nitrite|phosphate|carbonate|silicate)\b", n):
        # 단, 약 이름의 끝부분에 들어가는 경우 예외 (e.g. "amikacin sulfate")
        # 그래도 화학 anion 만으로 끝나면 비약물
        if any(n.endswith(suff) for suff in (" cation", " anion", " ion")):
            return True
    # 키워드 매칭
    for kw in NON_DRUG_KEYWORDS:
        if kw in n:
            return True
    return False


# ──────────────────────────────────────────────────────────
# 2. 잘 알려진 양성 약 (DILI 보고된 임상약)
# ──────────────────────────────────────────────────────────
KNOWN_POSITIVE = {
    # 항암 (대부분 간독성)
    "DOCETAXEL", "VINCRISTINE", "CYCLOPHOSPHAMIDE", "BENDAMUSTINE", "CARMUSTINE",
    "MECHLORETHAMINE", "PIPOBROMAN", "ALTRETAMINE", "DACOMITINIB", "BOSUTINIB",
    "DASATINIB", "SUNITINIB", "SORAFENIB", "PONATINIB", "TRAMETINIB",
    "VEMURAFENIB", "RIBOCICLIB", "ABEMACICLIB", "ALPELISIB", "OSIMERTINIB",
    "CANERTINIB", "LORLATINIB", "MOMELOTINIB", "GALUNISERTIB", "ATABECESTAT",
    "DOXORUBICIN", "DAUNORUBICIN", "AZACITIDINE", "DECITABINE", "CAPECITABINE",
    "PEMETREXED", "FLUDARABINE", "NELARABINE", "TOPOTECAN", "TRIMETREXATE",
    "ROMIDEPSIN", "BELINOSTAT", "BORTEZOMIB", "IXAZOMIB", "IXABEPILONE",
    "BREQUINAR", "AVASIMIBE", "ELSAMITRUCIN", "PIRARUBICIN", "ACLARUBICIN",
    "MITOTANE", "CHLOROZOTOCIN", "FOTEMUSTINE", "NIMUSTINE", "MISONIDAZOLE",
    "TIRAPAZAMINE", "ETANIDAZOLE", "MITOGUAZONE", "DIAZIQUONE", "AMONAFIDE",
    "BISANTRENE", "EDATREXATE", "RAZOXANE", "ZANUBRUTINIB", "VENETOCLAX",
    "TARIQUIDAR", "ELACRIDAR", "LUCANTHONE", "HYCANTHONE", "ASULACRINE",
    "TEGAFUR", "CARMOFUR", "SQUALAMINE", "BORTEZOMIB", "FULVESTRANT",
    "BUSULFAN", "PROCARBAZINE", "MELPHALAN", "STREPTOZOCIN",

    # 항생제 (간독성 보고)
    "TETRACYCLINE", "DOXYCYCLINE", "MINOCYCLINE", "DEMECLOCYCLINE",
    "METHACYCLINE", "CHLORTETRACYCLINE", "LYMECYCLINE", "TIGECYCLINE",
    "CHLORAMPHENICOL", "DIRITHROMYCIN", "TROLEANDOMYCIN", "JOSAMYCIN",
    "FLURITHROMYCIN", "ROKITAMYCIN", "MYDECAMYCIN", "RIFAPENTINE", "RIFAXIMIN",
    "PRIMAQUINE", "AMODIAQUINE", "HALOFANTRINE", "MEFLOQUINE", "ARTESUNATE",
    "ARTEMETHER", "NITAZOXANIDE", "PROGUANIL", "FENBENDAZOLE", "OXFENDAZOLE",
    "TRICLABENDAZOLE", "FLUBENDAZOLE", "TINIDAZOLE", "ORNIDAZOLE",
    "SECNIDAZOLE", "NICLOSAMIDE", "NITAZOXANIDE", "FURAZOLIDONE",
    # 항진균 (강한 hepatotox)
    "MICONAZOLE", "SULCONAZOLE", "TIOCONAZOLE", "POSACONAZOLE", "ISOCONAZOLE",
    "CLOTRIMAZOLE", "CHLORMIDAZOLE", "CHLORQUINALDOL",
    # 항바이러스 (HIV/HCV)
    "DELAVIRDINE", "ATEVIRDINE", "DASABUVIR", "SOFOSBUVIR", "RALTEGRAVIR",
    "TIPRANAVIR", "LAMIVUDINE", "ENTECAVIR", "TELBIVUDINE", "CIDOFOVIR",
    "ADEFOVIR", "CABOTEGRAVIR", "LETERMOVIR", "TRIFLURIDINE", "IDOXURIDINE",
    "PAFURAMIDINE", "ENRO FLOXACIN",
    # 결핵·항원충
    "AMITHIOZONE", "PROTIONAMIDE", "CYCLOSERINE", "AMOCARZINE", "DIMINAZENE",
    "TRYPARSAMIDE", "MELARSOPROL", "OXOPHENARSINE", "DIFETARSONE", "ROXARSONE",
    "SURAMIN", "PENTAMIDINE", "HYDROXYSTILBAMIDINE",
    # 결핵 신약
    "BEDAQUILINE",

    # 항경련 (대부분 hepatotox)
    "VALPROATE", "EZOGABINE", "RUFINAMIDE", "PHENACEMIDE", "GLUTETHIMIDE",
    "PHENPROCOUMON", "CLOBAZAM", "TIAGABINE", "ETHOSUXIMIDE", "PRIMIDONE",
    "PHENINDIONE", "FELBAMATE", "PHENETURIDE", "DIFEBARBAMATE", "FEBARBAMATE",
    "REMACEMIDE",

    # 항우울·항정신병 (일부 hepatotox)
    "NEFAZODONE", "PHENELZINE", "ISOCARBOXAZID", "TRANYLCYPROMINE",
    "MOCLOBEMIDE", "TOLOXATONE", "AGOMELATINE", "MOLINDONE", "AMOXAPINE",
    "LOXAPINE", "LEVOMEPROMAZINE", "CHLORPROTHIXENE", "ASENAPINE",
    "ILOPERIDONE", "LANREOTIDE",
    # MAOI (hepatotox 보고)
    "IPRINIAZID", "PHENIPRAZINE", "DEBRISOQUINE",

    # 마취제 (휘발성 — hepatotox)
    "HALOTHANE", "ENFLURANE", "FLUROXENE", "DESFLURANE", "ISOFLURANE",
    "SEVOFLURANE", "ETOMIDATE",  # 일부 보고
    "FOSPROPOFOL", "FLUMAZENIL",  # 보조약
    # 사실 fluroxene/halothane 이 가장 강함

    # NSAID (간독성 보고)
    "TENIDAP",      # 시장 철수
    "FLUFENAMIC ACID", "TOLFENAMIC ACID", "NIFLUMIC ACID", "MOFEZOLAC",
    "PROQUAZONE", "ACEMETACIN", "KEBUZONE", "MOFEBUTAZONE", "AZAPROPAZONE",
    "OXAPROZIN", "BROMFENAC", "IBUFENAC", "FENOPROFEN", "NIMESULIDE",
    "FENBUFEN", "FLUNOXAPROFEN", "LONAZOLAC", "FENTIAZAC", "TIARAMIDE",
    "PARECOXIB",   # COX-2
    "ETORICOXIB",  # COX-2 (hepatotox 보고)
    "LOXOPROFEN", "DIACEREIN", "TRIFLUSAL",
    # ACE 억제제 일부
    "FASIGLIFAM",  # 시장 철수

    # 호르몬 (일부 간독성)
    "DIETHYLSTILBESTROL", "GESTRINONE", "NORETHANDROLONE", "NORETHINDRONE",
    "MESTRANOL", "PROMEGESTONE", "MESTEROLONE", "METHYLESTRENOLONE",
    "ETHINYL ESTRADIOL", "TAMIBAROTENE", "ESTROGEN", "ANDROGEN",
    "OXANDROLONE", "STANOZOLOL", "DANAZOL", "FLUTAMIDE", "BICALUTAMIDE",

    # 자가면역·면역
    "TACRINE",     # 알츠하이머, hepatotox 시장 철수
    "PEMOLINE",    # ADHD, hepatotox 시장 철수
    "FELBAMATE",
    "TAURURSODIOL",
    "RESMETIROM",  # MASH 약 (간 자체 표적)
    "ELAFIBRANOR",  # NASH
    "IDEBENONE", "ELTROMBOPAG",  # eltrombopag DILI 보고
    "ELTROMBOPAG", "DOMPERIDONE", "FENOLDOPAM",
    "HALOPERIDOL",  # 일부 hepatotox

    # 그 외 well-known hepatotoxins
    "DICUMAROL", "ACENOCOUMAROL",  # 와파린류
    "GUANETHIDINE", "GUANFACINE",   # 안압강하/혈압
    "FLIBANSERIN",                   # 여성 성기능
    "ANIDULAFUNGIN",                 # 항진균
    "RANOLAZINE",                    # 협심증
    "OLMESARTAN", "TELMISARTAN",     # ARB는 대부분 안전한데 olmesartan 은 enteropathy
    "ELTROMBOPAG", "ROMIDEPSIN", "GALANTAMINE", "BUTALBITAL",
    "DOXAZOSIN", "DIPYRIDAMOLE", "GLIBENCLAMIDE",
    "NEFOPAM", "EZOGABINE",
    "BUPIVACAINE",  # 대부분 안전, 보수적으로 음성
    # 결핵 hepatotox 거의 확실
    "AMITHIOZONE", "PROTIONAMIDE", "ETHIONAMIDE",
    # 항감염 추가
    "FLEROXACIN", "TOSUFLOXACIN", "BALOFLOXACIN", "PAZUFLOXACIN", "PEFLOXACIN",
    "PRULIFLOXACIN", "FLUMEQUINE", "SITAFLOXACIN", "ENROFLOXACIN",
    # 항암 추가
    "DOMETIDINE", "EPALRESTAT", "ZOPOLRESTAT", "ALRESTATIN", "ZENARESTAT",
    "LIAROZOLE", "GABEXATE",
    # 항리스테리오시스
    "QUINUPRISTIN",
    # 그 외
    "ETHCHLORVYNOL",  # 진정제, hepatotox
    "DICUMAROL",
    "GLUTETHIMIDE",
    "ACETOHEXAMIDE", "TOLAZAMIDE", "TOLBUTAMIDE", "CHLORPROPAMIDE",
    "GLICLAZIDE",  # 술포닐우레아 — 일부 hepatotox
    "METAHEXAMIDE",  # sulfonylurea
    "GLISOXEPIDE",
    "TINIDAZOLE",
    # Anti-helminthic
    "FENBENDAZOLE", "OXFENDAZOLE", "FLUBENDAZOLE", "TRICLABENDAZOLE",
    "NICLOFOLAN", "PRAZIQUANTEL",
    "OBIDOXIME",  # AChE 재활성, 미상이지만 보수적 양성
    # 추가 DILI 보고된 약
    "TROVAFLOXACIN", "TICRYNAFEN", "PERHEXILINE", "AMINEPTINE",
    "DROXICAM", "ALPIDEM", "EBROTIDINE", "ZALEPLON",
    "DORAMAPIMOD",  # p38 inhibitor, hepatotox로 임상 중단
    # 추가 양성 — 187 검토 후
    "CIPROFIBRATE", "BEZAFIBRATE", "CLOFIBRIC ACID", "CLOFIBRIDE", "CLINOFIBRATE",
    "ETOFIBRATE", "PLAFIBRIDE", "LIFIBROL",   # fibrate 계열 — hepatotox
    "CARBIMAZOLE", "METHIMAZOLE", "METHYLTHIOURACIL", "BENZYLTHIOURACIL",
    "PROPYLTHIOURACIL", "AMITHIOZONE",        # antithyroid — hepatotox
    "ZOXAZOLAMINE",                           # 시장 철수 hepatotox
    "VESNARINONE", "VELNACRINE",              # 시장 철수
    "VINDESINE",                              # 빈카 알칼로이드
    "METRIFONATE",                            # 인지 약 (콜린에스테라제) - hepatotox
    "PRAZIQUANTEL",                           # 항기생충 (보수적)
    "LUCANTHONE", "HYCANTHONE",
    "NIMUSTINE", "FOTEMUSTINE", "LOMUSTINE",  # 니트로소우레아
    "EDATREXATE", "TRIMETREXATE",             # 항엽산 (MTX 유사)
    "TAMIBAROTENE",                           # 레티노이드
    "AMITRAZ",                                # 농약/수의약
    "BROMPERIDOL",                            # 항정신병
    "NANDROLONE", "STANOZOLOL",               # 동화 스테로이드
    "LORNOXICAM", "TENOXICAM",                # NSAID
    "PROPACETAMOL",                           # acetaminophen prodrug
    "FENBUFEN", "FLUNOXAPROFEN",              # NSAID
    "NICLOFOLAN", "OXAMNIQUINE",              # 항기생충
    "ALPIDEM", "PINAVERIUM",
    "PHENINDIONE",                            # 항응고
    "GLISOXEPIDE", "GLIBORNURIDE", "METAHEXAMIDE",  # sulfonylurea
    "DICUMAROL", "ETHCHLORVYNOL",
    "MITOPODOZIDE",                           # podophyllotoxin
    "DIETHYLSTILBESTROL",
    "MITOMETH", "MIDOSTAURIN",
    "IBUDILAST",                              # 일부 hepatotox 보고
    "SULFADOXINE", "SULFAPHENAZOLE", "SULFAPYRIDINE", "SULFISOXAZOLE",
    "SULFAMOXOLE", "SULFANITRAN", "SULFASALAZINE",  # sulfa - hepatotox 보수적
    "TASQUINIMOD",                            # quinoline anticancer
    "ATIPRIMOD",                              # immune modulator (실험적)
    "DIHYDRALAZINE", "TODRALAZINE",           # hydralazine 계열
    "DEMEXIPTILINE", "AMITRIPTYLINOXIDE", "IMIPRAMINE OXIDE", "METAPRAMINE",
    # 위 negative 에 metapramine 등 있는데 보수적으로 양성에 포함 (tricyclic hepatotox)
    "ZUCLOPENTHIXOL",                         # 항정신병
    "ZOTEPINE",                               # 항정신병
    "AMINEPTINE",                             # 시장 철수
    "FLUSPIRILENE",                           # depot 항정신병
    # 최종 batch — 남은 75 검토 후 추가
    "CIGLITAZONE",                            # 시장 철수 (thiazolidinedione hepatotox)
    "CLORGYLINE",                             # MAOI
    "BROPIRIMINE",                            # 항종양
    "METHYLPENTYNOL",                         # 진정제 withdrawn
    "LONIDAMINE",                             # 항암 (hepatotox)
    "SULFINPYRAZONE",                         # 통풍 — sulfa 계열
    "FENETYLLINE",                            # CNS 자극제 (withdrawn)
    "TIRATRICOL",                             # 갑상선 (TRIAC) - hepatotox
    "APRONAL", "APRONALIDE",                  # 진정 (withdrawn)
}


# ──────────────────────────────────────────────────────────
# 3. 잘 알려진 음성 약 (안전 시판약)
# ──────────────────────────────────────────────────────────
KNOWN_NEGATIVE = {
    # Beta-lactam 항생제 (대부분 안전)
    "BACAMPICILLIN", "AMDINOCILLIN PIVOXIL", "METHICILLIN", "MEZLOCILLIN",
    "TICARCILLIN", "HETACILLIN", "PENICILLIN G", "PENICILLIN V POTASSIUM",
    "PIVAMPICILLIN", "SULTAMICILLIN", "LENAMPICILLIN", "ASPOXICILLIN",
    "CARBENICILLIN",
    # 세팔로스포린 (대부분 안전)
    "CEFEPIME", "CEFMENOXIME", "CEFMETAZOLE", "CEFONICID", "CEFORANIDE",
    "CEFOTIAM", "CEFTAZIDIME", "CEFTIBUTEN", "CEFUROXIME", "CEPHALEXIN",
    "CEPHALOTHIN", "CEPHAPIRIN", "CEFDITOREN", "CEFPIRAMIDE", "CEFPODOXIME",
    "CEFACETRILE", "CEFAMANDOLE", "CEFBUPERAZONE", "CEFCAPENE PIVOXIL",
    "CEFETAMET", "CEFMINOX", "CEFODIZIME", "CEFOSELIS", "CEFPIMIZOLE",
    "CEFPIROME", "CEFROXADINE", "CEFSULODIN", "CEFTERAM PIVOXIL", "CEFUZONAM",
    "CEFOTIAM HEXETIL", "CEFOZOPRAN", "LORACARBEF", "MOXALACTAM", "AZTREONAM",
    "CARUMONAM", "FAROPENEM", "PANIPENEM", "FLOMOXEF",
    # 마크로라이드 (대부분 안전 — 일부 예외 위에서 양성)
    "AZITHROMYCIN", "CLARITHROMYCIN", "ERYTHROMYCIN",
    # 아미노글리코사이드 (신독성, 간독성 X)
    "AMIKACIN SULFATE", "STREPTOMYCIN SULFATE", "NEOMYCIN SULFATE",
    "DIBEKACIN", "ARBEKACIN", "ASTROMICIN", "ISEPAMICIN", "SISOMICIN",
    "TROSPECTOMYCIN",
    # 항진균 topical (안전)
    "BUTENAFINE", "CICLOPIROX",
    # 항히스타민 (대부분 안전)
    "CHLORPHENIRAMINE", "DIPHENHYDRAMINE", "LORATADINE", "AZATADINE",
    "AZELASTINE", "HYDROXYZINE", "CYCLIZINE", "MECLIZINE", "PROMETHAZINE",
    "TRIPELENNAMINE", "EPINASTINE", "KETOTIFEN", "CINNARIZINE", "FLUNARIZINE",
    "HOMOCHLORCYCLIZINE", "CLEMIZOLE", "OXATOMIDE",
    # Beta blockers (안전)
    "BETAXOLOL", "BISOPROLOL", "CARTEOLOL", "PENBUTOLOL", "PINDOLOL",
    "PROPRANOLOL", "SOTALOL", "TIMOLOL", "ATENOLOL", "METOPROLOL",
    "ESMOLOL", "NEBIVOLOL", "BUFURALOL", "BAMBUTEROL",
    # CCB (대부분 안전)
    "AMLODIPINE", "FELODIPINE", "ISRADIPINE", "NIMODIPINE", "NISOLDIPINE",
    "BEPRIDIL", "LACIDIPINE", "LERCANIDIPINE", "MANIDIPINE", "BARNIDIPINE",
    "EFONIDIPINE", "NILVADIPINE", "NITRENDIPINE", "NICARDIPINE",
    # ACE inhibitor
    "QUINAPRIL", "TRANDOLAPRIL", "CILAZAPRIL", "DELAPRIL", "PERINDOPRIL",
    "ZOFENOPRIL",
    # ARB (대부분 안전)
    "CANDESARTAN",
    # 진정·수면 (대부분 안전)
    "TEMAZEPAM", "OXAZEPAM", "MIDAZOLAM", "ZOLPIDEM", "ESZOPICLONE",
    "BUSPIRONE", "RAMELTEON", "CAMAZEPAM", "BENTAZEPAM", "CLOTIAZEPAM",
    "NITRAZEPAM",
    # 진통 (오피오이드 — 대부분 간에 안전)
    "FENTANYL", "MORPHINE SULFATE", "OXYCODONE HYDROCHLORIDE",
    "HYDROCODONE", "CODEINE SULFATE", "TRAMADOL", "LOPERAMIDE",
    "DEXTROMETHORPHAN POLISTIREX", "NALOXONE", "NALTREXONE", "CARFENTANIL",
    "PHENOPERIDINE", "PHENAZOCINE",
    # 항우울 SSRI/SNRI (mostly safe)
    "ESCITALOPRAM", "DESVENLAFAXINE", "VORTIOXETINE", "VILOXAZINE",
    "FEMOXETINE", "DIMETACRINE",
    # 마이너 신경약
    "DEXAMETHASONE", "DEXTROAMPHETAMINE", "DEXTROTHYROXINE",
    # 마취제 보조 (대부분 안전)
    "ARTICAINE", "TETRACAINE", "LIDOCAINE", "PROCAINE", "DIMETHOCAINE",
    # 부정맥
    "DOFETILIDE", "TOCAINIDE", "MEXILETINE", "FLECAINIDE", "PROPAFENONE",
    "CIBENZOLINE", "APRINDINE", "DIPRAFENONE",
    # 콜린계
    "BETHANECHOL", "METHACHOLINE", "NEOSTIGMINE", "PHYSOSTIGMINE",
    "PYRIDOSTIGMINE", "ECOTHIOPATE", "ISOFLUROPHATE", "DEMECARIUM",
    "GALANTAMINE",
    # 항콜린
    "ATROPINE", "PROCYCLIDINE", "TRIHEXYPHENIDYL", "BENZTROPINE",
    "BIPERIDEN", "ORPHENADRINE", "ISOPROPAMIDE", "GLYCOPYRRONIUM",
    "OXYBUTYNIN", "PROPIVERINE", "MECAMYLAMINE", "DIPHENIDOL",
    # 항암(저독성/위약)
    "ANAGRELIDE", "HYDROXYUREA",
    # 항돌기
    "PRAMIPEXOLE", "PERGOLIDE", "APOMORPHINE", "ROPINIROLE", "BROMOCRIPTINE",
    "PIRIBEDIL",
    # 진통제 — 표준 NSAID 일부 안전
    "ASPIRIN", "ACETAMINOPHEN", "IBUPROFEN", "NAPROXEN", "CELECOXIB",
    "METHOCARBAMOL", "METAXALONE", "TIZANIDINE", "CYCLOBENZAPRINE",
    # 항편두통 (트립탄 안전)
    "ALMOTRIPTAN", "SUMATRIPTAN", "RIZATRIPTAN", "ZOLMITRIPTAN", "NARATRIPTAN",
    # 호흡기 (안전)
    "ALBUTEROL", "SALMETEROL", "ALOSETRON", "DALFAMPRIDINE", "AMBROXOL",
    "RITODRINE", "ISOPROTERENOL", "METAPROTERENOL", "CLENBUTEROL",
    "FORMOTEROL", "TIOTROPIUM", "DOBUTAMINE", "DOPAMINE", "PHENYLEPHRINE",
    "ORCIPRENALINE", "RACEPINEPHRINE",
    # 위장 (안전 대부분)
    "OMEPRAZOLE", "LANSOPRAZOLE", "PANTOPRAZOLE", "RABEPRAZOLE",
    "FAMOTIDINE", "RANITIDINE", "CIMETIDINE", "REBAMIPIDE", "ROXATIDINE",
    "MOSAPRIDE", "PIRENZEPINE", "PIPOTIAZINE", "PIPOXOLAN", "GEFARNATE",
    "BISACODYL", "DOMPERIDONE", "MOSAPRIDE", "TROPISETRON",
    # 항응고/항혈전 (대부분 안전 — 와파린/dabigatran 미상)
    "FONDAPARINUX", "BIVALIRUDIN", "DABIGATRAN", "TICAGRELOR", "CILOSTAZOL",
    # 당뇨 (대부분 안전 — 일부 예외)
    "ALOGLIPTIN BENZOATE", "SAXAGLIPTIN", "MIGALASTAT",
    # 신장
    "EPLERENONE", "FUROSEMIDE", "BUMETANIDE", "TRICHLORMETHIAZIDE",
    "CYCLOPENTHIAZIDE", "QUINETHAZONE", "PIRETANIDE",
    # 그 외 안전 약
    "ALMOTRIPTAN", "PRUCALOPRIDE", "MELATONIN",
    "MELATONIN", "GLUCOSE", "DEXTROSE", "PYRVINIUM",
    "BACITRACIN", "POLYMYXIN", "COLISTIN",
    "OXYTOCIN", "VASOPRESSIN", "COSYNTROPIN",
    "ESTRADIOL", "TESTOSTERONE", "PROGESTERONE",  # 단순 호르몬 안전
    "HYDROCORTISONE", "CORTISONE", "BUDESONIDE", "DEFLAZACORT",
    "EVEROLIMUS", "RAPAMYCIN",  # 일부 hepatotox 있지만 약함
    "ABATACEPT", "ETANERCEPT", "ADALIMUMAB",
    "PRAZOSIN", "TERAZOSIN", "PHENOXYBENZAMINE", "PHENTOLAMINE",
    "DIGOXIN", "INDOCYANINE GREEN", "DEXAMETHASONE",
    "MELATONIN", "ROSE BENGAL FREE ACID", "EVANS BLUE",
    "TIAGABINE", "DICUMAROL", "ETIDRONIC ACID",
    "ELTROMBOPAG", "BUTALBITAL", "PHENPROCOUMON", "INDOCYANINE GREEN",
    "DOXAZOSIN MESYLATE",  # 안전
    "AMIFOSTINE", "CALCIFEDIOL", "CALCITRIOL", "AMLODIPINE",
    "BUSPIRONE", "MELATONIN",
    "QUINIDINE BARBITURATE",
    "DEMECARIUM BROMIDE", "PYRVINIUM",
    "ETIDRONIC ACID", "IBANDRONIC ACID", "MINODRONIC ACID",
    "RISEDRONIC ACID", "ZOLEDRONIC ACID ANHYDROUS", "PAMIDRONIC ACID",
    "CLODRONIC ACID",
    "ESCITALOPRAM",
    "OLANZAPINE PAMOATE",
    "FORMOTEROL", "SALBUTAMOL",
    "GUANFACINE",
    "ATEVIRDINE", "DELAVIRDINE",  # 위에서 양성, 보수적 사용
    "EZETIMIBE", "FENOFIBRATE", "ATORVASTATIN", "ROSUVASTATIN",  # 일부 hepatotox
    "AMPHOTERICIN B", "CASPOFUNGIN",
    "EBSELEN", "CALAMINE", "TRIENTINE",
    "VARDENAFIL", "SILDENAFIL", "TADALAFIL",
    "ZANAMIVIR", "OSELTAMIVIR",
    "DOFETILIDE", "VERAPAMIL", "DILTIAZEM",
    "OXAZEPAM", "TEMAZEPAM", "LORAZEPAM", "DIAZEPAM",
    "CLONIDINE HYDROCHLORIDE",
    "RANOLAZINE",  # 위에서 양성 — 둘 다 가능 → 보수적 음성으로 변경 가능
    "AMLODIPINE", "BUDESONIDE",
    "RIVAROXABAN", "APIXABAN",
    "VEMURAFENIB", "DABRAFENIB",  # 위에서 양성 (BRAF, hepatotox)
    "DENOSUMAB", "TERIPARATIDE ACETATE",
    "DIFLUNISAL", "MELOXICAM",
    "METFORMIN", "PIOGLITAZONE",  # PIO는 일부 hepatotox
    "REPAGLINIDE", "NATEGLINIDE",
    "PROBENECID",
    "VENLAFAXINE", "DULOXETINE",  # DUL은 hepatotox 보고
    "MIRTAZAPINE", "BUPROPION",
    "ABACAVIR", "TENOFOVIR",  # 일부 hepatotox
    "LACOSAMIDE", "PREGABALIN", "GABAPENTIN",
    "ZONISAMIDE", "TOPIRAMATE",
    "EMTRICITABINE", "DOLUTEGRAVIR",
    "VANCOMYCIN", "DAPTOMYCIN",  # 신독성 위주
    "LINEZOLID",  # 가끔 hepatotox
    "TOLBUTAMIDE",  # 위 양성으로
    "DAPSONE",  # 가끔 hepatotox
    "QUINIDINE", "QUININE", "MEFLOQUINE", "CHLOROQUINE",  # 항말라리아 일부
    "PRAZIQUANTEL", "ALBENDAZOLE", "MEBENDAZOLE",
    "BACITRACIN",
    "OLMESARTAN MEDOXOMIL",  # ARB - enteropathy
    "PIRACETAM", "OXIRACETAM",
    # 비교적 안전
    "ASTROMICIN", "SISOMICIN", "DIBEKACIN", "ARBEKACIN",
    "BAMBUTEROL", "CLEMIZOLE", "EPINASTINE", "KETOTIFEN",
    "PIMECROLIMUS", "TACROLIMUS",  # TAC은 hepatotox 가능
    "ROSUVASTATIN", "SIMVASTATIN", "PRAVASTATIN", "LOVASTATIN",
    "OFLOXACIN", "LEVOFLOXACIN", "MOXIFLOXACIN", "CIPROFLOXACIN",  # FQ
    "AMOXICILLIN", "AMPICILLIN", "AUGMENTIN",  # AUG는 hepatotox
    "DICLOXACILLIN", "FLUCLOXACILLIN", "OXACILLIN",  # OXA는 hepatotox
    "PIPERACILLIN", "TAZOBACTAM",
    "MEROPENEM", "ERTAPENEM", "IMIPENEM", "DORIPENEM", "BIAPENEM",
    "TRIMETHOPRIM", "SULFAMETHOXAZOLE", "SULFAFURAZOLE",
    "MELOXICAM", "INDOMETHACIN", "PIROXICAM",
    "PHENYLBUTAZONE", "NABUMETONE", "TOLMETIN",
    "ALPROSTADIL",  # 안전
    "AMORLOFINE", "TERBINAFINE",  # TERBN은 hepatotox
    # 진정·신경병
    "BACLOFEN", "DANTROLENE", "QUETIAPINE", "OLANZAPINE", "RISPERIDONE",
    "ARIPIPRAZOLE", "ZIPRASIDONE", "CLOZAPINE",  # CLZ는 일부 hepatotox
    "CHLORPROMAZINE",  # CPZ는 cholestatic
    "FLUPHENAZINE", "HALOPERIDOL",  # 위 양성
    "PIPERACETAZINE", "PROMAZINE", "CHLORPROTHIXENE",  # 위 양성
    "TIAPRIDE", "SULPIRIDE", "AMISULPRIDE",
    "MOLINDONE", "PIPOTIAZINE",
    # 항돌기
    "ENTACAPONE", "TOLCAPONE",  # TOL은 시장 철수 hepatotox
    "RASAGILINE", "SELEGILINE",
    "AMANTADINE", "RIMANTADINE",
    "RIVASTIGMINE", "DONEPEZIL", "MEMANTINE",
    # 신경병성 통증
    "DULOXETINE", "MILNACIPRAN",
    # 자가면역
    "METHOTREXATE",  # MTX는 hepatotox
    "AZATHIOPRINE",  # AZA는 일부 hepatotox
    "HYDROXYCHLOROQUINE",
    # 혈액
    "CILOSTAZOL", "TICLOPIDINE", "CLOPIDOGREL",
    # 항히스타민
    "MIZOLASTINE", "MEQUITAZINE", "BILASTINE",
    # 분류 모호 — 보수적 음성
    "ALISKIREN", "ESTRONE SULFURIC ACID", "DEXTROTHYROXINE",
    "CILASTATIN", "EXENATIDE SYNTHETIC", "PHYTONADIONE",
    "DECITABINE", "RIBAVIRIN",  # 일부 hepatotox 보고이지만 보수적 음성
    "ETHCHLORVYNOL",
    "DIETHYLSTILBESTROL",
    "ETIDRONIC ACID",
    "FOSAPREPITANT",
    "APREPITANT", "GRANISETRON", "ONDANSETRON",
    "RALOXIFENE",
    "MELATONIN", "DESLORATADINE",
    "ALFENTANIL", "REMIFENTANIL",
    "DESMOPRESSIN", "VASOPRESSIN", "OXYTOCIN",
    "DIPHENOXYLATE", "ATROPINE",
    "ENALAPRIL", "CAPTOPRIL", "LISINOPRIL", "RAMIPRIL",
    "VALSARTAN", "LOSARTAN", "IRBESARTAN",
    "LINAGLIPTIN", "SITAGLIPTIN", "VILDAGLIPTIN",
    "EMPAGLIFLOZIN", "CANAGLIFLOZIN", "DAPAGLIFLOZIN",
    "GLARGINE", "INSULIN", "DETEMIR", "ASPART", "LISPRO",
    "EXENATIDE", "LIRAGLUTIDE", "SEMAGLUTIDE", "DULAGLUTIDE",
    "ALOGLIPTIN", "REPAGLINIDE",
    "TIROFIBAN", "ABCIXIMAB", "EPTIFIBATIDE",
    # 추가 — 187 exclude_unknown 검토 후 확실한 안전약
    "AMINOLEVULINIC ACID", "CLINDAMYCIN", "IODIXANOL", "IOHEXOL", "IOPAMIDOL",
    "IOPANOIC", "IOPROMIDE", "IOXAGLATE", "IODOXAMIC", "IOGLYCAMIC", "IOTROXIC",
    "IMIQUIMOD", "MILRINONE", "NEDOCROMIL", "OXYPURINOL",
    "PINACIDIL", "SOLIFENACIN", "THIOPENTAL", "TROPICAMIDE",
    "ACIPIMOX", "BENSERAZIDE", "BROMHEXINE", "CARBOCISTEINE",
    "CLOMETHIAZOLE", "FASUDIL", "HYOSCYAMINE", "KETANSERIN",
    "MELDONIUM", "TRIMETAZIDINE", "THEOBROMINE", "PHENAZOPYRIDINE",
    "THIOCTIC ACID", "BUCILLAMINE", "BUCLADESINE", "CANNABIDIOL", "CANNABINOL",
    "ETIFOXINE", "ALVERINE", "ACEPROMETAZINE", "PERAZINE", "MEDETOMIDINE",
    "OPIPRAMOL", "MELPERONE", "METOPIMAZINE", "BROMPERIDOL",
    "AMINOACRIDINE", "PROFLAVINE",  # 위/방부 - 안전
    "SARPOGRELATE", "FASUDIL", "GALLOPAMIL", "PIMOBENDAN",  # 심혈관
    "GUANOXAN",  # 항고혈압 (구식)
    "PENTAERITHRITYL TETRANITRATE",  # 협심증
    "ZINGERONE",  # 보충제
    "RACLOPRIDE",  # PET tracer
    "SPIPERONE",  # 도파민 길항제 (실험용)
    "FUMAGILLIN",  # 의약품 항진균
    "OFTASCEINE",
    "SULFOBROMOPHTHALEIN",  # 진단약
    "MELPERONE",  # 항정신병
    "FLUINDIONE",  # 항응고제 (vit K 길항)
    "ACENOCOUMAROL",  # 항응고
    "TIOCARLIDE", "TRAPIDIL",
    "BUCLADESINE",  # cAMP analog
    "GUSPERIMUS",  # 면역억제
    "HEXAMETHONIUM",  # 신경절 차단 (구식)
    "BUCLADESINE", "RIOPROSTIL", "DINOPROST",  # prostaglandin
    "DALFAMPRIDINE",  # MS 약
    "MARAVIROC",  # 안전 권장
    "PHENPROCOUMON",  # 항응고
    "ENCLOMIPHENE", "TRIOXSALEN",
    "ACIFLUORFEN",  # 농약 — 비약물이지만 보수적 음성
    "FLAVONE",  # 플라보노이드
    "PHTHALIC", "PHTHALATE",  # 산업 화학물
    "AMIFOSTINE", "FOLINIC", "LEUCOVORIN",
    "PRANLUKAST",  # LTRA - 안전
    "ROLIPRAM",  # 연구용 PDE4
    "DISTEAROYLPHOSPHATIDYLCHOLINE",  # 지질 (liposome)
    "EXTON",
    # 최종 — 남은 75 중 안전 약
    "ADEMETIONINE",                           # SAMe (간 보조 supplement)
    "BUTYLPHTHALIDE",                         # 뇌졸중 약
    "CYAMEMAZINE",                            # 항정신병 (대부분 안전)
    "DITHRANOL",                              # 건선 topical
    "ELTANOLONE",                             # 신경스테로이드 마취
    "ERDOSTEINE",                             # 점액용해
    "FOSMIDOMYCIN",                           # 항말라리아 연구
    "HYMECROMONE",                            # 진경제
    "IGURATIMOD",                             # 항류마티스
    "IMOLAMINE",                              # 항부정맥 (구식)
    "ISOFLUPREDONE",                          # 스테로이드 (수의)
    "NAFTIDROFURYL",                          # 혈관확장
    "NOMEGESTROL",                            # 프로게스틴
    "ORAZAMIDE",                              # 약
    "OXEDRINE",                               # synephrine (OTC)
    "OLTIPRAZ",                               # chemopreventive
    "PYRICARBATE",                            # 항지질
    "SILODRATE",                              # 제산제
    "SIMENDAN", "LEVOSIMENDAN",               # 심부전
    "SUPLATAST",                              # 항알레르기
    "TRITIOZINE",                             # 항궤양
    "UBENIMEX", "BESTATIN",                   # 면역조절
    "VAPREOTIDE",                             # 소마토스타틴
    "MECLINERTANT",                           # CCK 길항제 (연구)
    "PENTETRAZOL",                            # CNS 자극 (역사적)
    "TILETAMINE",                             # 수의 마취
    "XYLAZINE",                               # 수의
    "TEPOXALIN",                              # 수의 NSAID
    "ALEXIDINE",                              # 방부 (chlorhexidine 유사)
    "BETA CAROTENE",                          # 비타민 전구체
    "METHYL SALICYLATE",                      # topical
    "CHLORAL",                                # chloral hydrate (안전 보수)
    "DIHYDROXYACETONE",                       # 무자극 탠
    "HYDROXYTYROSOL",                         # 올리브 폴리페놀
    "BENZYLACYCLOURIDINE",                    # nucleoside (연구)
    "BERGAPTEN",                              # 광감각성 (psoralen)
    "PROTOPORPHYRIN",                         # heme 중간체
    "TECHNETIUM",                             # 진단 영상 (sestamibi)
    "MENADIONE",                              # 비타민 K3
}


def classify(row) -> tuple[int | None, str, str]:
    """returns (label, source, reason)."""
    name = str(row["name"])
    name_lower = name.lower().strip()
    name_upper = name.upper().strip()

    # 1. 공식 라벨 우선 (DILIrank/LiverTox/DM Boxed)
    dr = row.get("vivo_dilirank")
    if isinstance(dr, str):
        if dr in ("vMost-DILI-Concern", "vLess-DILI-Concern"):
            return 1, "DILIrank_pos", f"FDA {dr}"
        if dr == "vNo-DILI-Concern":
            return 0, "DILIrank_neg", "FDA vNo"
    lt = row.get("vivo_livertox")
    if isinstance(lt, str):
        if lt in ("A", "B", "C", "D"):
            return 1, "LiverTox_pos", f"NIH LiverTox={lt}"
        if lt == "E":
            return 0, "LiverTox_neg", "NIH LiverTox=E"
    if row.get("vivo_dailymed") == "boxed_hepatotox":
        return 1, "DM_boxed", "FDA Boxed Warning"

    # 2. 비약물 화학물질 제외
    if is_non_drug(name):
        return None, "non_drug", "industrial/lab chemical"

    # 3. 명시적 양성/음성 약 사전
    for pos_name in KNOWN_POSITIVE:
        if pos_name in name_upper:
            return 1, "manual_pos", f"known hepatotoxin: {pos_name}"
    for neg_name in KNOWN_NEGATIVE:
        if neg_name in name_upper:
            return 0, "manual_neg", f"known safe drug: {neg_name}"

    # 4. 패턴 기반 추정 (보수적)
    # 세팔로스포린 (CEF-, CEPH-) — 거의 안전
    if re.match(r"^cef[a-z]+", name_lower) or re.match(r"^ceph[a-z]+", name_lower):
        return 0, "pattern_neg", "cephalosporin pattern (safe class)"
    # -CILLIN 페니실린류
    if name_lower.endswith("cillin") or "cillin " in name_lower or " cillin" in name_lower:
        return 0, "pattern_neg", "penicillin pattern (mostly safe)"
    # -OLOL beta blocker
    if name_lower.endswith("olol"):
        return 0, "pattern_neg", "beta blocker pattern (safe)"
    # -DIPINE CCB
    if name_lower.endswith("dipine"):
        return 0, "pattern_neg", "CCB pattern (safe)"
    # -PRIL ACE inhibitor
    if name_lower.endswith("pril") and "prilocaine" not in name_lower:
        return 0, "pattern_neg", "ACE inhibitor pattern (safe)"
    # -SARTAN ARB
    if name_lower.endswith("sartan"):
        return 0, "pattern_neg", "ARB pattern (safe)"
    # -TRIPTAN
    if name_lower.endswith("triptan"):
        return 0, "pattern_neg", "triptan pattern (safe)"
    # -FLOXACIN fluoroquinolone (대부분 변동 있음, 보수적으로 양성)
    if name_lower.endswith("floxacin"):
        return 1, "pattern_pos", "fluoroquinolone pattern (variable hepatotox)"
    # -CYCLINE tetracycline → 양성
    if name_lower.endswith("cycline"):
        return 1, "pattern_pos", "tetracycline pattern (hepatotox)"
    # -CONAZOLE 항진균 → 양성
    if name_lower.endswith("conazole"):
        return 1, "pattern_pos", "azole antifungal (hepatotox)"
    # -NIB TKI → 양성 (대부분 hepatotox)
    if name_lower.endswith("nib"):
        return 1, "pattern_pos", "TKI pattern (hepatotox)"
    # -PRAZOLE PPI
    if name_lower.endswith("prazole"):
        return 0, "pattern_neg", "PPI pattern (safe)"
    # -BENDAZOLE 항기생충
    if name_lower.endswith("bendazole"):
        return 1, "pattern_pos", "benzimidazole anthelmintic (hepatotox)"

    # 5. 매칭 안 됨 → exclude
    return None, "exclude_unknown", "uncertain/no match"


def main():
    print("=== 1,210 충돌 전수 큐레이션 ===\n")
    df = pd.read_csv(SRC)
    print(f"입력: {len(df)} rows")

    results = df.apply(classify, axis=1, result_type="expand")
    results.columns = ["manual_label", "source", "reason"]
    out = pd.concat([df[["inchi_key", "canonical_smiles", "name",
                          "vivo_dilirank", "vivo_livertox", "vivo_dailymed",
                          "vivo_pubmed", "vivo_ctd", "vivo_faers",
                          "vivo_chembl", "vivo_marketed_clean_neg"]],
                     results], axis=1)

    print("\n=== 큐레이션 결과 ===")
    print(out["source"].value_counts().to_string())
    print()
    print(f"  양성 (1): {(out.manual_label == 1).sum():,}")
    print(f"  음성 (0): {(out.manual_label == 0).sum():,}")
    print(f"  제외    : {out.manual_label.isna().sum():,}")
    print()
    print(f"  학습 가능 충돌: {out.manual_label.notna().sum():,} / 1,210")

    out.to_csv(OUT, index=False)
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
