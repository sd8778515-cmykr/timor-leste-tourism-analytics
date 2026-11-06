"""
annotate_places.py
==================
ANOTASAUN MANUAL (GROUND TRUTH) ba 4 kategoria fixu husi dosente:
    SEJARAH | AGAMA | TRADISIONAL | MODEREN

Tanba sa presiza?
-----------------
Manual seksaun 14 ezije metrika ba Classification (Accuracy, Precision, Recall,
F1-score, Confusion Matrix). Metrika sira-ne'e SO bele kalkula bainhira iha
"ground truth" (label lolos). Dataset Google Maps la iha kolona kategoria
Historical/Religious/Traditional/Modern, tanba ne'e grupu halo ANOTASAUN MANUAL ba
kada lokal turizmu (50 lokal orijinál — dataset foun agora iha 569 lokal, maibé
`EXPERT_LABELS` kraik seidauk kobre lokal foun sira, haree README seksaun 5),
bazeia ba:
    1. Naran lokal + kategoria Google
    2. Deskrisaun ofisial husi lokal (kampu `description`/`ownerDescription`)
    3. Evidensia husi teksu review rasik (sita iha kolona `label_evidence`)
    4. Konhesimentu kontekstu Timor-Leste

Rezultadu: `data/place_category_labels.csv` — ida-ne'e mak FICHEIRU DADUS ne'ebe
pipeline le'e. Estudante/dosente BELE EDITA CSV ne'e ho Excel se la konkorda ho
label ruma; pipeline la presiza muda. Halao script ne'e fila fali DEIT bainhira
hakarak rejenera CSV husi zero (sei sobrescreve edisaun manual).

Nivel konfiansa (`label_confidence`):
    high   = evidensia diretu (naran esplisitu / review sita marka klaru)
    medium = interpretasaun kontekstu (bele diskuti; dokumenta iha relatoriu)

Halao:
    py -3.11 src/annotate_places.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PLACE_LABELS_CSV, ensure_dirs
from data_collection import load_raw_data

# naran lokal -> (kategoria, konfiansa, evidensia/justifikasaun)
EXPERT_LABELS = {
    # ---------------------------------------------------------- SEJARAH ----
    "Memorial do Massacre de Aifu": (
        "Historical", "high",
        "Memorial ba masakre - sitiu memoria istorika kona-ba periodu okupasaun.",
    ),
    "Pousada de Maubara": (
        "Historical", "high",
        "Review (pt): 'Edificio colonial portugues renovado mas abandonado' - "
        "edifisiu era kolonial portuges.",
    ),
    "Antigo Mercado de Ainaro": (
        "Historical", "high",
        "Naran 'Antigo Mercado' = merkadu antigu, edifisiu heransa era kolonial.",
    ),
    "Marobo Hot Spring": (
        "Historical", "medium",
        "Review (en): 'developed by the Portuguese, with a resort complex built "
        "around the pools, but fell into disrepair' - kompleksu termal konstrui "
        "iha era kolonial. NOTA: lokal ne'e mos iha komponente natureza no lisan "
        "(review seluk temi uma lulik no tebe-tebe); label bele diskuti.",
    ),

    # ------------------------------------------------------------ AGAMA ----
    "Gruta Soro": (
        "Religious", "high",
        "'Gruta' iha kontekstu Timor-Leste = santuariu Marianu katoliku "
        "(fatin orasaun no peregrinasaun).",
    ),
    "Flecha(Fleixa)": (
        "Religious", "medium",
        "Review (en): 'This's the start hiking point to Mt. Ramelau' - pontu "
        "hahu dalan peregrinasaun ba estatua Nossa Senhora de Ramelau, "
        "peregrinasaun katoliku boot liu iha Timor-Leste.",
    ),

    # ------------------------------------------------------ TRADISIONAL ----
    'UMA LISAN "DATOMIR"': (
        "Traditional", "high",
        "'Uma Lisan' = uma lulik/uma adat - arkitetura no sistema lisan tradisional.",
    ),
    "Casas tradicionais de Suailoro": (
        "Traditional", "high",
        "Naran esplisitu 'casas tradicionais'; review (pt): 'Uma das Aldeias mais "
        "bonita de Timor'.",
    ),
    "Lisan Lol-Hae Leotelo Ainaro": (
        "Traditional", "high",
        "'Lisan' = lisan/adat - fatin kultura tradisional komunidade.",
    ),
    "Negosiasi Artesenato": (
        "Traditional", "high",
        "Deskrisaun ofisial: promosaun produtu artezanatu lokal husi komunidade.",
    ),
    "Bei Luruk": (
        "Traditional", "medium",
        "'Bei' iha Tetun = bei-ala (ancestral) - indika fatin lisan/ancestral.",
    ),
    "Johnny the crocodile": (
        "Traditional", "medium",
        "Lafaek ne'ebe komunidade lokal konsidera lulik (mitu 'abo lafaek', "
        "mitu fundasaun rai Timor).",
    ),
    "Atabae Rice Fields Lookout": (
        "Traditional", "medium",
        "Paisajen natar - moduarian tradisional komunidade.",
    ),
    "Ainaro Hatubulico": (
        "Traditional", "medium",
        "Review (en): 'A scenic village in the valley. The villagers are very "
        "nice and friendly' - suku tradisional iha foho-leten.",
    ),
    "Palco Haksolok": (
        "Traditional", "medium",
        "'Palco Haksolok' = palku ba eventu kultural/dansa komunidade.",
    ),

    # ---------------------------------------------------------- MODEREN ----
    # Fallback dokumentadu: atrasaun natureza / eco-tourism / lazer / fasilidade
    # publiku ne'ebe LA IHA marka istorika, relijioza ka lisan espesifiku.
    "Mamorai Waterfall": ("Modern", "high", "Cachoeira iha Kutulau Protected Area - atrasaun natureza/lazer."),
    "Ikan Kail Fresco": ("Modern", "medium", "Fatin peska/rekreasaun."),
    "Lagoa masin maubara": ("Modern", "medium", "Lagoa natural - eco-tourism."),
    "Lissa-ico": ("Modern", "medium", "Atrasaun natureza sem marka kultural iha dadus."),
    "Mota hatulihu": ("Modern", "medium", "Mota/rio - atrasaun natureza."),
    "Nojhe's garden": ("Modern", "medium", "Jardin/espasu lazer."),
    "Plila": ("Modern", "medium", "Atrasaun natureza sem marka kultural iha dadus."),
    "Tibalau Maubara": ("Modern", "high", "Kategoria Google 'National reserve' + area mangrove - eco-tourism."),
    "Bazartete Waterfall": ("Modern", "high", "Cachoeira - atrasaun natureza."),
    "Liquica lookout and cave": ("Modern", "high", "Miradouru no fatin natureza."),
    "Lebukoe Delecuvati": ("Modern", "medium", "Atrasaun natureza sem marka kultural iha dadus."),
    "Maubara Beach": ("Modern", "medium", "Praia + shore diving; review foka ba kualidade praia, laos forte Maubara."),
    "Lagoa Tepha": ("Modern", "medium", "Lagoa natural."),
    "Damuk Ai Teka Laran": ("Modern", "medium", "Atrasaun natureza."),
    "Fatin Kail": ("Modern", "medium", "Fatin peska."),
    "Suai Loro Beach": ("Modern", "high", "Praia - atrasaun natureza/lazer."),
    "Cumu-Oan Lagoon": ("Modern", "high", "Lagoa ho mangrove - relax no peska."),
    "croc spot": ("Modern", "medium", "Fatin observasaun lafaek - wildlife/natureza."),
    "LAEM TOLU": ("Modern", "high", "Wetland no hiking - eco-tourism."),
    "Kampus Unpaz": ("Modern", "high", "Kampus universidade - infraestrutura moderna."),
    "Lagoa Keimau": ("Modern", "medium", "Lagoa natural."),
    "Manu-babi Waterfall": ("Modern", "high", "Cachoeira."),
    "Foelora Tutun": ("Modern", "medium", "Fatin panoramiku/natureza."),
    "Waterfall Ertis": ("Modern", "high", "Cachoeira."),
    "Crocodilo spot": ("Modern", "high", "Review (pt): 'Habitat natural de crocodilos'."),
    "Waterfall Manubabi": ("Modern", "high", "Cachoeira."),
    "Mirbuti Bee Malirin Natural": ("Modern", "high", "Bee malirin - fonte bee natural."),
    "Flixa": ("Modern", "medium", "Atrasaun natureza."),
    "Maubissi Kemping": ("Modern", "high", "Camping ground - fasilidade turizmu moderna."),
    "Be Tuda Aituto": ("Modern", "high", "Serie cachoeira ho asesu husi estrada."),
    "Kilelo Waterfall": ("Modern", "high", "Cachoeira."),
    "Pixina natural Leotelo": ("Modern", "high", "Pixina natural - lazer/natureza."),
    "Ersalibuti waterfall": ("Modern", "high", "Cachoeira iha meiu plantasaun kafe."),
    "Bandeira Waterfall": ("Modern", "high", "Cachoeira aas liu iha Timor-Leste (~200m)."),
    "Dokomali Falls/ Bee Tudak Dokomali": ("Modern", "high", "Cachoeira - trilha no ponte bambu."),
}

DEFAULT_LABEL = ("Modern", "low",
                 "Sem evidensia istorika/relijioza/lisan iha dadus - fallback "
                 "dokumentadu ba atrasaun natureza/lazer.")


def _normalize(name: str) -> str:
    """Normaliza naran atu match la depende asentu (ex: 'Bee' vs 'Bee')."""
    if not isinstance(name, str):
        return ""
    trans = str.maketrans("áàâãäéèêëíìîïóòôõöúùûüçÁÀÂÃÄÉÈÊËÍÌÎÏÓÒÔÕÖÚÙÛÜÇ",
                          "aaaaaeeeeiiiiooooouuuucAAAAAEEEEIIIIOOOOOUUUUC")
    return name.translate(trans).strip().lower()


_NORMALIZED_LABELS = {_normalize(k): v for k, v in EXPERT_LABELS.items()}


def label_for_place(title: str) -> tuple:
    return _NORMALIZED_LABELS.get(_normalize(title), DEFAULT_LABEL)


def build_labels_csv(dataset_path=None, out_path=None) -> pd.DataFrame:
    """`dataset_path=None` (default) -> le'e no hamutuk TODO `*.json` iha
    `data/raw/` (haree `data_collection.load_raw_data()`), atu lokal FOUN
    husi kualker ficheiru raw automatikamente simu linha anotasaun."""
    ensure_dirs()
    out_path = Path(out_path) if out_path else PLACE_LABELS_CSV
    places = load_raw_data(dataset_path)

    rows = []
    for p in places:
        title = p.get("title")
        category, confidence, evidence = label_for_place(title)
        rows.append({
            "place_id": p.get("placeId"),
            "place_name": title,
            "city": p.get("city"),
            "google_category": p.get("categoryName"),
            "category_label": category,
            "label_confidence": confidence,
            "label_evidence": evidence,
        })

    df = pd.DataFrame(rows).sort_values(["category_label", "place_name"])
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    return df


if __name__ == "__main__":
    df = build_labels_csv()
    print(f"Salva {len(df)} lokal -> {PLACE_LABELS_CSV}")
    print()
    print(df["category_label"].value_counts().to_string())
    print()
    print(df["label_confidence"].value_counts().to_string())
    unlabeled = df[df["label_evidence"] == DEFAULT_LABEL[2]]
    if len(unlabeled):
        print(f"\nLokal ho fallback (presiza revizaun manual): {len(unlabeled)}")
        for name in unlabeled["place_name"]:
            print("  -", name)
