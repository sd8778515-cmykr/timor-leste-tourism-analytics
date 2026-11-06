"""
category_classifier.py
======================
LO4 | Manual seksaun 13 - Text Mining / Classification.

MODEL 1 (baseline interpretavel): klasifikasaun RULE-BASED ho lexicon keyword
multi-lingua (Tetun / Indonesia / Portugues / Ingles) ba 4 kategoria fixu husi
dosente:

        SEJARAH  |  AGAMA  |  TRADISIONAL  |  MODEREN

Tanba sa rule-based hanesan baseline?
  1. Kategoria fixu no defini ona husi dosente -> lojika keyword bele
     justifika, audita no reproduz (LO7 reproducibility).
  2. Ground truth manual deit kobre 49/569 lokal (125 review ho teksu) -> model
     estatistiku (Model 2) sei treinu ho amostra ki'ik risku overfitting;
     baseline transparente presiza ba komparasaun (haree README seksaun 5).
  3. Manual seksaun 13 lista "TF-IDF / Text Mining" ba problema tipu ida-ne'e;
     lexicon mak nia forma simples liu no serve hanesan baseline ba MODEL 2
     (Naive Bayes / Logistic Regression / SVM iha `supervised_models.py`).

DUA MODU PREDISAUN (importante ba evaluation honestu):
  - `predict_from_text`      : uza DEIT teksu review. Ida-ne'e mak teste lolos -
                               "review nia liafuan rasik bele fo sai kategoria?"
  - `predict_from_context`   : uza teksu + naran lokal + kategoria Google.
                               Ida-ne'e mak modu PRODUSAUN (informasaun barak
                               liu), maibe nia rezultadu la bele kompara
                               diretamente ho ground truth tanba ground truth
                               rasik mai husi naran/deskrisaun lokal (data
                               leakage) - ne'e dokumenta iha relatoriu.

Matching uza WORD BOUNDARY (`\\b...\\b`), laos substring simples, atu evita
false positive hanesan 'reza' ne'ebe kaer iha 'natureza'.
"""

import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORIES

CATEGORY_KEYWORDS = {
    "Historical": [
        "sejarah", "historia", "history", "historical", "historiku", "istoriku",
        "colonial", "kolonial", "portugues", "portuguese", "portugal",
        "japanese", "japones", "monumento", "monument", "museum", "museu",
        "memorial", "massacre", "masakre", "guerra", "war", "invasion",
        "invasaun", "occupation", "okupasaun", "resistencia", "resistance",
        "referendum", "referendo", "heroi", "hero", "libertasaun",
        "kemerdekaan", "independencia", "independence", "antigo", "antiga",
        "ancient", "peninggalan", "warisan", "heritage", "patrimonio",
        "fort", "forte", "fortaleza", "benteng", "ruina", "ruinas", "ruin",
        "ruins", "abandonado", "dilapidated", "old building", "edificio",
    ],
    "Religious": [
        "igreja", "church", "gereja", "catedral", "cathedral", "cristo rei",
        "cristo", "jesus", "kristus", "katolik", "catholic", "catolica",
        "doa", "pray", "prayer", "reza", "orasaun", "religiaun", "religiao",
        "religi", "religious", "religioso", "santa", "santo", "saint",
        "missa", "misa", "mass", "papa", "pope", "rosario", "biara",
        "monastery", "convento", "kapela", "chapel", "capela", "ziarah",
        "pilgrimage", "peregrinasaun", "peregrinacao", "altar", "padre",
        "priest", "paroquia", "parish", "gruta", "grotto", "nossa senhora",
        "virgem", "maria", "ramelau", "lourdes", "sagrado coracao",
    ],
    "Traditional": [
        "tradisional", "traditional", "tradicional", "adat", "kultura",
        "cultura", "culture", "cultural", "uma lulik", "uma lisan", "lisan",
        "lulik", "sacred house", "casa sagrada", "tais", "tenun", "weaving",
        "dansa", "dance", "tarian", "upacara", "ceremony", "cerimonia",
        "ritual", "aldeia", "kampung", "village", "suku", "artesanato",
        "artesenato", "handicraft", "kerajinan", "batik", "tebe", "tebedai",
        "bei ala", "ancestral", "ancestor", "lia nain", "sasan lisan",
        "natar", "rice field", "sawah", "moduarian", "custom", "kostumi",
    ],
    "Modern": [
        "moderno", "modern", "cafe", "kafe", "restaurant", "restoran",
        "restaurante", "hotel", "resort", "pousada", "spa", "wifi",
        "fasilidade", "facility", "facilities", "facilidades", "bersih",
        "clean", "limpo", "nyaman", "comfortable", "confortavel", "teknologi",
        "technology", "instagramable", "photo spot", "viewpoint", "lookout",
        "sunset", "sunrise", "swimming", "swimming pool", "piscina", "pixina",
        "diving", "snorkeling", "kemping", "camping", "playground", "parking",
        "estacionamento", "toilet", "signage", "infraestrutura",
        "infrastructure", "waterfall", "cascata", "cachoeira", "air terjun",
        "beach", "praia", "pantai", "tasi", "lagoon", "lagoa", "hot spring",
        "bee malirin", "mangrove", "hiking", "hike", "trail", "trilha",
        "nature", "natureza", "naturiza", "natural", "scenery", "pemandangan",
        "eco", "wildlife", "crocodile", "lafaek", "crocodilo",
    ],
}

# Se score empata, kategoria liu espesifiku manan (marka relijioza/istorika/
# lisan sira raru liu duke marka lazer jeral).
TIE_BREAK_ORDER = ["Religious", "Historical", "Traditional", "Modern"]
FALLBACK_CATEGORY = "Modern"

# Keyword badak/ambiguu ne'ebe presiza padraun liu espesifiku
_AMBIGUOUS_OVERRIDES = {
    "maria": r"\b(santa|nossa senhora|virgem)\s+maria\b",
    "santa": r"\bsanta\s+[a-z]+",
    "santo": r"\bsanto\s+[a-z]+",
    "papa": r"\b(papa|pope)\s+(francisco|joao|john|paulo)\b",
    "natar": r"\bnatar\b",
}


def _keyword_pattern(kw: str) -> re.Pattern:
    if kw in _AMBIGUOUS_OVERRIDES:
        return re.compile(_AMBIGUOUS_OVERRIDES[kw])
    return re.compile(r"\b" + re.escape(kw) + r"\b")


_COMPILED = {
    cat: [(kw, _keyword_pattern(kw)) for kw in kws]
    for cat, kws in CATEGORY_KEYWORDS.items()
}


def _score(blob: str) -> dict:
    """Kalkula score kada kategoria = kuantidade keyword uniku ne'ebe match."""
    return {
        cat: sum(1 for _, pat in pats if pat.search(blob))
        for cat, pats in _COMPILED.items()
    }


def _matched_keywords(blob: str, category: str) -> list:
    return [kw for kw, pat in _COMPILED[category] if pat.search(blob)]


def classify(blob: str) -> tuple:
    """
    Klasifika teksu ida ba 1 kategoria.
    Retorna (kategoria, scores_dict, matched_keywords, match_type).
    """
    blob = blob.lower() if isinstance(blob, str) else ""
    scores = _score(blob)
    top = max(scores.values()) if scores else 0
    if top == 0:
        return FALLBACK_CATEGORY, scores, [], "fallback"

    winners = [c for c, s in scores.items() if s == top]
    chosen = winners[0] if len(winners) == 1 else next(
        c for c in TIE_BREAK_ORDER if c in winners
    )
    match_type = "keyword" if len(winners) == 1 else "keyword_tiebreak"
    return chosen, scores, _matched_keywords(blob, chosen), match_type


def predict_from_text(text_clean: str) -> tuple:
    """Predisaun uza DEIT teksu review (modu evaluation - la iha leakage)."""
    return classify(text_clean)


def predict_from_context(text_clean: str, place_name: str = "",
                         google_category: str = "") -> tuple:
    """Predisaun uza teksu + naran lokal + kategoria Google (modu produsaun)."""
    parts = [text_clean, place_name, google_category]
    blob = " ".join(p if isinstance(p, str) else "" for p in parts)
    return classify(blob)


def apply_category_classification(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aplika Model 1 (rule-based) ba DataFrame. Hatama kolona:
        category_rule_text     -> predisaun husi teksu deit
        category_rule_context  -> predisaun husi teksu + naran lokal + kat. Google
        category_rule_match    -> keyword / keyword_tiebreak / fallback
        category_rule_keywords -> lista keyword ne'ebe match (transparensia)
        category               -> KATEGORIA FINAL ba analiza/dashboard =
                                  `category_label` (anotasaun manual) bainhira iha,
                                  senaun `category_rule_context`.
    """
    df = df.copy()

    text_res = df["text_clean"].apply(predict_from_text)
    df["category_rule_text"] = text_res.apply(lambda r: r[0])
    df["category_rule_text_match"] = text_res.apply(lambda r: r[3])

    ctx_res = df.apply(
        lambda row: predict_from_context(
            row.get("text_clean", ""),
            row.get("place_name", ""),
            row.get("place_category_google", ""),
        ),
        axis=1,
    )
    df["category_rule_context"] = ctx_res.apply(lambda r: r[0])
    df["category_rule_match"] = ctx_res.apply(lambda r: r[3])
    df["category_rule_keywords"] = ctx_res.apply(lambda r: ", ".join(r[2][:6]))

    # Kategoria FINAL ba EDA/dashboard/Power BI: uza anotasaun manual (ground
    # truth) tanba kategoria mak PROPRIEDADE husi LOKAL turizmu (destinasaun ida
    # nia natureza tematika), la muda tuir kada komentariu. Model sira nia
    # servisu mak PREDIZ kategoria ne'e husi teksu, no sira nia kualidade
    # avalia iha `model_evaluation.py` (Manual seksaun 14).
    # Decide once per destination. Review-level context predictions may differ,
    # but must not change a destination's category across rows or filters.
    place_categories = {}
    for place_id, group in df.groupby("place_id", sort=False):
        metadata = group.iloc[0]
        texts = sorted(set(group["text_clean"].dropna().astype(str)))
        place_categories[place_id] = predict_from_context(
            " ".join(texts), metadata.get("place_name", ""),
            metadata.get("place_category_google", ""),
        )[0]
        if "category_label" in group:
            manual = group["category_label"].dropna().unique()
            if len(manual) > 1:
                raise ValueError(f"Conflicting manual categories for {place_id}")
            if len(manual):
                place_categories[place_id] = manual[0]
    df["category"] = df["place_id"].map(place_categories)
    if "category_label" in df.columns:
        df["category_source"] = df["category_label"].notna().map(
            {True: "anotasaun_manual", False: "rule_based_fallback"}
        )
    else:
        df["category_source"] = "rule_based_fallback"

    df["category"] = pd.Categorical(df["category"], categories=CATEGORIES, ordered=False)
    return df


if __name__ == "__main__":
    from data_collection import collect
    from preprocessing import preprocess

    raw = collect()
    clean, _ = preprocess(raw)
    out = apply_category_classification(clean)

    print("Kategoria FINAL (anotasaun manual):")
    print(out["category"].value_counts().to_string())
    print("\nPredisaun rule-based husi TEKSU deit:")
    print(out.loc[out["has_text"], "category_rule_text"].value_counts().to_string())
    print("\nTipu match (modu kontekstu):")
    print(out["category_rule_match"].value_counts().to_string())
    print("\nExemplu review ho keyword match:")
    sample = out[(out["category_rule_text_match"] != "fallback") & out["has_text"]]
    print(sample[["place_name", "category", "category_rule_text",
                  "category_rule_keywords"]].head(12).to_string())
