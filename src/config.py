"""
config.py
=========
Konfigurasaun sentral ba sistema tomak (path, konstante, metadata fonte dadus).

Referensia Manual Capstone Project - Mining and Exploring (DIT 2026):
  - Seksaun 10 "Kolesaun no Jestaun Dadus"  -> DATA_SOURCE metadata + dataset layers
  - Seksaun 16 "Source Code no GitHub"      -> estrutura folder

Dataset Layer (Manual seksaun 10):
    Raw       -> data/raw/         (dadus orijinal husi scraper/API, la muda)
    Processed -> data/processed/   (dadus ne'ebe limpa no transforma)
    Feature   -> data/features/    (variavel ne'ebe prepara ba model)
    Output    -> data/output/      (prediction, cluster, sentiment)

NOTA IMPORTANTE (status projetu):
    Grupu SEIDAUK iha asesu ba API komersial (Google Places API presiza billing).
    Tanba ne'e, fonte dadus atual mak SNAPSHOT JSON ida ne'ebe koleta liu husi
    Apify Google Maps Reviews Scraper. Tuir Manual seksaun 5 nia nota, sistema
    ne'e KLASIFIKA nia-an nudar "batch / periodic re-collection", LAOS "real-time".
    Modulu data_collection.py ona prepara ho interface ida ne'ebe permite troka
    fonte ba API polling iha futuru sem muda modulu seluk (haree README seksaun 12).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------- PATHS ----
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")


def _runtime_secret(name: str, default: str = "") -> str:
    """Read a local .env value or a Streamlit Community Cloud secret.

    Streamlit exposes root-level secrets as environment variables in most
    deployments. The explicit ``st.secrets`` fallback keeps the Apify feature
    working when the hosting environment does not export those values.
    """
    value = os.getenv(name)
    if value:
        return value.strip()

    try:
        import streamlit as st

        return str(st.secrets.get(name, default)).strip()
    except Exception:
        return default

# ------------------------------------------------------- APIFY (seksaun 5/10) --
# Fonte TUNGGAL: destinasaun + review, via Apify actor `compass/crawler-google-places`.
# Token sei LA iha .env -> sistema book funsiona hanesan antes (batch snapshot local
# iha data/raw/), haree data_collection.fetch_from_api().
APIFY_API_TOKEN = _runtime_secret("APIFY_API_TOKEN")
APIFY_ACTOR_ID = _runtime_secret("APIFY_ACTOR_ID", "compass/crawler-google-places")
APIFY_API_BASE = "https://api.apify.com/v2"
# Ficheiru ne'ebe ema bele edita MANUALMENTE (naran lokal/kota/kantidade) atu
# kontrola saida ne'ebe request husi Apify - haree apify_request.json iha
# project root. Se la iha, sistema book nia-an uza INPUT husi run successu
# ikus liu (haree data_collection.fetch_from_api()).
APIFY_INPUT_CONFIG = PROJECT_ROOT / "apify_request.json"
COLLECTION_POLL_INTERVAL_HOURS = float(os.getenv("COLLECTION_POLL_INTERVAL_HOURS", "24"))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
FEATURES_DIR = DATA_DIR / "features"
OUTPUT_DIR = DATA_DIR / "output"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

POWERBI_DIR = PROJECT_ROOT / "powerbi"
MODEL_DIR = PROJECT_ROOT / "Model"
SENTIMENT_MODEL_PATH = MODEL_DIR / "sentiment_model.joblib"
CATEGORY_MODEL_PATH = MODEL_DIR / "category_model.joblib"

# Raw dataset (snapshot Apify). Fallback ba lokal tuan atu la kesi kodigu bainhira
# ema seluk kopia dataset ba `data/` diretamente.
RAW_DATASET = RAW_DIR / "dataset_turris.json"
RAW_DATASET_FALLBACK = DATA_DIR / "dataset_turris.json"

# Tabela anotasaun manual (ground truth kategoria nivel-lokal) - haree
# `data/place_category_labels.csv` no README seksaun 5.
PLACE_LABELS_CSV = DATA_DIR / "place_category_labels.csv"

# Output ficheiru prinsipal
PROCESSED_CSV = PROCESSED_DIR / "reviews_processed.csv"
FEATURES_CSV = FEATURES_DIR / "review_features.csv"
OUTPUT_CSV = OUTPUT_DIR / "reviews_mined.csv"
EDA_SUMMARY_JSON = REPORTS_DIR / "eda_summary.json"
EVALUATION_JSON = REPORTS_DIR / "model_evaluation.json"
COLLECTION_LOG_JSON = REPORTS_DIR / "data_collection_log.json"
# Rezultadu fetch Apify ikus liu (haree data_collection.collect_from_api()) -
# dashboard le'e ficheiru ne'e atu hatudu "lokal saida mak foin mai" hanesan
# Apify Console nia Results tab.
APIFY_FETCH_LOG = REPORTS_DIR / "apify_fetch_log.json"

ALL_DIRS = [
    RAW_DIR, PROCESSED_DIR, FEATURES_DIR, OUTPUT_DIR,
    REPORTS_DIR, FIGURES_DIR, TABLES_DIR, POWERBI_DIR, MODEL_DIR,
]


def ensure_dirs() -> None:
    """Kria folder hotu-hotu se seidauk iha (idempotente)."""
    for d in ALL_DIRS:
        d.mkdir(parents=True, exist_ok=True)


def resolve_raw_dataset() -> Path:
    """Hetan lokal dataset raw PRINSIPAL (data/raw/dataset_turris.json uluk,
    depois data/) - uza nudar destinu backup/write ba collect_from_api().
    NOTA: pipeline la restrik ba ficheiru ida-ne'e deit, haree
    `list_raw_dataset_files()` ne'ebe le'e TODO *.json iha data/raw/."""
    if RAW_DATASET.exists():
        return RAW_DATASET
    if RAW_DATASET_FALLBACK.exists():
        return RAW_DATASET_FALLBACK
    raise FileNotFoundError(
        f"Dataset raw la hetan. Tau `dataset_turris.json` iha: {RAW_DATASET}"
    )


def list_raw_dataset_files() -> list:
    """Hetan TODO ficheiru `.json` iha data/raw/ (ordenadu husi tuan ba foun
    liu husi mtime), atu sistema prosesa KADA dadus ne'ebe iha-ne'ebá - laos
    deit ficheiru ho naran fixu `dataset_turris.json`."""
    return sorted(RAW_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime)


# ------------------------------------------------- DATA SOURCE METADATA ----
# Manual seksaun 10: "Dokumenta fonte dadus, URL/API, data/time collection no
# update interval."
DATA_SOURCE = {
    "source_name": "Google Maps - Places & Reviews (Timor-Leste tourist attractions)",
    "access_method": "Apify actor `compass/crawler-google-places` (export JSON)",
    "source_url": "https://www.google.com/maps",
    "api_status": (
        "SEIDAUK IHA API KEY. Google Places API presiza billing account, tanba ne'e "
        "grupu uza snapshot JSON husi scraper hanesan fonte dadus ba versaun ida-ne'e."
    ),
    "collection_mode": "batch snapshot (manual re-collection)",
    "update_interval": (
        "Manual/periodiku - rekomenda re-koleta kada 7 loron (weekly batch). "
        "Sistema LA'OS real-time; bainhira API ho polling disponivel, troka deit "
        "`data_collection.load_raw_data()` ba fetch HTTP + interval 5 minutu "
        "(near-real-time) tuir Manual seksaun 5 nia nota."
    ),
    "collection_timestamp_field": "scrapedAt",
    "unit_of_analysis": "1 baris = 1 review (komentariu turista) husi 1 lokal turizmu",
    "terms_of_service_note": (
        "Dadus publiku (review anonimu, la inklui email/telefone/dadus pessoal "
        "sensivel). Naran reviewer mantein deit ba dedupe teknika no LA hatudu iha "
        "dashboard - tuir Manual seksaun 10: 'Evita kolekta informasaun pessoal "
        "ne'ebe la nesesariu'."
    ),
}

# ------------------------------------------------------- BUSINESS RULES ----
# 4 kategoria fixu (pedidu dosente). Ordem ne'e uza ba grafiku/tabela hotu.
CATEGORIES = ["Historical", "Religious", "Traditional", "Modern"]

SENTIMENTS = ["Positive", "Neutral", "Negative"]
SENTIMENTS_WITH_UNKNOWN = SENTIMENTS + ["Unknown"]

# Kor konsistente entre dashboard Streamlit no Power BI
SENTIMENT_COLORS = {
    "Positive": "#2E9E5B",
    "Neutral": "#E2A33C",
    "Negative": "#D1495B",
    "Unknown": "#8A93A0",
}
CATEGORY_COLORS = {
    "Historical": "#6A4C93",
    "Religious": "#1B6CA8",
    "Traditional": "#C1666B",
    "Modern": "#2A9D8F",
}

RANDOM_STATE = 42
N_CLUSTERS = 4          # hanesan numeru kategoria, atu bele kompara
K_SWEEP = range(2, 9)   # elbow / silhouette sweep
TEST_SIZE = 0.25        # Manual seksaun 11: Train/Test Split
CV_FOLDS = 5


if __name__ == "__main__":
    ensure_dirs()
    print("PROJECT_ROOT :", PROJECT_ROOT)
    print("RAW dataset  :", resolve_raw_dataset())
    print("Kategoria    :", CATEGORIES)
