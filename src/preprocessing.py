"""
preprocessing.py
================
LO2 | Manual seksaun 11 - Data Preprocessing.

Implementa ETAPA HOTU-HOTU husi tabela Manual seksaun 11:

    Etapa                Atividade iha sistema ida-ne'e
    -------------------  ------------------------------------------------------
    Cleaning             Remove duplicate review, hasai baris invalidu
    Missing Values       Tratamentu esplisitu (marka `has_text`, `sentiment_source`,
                         imputasaun `city` husi address, la halo drop siega)
    Transformation       Date parsing (UTC), normalizasaun teksu, encoding kategoria
    Outliers             Deteta ho metodu IQR ba tamanhu teksu -> FLAG (la hasai)
    Text                 Lowercase, hasai URL/emoji/simbolu, tokenization, stopwords
    Feature Engineering  n_tokens, n_chars, review_year/month, days_since_review, ...
    Train/Test Split     Kolona `split` (train/test) ho stratifikasaun

Prinsipiu: LA HASAI dadus sem dokumenta. Kada baris ne'ebe hasai konta iha
`meta` no hatudu iha dashboard (transparensia - LO7).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import PROCESSED_CSV, RANDOM_STATE, TEST_SIZE, ensure_dirs

# ---------------------------------------------------------------------------
# Stopwords multi-lingua: Tetun, Indonesia/Malay, Portugues, Ingles.
# Dataset ida-ne'e mistura lingua hat, tanba turista internasional no lokal
# hakerek iha lingua sira-ne'e (haree EDA: distribuisaun `review_language`).
# ---------------------------------------------------------------------------
STOPWORDS = set("""
a o os as um uma uns umas de da do das dos em no na nos nas para por com sem
que e eh sao foi ser esta este isto essa esse aqui ali muito mais menos tambem
mas ou como ja nao ao aos pelo pela seu sua
di ke yang dan atau dengan untuk dari pada ini itu adalah tidak ada juga saja
akan bisa saya kami kita anda dia mereka sangat sekali karena kalau
ida nee neebe nebee la iha ho tan liu husi mak ba nia sira karik uma ne
ita hau ami imi maibe atu hanesan tuir hotu deit tebes bele
the an is are was were of to in on at for and or with this that it its
be been being have has had do does did not no so very there their they you
your we our us he she him her from by as if then than too can will just
""".split())

URL_RE = re.compile(r"http\S+|www\.\S+")
EMOJI_RE = re.compile(
    "["
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F1E6-\U0001F1FF"
    "\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)
NON_ALPHA_RE = re.compile(r"[^a-zA-ZÀ-ÿ\s]")
MULTISPACE_RE = re.compile(r"\s+")


# ------------------------------------------------------------------ TEXT ----
def clean_text(text: str) -> str:
    """Normaliza 1 review: lowercase, hasai url/emoji/numeru/simbolu, espasu."""
    if not isinstance(text, str) or not text.strip():
        return ""
    t = text.lower()
    t = URL_RE.sub(" ", t)
    t = EMOJI_RE.sub(" ", t)
    t = NON_ALPHA_RE.sub(" ", t)
    t = MULTISPACE_RE.sub(" ", t).strip()
    return t


def tokenize(text_clean: str) -> list:
    """Tokenization + hasai stopwords + hasai token badak liu (< 2 karakter)."""
    if not text_clean:
        return []
    return [tok for tok in text_clean.split() if tok not in STOPWORDS and len(tok) > 1]


# -------------------------------------------------------------- OUTLIERS ----
def detect_outliers_iqr(series: pd.Series, factor: float = 1.5) -> tuple:
    """
    Deteta outlier ho metodu IQR (Manual seksaun 11: 'Detect and document
    treatment'). Retorna (mask_boolean, info_dict).

    TRATAMENTU ne'ebe eskolhe: FLAG deit, LA HASAI. Razaun: review naruk
    (ex: 1.500 karakter) mak informasaun valiozu tebes ba text mining, laos
    erru dadus. Hasai sira sei lakon insight. Outlier flag uza deit atu
    dokumenta no filtra iha dashboard se presiza.
    """
    valid = series.dropna()
    if valid.empty:
        return pd.Series(False, index=series.index), {"status": "no_data"}
    q1, q3 = float(valid.quantile(0.25)), float(valid.quantile(0.75))
    iqr = q3 - q1
    low, high = q1 - factor * iqr, q3 + factor * iqr
    mask = (series < low) | (series > high)
    mask = mask.fillna(False)
    return mask, {
        "q1": round(q1, 2),
        "q3": round(q3, 2),
        "iqr": round(iqr, 2),
        "lower_bound": round(low, 2),
        "upper_bound": round(high, 2),
        "n_outliers": int(mask.sum()),
        "treatment": "flag_only_not_removed",
    }


# ----------------------------------------------------------- MAIN ROUTINE ----
def preprocess(df: pd.DataFrame, save: bool = False) -> tuple:
    """
    Aplika pipeline preprocessing kompletu ba DataFrame nivel-review husi
    `data_collection.collect()`.

    Retorna (df_processed, meta_dict).
    """
    df = df.copy()
    meta = {}

    # --- 1. CLEANING -------------------------------------------------------
    rows_before = len(df)

    # (a) duplicate ho review_id hanesan iha lokal hanesan
    df = df.drop_duplicates(subset=["place_id", "review_id"], keep="first")
    dup_by_id = rows_before - len(df)

    # (b) duplicate "logika": lokal + teksu + reviewer hanesan (review kopia)
    n = len(df)
    df = df.drop_duplicates(subset=["place_id", "review_text", "reviewer_name"], keep="first")
    dup_by_content = n - len(df)

    # (c) baris invalidu: la iha place_id (imposivel liga ba lokal)
    n = len(df)
    df = df[df["place_id"].notna()]
    invalid_rows = n - len(df)

    meta["cleaning"] = {
        "rows_before": rows_before,
        "duplicates_by_review_id": dup_by_id,
        "duplicates_by_content": dup_by_content,
        "invalid_rows_removed": invalid_rows,
        "rows_after": len(df),
    }

    # --- 2. MISSING VALUES -------------------------------------------------
    # Estratejia: tratamentu ESPLISITU (la halo drop), tanba review sem teksu
    # (rating-only) mantein valor analitiku ba EDA no ba sentiment via rating.
    df["has_text"] = df["review_text"].apply(
        lambda x: isinstance(x, str) and x.strip() != ""
    )
    df["has_rating"] = df["review_stars"].notna()

    # Imputasaun `city`: se mamuk, foti liafuan ikus husi address (ex: "..., Ainaro")
    def _city_from_address(row):
        if isinstance(row.get("city"), str) and row["city"].strip():
            return row["city"]
        addr = row.get("address")
        if isinstance(addr, str) and "," in addr:
            return addr.split(",")[-1].strip()
        return "La hatene"

    city_missing_before = int(df["city"].isna().sum())
    df["city"] = df.apply(_city_from_address, axis=1)

    meta["missing_values"] = {
        "review_text_missing": int((~df["has_text"]).sum()),
        "review_text_missing_pct": round(float((~df["has_text"]).mean() * 100), 2),
        "review_stars_missing": int((~df["has_rating"]).sum()),
        "city_missing_before_imputation": city_missing_before,
        "city_still_unknown": int((df["city"] == "La hatene").sum()),
        "lat_lng_missing": int(df["lat"].isna().sum()),
        "strategy": (
            "Tratamentu esplisitu: la halo drop. `has_text`/`has_rating` marka "
            "disponibilidade; `city` imputa husi `address`; koordenada mamuk "
            "esklui deit husi mapa."
        ),
    }

    # --- 3. TRANSFORMATION -------------------------------------------------
    df["review_date_parsed"] = pd.to_datetime(
        df["review_date"], errors="coerce", utc=True, format="mixed"
    )
    df["scraped_at_parsed"] = pd.to_datetime(
        df["scraped_at"], errors="coerce", utc=True, format="mixed"
    )
    df["review_stars"] = pd.to_numeric(df["review_stars"], errors="coerce")
    df["place_avg_rating"] = pd.to_numeric(df["place_avg_rating"], errors="coerce")

    # --- 4. TEXT PROCESSING ------------------------------------------------
    df["text_clean"] = df["review_text"].apply(clean_text)
    df["tokens"] = df["text_clean"].apply(tokenize)
    df["tokens_joined"] = df["tokens"].apply(lambda t: " ".join(t))
    df["n_tokens"] = df["tokens"].apply(len)
    df["n_chars"] = df["review_text"].apply(
        lambda x: len(x) if isinstance(x, str) else 0
    )

    # Review ho teksu maibe depois cleaning sai mamuk (ex: emoji deit ka numeru deit)
    df["text_empty_after_clean"] = df["has_text"] & (df["tokens_joined"].str.len() == 0)

    meta["text"] = {
        "reviews_with_text": int(df["has_text"].sum()),
        "empty_after_cleaning": int(df["text_empty_after_clean"].sum()),
        "avg_tokens_per_review": round(
            float(df.loc[df["has_text"], "n_tokens"].mean()), 2
        ) if df["has_text"].any() else 0.0,
        "vocabulary_size": int(len(set(
            tok for toks in df.loc[df["has_text"], "tokens"] for tok in toks
        ))),
        "steps": ["lowercase", "remove_url", "remove_emoji", "remove_non_alpha",
                  "tokenize", "remove_stopwords", "drop_tokens_len_lt_2"],
    }

    # --- 5. OUTLIERS -------------------------------------------------------
    len_mask, len_info = detect_outliers_iqr(
        df.loc[df["has_text"], "n_chars"].reindex(df.index)
    )
    df["is_outlier_length"] = len_mask
    meta["outliers"] = {"review_length_chars": len_info}

    # --- 6. FEATURE ENGINEERING -------------------------------------------
    df["is_long_review"] = df["n_tokens"] >= 8
    df["review_year"] = df["review_date_parsed"].dt.year
    df["review_month"] = df["review_date_parsed"].dt.month
    df["review_yearmonth"] = df["review_date_parsed"].dt.tz_localize(None).dt.to_period("M").astype(str)
    df["review_weekday"] = df["review_date_parsed"].dt.day_name()

    ref = df["scraped_at_parsed"].max()
    df["days_since_review"] = (
        (ref - df["review_date_parsed"]).dt.days if pd.notna(ref) else np.nan
    )
    df["is_local_guide"] = df["is_local_guide"].fillna(False).astype(bool)
    df["review_language"] = df["review_language"].fillna("unknown")
    df["rating_deviation"] = df["review_stars"] - df["place_avg_rating"]

    meta["feature_engineering"] = {
        "created": [
            "has_text", "has_rating", "n_tokens", "n_chars", "is_long_review",
            "is_outlier_length", "review_year", "review_month", "review_yearmonth",
            "review_weekday", "days_since_review", "rating_deviation",
        ],
        "reference_date_for_recency": str(ref) if pd.notna(ref) else None,
    }

    # --- 7. TRAIN/TEST SPLIT ----------------------------------------------
    # Split halao DEIT ba review ho teksu (sira mak entrada ba model supervisadu).
    df["split"] = "not_used"
    modelable = df.index[df["has_text"] & (df["tokens_joined"].str.len() > 0)]
    if len(modelable) >= 8:
        strat = df.loc[modelable, "category_label"] if "category_label" in df.columns else None
        try:
            train_idx, test_idx = train_test_split(
                modelable, test_size=TEST_SIZE,
                random_state=RANDOM_STATE, stratify=strat,
            )
        except ValueError:
            # Klase ho membru < 2 -> stratifikasaun la posivel
            train_idx, test_idx = train_test_split(
                modelable, test_size=TEST_SIZE, random_state=RANDOM_STATE,
            )
        df.loc[train_idx, "split"] = "train"
        df.loc[test_idx, "split"] = "test"

    meta["train_test_split"] = {
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": int((df["split"] == "train").sum()),
        "n_test": int((df["split"] == "test").sum()),
        "n_not_used": int((df["split"] == "not_used").sum()),
        "note": "Split aplika deit ba review ho teksu (input ba model supervisadu).",
    }

    if save:
        ensure_dirs()
        save_cols = [c for c in df.columns if c != "tokens"]
        df[save_cols].to_csv(PROCESSED_CSV, index=False, encoding="utf-8-sig")
        meta["saved_to"] = str(PROCESSED_CSV)

    return df, meta


if __name__ == "__main__":
    from data_collection import collect

    raw = collect()
    clean, meta = preprocess(raw, save=True)
    import json
    print(json.dumps(meta, indent=2, ensure_ascii=False, default=str))
    print()
    print(clean[["place_name", "category_label", "review_text", "n_tokens", "split"]].head(10).to_string())
