"""
export_powerbi.py
=================
LO6 | Manual seksaun 15 - Dashboard & Visualization (ferramenta: Power BI).

Exporta rezultadu pipeline ba konjuntu CSV ho MODELU ESTRELA (star schema)
ne'ebe Power BI Desktop bele import diretamente:

    FACT
      fact_reviews.csv            1 baris = 1 review (grain analiza)
    DIMENSAUN
      dim_place.csv               lokal turizmu (lat/lng ba mapa)
      dim_date.csv                tabela data kompletu (time intelligence)
      dim_category.csv            4 kategoria + kor + orden
      dim_sentiment.csv           4 label sentiment + kor + orden
      dim_cluster.csv             cluster K-Means + top terms
    AGREGASAUN / MINING RESULT
      agg_category_sentiment.csv  matriz kategoria x sentiment (+ %)
      agg_place_summary.csv       rezumu kada lokal
      agg_monthly_trend.csv       trend fulan x sentiment
      cluster_points.csv          koordenada 2D ba SCATTER PLOT cluster
      cluster_ksweep.csv          elbow + silhouette tuir k
    EVALUATION
      model_metrics.csv           metrika model hotu (Manual seksaun 14)
      confusion_matrix.csv        confusion matrix formatu tidy (matrix visual)
    METADATA
      meta_pipeline.csv           timestamp, fonte dadus, versaun (KPI "Last updated")
      data_dictionary.csv         diksionariu dadus (Manual seksaun 10)

Nota teknika: hotu grava ho encoding **utf-8-sig** tanba Power BI Desktop
presiza BOM atu le'e asentu (á, é, í, ó, ú, ç) ho loloos.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CATEGORIES,
    CATEGORY_COLORS,
    DATA_SOURCE,
    POWERBI_DIR,
    SENTIMENT_COLORS,
    SENTIMENTS_WITH_UNKNOWN,
    ensure_dirs,
)

ENC = "utf-8-sig"

# Kolona ne'ebe la exporta ba Power BI (interno/redundante ka dadus pessoal)
DROP_COLS = ["tokens", "reviewer_name", "address", "label_evidence"]

FACT_COLS = [
    "review_key", "place_id", "review_id", "date_key", "review_date_parsed",
    "review_year", "review_month", "review_yearmonth", "review_weekday",
    "category", "category_source", "category_label", "label_confidence", "category_rule_text",
    "category_rule_context", "category_rule_match", "category_rule_keywords",
    "category_pred_ml", "sentiment", "sentiment_source", "sentiment_lexicon",
    "sentiment_true_proxy", "sentiment_pred_ml", "sentiment_score",
    "sentiment_n_pos", "sentiment_n_neg", "review_stars", "rating_deviation",
    "review_text", "review_language", "n_tokens", "n_chars", "has_text",
    "has_rating", "is_long_review", "is_outlier_length", "is_local_guide",
    "days_since_review", "cluster", "cluster_hier", "split", "has_review", "sentiment_split", "category_split",
]


def _write(df: pd.DataFrame, name: str) -> str:
    path = POWERBI_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding=ENC)
    return str(path)


# ------------------------------------------------------------------- FACT ---
def build_fact_reviews(df: pd.DataFrame) -> pd.DataFrame:
    f = df.copy()
    f["date_key"] = f["review_date_parsed"].dt.strftime("%Y%m%d")
    f["review_date_parsed"] = f["review_date_parsed"].dt.tz_localize(None)
    f["category"] = f["category"].astype(str)

    # Teksu review limita ba 1000 karakter (Power BI la gosta teksu naruk demais)
    f["review_text"] = f["review_text"].astype("object").where(
        f["review_text"].notna(), None
    ).apply(lambda t: t[:1000] if isinstance(t, str) else None)

    cols = [c for c in FACT_COLS if c in f.columns]
    return f[cols]


# ------------------------------------------------------------- DIMENSAUN ----
def build_dim_place(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("place_id").agg(
        place_name=("place_name", "first"),
        city=("city", "first"),
        google_category=("place_category_google", "first"),
        lat=("lat", "first"),
        lng=("lng", "first"),
        place_avg_rating=("place_avg_rating", "first"),
        place_url=("place_url", "first"),
        category=("category", lambda s: str(s.iloc[0])),
        label_confidence=("label_confidence", "first"),
        n_reviews=("review_id", "count"),
        n_reviews_with_text=("has_text", "sum"),
        avg_review_stars=("review_stars", "mean"),
    ).reset_index()

    known = df[df["sentiment"].isin(["Positive", "Neutral", "Negative"])]
    pos = known.assign(is_pos=known["sentiment"].eq("Positive")).groupby("place_id")["is_pos"].mean()
    neg = known.assign(is_neg=known["sentiment"].eq("Negative")).groupby("place_id")["is_neg"].mean()
    agg["pct_positive"] = (agg["place_id"].map(pos) * 100).round(1)
    agg["pct_negative"] = (agg["place_id"].map(neg) * 100).round(1)
    agg["avg_review_stars"] = agg["avg_review_stars"].round(2)
    return agg


def build_dim_date(df: pd.DataFrame) -> pd.DataFrame:
    dates = df["review_date_parsed"].dropna()
    if dates.empty:
        return pd.DataFrame(columns=["date_key", "date"])
    start = dates.min().tz_localize(None).normalize()
    end = dates.max().tz_localize(None).normalize()
    idx = pd.date_range(start, end, freq="D")
    d = pd.DataFrame({"date": idx})
    d["date_key"] = d["date"].dt.strftime("%Y%m%d")
    d["year"] = d["date"].dt.year
    d["quarter"] = "Q" + d["date"].dt.quarter.astype(str)
    d["month_number"] = d["date"].dt.month
    d["month_name"] = d["date"].dt.strftime("%B")
    d["year_month"] = d["date"].dt.strftime("%Y-%m")
    d["day"] = d["date"].dt.day
    d["weekday_number"] = d["date"].dt.dayofweek + 1
    d["weekday_name"] = d["date"].dt.day_name()
    d["is_weekend"] = d["weekday_number"].isin([6, 7])
    return d[["date_key", "date", "year", "quarter", "month_number", "month_name",
              "year_month", "day", "weekday_number", "weekday_name", "is_weekend"]]


def build_dim_category() -> pd.DataFrame:
    desc = {
        "Historical": "Sitiu istoriku: memorial, edifisiu era kolonial, marka guerra/rezistensia",
        "Religious": "Sitiu relijiozu: gruta, igreja, dalan peregrinasaun",
        "Traditional": "Sitiu kultura lisan: uma lulik/lisan, artezanatu, aldeia tradisional",
        "Modern": "Atrasaun natureza, lazer no fasilidade turizmu kontemporaneu",
    }
    return pd.DataFrame([{
        "category": c,
        "category_sort": i + 1,
        "category_color": CATEGORY_COLORS[c],
        "category_description": desc[c],
    } for i, c in enumerate(CATEGORIES)])


def build_dim_sentiment() -> pd.DataFrame:
    score = {"Positive": 1, "Neutral": 0, "Negative": -1, "Unknown": None}
    return pd.DataFrame([{
        "sentiment": s,
        "sentiment_sort": i + 1,
        "sentiment_color": SENTIMENT_COLORS[s],
        "sentiment_score_value": score[s],
    } for i, s in enumerate(SENTIMENTS_WITH_UNKNOWN)])


def build_dim_cluster(cluster_results: dict) -> pd.DataFrame:
    profile = cluster_results.get("profile", [])
    if not profile:
        return pd.DataFrame(columns=["cluster", "cluster_label"])
    rows = []
    for p in profile:
        terms = p.get("top_terms", [])
        rows.append({
            "cluster": p["cluster"],
            "cluster_label": f"Cluster {p['cluster']}: " + ", ".join(terms[:3]),
            "size": p["size"],
            "share_pct": p["share_pct"],
            "avg_rating": p["avg_rating"],
            "avg_tokens": p["avg_tokens"],
            "dominant_category": p["dominant_category"],
            "dominant_sentiment": p["dominant_sentiment"],
            "dominant_language": p["dominant_language"],
            "top_terms": ", ".join(terms),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------ AGREGASAUN ----
def build_agg_category_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["sentiment"].isin(["Positive", "Neutral", "Negative"])]
    cross = pd.crosstab(df["category"].astype(str), df["sentiment"])
    cross = cross.reindex(index=CATEGORIES, fill_value=0)
    for s in SENTIMENTS_WITH_UNKNOWN:
        if s not in cross.columns:
            cross[s] = 0
    cross = cross[SENTIMENTS_WITH_UNKNOWN]

    tidy = cross.reset_index(names="category").melt(
        id_vars="category", var_name="sentiment", value_name="n_reviews"
    )
    total = tidy.groupby("category")["n_reviews"].transform("sum")
    tidy["pct_within_category"] = (tidy["n_reviews"] / total.replace(0, np.nan) * 100).round(2)
    tidy["total_category_reviews"] = total
    tidy = tidy.merge(build_dim_category()[["category", "category_sort"]], on="category")
    tidy = tidy.merge(build_dim_sentiment()[["sentiment", "sentiment_sort"]], on="sentiment")
    return tidy.sort_values(["category_sort", "sentiment_sort"])


def build_agg_monthly(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.dropna(subset=["review_date_parsed"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["year_month", "category", "sentiment", "n_reviews"])
    sub["year_month"] = sub["review_date_parsed"].dt.tz_localize(None).dt.strftime("%Y-%m")
    out = (sub.groupby(["year_month", sub["category"].astype(str), "sentiment"])
           .agg(n_reviews=("review_id", "count"),
                avg_stars=("review_stars", "mean"))
           .reset_index())
    out.columns = ["year_month", "category", "sentiment", "n_reviews", "avg_stars"]
    out["avg_stars"] = out["avg_stars"].round(2)
    return out


def build_cluster_points(df: pd.DataFrame) -> pd.DataFrame:
    """Koordenada 2D ba SCATTER PLOT cluster K-Means iha Power BI."""
    sub = df.dropna(subset=["cluster_x", "cluster_y"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["review_key", "cluster_x", "cluster_y", "cluster"])
    cols = ["review_key", "cluster", "cluster_x", "cluster_y", "place_name",
            "city", "category", "sentiment", "review_stars", "n_tokens",
            "review_text"]
    out = sub[[c for c in cols if c in sub.columns]].copy()
    out["category"] = out["category"].astype(str)
    out["review_text"] = out["review_text"].apply(
        lambda t: t[:200] if isinstance(t, str) else None
    )
    out["cluster"] = out["cluster"].astype(int)
    return out


# ------------------------------------------------------------ EVALUATION ----
def build_confusion_tidy(evaluation: dict) -> pd.DataFrame:
    """Konverte confusion matrix hotu ba formatu tidy (actual, predicted, n)."""
    rows = []

    def _add(block, task):
        if not block or block.get("status") != "ok":
            return
        labs = block["confusion_matrix_labels"]
        cm = block["confusion_matrix"]
        for i, actual in enumerate(labs):
            for j, pred in enumerate(labs):
                rows.append({
                    "task": task,
                    "model": block["model"],
                    "actual": actual,
                    "predicted": pred,
                    "n": int(cm[i][j]),
                    "is_correct": actual == pred,
                })

    _add(evaluation["sentiment"].get("model_1_lexicon"), "Sentiment")
    _add(evaluation["category"].get("model_1_rule_text"), "Kategoria")
    _add(evaluation["category"].get("model_1_rule_context"), "Kategoria")

    for task, key in (("Sentiment", "sentiment"), ("Kategoria", "category")):
        _add(evaluation[key].get("majority_baseline"), task)
        res = evaluation[key].get("model_2_supervised", {})
        if res.get("status") == "ok":
            for name, block in res["models"].items():
                b = dict(block)
                b["model"] = f"Model 2 - {name}"
                b["status"] = "ok"
                _add(b, task)
    return pd.DataFrame(rows)


def build_model_metrics(evaluation: dict) -> pd.DataFrame:
    return pd.DataFrame(evaluation.get("metrics_table", []))


def build_per_class_metrics(evaluation: dict) -> pd.DataFrame:
    """Precision/Recall/F1 kada klase - ba tabela detallu iha Power BI."""
    rows = []

    def _add(block, task):
        if not block or block.get("status") != "ok":
            return
        for lab, m in block.get("per_class", {}).items():
            rows.append({"task": task, "model": block["model"], "class": lab,
                         **m})

    _add(evaluation["sentiment"].get("model_1_lexicon"), "Sentiment")
    _add(evaluation["category"].get("model_1_rule_text"), "Kategoria")
    _add(evaluation["category"].get("model_1_rule_context"), "Kategoria")
    for task, key in (("Sentiment", "sentiment"), ("Kategoria", "category")):
        _add(evaluation[key].get("majority_baseline"), task)
        res = evaluation[key].get("model_2_supervised", {})
        if res.get("status") == "ok":
            for name, block in res["models"].items():
                b = dict(block)
                b["model"] = f"Model 2 - {name}"
                b["status"] = "ok"
                _add(b, task)
    return pd.DataFrame(rows)


# -------------------------------------------------------------- METADATA ----
def build_meta(df: pd.DataFrame, evaluation: dict, cluster_results: dict) -> pd.DataFrame:
    last_collect = df["scraped_at_parsed"].max()
    cm = cluster_results.get("metrics", {})
    sent_ml = evaluation["sentiment"].get("model_2_supervised", {})
    rows = [
        ("pipeline_run_utc", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")),
        ("last_data_collection_utc",
         last_collect.strftime("%Y-%m-%d %H:%M:%S") if pd.notna(last_collect) else "N/A"),
        ("data_source", DATA_SOURCE["source_name"]),
        ("access_method", DATA_SOURCE["access_method"]),
        ("collection_mode", DATA_SOURCE["collection_mode"]),
        ("update_interval", DATA_SOURCE["update_interval"]),
        ("total_places", str(df["place_id"].nunique())),
        ("total_reviews", str(int(df["has_review"].sum()))),
        ("reviews_with_text", str(int(df["has_text"].sum()))),
        ("clustering_algorithm", str(cm.get("algorithm", "N/A"))),
        ("clustering_k", str(cm.get("k_used", "N/A"))),
        ("silhouette_score", str(cm.get("silhouette_score", "N/A"))),
        ("best_sentiment_model", str(sent_ml.get("best_model", "N/A"))),
        ("best_sentiment_f1_macro", str(sent_ml.get("best_f1_macro", "N/A"))),
        ("python_version", "3.11"),
    ]
    return pd.DataFrame(rows, columns=["metric_name", "metric_value"])


DATA_DICTIONARY = [
    ("place_id", "text", "ID uniku lokal turizmu (Google placeId)", "-", "Nunka mamuk"),
    ("place_name", "text", "Naran lokal turizmu", "-", "Nunka mamuk"),
    ("city", "text", "Cidade/munisipiu (imputa husi address se mamuk)", "-", "296 seidauk hetan depois imputasaun"),
    ("lat / lng", "decimal", "Koordenada GPS lokal", "grau", "0"),
    ("place_avg_rating", "decimal", "Rating mediu lokal iha Google", "estrela 1-5", "Bele mamuk"),
    ("review_id", "text", "ID uniku review", "-", "Mamuk ba lokal sem review"),
    ("review_text", "text", "Teksu komentariu orijinal turista", "-", "~42% mamuk (rating-only/sem review)"),
    ("review_stars", "integer", "Rating individual husi review", "estrela 1-5", "262 mamuk"),
    ("review_date_parsed", "datetime", "Data-oras review publika (UTC)", "-", "Bele mamuk"),
    ("review_language", "text", "Kodigu lingua orijinal (en/id/pt/tet/...)", "-", "'unknown' se mamuk"),
    ("text_clean", "text", "Teksu depois lowercase + hasai url/emoji/simbolu", "-", "'' se sem teksu"),
    ("tokens_joined", "text", "Token depois hasai stopwords (input ba TF-IDF)", "-", "'' se sem teksu"),
    ("n_tokens", "integer", "Kuantidade token depois preprocessing", "token", "0 se sem teksu"),
    ("n_chars", "integer", "Kuantidade karakter teksu orijinal", "karakter", "0 se sem teksu"),
    ("category_label", "text", "ANOTASAUN MANUAL kategoria (ground truth, 49/569 lokal deit)", "Historical/Religious/Traditional/Modern", "3.145 mamuk (520 lokal seidauk anotadu)"),
    ("label_confidence", "text", "Nivel konfiansa anotasaun manual", "high/medium/low", "Mamuk se category_label mamuk"),
    ("category", "text", "Kategoria FINAL (= category_label se iha, senaun fallback rule-based)", "4 kategoria", "Nunka mamuk"),
    ("category_rule_text", "text", "Predisaun Model 1 husi teksu review deit", "4 kategoria", "Nunka mamuk"),
    ("category_rule_context", "text", "Predisaun Model 1 husi teksu + naran lokal", "4 kategoria", "Nunka mamuk"),
    ("category_pred_ml", "text", "Predisaun Model 2 (supervisadu) - deit ba test set", "4 kategoria", "Mamuk se laos test set"),
    ("sentiment", "text", "Sentiment FINAL (teksu > rating > Unknown)", "Positive/Neutral/Negative/Unknown", "Nunka mamuk"),
    ("sentiment_source", "text", "Fonte label sentiment", "text/rating/none", "Nunka mamuk"),
    ("sentiment_lexicon", "text", "Predisaun Model 1 lexicon husi teksu", "3 label", "Mamuk se sem teksu"),
    ("sentiment_true_proxy", "text", "Ground truth PROXY deriva husi rating", "3 label", "Mamuk se sem rating"),
    ("sentiment_pred_ml", "text", "Predisaun Model 2 (supervisadu) - deit ba test set", "3 label", "Mamuk se laos test set"),
    ("sentiment_score", "integer", "n_liafuan_positivu - n_liafuan_negativu", "kontajen", "0 se sem teksu"),
    ("cluster", "integer", "ID cluster K-Means", "0..k-1", "Mamuk se sem teksu"),
    ("cluster_x / cluster_y", "decimal", "Koordenada TruncatedSVD 2D ba scatter plot", "-", "Mamuk se sem teksu"),
    ("cluster_hier", "integer", "ID cluster Hierarchical (komparasaun)", "0..k-1", "Mamuk se sem teksu"),
    ("split", "text", "Split preliminar preprocessing; evaluation uza sentiment_split/category_split", "train/test/not_used", "Nunka mamuk"),
    ("sentiment_split", "text", "Split atual ba tarefa sentiment; estratifikadu tuir proxy rating", "train/test/not_used", "Nunka mamuk"),
    ("category_split", "text", "Split atual ba kategoria; lokal train no test separadu", "train/test/not_used", "Nunka mamuk"),
    ("has_review", "boolean", "True se baris iha review; False se lokal sem review", "true/false", "Nunka mamuk"),
    ("category_source", "text", "Fonte kategoria final nivel lokal", "anotasaun_manual/rule_based_fallback", "Nunka mamuk"),
    ("is_outlier_length", "boolean", "Flag outlier naruk teksu (metodu IQR)", "true/false", "Nunka mamuk"),
    ("scraped_at_parsed", "datetime", "Timestamp kolesaun dadus (Last updated)", "-", "Nunka mamuk"),
]


def build_data_dictionary() -> pd.DataFrame:
    return pd.DataFrame(DATA_DICTIONARY, columns=[
        "column", "data_type", "description", "unit_or_values", "missing_values"
    ])


# ----------------------------------------------------------- ORKESTRASAUN ---
def export_all(df: pd.DataFrame, eda: dict, evaluation: dict,
               cluster_results: dict) -> dict:
    """Exporta ficheiru hotu ba folder powerbi/. Retorna map naran -> path."""
    ensure_dirs()
    out = {}

    # Xave uniku nivel-review, konsistente entre fact_reviews no cluster_points
    # (relasaun 1:1 iha modelu Power BI).
    df = df.reset_index(drop=True).copy()
    df["review_key"] = np.arange(1, len(df) + 1)

    out["fact_reviews"] = _write(build_fact_reviews(df), "fact_reviews")
    out["dim_place"] = _write(build_dim_place(df), "dim_place")
    out["dim_date"] = _write(build_dim_date(df), "dim_date")
    out["dim_category"] = _write(build_dim_category(), "dim_category")
    out["dim_sentiment"] = _write(build_dim_sentiment(), "dim_sentiment")
    out["dim_cluster"] = _write(build_dim_cluster(cluster_results), "dim_cluster")

    out["agg_category_sentiment"] = _write(
        build_agg_category_sentiment(df), "agg_category_sentiment")
    out["agg_place_summary"] = _write(build_dim_place(df), "agg_place_summary")
    out["agg_monthly_trend"] = _write(build_agg_monthly(df), "agg_monthly_trend")

    out["cluster_points"] = _write(build_cluster_points(df), "cluster_points")
    out["cluster_ksweep"] = _write(
        pd.DataFrame(cluster_results.get("k_sweep", [])), "cluster_ksweep")

    out["model_metrics"] = _write(build_model_metrics(evaluation), "model_metrics")
    out["model_metrics_per_class"] = _write(
        build_per_class_metrics(evaluation), "model_metrics_per_class")
    out["confusion_matrix"] = _write(build_confusion_tidy(evaluation), "confusion_matrix")

    out["eda_descriptive_stats"] = _write(
        pd.DataFrame(eda.get("descriptive_statistics", [])), "eda_descriptive_stats")
    out["meta_pipeline"] = _write(
        build_meta(df, evaluation, cluster_results), "meta_pipeline")
    out["data_dictionary"] = _write(build_data_dictionary(), "data_dictionary")

    return out


if __name__ == "__main__":
    from category_classifier import apply_category_classification
    from clustering import run_clustering
    from data_collection import collect
    from eda import run_eda
    from preprocessing import preprocess
    from sentiment_analysis import apply_sentiment_analysis
    from supervised_models import run_supervised_models
    from model_evaluation import run_evaluation

    raw = collect()
    clean, _ = preprocess(raw)
    d = apply_category_classification(clean)
    d = apply_sentiment_analysis(d)
    d, cres = run_clustering(d)
    d, sres = run_supervised_models(d)
    eda_res = run_eda(d, make_figures=False)
    ev = run_evaluation(d, cres, sres, make_figures=False)
    paths = export_all(d, eda_res, ev, cres)
    for name, p in paths.items():
        print(f"  {name:26s} -> {p}")
