"""
eda.py
======
LO3 | Manual seksaun 12 - Exploratory Data Analysis.

Implementa PONTU HOTU-HOTU husi lista Manual seksaun 12:

    [x] Descriptive statistics : count, mean, median, min, max, std
    [x] Distribution analysis  : histogram, boxplot, density
    [x] Trend analysis         : line chart tuir tempu
    [x] Relationship analysis  : correlation matrix, scatterplot
    [x] Category analysis      : frequency no ranking
    [x] Geographical analysis  : dataset iha koordenada lat/lng
    [x] Insight prinsipal      : lista pergunta ne'ebe data mining sei responde

Output:
    reports/figures/*.png   -> figura ba relatoriu no PowerPoint
    reports/tables/*.csv    -> tabela tidy (bele import ba Power BI/Word)
    dict `eda`              -> grava iha reports/eda_summary.json
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # la presiza janela GUI (bele halao iha servidor/CI)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CATEGORIES,
    CATEGORY_COLORS,
    FIGURES_DIR,
    SENTIMENT_COLORS,
    SENTIMENTS_WITH_UNKNOWN,
    TABLES_DIR,
    ensure_dirs,
)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 120,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

NUMERIC_COLS = [
    "review_stars", "place_avg_rating", "n_tokens", "n_chars",
    "days_since_review", "rating_deviation", "place_reviews_count",
]


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def _save_table(df: pd.DataFrame, name: str) -> str:
    path = TABLES_DIR / f"{name}.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


# ------------------------------------------------------------------------- #
# 1. DESCRIPTIVE STATISTICS
# ------------------------------------------------------------------------- #
def descriptive_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """count, mean, median, min, max, std ba kolona numeriku hotu."""
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    stats = df[cols].describe().T
    stats["median"] = df[cols].median()
    stats["missing"] = df[cols].isna().sum()
    stats["missing_pct"] = (df[cols].isna().mean() * 100).round(2)
    stats = stats.rename(columns={"50%": "p50", "25%": "p25", "75%": "p75"})
    order = ["count", "missing", "missing_pct", "mean", "median", "std",
             "min", "p25", "p50", "p75", "max"]
    stats = stats[[c for c in order if c in stats.columns]].round(2)
    return stats.reset_index(names="variable")


# ------------------------------------------------------------------------- #
# 2-7. FIGURA SIRA
# ------------------------------------------------------------------------- #
def plot_rating_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    ratings = df["review_stars"].dropna()
    axes[0].hist(ratings, bins=np.arange(0.5, 6.5, 1), color="#2A9D8F",
                 edgecolor="white")
    axes[0].set_title("Histogram rating review")
    axes[0].set_xlabel("Estrela")
    axes[0].set_ylabel("Kuantidade review")
    axes[0].set_xticks([1, 2, 3, 4, 5])

    data = [df.loc[df["category"].astype(str) == c, "review_stars"].dropna()
            for c in CATEGORIES]
    bp = axes[1].boxplot(data, tick_labels=CATEGORIES, patch_artist=True)
    for patch, cat in zip(bp["boxes"], CATEGORIES):
        patch.set_facecolor(CATEGORY_COLORS[cat])
        patch.set_alpha(0.75)
    axes[1].set_title("Boxplot rating tuir kategoria")
    axes[1].set_ylabel("Estrela")
    axes[1].tick_params(axis="x", rotation=15)
    fig.tight_layout()
    return _save(fig, "01_distribuisaun_rating")


def plot_text_length_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    lengths = df.loc[df["has_text"], "n_chars"]
    axes[0].hist(lengths, bins=30, color="#6A4C93", edgecolor="white")
    axes[0].set_title("Distribuisaun naruk teksu review (karakter)")
    axes[0].set_xlabel("Karakter")
    axes[0].set_ylabel("Kuantidade review")

    lengths.plot(kind="density", ax=axes[1], color="#6A4C93")
    axes[1].set_title("Density - naruk teksu review")
    axes[1].set_xlabel("Karakter")
    axes[1].set_xlim(left=0)
    fig.tight_layout()
    return _save(fig, "02_distribuisaun_naruk_teksu")


def plot_category_sentiment(df: pd.DataFrame) -> tuple:
    """FIGURA XAVE ba pedidu dosente: sentiment iha kategoria ida-idak."""
    df = df[df["sentiment"].isin(["Positive", "Neutral", "Negative"])]
    cross = pd.crosstab(df["category"].astype(str), df["sentiment"])
    cross = cross.reindex(index=CATEGORIES, fill_value=0)
    cross = cross.reindex(columns=[s for s in SENTIMENTS_WITH_UNKNOWN
                                   if s in cross.columns], fill_value=0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    bottom = np.zeros(len(cross))
    x = np.arange(len(cross))
    for sent in cross.columns:
        axes[0].bar(x, cross[sent].values, bottom=bottom, label=sent,
                    color=SENTIMENT_COLORS.get(sent, "#999"))
        bottom += cross[sent].values
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(cross.index, rotation=15)
    axes[0].set_title("Sentiment tuir kategoria (absolutu)")
    axes[0].set_ylabel("Kuantidade review")
    axes[0].legend(fontsize=7)

    pct = cross.div(cross.sum(axis=1).replace(0, np.nan), axis=0) * 100
    bottom = np.zeros(len(pct))
    for sent in pct.columns:
        vals = pct[sent].fillna(0).values
        axes[1].bar(x, vals, bottom=bottom, label=sent,
                    color=SENTIMENT_COLORS.get(sent, "#999"))
        bottom += vals
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(pct.index, rotation=15)
    axes[1].set_title("Sentiment tuir kategoria (%)")
    axes[1].set_ylabel("% review")
    fig.tight_layout()
    path = _save(fig, "03_kategoria_vs_sentiment")

    tidy = cross.reset_index(names="category").melt(
        id_vars="category", var_name="sentiment", value_name="n_reviews"
    )
    total = tidy.groupby("category")["n_reviews"].transform("sum")
    tidy["pct_within_category"] = (tidy["n_reviews"] / total.replace(0, np.nan) * 100).round(2)
    return path, cross, tidy


def plot_trend(df: pd.DataFrame) -> str:
    trend_df = df.dropna(subset=["review_date_parsed"]).copy()
    if trend_df.empty:
        return ""
    trend_df["ym"] = trend_df["review_date_parsed"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 5.6), sharex=True)
    monthly = trend_df.groupby("ym").size()
    axes[0].plot(monthly.index, monthly.values, marker="o", color="#2A9D8F", lw=1.5)
    axes[0].set_title("Volume review kada fulan")
    axes[0].set_ylabel("Kuantidade review")

    by_sent = trend_df.groupby(["ym", "sentiment"]).size().unstack(fill_value=0)
    for sent in by_sent.columns:
        axes[1].plot(by_sent.index, by_sent[sent], marker="o", lw=1.3,
                     label=sent, color=SENTIMENT_COLORS.get(sent, "#999"))
    axes[1].set_title("Trend sentiment kada fulan")
    axes[1].set_ylabel("Kuantidade review")
    axes[1].set_xlabel("Fulan")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "04_trend_tempu")


def plot_correlation(df: pd.DataFrame) -> tuple:
    cols = [c for c in NUMERIC_COLS if c in df.columns]
    corr = df[cols].corr(numeric_only=True).round(3)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    im = axes[0].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    axes[0].set_xticks(range(len(cols)))
    axes[0].set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(len(cols)))
    axes[0].set_yticklabels(cols, fontsize=7)
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr.values[i, j]
            if not np.isnan(val):
                axes[0].text(j, i, f"{val:.2f}", ha="center", va="center",
                             fontsize=6,
                             color="white" if abs(val) > 0.55 else "#222")
    axes[0].set_title("Matriz korelasaun (Pearson)")
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], fraction=0.046)

    sub = df[df["has_text"]]
    for cat in CATEGORIES:
        s = sub[sub["category"].astype(str) == cat]
        axes[1].scatter(s["n_tokens"], s["review_stars"], s=22, alpha=0.7,
                        label=cat, color=CATEGORY_COLORS[cat],
                        edgecolors="white", linewidths=0.4)
    axes[1].set_xlabel("Kuantidade token (naruk review)")
    axes[1].set_ylabel("Rating (estrela)")
    axes[1].set_title("Scatter: naruk review vs rating")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    return _save(fig, "05_korelasaun_scatter"), corr


def plot_category_ranking(df: pd.DataFrame) -> str:
    df = df[df["has_review"]]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    cat = df["category"].astype(str).value_counts().reindex(CATEGORIES, fill_value=0)
    axes[0].bar(cat.index, cat.values,
                color=[CATEGORY_COLORS[c] for c in cat.index])
    for i, v in enumerate(cat.values):
        axes[0].text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Frekuensia kategoria")
    axes[0].tick_params(axis="x", rotation=15)

    top_places = df.groupby("place_name").size().sort_values(ascending=False).head(10)
    axes[1].barh(top_places.index[::-1], top_places.values[::-1], color="#2A9D8F")
    axes[1].set_title("Top 10 lokal ho review barak liu")
    axes[1].tick_params(axis="y", labelsize=7)

    top_city = df["city"].value_counts().head(10)
    axes[2].barh(top_city.index[::-1], top_city.values[::-1], color="#E2A33C")
    axes[2].set_title("Top 10 cidade/munisipiu")
    axes[2].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return _save(fig, "06_ranking_kategoria_lokal")


def plot_geographic(df: pd.DataFrame) -> str:
    places = df.dropna(subset=["lat", "lng"]).drop_duplicates(subset=["place_id"])
    if places.empty:
        return ""
    fig, ax = plt.subplots(figsize=(8, 5))
    sizes = df.groupby("place_id")["has_review"].sum()
    places = places.assign(n_reviews=places["place_id"].map(sizes).fillna(0))
    for cat in CATEGORIES:
        s = places[places["category"].astype(str) == cat]
        ax.scatter(s["lng"], s["lat"], s=20 + s["n_reviews"] * 4, alpha=0.75,
                   label=cat, color=CATEGORY_COLORS[cat],
                   edgecolors="white", linewidths=0.6)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Distribuisaun jeografika lokal turizmu\n(tamanhu = kuantidade review)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, "07_mapa_jeografiku")


def plot_language(df: pd.DataFrame) -> str:
    langs = df.loc[df["has_text"], "review_language"].value_counts().head(10)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.bar(langs.index.astype(str), langs.values, color="#1B6CA8")
    for i, v in enumerate(langs.values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=8)
    ax.set_title("Lingua orijinal review (top 10)")
    ax.set_ylabel("Kuantidade review")
    fig.tight_layout()
    return _save(fig, "08_lingua_review")


# ------------------------------------------------------------------------- #
# ORKESTRASAUN
# ------------------------------------------------------------------------- #
def run_eda(df: pd.DataFrame, make_figures: bool = True) -> dict:
    """Halao EDA kompletu; retorna dict rezumu no grava figura + tabela."""
    ensure_dirs()

    stats = descriptive_statistics(df)
    reviews = df[df["has_review"]]
    known = df[df["sentiment"].isin(["Positive", "Neutral", "Negative"])]
    cat_counts = reviews["category"].astype(str).value_counts().reindex(CATEGORIES, fill_value=0)
    sent_counts = df["sentiment"].value_counts()

    figures, tables = {}, {}
    cross_tidy = None

    if make_figures:
        figures["rating_distribution"] = plot_rating_distribution(df)
        figures["text_length"] = plot_text_length_distribution(df)
        fig_cs, cross, cross_tidy = plot_category_sentiment(df)
        figures["category_vs_sentiment"] = fig_cs
        figures["trend"] = plot_trend(df)
        fig_corr, corr = plot_correlation(df)
        figures["correlation"] = fig_corr
        figures["ranking"] = plot_category_ranking(df)
        figures["geographic"] = plot_geographic(df)
        figures["language"] = plot_language(df)

        tables["descriptive_statistics"] = _save_table(stats, "descriptive_statistics")
        tables["category_sentiment"] = _save_table(cross_tidy, "category_sentiment")
        tables["correlation_matrix"] = _save_table(
            corr.reset_index(names="variable"), "correlation_matrix"
        )
    else:
        cross = pd.crosstab(known["category"].astype(str), known["sentiment"])
        corr = df[[c for c in NUMERIC_COLS if c in df.columns]].corr(numeric_only=True).round(3)

    # ------------------------------------------------------------- rezumu ---
    with_text = df[df["has_text"]]
    dates = df["review_date_parsed"]

    # Insight automatiku: kategoria ho % positivu aas liu / kraik liu
    pct_pos = (
        known.assign(is_pos=known["sentiment"].eq("Positive"))
        .groupby(known["category"].astype(str))["is_pos"].mean() * 100
    ).round(1).reindex(CATEGORIES).dropna()

    eda = {
        "overview": {
            "total_places": int(df["place_id"].nunique()),
            "total_rows": int(len(df)),
            "total_reviews": int(df["has_review"].sum()) if "has_review" in df else int(len(df)),
            "reviews_with_text": int(df["has_text"].sum()),
            "reviews_rating_only": int((df["has_review"] & ~df["has_text"] & df["has_rating"]).sum()),
            "places_without_review": int((~df["has_review"]).sum()),
            "avg_rating_overall": round(float(df["review_stars"].dropna().mean()), 2)
            if df["review_stars"].notna().any() else None,
            "median_rating": float(df["review_stars"].dropna().median())
            if df["review_stars"].notna().any() else None,
            "avg_tokens_with_text": round(float(with_text["n_tokens"].mean()), 2)
            if len(with_text) else 0.0,
        },
        "descriptive_statistics": stats.to_dict(orient="records"),
        "category_distribution": cat_counts.to_dict(),
        "sentiment_distribution": sent_counts.to_dict(),
        "sentiment_source_distribution": df["sentiment_source"].value_counts().to_dict(),
        "category_x_sentiment": cross.to_dict(),
        "category_positive_pct": pct_pos.to_dict(),
        "avg_rating_by_category": df.groupby(df["category"].astype(str))["review_stars"]
            .mean().round(2).reindex(CATEGORIES).dropna().to_dict(),
        "top_places_by_reviews": reviews.groupby("place_name").size()
            .sort_values(ascending=False).head(10).to_dict(),
        "reviews_per_city": reviews["city"].value_counts().head(10).to_dict(),
        "language_distribution": df.loc[df["has_text"], "review_language"]
            .value_counts().head(10).to_dict(),
        "correlation_highlights": {
            "n_tokens_vs_review_stars": float(corr.loc["n_tokens", "review_stars"])
            if "n_tokens" in corr.index and "review_stars" in corr.columns else None,
            "review_stars_vs_place_avg_rating": float(
                corr.loc["review_stars", "place_avg_rating"])
            if "review_stars" in corr.index and "place_avg_rating" in corr.columns else None,
        },
        "date_range": {
            "min": str(dates.min()) if dates.notna().any() else None,
            "max": str(dates.max()) if dates.notna().any() else None,
        },
        "last_data_collection": str(df["scraped_at_parsed"].max())
        if df["scraped_at_parsed"].notna().any() else None,
        "figures": figures,
        "tables": tables,
        "key_questions_for_mining": [
            "Kategoria destinasaun ne'ebe simu proporsaun sentiment positivu aas liu?",
            "Kategoria ne'ebe hetan komentariu negativu barak liu, no tanba sa?",
            "Bainhira ita husik dadus koalia mesak (clustering), tema natural hira mak mosu, no hanesan ka la'os ho 4 kategoria fixu?",
            "Teksu review nia liafuan bele preve sentiment no kategoria ho presizaun hira?",
            "Lokal ka munisipiu ne'ebe presiza atensaun (rating kraik / sentiment negativu makaas)?",
        ],
    }
    return eda


if __name__ == "__main__":
    import json

    from category_classifier import apply_category_classification
    from data_collection import collect
    from preprocessing import preprocess
    from sentiment_analysis import apply_sentiment_analysis

    raw = collect()
    clean, _ = preprocess(raw)
    d = apply_category_classification(clean)
    d = apply_sentiment_analysis(d)
    result = run_eda(d)

    print(json.dumps(result["overview"], indent=2, ensure_ascii=False))
    print("\nKategoria x Sentiment:")
    print(pd.DataFrame(result["category_x_sentiment"]).to_string())
    print("\n% Positivu kada kategoria:", result["category_positive_pct"])
    print("\nFigura ne'ebe grava:")
    for k, v in result["figures"].items():
        print(f"  {k:24s} -> {v}")
