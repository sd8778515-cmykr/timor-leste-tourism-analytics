"""
dashboard/app.py
================
LO6 | Manual seksaun 15 - Dashboard & Visualization (Streamlit).

Komponente ne'ebe Manual seksaun 15 rekomenda no oinsa implementa:
    [x] KPI cards                 -> liña KPI iha leten
    [x] Trend chart               -> Tab "Explorasaun (EDA)"
    [x] Comparison chart          -> Category x Sentiment (pedidu dosente)
    [x] Map                       -> mapa lokal turizmu ho lat/lng
    [x] Filter date/location/cat  -> sidebar
    [x] Prediction/Mining result  -> Tab "Clustering" no Tab "Model Evaluation"
    [x] Last updated timestamp    -> badge iha hero

Halao:
    py -3.11 -m streamlit run dashboard/app.py --server.address 127.0.0.1
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from config import (  # noqa: E402
    APIFY_API_TOKEN,
    APIFY_FETCH_LOG,
    APIFY_INPUT_CONFIG,
    CATEGORIES,
    CATEGORY_COLORS,
    DATA_SOURCE,
    EDA_SUMMARY_JSON,
    EVALUATION_JSON,
    OUTPUT_CSV,
    SENTIMENT_COLORS,
    SENTIMENTS,
    SENTIMENTS_WITH_UNKNOWN,
    PLACE_LABELS_CSV,
    PROCESSED_CSV,
    TABLES_DIR,
    resolve_raw_dataset,
)
from data_collection import collect_from_api  # noqa: E402
from export_powerbi import build_data_dictionary  # noqa: E402
from pipeline import run_pipeline  # noqa: E402

st.set_page_config(
    page_title="Timor-Leste Tourism Analytics",
    page_icon="🇹🇱",
    layout="wide",
)

# --------------------------------------------------------------- ESTILU ----
components.html(
    Path(__file__).with_name("cursor.html").read_text(encoding="utf-8"),
    height=0,
)
st.markdown(
    """
    <style>
    :root {--ink:#d9f5ff;--muted:#91b5c9;--paper:#050e1c;--panel:#0a1b2e;--line:#204b65;}
    .stApp,[data-testid="stAppViewContainer"] {background:#050e1c;color:var(--ink);}
    [data-testid="stMain"] {background:radial-gradient(ellipse at 50% 0%,#103447 0%,transparent 65%), repeating-linear-gradient(0deg,transparent,transparent 47px,#12304733 48px);}
    [data-testid="stHeader"] {background:#050e1ce8;}
    .block-container {max-width:1800px;padding:1.4rem 2rem 3rem;}
    [data-testid="stSidebar"] {background:#081626;border-right:1px solid #204b65;}
    .sidebar-brand {padding:1rem 0;border-bottom:1px solid #204b65;margin-bottom:1rem;}
    .sidebar-brand strong {display:block;color:#5ce7ed;letter-spacing:.08em;font-size:1rem;}
    .sidebar-brand span {color:#91b5c9;font-size:.8rem;}
    .dashboard-hero {text-align:center;padding:1.4rem 1rem;margin-bottom:1.4rem;border:1px solid #367b96;border-top:3px solid #37dfe9;background:linear-gradient(100deg,#091a2e,#0b3041,#091a2e);box-shadow:0 0 24px #00d4ff12;clip-path:polygon(1% 0,99% 0,100% 14%,99% 100%,1% 100%,0 14%);}
    .hero-eyebrow {color:#83b5c9;text-transform:uppercase;letter-spacing:.18em;font-size:.7rem;}
    .dashboard-hero h1 {color:#8ff4ff;font-size:clamp(1.6rem,3vw,2.8rem);letter-spacing:.02em;margin:.4rem 0;text-shadow:0 0 18px #29d7ef44;}
    .hero-subtitle {color:#b1cddb;max-width:850px;margin:.6rem auto;font-size:.9rem;}
    .hero-status {display:inline-block;color:#74e8e1;font-size:.75rem;border:1px solid #24576a;padding:.3rem .8rem;margin-top:.4rem;}
    [data-testid="stMetric"] {background:linear-gradient(135deg,#0d263b,#091727);border:1px solid #28536c;border-top:2px solid #42dce7;padding:1rem;min-height:110px;box-shadow:inset 0 0 24px #1ccadd07;}
    [data-testid="stMetricLabel"] {color:#a3c5d8;font-size:.8rem;}
    [data-testid="stMetricValue"] {color:#65eee9;font-variant-numeric:tabular-nums;}
    h2,h3 {color:#beeef7!important;}
    .section-kicker {color:#65e5eb;text-transform:uppercase;letter-spacing:.14em;font-size:.7rem;margin-top:1rem;}
    .stPlotlyChart,[data-testid="stPlotlyChart"] {border:1px solid #26536e;border-top:2px solid #459aba;background:#081828;}
    .stTabs [data-baseweb="tab-list"] {gap:1rem;border-bottom:1px solid #28536c;margin-top:1rem;}
    .stTabs [data-baseweb="tab"] {color:#9ab8cb;padding:.7rem 1rem;}
    .stTabs [aria-selected="true"] {color:#69f4ef!important;background:#103045;}
    [data-testid="stExpander"] {border-color:#28536c;background:#0a1b2e;}
    [data-testid="stCaptionContainer"] {color:#9ab8cb;}
    @media(max-width:800px) {.block-container {padding:1rem;} .dashboard-hero {clip-path:none;}}
    </style>
    """,
    unsafe_allow_html=True,
)


def polish(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#081828", plot_bgcolor="#081828",
        font={"family": "Trebuchet MS, sans-serif", "color": "#c6eaf4"},
        margin={"l": 24, "r": 24, "t": 40, "b": 30},
        hoverlabel={"bgcolor": "#173b3f", "font_color": "#fffaf0"},
    )
    fig.update_xaxes(showgrid=False, linecolor="#28536c")
    fig.update_yaxes(gridcolor="#1c3549", zeroline=False)
    return fig


# ------------------------------------------------- APIFY FETCH RESULTS ----
def render_apify_fetch_results() -> None:
    """Hatudu dadus RAW ne'ebe foin mai husi Apify (seidauk cleaning/dedup),
    hanesan Apify Console nia Results tab - ema bele klik 1 liña atu haree
    komentariu lolos no JSON raw kompletu, prova katak Manual sek. 10 nia
    kamada RAW ne'e lolos la'o, antes Pasu 2 (pipeline) altera buat ida."""
    if not APIFY_FETCH_LOG.exists():
        st.caption("No Apify fetch has been run yet.")
        return

    with open(APIFY_FETCH_LOG, "r", encoding="utf-8") as fh:
        fetch_log = json.load(fh)
    fetched = fetch_log.get("fetched_places", [])
    ts = (fetch_log.get("run_timestamp_utc") or "").replace("T", " ")[:16]
    st.markdown(
        f"**Last fetch:** {ts} UTC · "
        f"{fetch_log.get('new_places', 0)} new place(s), "
        f"{fetch_log.get('merged_places', 0)} updated, "
        f"{fetch_log.get('new_reviews', 0)} new review(s)"
    )
    st.caption(
        "⚠️ This is the **RAW layer** — exactly what Apify returned, written "
        "as-is to `data/raw/dataset_turris.json`. Duplicates, nulls and messy "
        "fields are expected and normal here. Nothing below has been cleaned, "
        "deduplicated or transformed yet — that only happens in Step 2."
    )
    if not fetched:
        st.caption("The last Apify run returned no places.")
        return

    fetch_df = pd.DataFrame(fetched).rename(columns={
        "title": "Place", "category": "Category", "city": "City",
        "rating": "Rating", "reviews_count_google": "Reviews (Google)",
        "reviews_fetched": "Reviews fetched", "status": "Status",
        "url": "Google Maps",
    })
    fetch_df["Status"] = fetch_df["Status"].map(
        {"baru": "🆕 New", "update": "🔄 Updated"}
    ).fillna(fetch_df["Status"])
    cols = ["Status", "Place", "Category", "City", "Rating",
            "Reviews (Google)", "Reviews fetched", "Google Maps"]
    cols = [c for c in cols if c in fetch_df.columns]

    st.caption("Click a row to open its raw comments/reviews and full raw "
               "JSON below — proof the data really came in from Apify.")
    selection = st.dataframe(
        fetch_df[cols], width="stretch", hide_index=True,
        column_config={
            "Google Maps": st.column_config.LinkColumn(
                "Google Maps", display_text="Open page ↗"
            ),
        },
        on_select="rerun", selection_mode="single-row",
        key="apify_fetch_table",
    )

    picked_rows = selection.selection.rows if selection and selection.selection else []
    if picked_rows:
        chosen = fetched[picked_rows[0]]
        st.markdown(f"#### 💬 Raw comments scraped for: {chosen['title']}")
        reviews = chosen.get("reviews") or []
        if reviews:
            rev_df = pd.DataFrame(reviews).rename(columns={
                "reviewer": "Reviewer", "stars": "Stars",
                "date": "Date", "text": "Comment",
            })
            dup_count = len(reviews) - rev_df.drop_duplicates(
                subset=["Reviewer", "Date", "Comment"]
            ).shape[0] if not rev_df.empty else 0
            st.dataframe(
                rev_df[["Reviewer", "Stars", "Date", "Comment"]],
                width="stretch", hide_index=True, height=300,
            )
            if dup_count:
                st.caption(
                    f"🔎 Note: {dup_count} look like duplicate/empty rows in "
                    "this raw batch — this is exactly what Step 2's cleaning "
                    "step removes."
                )
        else:
            st.caption("No reviews were fetched for this place (0 reviews).")
        if chosen.get("url"):
            st.link_button("🔗 Verify on the real Google Maps page", chosen["url"])
        with st.expander("🔬 View full raw JSON (exactly as stored in data/raw/)"):
            st.json(chosen.get("raw", {}), expanded=False)
    else:
        st.caption("👆 No row selected yet — click one above to jump straight to its raw comments.")


# ----------------------------------------------------------------- DATA ----
@st.cache_data(show_spinner="Loading validated pipeline outputs…")
def load(input_versions: tuple):
    missing = [p for p in [PROCESSED_CSV, OUTPUT_CSV, EDA_SUMMARY_JSON, EVALUATION_JSON] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Pipeline outputs are incomplete. Run first: python src/pipeline.py. "
            + "Missing: " + ", ".join(str(p) for p in missing)
        )

    processed = pd.read_csv(PROCESSED_CSV)
    output = pd.read_csv(OUTPUT_CSV)
    if len(processed) != len(output):
        raise ValueError(
            f"Processed and output layers have different row counts: {len(processed)} vs {len(output)}. "
            "Run again: python src/pipeline.py"
        )

    keys = [c for c in ["place_id", "review_id"] if c in processed.columns and c in output.columns]
    for key in keys:
        left = processed[key].fillna("").astype(str).reset_index(drop=True)
        right = output[key].fillna("").astype(str).reset_index(drop=True)
        if not left.equals(right):
            raise ValueError(
                f"Processed and output layers are not aligned on {key}. "
                "Run again: python src/pipeline.py"
            )

    df = processed.copy()
    for col in output.columns:
        if col not in {"place_id", "place_name", "city", "lat", "lng", "review_id", "review_text"}:
            df[col] = output[col]
        elif col not in df.columns:
            df[col] = output[col]

    for col in ["review_date_parsed", "scraped_at_parsed"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce", utc=True, format="mixed")
    for col in ["has_review", "has_text", "has_rating"]:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)
    for col in ["review_stars", "lat", "lng", "cluster", "cluster_hier", "cluster_x", "cluster_y"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    with open(EDA_SUMMARY_JSON, "r", encoding="utf-8") as f:
        eda_report = json.load(f)
    with open(EVALUATION_JSON, "r", encoding="utf-8") as f:
        evaluation = json.load(f)

    return {
        "data": df,
        "eda": eda_report["eda"],
        "evaluation": evaluation,
        "clustering": evaluation.get("clustering", {}),
        "preprocessing": eda_report["preprocessing"],
        "run_info": eda_report.get("run_info", {}),
    }


try:
    dataset_path = resolve_raw_dataset()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

cache_inputs = [
    dataset_path,
    PLACE_LABELS_CSV,
    TABLES_DIR / "sample_validasaun_manual.csv",
    PROCESSED_CSV,
    OUTPUT_CSV,
    EDA_SUMMARY_JSON,
    EVALUATION_JSON,
]
result = load(tuple((str(p), p.stat().st_mtime_ns) for p in cache_inputs if p.exists()))
df = result["data"].copy()
eda = result["eda"]
evaluation = result["evaluation"]
cluster_results = result["clustering"]
prep_meta = result["preprocessing"]

df["category"] = df["category"].astype(str)
df["sentiment"] = df["sentiment"].fillna("Unknown")

last_collect = df["scraped_at_parsed"].max()
last_updated = last_collect.strftime("%d-%m-%Y %H:%M UTC") if pd.notna(last_collect) else "N/A"

# ----------------------------------------------------------------- HERO ----
st.markdown(
    f"""
    <div class="dashboard-hero">
        <div class="hero-eyebrow">Mining &amp; Exploring · Capstone Project · DIT 2026</div>
        <h1>Timor-Leste Tourism Analytics</h1>
        <p class="hero-subtitle">
            Analysis of {int(df['has_review'].sum()):,} tourist reviews from
            {df['place_id'].nunique():,} destinations using sentiment analysis,
            four-category classification, and text clustering.
        </p>
        <span class="hero-status">Dataset last collected: {last_updated}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------------------------------------------- SIDEBAR ----
st.sidebar.markdown(
    """
    <div class="sidebar-brand">
        <strong>TL TOURISM ANALYTICS</strong>
        <span>Capstone Project · DIT 2026</span>
    </div>
    """,
    unsafe_allow_html=True,
)
st.sidebar.header("Filters")

cities = sorted(df["city"].dropna().unique().tolist())
search = st.sidebar.text_input("Search destination", placeholder="Enter destination name")
sel_cities = st.sidebar.multiselect("Municipality", cities)
sel_categories = st.sidebar.multiselect("Category", CATEGORIES)
sel_sentiments = st.sidebar.multiselect("Sentiment", SENTIMENTS)
clusters_available = sorted(df["cluster"].dropna().astype(int).unique().tolist())
sel_clusters = st.sidebar.multiselect("K-Means cluster", clusters_available)
min_rating = st.sidebar.selectbox("Minimum destination rating", ["Any", "3.0+", "4.0+", "4.5+"])
only_text = st.sidebar.checkbox("Reviews with text only", value=False)
include_undated = st.sidebar.checkbox("Include records without a review date", value=True)

date_min, date_max = df["review_date_parsed"].min(), df["review_date_parsed"].max()
date_range = None
if pd.notna(date_min) and pd.notna(date_max):
    date_range = st.sidebar.date_input(
        "Review period",
        value=(date_min.date(), date_max.date()),
        min_value=date_min.date(), max_value=date_max.date(),
    )

f = df.copy()
if search.strip():
    f = f[f["place_name"].fillna("").astype(str).str.contains(
        search.strip(), case=False, regex=False
    )]
if sel_cities:
    f = f[f["city"].isin(sel_cities)]
if sel_categories:
    f = f[f["category"].isin(sel_categories)]
if sel_sentiments:
    f = f[f["sentiment"].isin(sel_sentiments)]
if sel_clusters:
    f = f[f["cluster"].isin(sel_clusters)]
if only_text:
    f = f[f["has_text"]]
if date_range and isinstance(date_range, tuple) and len(date_range) == 2:
    start, end = date_range
    start_utc = pd.Timestamp(start).tz_localize("UTC")
    end_utc = pd.Timestamp(end).tz_localize("UTC") + pd.Timedelta(days=1)
    in_range = f["review_date_parsed"].ge(start_utc) & f["review_date_parsed"].lt(end_utc)
    f = f[in_range | (include_undated & f["review_date_parsed"].isna())]
if min_rating != "Any":
    place_ratings = df.groupby("place_id")["review_stars"].mean()
    f = f[f["place_id"].map(place_ratings).ge(float(min_rating.rstrip("+")))]

st.sidebar.markdown("---")
st.sidebar.caption(f"Displayed records: **{len(f):,} / {len(df):,}**")
st.sidebar.markdown("---")
st.sidebar.caption(
    f"**Data source**\n\n{DATA_SOURCE['source_name']}\n\n"
    f"**Collection mode:** {DATA_SOURCE['collection_mode']}\n\n"
    f"**Python:** 3.11"
)

st.sidebar.markdown("---")
st.sidebar.caption("📥 To request new data from Apify, open the "
                    "**\"Data Collection\"** tab (first tab below).")

if f.empty:
    st.warning("No records match the selected filters.")
    st.stop()

# Subset ho sentiment DETERMINADU deit (Positive/Neutral/Negative) — hasai
# "Unknown" (lokal/review sem review_excerpt no sem rating) husi TODU grafiku no
# metrika sentiment, atu la "diminui" persentajen ho baris sem evidensia.
f_sent = f[f["sentiment"].isin(SENTIMENTS)]
f_reviews = f[f["has_review"]]

# ------------------------------------------------------------ KPI CARDS ----
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Destinations", f["place_id"].nunique())
k2.metric("Valid reviews", int(f["has_review"].sum()))
k3.metric("Average rating",
          f"{f['review_stars'].dropna().mean():.2f}" if f["review_stars"].notna().any() else "N/A")
k4.metric("Positive sentiment", f"{(f_sent['sentiment'] == 'Positive').mean() * 100:.1f}%" if not f_sent.empty else "N/A")
k5.metric("Negative sentiment", f"{(f_sent['sentiment'] == 'Negative').mean() * 100:.1f}%" if not f_sent.empty else "N/A")
sil = cluster_results.get("metrics", {}).get("silhouette_score")
k6.metric("Silhouette score", sil if sil is not None else "N/A")
st.caption("Sentiment percentages exclude 206 destination records without review evidence.")

tab_collect, tab_eda, tab_cluster, tab_eval, tab_data = st.tabs([
    "📥 Data Collection",
    "Overview & EDA",
    "Text Clustering",
    "Model Evaluation",
    "Data & Methodology",
])

# ============================================ TAB 0 — DATA COLLECTION ====
with tab_collect:
    st.markdown(
        "Mirrors the Manual's data layers (sek. 10): **1)** pull new data "
        "straight from Apify and show it exactly as it lands in the **RAW** "
        "layer (`data/raw/`) — untouched, duplicates and all, proving the "
        "data genuinely came in — **2)** only after that, run the pipeline "
        "so it cleans, deduplicates and mines it into the Processed/Feature/"
        "Output layers that power the analytics tabs."
    )

    st.markdown("##### 1️⃣ Request RAW data from Apify")
    if not APIFY_API_TOKEN:
        st.caption("⚠️ `APIFY_API_TOKEN` not set in `.env` — cannot request new data.")
    else:
        req_col, btn_col = st.columns([1, 2])
        with req_col:
            jumlah_data = st.number_input(
                "How many new places?", min_value=1, max_value=200, value=1, step=1,
                help="Exact number of NEW places to fetch from Google Maps via "
                     "Apify (not multiplied by anything).",
            )
        with btn_col:
            st.caption(
                "Only fetches from Apify and shows the results below — it does "
                "NOT touch the dashboard charts yet. Can take a few minutes and "
                "spends your Apify monthly budget."
            )
            if st.button("📡 Request data from Apify", type="primary"):
                request_body = {"jumlah_data": int(jumlah_data)}
                with open(APIFY_INPUT_CONFIG, "w", encoding="utf-8") as fh:
                    json.dump(request_body, fh, indent=2, ensure_ascii=False)
                try:
                    with st.spinner(f"Requesting {jumlah_data} place(s) from Apify…"):
                        collect_from_api()
                    st.success("Done — see what came in below.")
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to fetch new data: {exc}")

    st.markdown("---")
    st.markdown("##### 📊 RAW data that just came in (`data/raw/`)")
    render_apify_fetch_results()

    st.markdown("---")
    st.markdown("##### 2️⃣ Run the pipeline — clean, deduplicate & mine the RAW data above")
    st.caption(
        "Only run this once the RAW data above has been shown/confirmed. It "
        "removes duplicates, cleans text, classifies category + sentiment, "
        "clusters, trains models, and runs EDA + evaluation — then refreshes "
        "every chart and KPI in the tabs to the right."
    )
    if st.button("⚙️ Run full pipeline & refresh dashboard"):
        try:
            with st.spinner("Running the full pipeline…"):
                pipeline_result = run_pipeline(
                    make_figures=False, verbose=False,
                    write_outputs=True, fetch_api=False,
                )
            n_places = pipeline_result["data"]["place_id"].nunique()
            st.success(f"Pipeline complete. Dashboard now reflects {n_places:,} places.")
            st.cache_data.clear()
            st.rerun()
        except Exception as exc:
            st.error(f"Pipeline failed: {exc}")

# ========================================================== TAB 1 — EDA ====
with tab_eda:
    left_panel, map_panel, right_panel = st.columns([1, 1.8, 1])
    with left_panel:
        counts = f_reviews["category"].value_counts().reindex(CATEGORIES, fill_value=0)
        overview_category = px.bar(
            x=counts.values, y=counts.index, orientation="h",
            labels={"x": "Reviews", "y": ""}, title="Tourism categories",
            color=counts.index, color_discrete_map=CATEGORY_COLORS,
        )
        overview_category.update_layout(height=340, showlegend=False)
        st.plotly_chart(polish(overview_category), width="stretch")
    with map_panel:
        locations = f.dropna(subset=["lat", "lng"]).drop_duplicates("place_id")
        # Coordinate view is available offline and does not imply live tracking.
        location_chart = px.scatter(
            locations, x="lng", y="lat", color="category",
            color_discrete_map=CATEGORY_COLORS, hover_name="place_name",
            labels={"lng": "Longitude", "lat": "Latitude", "category": "Category"},
            title="Timor-Leste · Destination coverage",
        )
        location_chart.update_traces(marker=dict(size=8, opacity=.8))
        location_chart.update_layout(height=340, legend=dict(orientation="h", y=-.25))
        location_chart.update_yaxes(scaleanchor="x", scaleratio=1)
        st.plotly_chart(polish(location_chart), width="stretch")
    with right_panel:
        shares = f_sent["sentiment"].value_counts()
        sentiment_ring = px.pie(
            names=shares.index, values=shares.values, color=shares.index,
            color_discrete_map=SENTIMENT_COLORS, hole=.72, title="Visitor sentiment",
        )
        sentiment_ring.update_layout(height=340, legend=dict(orientation="h", y=-.15))
        sentiment_ring.update_traces(textinfo="percent", textposition="inside")
        st.plotly_chart(polish(sentiment_ring), width="stretch")
    st.caption("Dataset snapshot · Location points represent recorded destinations, not live activity.")
    st.markdown('<div class="section-kicker">Lecturer requirement</div>', unsafe_allow_html=True)
    st.subheader("Sentiment by tourism category")

    # Hasi tiha "Unknown" husi crosstab ne'e — kategoria x sentiment ne'e
    # deit ba review ho sentiment DETERMINADU (Positive/Neutral/Negative).
    cross = pd.crosstab(f_sent["category"], f_sent["sentiment"]).reindex(
        index=CATEGORIES, fill_value=0)
    cross = cross.reindex(
        columns=[s for s in SENTIMENTS if s in cross.columns], fill_value=0)

    c1, c2 = st.columns(2)
    with c1:
        tidy = cross.reset_index(names="Category").melt(
            id_vars="Category", var_name="Sentiment", value_name="Total")
        fig = px.bar(tidy, x="Category", y="Total", color="Sentiment",
                     color_discrete_map=SENTIMENT_COLORS, text="Total",
                     category_orders={"Category": CATEGORIES},
                     title="Review count by category and sentiment")
        st.plotly_chart(polish(fig), width="stretch")
    with c2:
        pct = (cross.div(cross.sum(axis=1).replace(0, np.nan), axis=0) * 100).round(1)
        tidy_pct = pct.reset_index(names="Category").melt(
            id_vars="Category", var_name="Sentiment", value_name="Percentage")
        fig = px.bar(tidy_pct, x="Category", y="Percentage", color="Sentiment",
                     color_discrete_map=SENTIMENT_COLORS, text="Percentage",
                     category_orders={"Category": CATEGORIES},
                     title="Sentiment percentage within each category")
        st.plotly_chart(polish(fig), width="stretch")

    pos_pct = (f_sent.assign(p=f_sent["sentiment"].eq("Positive"))
               .groupby("category")["p"].mean() * 100).round(1)
    pos_pct = pos_pct.reindex(CATEGORIES).dropna()
    if not pos_pct.empty:
        best = pos_pct.idxmax()
        worst = pos_pct.idxmin()
        st.info(
            f"**Key finding:** **{best}** has the highest positive sentiment share "
            f"(**{pos_pct[best]}%**), while **{worst}** has the lowest "
            f"(**{pos_pct[worst]}%**). Category details: "
            + " · ".join(f"{k} {v}%" for k, v in pos_pct.items())
        )

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Category distribution")
        cat_counts = f_reviews["category"].value_counts().reindex(CATEGORIES, fill_value=0)
        fig = px.bar(x=cat_counts.index, y=cat_counts.values, color=cat_counts.index,
                     color_discrete_map=CATEGORY_COLORS, text=cat_counts.values,
                     labels={"x": "Category", "y": "Reviews"})
        fig.update_layout(showlegend=False)
        st.plotly_chart(polish(fig), width="stretch")
    with c2:
        st.subheader("Sentiment distribution")
        sc = f_sent["sentiment"].value_counts()
        fig = px.pie(names=sc.index, values=sc.values, hole=0.45,
                     color=sc.index, color_discrete_map=SENTIMENT_COLORS)
        st.plotly_chart(polish(fig), width="stretch")

    st.subheader("Review trend over time")
    trend = f_sent.dropna(subset=["review_date_parsed"]).copy()
    if not trend.empty:
        trend["fulan"] = trend["review_date_parsed"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp()
        agg = trend.groupby(["fulan", "sentiment"]).size().reset_index(name="Total")
        fig = px.line(agg, x="fulan", y="Total", color="sentiment", markers=True,
                      color_discrete_map=SENTIMENT_COLORS,
                      labels={"fulan": "Month", "sentiment": "Sentiment"})
        st.plotly_chart(polish(fig), width="stretch")
    else:
        st.info("No dated reviews match the selected filters.")

    st.subheader("Tourism destination map")
    map_df = f.dropna(subset=["lat", "lng"]).drop_duplicates(subset=["place_id"]).copy()
    if not map_df.empty:
        n_rev = f.groupby("place_id")["has_review"].sum()
        pct_pos_place = (f_sent.assign(p=f_sent["sentiment"].eq("Positive"))
                         .groupby("place_id")["p"].mean() * 100).round(1)
        map_df["n_reviews"] = map_df["place_id"].map(n_rev).fillna(0)
        map_df["marker_size"] = map_df["n_reviews"] + 1
        map_df["pct_positive"] = map_df["place_id"].map(pct_pos_place)
        fig = px.scatter_map(
            map_df, lat="lat", lon="lng", hover_name="place_name",
            hover_data={"city": True, "category": True, "n_reviews": True,
                        "pct_positive": True, "lat": False, "lng": False, "marker_size": False},
            color="category", size="marker_size", size_max=28,
            color_discrete_map=CATEGORY_COLORS, zoom=7.4, height=520,
        )
        fig.update_layout(map_style="open-street-map",
                          margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
    else:
        st.info("No coordinates match the selected filters.")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Top 10 destinations by review volume")
        top = (f_reviews.groupby("place_name").size()
               .sort_values(ascending=False).head(10)
               .reset_index(name="Reviews"))
        if top.empty:
            st.info("No reviews match the selected filters.")
        else:
            fig = px.bar(top, x="Reviews", y="place_name", orientation="h",
                         labels={"place_name": ""})
            fig.update_traces(marker_color="#2A9D8F")
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(polish(fig), width="stretch")
    with c2:
        st.subheader("Review languages")
        langs = f.loc[f["has_text"], "review_language"].value_counts().head(10)
        if not langs.empty:
            fig = px.bar(x=langs.index.astype(str), y=langs.values,
                         labels={"x": "Language", "y": "Reviews"})
            fig.update_traces(marker_color="#1B6CA8")
            st.plotly_chart(polish(fig), width="stretch")
        else:
            st.info("No text reviews match the selected filters.")

    with st.expander("Descriptive statistics (Manual Section 12)"):
        st.caption("Statistics describe the full dataset; the filtered records are available in Data & Methodology.")
        st.dataframe(pd.DataFrame(eda["descriptive_statistics"]),
                     hide_index=True, width="stretch")

# ===================================================== TAB 2 — CLUSTERING ==
with tab_cluster:
    cm = cluster_results.get("metrics", {})
    if cm.get("status") != "ok":
        st.warning(f"Clustering unavailable: {cm.get('status')}")
    else:
        st.markdown('<div class="section-kicker">Unsupervised mining</div>',
                    unsafe_allow_html=True)
        st.subheader("K-Means cluster visualization")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Number of clusters (k)", cm["k_used"])
        m2.metric("Silhouette Score", cm["silhouette_score"])
        m3.metric("Davies-Bouldin", cm["davies_bouldin_score"], help="Lower is better")
        m4.metric("Clustered text reviews", cm["n_texts_clustered"])

        pts = f.dropna(subset=["cluster_x", "cluster_y", "cluster"]).copy()
        if pts.empty:
            st.info("No text reviews match the selected filters.")
        else:
            pts["cluster"] = pts["cluster"].astype(int)
            terms = cluster_results.get("top_terms", {})
            pts["cluster_label"] = pts["cluster"].apply(
                lambda c: f"Cluster {c}: " + ", ".join(terms.get(str(c), [])[:3])
            )
            pts["review_excerpt"] = pts["review_text"].apply(
                lambda t: (t[:110] + "…") if isinstance(t, str) and len(t) > 110 else t
            )

            c1, c2 = st.columns(2)
            with c1:
                fig = px.scatter(
                    pts, x="cluster_x", y="cluster_y", color="cluster_label",
                    hover_name="place_name",
                    hover_data={"review_excerpt": True, "sentiment": True, "category": True,
                                "review_stars": True, "cluster_x": False, "cluster_y": False,
                                "cluster_label": False},
                    labels={"cluster_x": "SVD component 1",
                            "cluster_y": "SVD component 2",
                            "cluster_label": "Cluster"},
                    title="Natural review-text clusters (2D TruncatedSVD)",
                )
                fig.update_traces(marker=dict(size=10, opacity=.8,
                                              line=dict(width=.6, color="white")))
                st.plotly_chart(polish(fig), width="stretch")
            with c2:
                fig = px.scatter(
                    pts, x="cluster_x", y="cluster_y", color="category",
                    color_discrete_map=CATEGORY_COLORS, hover_name="place_name",
                    hover_data={"review_excerpt": True, "sentiment": True,
                                "cluster_x": False, "cluster_y": False},
                    labels={"cluster_x": "SVD component 1",
                            "cluster_y": "SVD component 2",
                            "category": "Category"},
                    title="Espasu hanesan, kolorida tuir KATEGORIA final",
                )
                fig.update_traces(marker=dict(size=10, opacity=.8,
                                              line=dict(width=.6, color="white")))
                st.plotly_chart(polish(fig), width="stretch")

            proj = cm.get("projection", {})
            st.caption(
                f"Projection: **{proj.get('method', 'TruncatedSVD(2)')}**. "
                f"Silhouette in 2D = **{proj.get('silhouette_in_2d', 'N/A')}** "
                f"(original TF-IDF space = {proj.get('silhouette_original_space', 'N/A')}). "
                f"Variance retained in 2D: **{100 * cm.get('svd_explained_variance_total', 0):.2f}%**. "
                "The scatter plot is illustrative; use zoom to inspect dense points."
            )

        st.markdown("---")
        st.subheader("Selecting the number of clusters")
        sweep = pd.DataFrame(cluster_results.get("k_sweep", []))
        if not sweep.empty:
            c1, c2, c3 = st.columns(3)
            with c1:
                fig = px.line(sweep, x="k", y="inertia", markers=True,
                              title="Elbow method (inertia / WCSS)")
                fig.add_vline(x=cm["k_used"], line_dash="dash", line_color="#D1495B")
                fig.update_traces(line_color="#2A9D8F")
                st.plotly_chart(polish(fig), width="stretch")
            with c2:
                fig = px.line(sweep, x="k", y="silhouette", markers=True,
                              title="Silhouette score by k")
                fig.add_vline(x=cm["k_used"], line_dash="dash", line_color="#D1495B")
                fig.update_traces(line_color="#6A4C93")
                st.plotly_chart(polish(fig), width="stretch")
            with c3:
                fig = px.line(sweep, x="k", y="davies_bouldin", markers=True,
                              title="Davies-Bouldin (lower is better)")
                fig.add_vline(x=cm["k_used"], line_dash="dash", line_color="#D1495B")
                fig.update_traces(line_color="#E2A33C")
                st.plotly_chart(polish(fig), width="stretch")
            st.caption(
                f"k = {cm['k_used']} is retained for comparison with the four configured categories. "
                f"The highest silhouette in the tested range occurs at k = {cm['k_best_by_silhouette']}. "
                "This analytical choice is documented in Chapter 6."
            )

        st.markdown("---")
        st.subheader("Cluster interpretation")
        profile = pd.DataFrame(cluster_results.get("profile", []))
        if not profile.empty:
            show = profile.copy()
            show["top_terms"] = show["top_terms"].apply(lambda t: ", ".join(t[:8]))
            show = show.rename(columns={
                "cluster": "Cluster", "size": "N", "share_pct": "%",
                "avg_rating": "Average rating", "avg_tokens": "Average tokens",
                "dominant_category": "Dominant category",
                "dominant_sentiment": "Dominant sentiment",
                "dominant_language": "Dominant language", "top_terms": "Top terms",
            })
            st.dataframe(show, hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Cluster × Category**")
            cx = pd.DataFrame(cluster_results.get("cluster_x_category", {}))
            if not cx.empty:
                st.dataframe(cx, width="stretch")
        with c2:
            st.markdown("**Cluster × Sentiment**")
            cs = pd.DataFrame(cluster_results.get("cluster_x_sentiment", {}))
            if not cs.empty:
                st.dataframe(cs, width="stretch")

        hier = cm.get("hierarchical", {})
        if hier.get("silhouette_score") is not None:
            st.caption(
                f"**Algorithm comparison:** Hierarchical clustering (Ward) achieves silhouette "
                f"{hier['silhouette_score']}, compared with K-Means {cm['silhouette_score']}."
            )

# ============================================== TAB 3 — MODEL EVALUATION ===
with tab_eval:
    st.markdown('<div class="section-kicker">Manual Section 14</div>',
                unsafe_allow_html=True)
    st.subheader("Results interpretation")
    interpretation = evaluation.get("interpretation", {})
    if interpretation:
        st.caption(interpretation.get("scope", ""))
        for task in interpretation.get("tasks", {}).values():
            with st.expander(task["title"], expanded=True):
                st.write(task["summary"])
                for limit in task.get("limitations", []):
                    st.write("• " + limit)
                if task.get("largest_confusions"):
                    st.caption("Largest errors from the selected model")
                    st.dataframe(pd.DataFrame(task["largest_confusions"]), hide_index=True, width="stretch")
        st.info(interpretation.get("clustering", ""))
    else:
        st.info(
            "Academic interpretation will appear after running `py -3.11 src/pipeline.py`. "
            "The dashboard reads validated outputs and does not retrain models automatically."
        )
    st.subheader("Model performance comparison")

    metrics = pd.DataFrame(evaluation["metrics_table"])
    if metrics.empty:
        st.info("No model metrics are available.")
    else:
        show = metrics.rename(columns={
            "task": "Task", "model": "Model", "type": "Type",
            "accuracy": "Accuracy", "precision_macro": "Precision (macro)",
            "recall_macro": "Recall (macro)", "f1_macro": "F1 (macro)",
            "f1_weighted": "F1 (weighted)", "n_samples": "N",
            "cv_f1_macro_mean": "CV F1 (macro)",
        })
        st.dataframe(show, hide_index=True, width="stretch")

        fig = px.bar(metrics, x="model", y=["accuracy", "f1_macro"],
                     facet_col="task", barmode="group",
                     color_discrete_sequence=["#2A9D8F", "#6A4C93"],
                     labels={"value": "Score", "model": "", "variable": "Metric"},
                     title="Model performance comparison")
        fig.update_xaxes(tickangle=-25)
        fig.update_yaxes(range=[0, 1])
        st.plotly_chart(polish(fig), width="stretch")

    # Keep methodological caveats available without dominating the dashboard.
    degen_msgs = []
    for task, key in (("Sentiment", "sentiment"), ("Category", "category")):
        res = evaluation[key].get("model_2_supervised", {})
        if res.get("status") != "ok":
            continue
        for name in res.get("degenerate_models", []):
            w = res["models"][name].get("warning", "")
            degen_msgs.append(f"**{task} · Model 2 — {name}:** {w}")
        if res.get("selection_rule"):
            pass
    with st.expander("Model limitations and evaluation notes", expanded=False):
        if degen_msgs:
            st.warning("\n\n".join(degen_msgs))
        else:
            st.success("No degenerate supervised model was detected.")
        sel_rule = (evaluation["sentiment"].get("model_2_supervised", {})
                    .get("selection_rule"))
        if sel_rule:
            st.caption(f"**Model selection rule:** {sel_rule}")
        st.caption("Model 1 and Model 2 use the same test partition. Cross-validation uses training data only.")
        cat_evaluation = evaluation["category"].get("model_2_supervised", {})
        if cat_evaluation.get("untested_classes"):
            st.warning("Classes not represented in the independent destination test set: "
                       + ", ".join(cat_evaluation["untested_classes"])
                       + ". More manually labelled destinations are required.")
        validation = evaluation.get("manual_validation", {})
        st.info(f"Human validation completed: sentiment {validation.get('sentiment_completed', 0)}/{validation.get('n_sample', 0)}; "
                f"category {validation.get('category_completed', 0)}/{validation.get('n_sample', 0)}. "
                "Star ratings are currently used as proxy sentiment labels.")
    validation = evaluation.get("manual_validation", {})
    for task, check in validation.get("results", {}).items():
        if check.get("n_invalid") or check.get("n_unmatched"):
            st.warning(f"Validation {task}: {check['n_invalid']} unrecognized labels; "
                       f"{check['n_unmatched']} reviews not matched to the dataset. Check the validation CSV.")
        if check.get("metrics"):
            with st.expander(f"Human validation results: {task} ({check['status']})"):
                st.dataframe(pd.DataFrame(check["metrics"]["per_class"]).T,
                             width="stretch")
                st.caption("Sample audit against human labels; this is not an independent evaluation of the full dataset.")

    st.markdown("---")
    st.subheader("Confusion Matrix")

    def cm_figure(block, title):
        if not block or block.get("status") != "ok":
            return None
        labels = block["confusion_matrix_labels"]
        z = np.array(block["confusion_matrix"])
        fig = go.Figure(data=go.Heatmap(
            z=z, x=labels, y=labels, colorscale="Blues",
            text=z, texttemplate="%{text}", showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>N: %{z}<extra></extra>",
        ))
        fig.update_layout(title=f"{title}<br><sub>acc={block['accuracy']} · "
                                f"F1 macro={block['f1_macro']}</sub>",
                          xaxis_title="Predicted",
                          yaxis_title="Actual", height=420)
        fig.update_yaxes(autorange="reversed")
        return fig

    sent_ml = evaluation["sentiment"].get("model_2_supervised", {})
    cat_ml = evaluation["category"].get("model_2_supervised", {})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sentiment**")
        fig = cm_figure(evaluation["sentiment"]["model_1_lexicon"],
                        "Model 1 — Lexicon")
        if fig:
            st.plotly_chart(polish(fig), width="stretch")
        if sent_ml.get("status") == "ok":
            best = sent_ml["best_model"]
            fig = cm_figure(dict(sent_ml["models"][best], status="ok"),
                            f"Model 2 — {best}")
            if fig:
                st.plotly_chart(polish(fig), width="stretch")
    with c2:
        st.markdown("**Category (4 classes)**")
        fig = cm_figure(evaluation["category"]["model_1_rule_text"],
                        "Model 1 — Rule-based (text)")
        if fig:
            st.plotly_chart(polish(fig), width="stretch")
        if cat_ml.get("status") == "ok":
            best = cat_ml["best_model"]
            fig = cm_figure(dict(cat_ml["models"][best], status="ok"),
                            f"Model 2 — {best}")
            if fig:
                st.plotly_chart(polish(fig), width="stretch")

    st.markdown("---")
    st.subheader("Per-class metrics (precision, recall, and F1)")
    rows = []
    for task, key, blocks in (
        ("Sentiment", "sentiment", [evaluation["sentiment"]["model_1_lexicon"]]),
        ("Category", "category", [evaluation["category"]["model_1_rule_text"]]),
    ):
        for b in blocks:
            if b.get("status") == "ok":
                for lab, m in b["per_class"].items():
                    rows.append({"Task": task, "Model": b["model"], "Class": lab, **m})
        res = evaluation[key].get("model_2_supervised", {})
        if res.get("status") == "ok":
            best = res["best_model"]
            for lab, m in res["models"][best]["per_class"].items():
                rows.append({"Task": task, "Model": f"Model 2 - {best}",
                             "Class": lab, **m})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.markdown("---")
    st.info(
        "**How to interpret these metrics:**\n\n"
        "- **Sentiment ground truth** uses star ratings as proxy labels rather than "
        "human text annotations. High accuracy with low macro F1 indicates strong "
        "performance on the majority Positive class but weak minority-class performance.\n"
        "- **Category ground truth** uses manually documented destination labels from "
        "`data/place_category_labels.csv`; remaining destinations use the rule-based fallback.\n"
        "- **Macro F1** is reported alongside accuracy because the classes are imbalanced.\n"
        "- The **human-validation sample** is available at "
        "`reports/tables/sample_validasaun_manual.csv`."
    )

# ================================================= TAB 4 — DADUS/TRANSP. ===
with tab_data:
    st.subheader("Detailed records")
    cols = ["place_name", "city", "category", "category_source", "label_confidence", "sentiment",
            "sentiment_source", "review_stars", "cluster", "review_language",
            "review_text", "review_date_parsed", "has_review", "sentiment_split", "category_split"]
    st.dataframe(f[[c for c in cols if c in f.columns]],
                 width="stretch", height=420)

    csv = f[[c for c in cols if c in f.columns]].to_csv(index=False).encode("utf-8-sig")
    st.download_button("⬇️ Download filtered data (CSV)", csv,
                       "turismo_filtered.csv", "text/csv")

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Data source (Manual Section 10)**")
        st.write(f"Source: {DATA_SOURCE['source_name']}")
        st.write(f"Collection mode: {DATA_SOURCE['collection_mode']}")
        st.write(f"Destinations in the raw snapshot: {df['place_id'].nunique():,}")
    with c2:
        st.markdown("**Preprocessing (Manual Section 11)**")
        cleaning = prep_meta.get("cleaning", {})
        missing = prep_meta.get("missing_values", {})
        st.write(f"Rows before cleaning: {cleaning.get('rows_before', 'N/A')}")
        st.write(f"Rows after cleaning: {cleaning.get('rows_after', 'N/A')}")
        st.write(f"Duplicate records removed: {cleaning.get('duplicates_by_content', 0) + cleaning.get('duplicates_by_review_id', 0)}")
        st.write(f"Records without review text: {missing.get('review_text_missing', 'N/A')}")

    st.markdown("---")
    st.subheader("Data dictionary (Manual Section 10)")
    st.dataframe(build_data_dictionary(), hide_index=True, width="stretch")

    st.markdown("---")
    st.subheader("Manual category annotations (ground truth)")
    labels = df.drop_duplicates(subset=["place_id"])[
        ["place_name", "city", "category_label", "label_confidence", "label_evidence"]
    ].sort_values(["category_label", "place_name"])
    st.dataframe(labels, hide_index=True, width="stretch", height=380)
    st.caption(
        "File: `data/place_category_labels.csv`. The research team can review these "
        "annotations in Excel; the pipeline reads the file directly."
    )

st.markdown("---")
st.caption(
    "Mining & Exploring System — Timor-Leste Tourism · Capstone Project DIT 2026 · "
    "Pipeline: Collection → Preprocessing → EDA → Data Mining (Rule-based + Lexicon + "
    "K-Means + NB/LR/SVM/RF) → Model Evaluation → Visualization. "
    "Source: Google Maps reviews (Apify snapshot). Python 3.11."
)
