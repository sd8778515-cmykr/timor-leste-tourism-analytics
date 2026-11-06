"""
pipeline.py
===========
ORKESTRADOR PRINSIPAL — implementa fluxu jeral husi Manual seksaun 5:

  PROBLEMA -> ONLINE DATA SOURCE -> DATA COLLECTION -> DATA STORAGE ->
  PREPROCESSING -> EDA -> DATA MINING -> MODEL EVALUATION -> VISUALIZATION ->
  DASHBOARD -> YOUTUBE DEMO -> FINAL SUBMISSION

Pasu ne'ebe script ne'e halao (pasu 1-9; pasu 10-12 halao manualmente):

   1. DATA SOURCE + COLLECTION   data_collection.collect()
   2. DATA STORAGE (raw)         data/raw/dataset_turris.json + log kolesaun
   3. PREPROCESSING              preprocessing.preprocess()      -> data/processed/
   4. FEATURE LAYER              grava feature ba model          -> data/features/
   5. DATA MINING (Model 1)      category_classifier + sentiment_analysis
   6. DATA MINING (unsupervised) clustering.run_clustering()
   7. DATA MINING (Model 2)      supervised_models.run_supervised_models()
   8. EDA                        eda.run_eda()                   -> reports/figures
   9. MODEL EVALUATION           model_evaluation.run_evaluation()
  10. OUTPUT LAYER + POWER BI    data/output/ + powerbi/*.csv

Halao:
    py -3.11 src/pipeline.py

Opsaun:
    py -3.11 src/pipeline.py --no-figures     # la halo figura PNG (lalais liu)
    py -3.11 src/pipeline.py --step-by-step    # para depois kada etapa ba demo
"""

import contextlib
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from category_classifier import apply_category_classification
from clustering import run_clustering
from config import (
    EDA_SUMMARY_JSON,
    EVALUATION_JSON,
    FEATURES_CSV,
    OUTPUT_CSV,
    PROCESSED_CSV,
    RAW_DIR,
    ensure_dirs,
    list_raw_dataset_files,
)
from data_collection import collect, collect_from_api
from eda import run_eda
from export_powerbi import export_all
from model_evaluation import run_evaluation
from evaluation_interpretation import write_interpretation
from preprocessing import preprocess
from sentiment_analysis import apply_sentiment_analysis
from supervised_models import run_supervised_models

# Kolona ba FEATURE LAYER (Manual seksaun 10: "Variavel ne'ebe prepara ba model")
FEATURE_COLS = [
    "place_id", "place_name", "city", "review_id", "tokens_joined", "n_tokens",
    "n_chars", "review_stars", "rating_deviation", "days_since_review",
    "is_long_review", "is_outlier_length", "is_local_guide", "review_language",
    "category_label", "sentiment_true_proxy", "split", "sentiment_split", "category_split",
]

# Kolona ba OUTPUT LAYER (Manual seksaun 10: "Prediction, cluster, sentiment")
OUTPUT_COLS = [
    "place_id", "place_name", "city", "lat", "lng", "review_id", "review_text",
    "review_date_parsed", "review_stars", "review_language",
    "category", "category_source", "category_label", "label_confidence", "category_rule_text",
    "category_rule_context", "category_rule_keywords", "category_pred_ml",
    "sentiment", "sentiment_source", "sentiment_lexicon", "sentiment_true_proxy",
    "sentiment_pred_ml", "sentiment_score",
    "cluster", "cluster_hier", "cluster_x", "cluster_y", "split",
    "sentiment_split", "category_split", "has_review", "has_text", "has_rating",
]

_STEP = 0
_STEP_BY_STEP = False


def _banner(title: str) -> float:
    global _STEP
    _STEP += 1
    print()
    print("=" * 74)
    print(f"  PASU {_STEP}. {title}")
    print("=" * 74)
    return time.perf_counter()


def _done(t0: float, msg: str = "") -> None:
    print(f"   [OK] {msg} ({time.perf_counter() - t0:.2f}s)")


def _pause_for_demo() -> None:
    """Pause between completed stages when running an interactive presentation."""
    if not _STEP_BY_STEP:
        return
    print()
    input("   Stage complete. Explain the result, then press ENTER to continue...")


def run_pipeline(make_figures: bool = True, verbose: bool = True,
                 write_outputs: bool = True, step_by_step: bool = False,
                 fetch_api: bool = False) -> dict:
    """
    Halao pipeline kompletu.

    Parametru:
        make_figures  - kria figura PNG iha reports/figures/
        verbose       - hatudu log pasu ida-idak iha terminal
        write_outputs - grava ba disku (CSV layer, JSON relatoriu, CSV Power BI).
                        Dashboard Streamlit uza **False** atu la sobrescreve
                        relatoriu ne'ebe pipeline ofisial produz ona (senaun,
                        loke dashboard sei hamenus figura husi eda_summary.json
                        no model_evaluation.json).
        fetch_api     - antes preprocessing, request dadus FOUN husi Apify API
                        (troka ba download+paste manual) no merge ba
                        data/raw/dataset_turris.json. Presiza APIFY_API_TOKEN
                        iha .env.
    """
    global _STEP_BY_STEP
    _STEP_BY_STEP = step_by_step
    kwargs = {"make_figures": make_figures, "write_outputs": write_outputs, "fetch_api": fetch_api}
    if verbose:
        return _run_pipeline(verbose=True, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()):
        return _run_pipeline(verbose=False, **kwargs)


def _run_pipeline(make_figures: bool = True, verbose: bool = True,
                  write_outputs: bool = True, fetch_api: bool = False) -> dict:
    global _STEP
    _STEP = 0
    ensure_dirs()
    started = datetime.now(timezone.utc)

    if verbose:
        print()
        print("#" * 74)
        print("#  SISTEMA MINING & EXPLORING — TURIZMU TIMOR-LESTE")
        print("#  Capstone Project | Mining and Exploring | S1 Siencia Komputador | DIT")
        print(f"#  Hahu: {started.strftime('%Y-%m-%d %H:%M:%S')} UTC   |   Python 3.11")
        print("#" * 74)

    # ---------------------------------------------------------------- 1-2 ---
    t0 = _banner("DATA SOURCE + DATA COLLECTION (Manual sek. 10)")
    if fetch_api:
        print("   Modu   : Apify REST API (request dadus FOUN ho APIFY_API_TOKEN)")
        collect_from_api()
    raw_files = list_raw_dataset_files()
    print(f"   Fonte  : {RAW_DIR} ({len(raw_files)} ficheiru .json: "
          f"{', '.join(f.name for f in raw_files)})")
    if not fetch_api:
        print(f"   Modu   : batch snapshot (haree README sek. 2; uza --fetch-api ba Apify API)")
    raw_df = collect(write_log=write_outputs)
    _done(t0, f"{raw_df['place_id'].nunique()} lokal, {len(raw_df)} baris")
    _pause_for_demo()

    # ------------------------------------------------------------------ 3 ---
    t0 = _banner("PREPROCESSING (Manual sek. 11)")
    df, prep_meta = preprocess(raw_df, save=write_outputs)
    c = prep_meta["cleaning"]
    print(f"   Cleaning        : {c['rows_before']} -> {c['rows_after']} baris "
          f"({c['duplicates_by_content'] + c['duplicates_by_review_id']} duplicate hasai)")
    print(f"   Missing values  : {prep_meta['missing_values']['review_text_missing']} "
          f"review sem teksu ({prep_meta['missing_values']['review_text_missing_pct']}%)")
    print(f"   Outliers        : {prep_meta['outliers']['review_length_chars']['n_outliers']} "
          f"(tratamentu: flag deit)")
    print(f"   Vocabulariu     : {prep_meta['text']['vocabulary_size']} liafuan uniku")
    print(f"   Train/Test      : {prep_meta['train_test_split']['n_train']} / "
          f"{prep_meta['train_test_split']['n_test']}")
    _done(t0, f"grava -> {PROCESSED_CSV.name}")
    _pause_for_demo()

    # ------------------------------------------------------------------ 4 ---
    t0 = _banner("DATA MINING — Model 1: Kategoria (rule-based) + Sentiment (lexicon)")
    df = apply_category_classification(df)
    df = apply_sentiment_analysis(df)
    print("   Kategoria final (anotasaun manual + rule-based nivel lokal):")
    for k, v in df["category"].astype(str).value_counts().items():
        print(f"      {k:14s}: {v}")
    print("   Sentiment (label final):")
    for k, v in df["sentiment"].value_counts().items():
        print(f"      {k:14s}: {v}")
    _done(t0)
    _pause_for_demo()

    # ------------------------------------------------------------------ 5 ---
    t0 = _banner("DATA MINING — Clustering K-Means (unsupervised)")
    df, cluster_results = run_clustering(df)
    cm = cluster_results.get("metrics", {})
    if cm.get("status") == "ok":
        print(f"   Algoritmu        : {cm['algorithm']}")
        print(f"   k uza            : {cm['k_used']}  (k di'ak liu tuir silhouette: {cm['k_best_by_silhouette']})")
        print(f"   Silhouette       : {cm['silhouette_score']}")
        print(f"   Davies-Bouldin   : {cm['davies_bouldin_score']}  (ki'ik = di'ak)")
        print(f"   Calinski-Harabasz: {cm['calinski_harabasz_score']}  (boot = di'ak)")
        for p in cluster_results["profile"]:
            print(f"      Cluster {p['cluster']} (n={p['size']:3d}, {p['share_pct']:4.1f}%) "
                  f"| {p['dominant_sentiment']:8s} | {', '.join(p['top_terms'][:4])}")
    else:
        print(f"   [!] {cm.get('status')}")
    _done(t0)
    _pause_for_demo()

    # ------------------------------------------------------------------ 6 ---
    t0 = _banner("DATA MINING — Model 2: Klasifikasaun supervisadu (NB/LR/SVM/RF)")
    df, supervised_results = run_supervised_models(df, save_models=write_outputs)
    for task, res in supervised_results.items():
        if res.get("status") == "ok":
            print(f"   {task.capitalize():10s}: model di'ak liu = {res['best_model']} "
                  f"(F1 macro = {res['best_f1_macro']}, train={res['n_train']}, test={res['n_test']})")
        else:
            print(f"   {task.capitalize():10s}: {res.get('status')}")
    _done(t0)
    _pause_for_demo()

    # ------------------------------------------------------------------ 7 ---
    t0 = _banner("EXPLORATORY DATA ANALYSIS (Manual sek. 12)")
    eda = run_eda(df, make_figures=make_figures)
    ov = eda["overview"]
    print(f"   Lokal / review   : {ov['total_places']} / {ov['total_rows']}")
    print(f"   Rating mediu     : {ov['avg_rating_overall']} (median {ov['median_rating']})")
    print("   % Positivu kada kategoria (PEDIDU DOSENTE):")
    for k, v in eda["category_positive_pct"].items():
        print(f"      {k:14s}: {v}%")
    if make_figures:
        print(f"   Figura EDA       : {len([v for v in eda['figures'].values() if v])} PNG "
              f"iha reports/figures/")
    _done(t0)
    _pause_for_demo()

    # ------------------------------------------------------------------ 8 ---
    t0 = _banner("MODEL EVALUATION (Manual sek. 14)")
    evaluation = run_evaluation(df, cluster_results, supervised_results,
                                make_figures=make_figures,
                                write_tables=write_outputs)
    metrics_df = pd.DataFrame(evaluation["metrics_table"])
    if not metrics_df.empty:
        show = metrics_df[["task", "model", "accuracy", "precision_macro",
                           "recall_macro", "f1_macro"]]
        print(show.to_string(index=False))
    _done(t0)
    _pause_for_demo()

    # ------------------------------------------------------------------ 9 ---
    report = {
        "run_info": {
            "started_utc": started.isoformat(timespec="seconds"),
            "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_required": "3.11",
            "dataset_files": [f.name for f in raw_files],
        },
        "preprocessing": prep_meta,
        "eda": eda,
    }
    paths = {}

    if not write_outputs:
        # Modu "read-only" (uza husi dashboard): la grava buat ida ba disku atu
        # la sobrescreve relatoriu/export ofisial husi `py -3.11 src/pipeline.py`.
        _banner("DATA STORAGE + EXPORT — HAKAT (write_outputs=False)")
        print("   Modu read-only: la grava ficheiru ba disku.")
    else:
        t0 = _banner("DATA STORAGE — Feature layer + Output layer (Manual sek. 10)")
        feat_cols = [c for c in FEATURE_COLS if c in df.columns]
        df[feat_cols].to_csv(FEATURES_CSV, index=False, encoding="utf-8-sig")
        print(f"   Feature layer -> {FEATURES_CSV}")

        out_cols = [c for c in OUTPUT_COLS if c in df.columns]
        df[out_cols].to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
        print(f"   Output layer  -> {OUTPUT_CSV}")

        with open(EDA_SUMMARY_JSON, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"   EDA summary   -> {EDA_SUMMARY_JSON}")

        with open(EVALUATION_JSON, "w", encoding="utf-8") as f:
            json.dump(evaluation, f, indent=2, ensure_ascii=False, default=str)
        write_interpretation(evaluation, EVALUATION_JSON.parent / "EVALUASI_DAN_INTERPRETASI.md")
        print(f"   Evaluation    -> {EVALUATION_JSON}")
        _done(t0)
        _pause_for_demo()

        # ------------------------------------------------------------- 10 ---
        t0 = _banner("VISUALIZATION — Export ba POWER BI (Manual sek. 15)")
        paths = export_all(df, eda, evaluation, cluster_results)
        for name in ["fact_reviews", "dim_place", "dim_date", "dim_category",
                     "dim_sentiment", "dim_cluster", "agg_category_sentiment",
                     "cluster_points", "cluster_ksweep", "model_metrics",
                     "confusion_matrix", "meta_pipeline", "data_dictionary"]:
            if name in paths:
                print(f"   {name:24s} -> powerbi/{name}.csv")
        _done(t0, f"{len(paths)} ficheiru CSV (encoding utf-8-sig ba Power BI)")
        _pause_for_demo()

    # -------------------------------------------------------------- REZUMU --
    if verbose:
        print()
        print("#" * 74)
        print("#  PIPELINE KOMPLETU")
        print("#" * 74)
        print("  Pasu tuir mai:")
        print("    1. Dashboard Streamlit : py -3.11 -m streamlit run dashboard/app.py")
        print("    2. Power BI            : loke Power BI Desktop -> Get Data -> Folder")
        print("                             -> hili folder `powerbi/` (haree powerbi/GUIA_POWERBI.md)")
        print("    3. Notebook EDA        : notebooks/01_eda_turismo.ipynb")
        print("    4. Figura ba relatoriu : reports/figures/*.png")
        print()

    return {
        "data": df,
        "eda": eda,
        "evaluation": evaluation,
        "clustering": cluster_results,
        "supervised": supervised_results,
        "preprocessing": prep_meta,
        "powerbi_files": paths,
    }


if __name__ == "__main__":
    make_figures = "--no-figures" not in sys.argv
    step_by_step = "--step-by-step" in sys.argv
    fetch_api = "--fetch-api" in sys.argv
    run_pipeline(make_figures=make_figures, step_by_step=step_by_step, fetch_api=fetch_api)
