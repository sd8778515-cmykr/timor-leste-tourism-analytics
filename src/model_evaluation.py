"""
model_evaluation.py
===================
LO5 | Manual seksaun 14 - Model Evaluation.

Tabela Manual seksaun 14 no oinsa sistema ne'e responde:

    Tipu Model       Metrics Rekomendadu (Manual)        Implementasaun
    ---------------  --------------------------------    -------------------------
    Classification   Accuracy, Precision, Recall,        evaluate_rule_based() no
                     F1-score, Confusion Matrix          supervised_models.py
    Clustering       Silhouette Score, cluster           evaluate_clustering() +
                     interpretation                      Davies-Bouldin, Calinski-
                                                         Harabasz, top terms, profile
    Sentiment        Accuracy/F1 se iha labeled data;    ground truth PROXY husi
                     manual validation se presiza        rating + validasaun manual
                                                         (sample_for_manual_validation)

Modulu ne'e mos produz FIGURA evaluation ba relatoriu/PowerPoint:
    09_confusion_matrix_sentiment.png
    10_confusion_matrix_kategoria.png
    11_elbow_silhouette.png
    12_cluster_scatter.png
    13_komparasaun_model.png
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from evaluation_interpretation import build_interpretation
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    CATEGORIES,
    CATEGORY_COLORS,
    FIGURES_DIR,
    RANDOM_STATE,
    SENTIMENTS,
    TABLES_DIR,
    ensure_dirs,
)

plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 120,
                     "savefig.bbox": "tight", "font.size": 9})


def _save(fig, name: str) -> str:
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return str(path)


# ------------------------------------------------------------------------- #
# 1. EVALUATION BA MODEL RULE-BASED (MODEL 1)
# ------------------------------------------------------------------------- #
def evaluate_rule_based(y_true, y_pred, labels: list, model_name: str) -> dict:
    """
    Metrika kompletu tuir Manual seksaun 14 ba klasifikasaun.
    Inklui mos Cohen's Kappa (agreement ne'ebe koreta ba akaso).
    """
    mask = pd.notna(y_true) & pd.notna(y_pred)
    y_true, y_pred = np.asarray(y_true)[mask], np.asarray(y_pred)[mask]
    if len(y_true) == 0:
        return {"status": "no_data", "model": model_name}

    present = list(labels)
    report = classification_report(y_true, y_pred, labels=present,
                                   output_dict=True, zero_division=0)
    return {
        "status": "ok",
        "model": model_name,
        "n_samples": int(len(y_true)),
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision_macro": round(float(precision_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "recall_macro": round(float(recall_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "f1_macro": round(float(f1_score(
            y_true, y_pred, labels=present, average="macro", zero_division=0)), 4),
        "f1_weighted": round(float(f1_score(
            y_true, y_pred, labels=present, average="weighted", zero_division=0)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "per_class": {
            lab: {
                "precision": round(float(report[lab]["precision"]), 4),
                "recall": round(float(report[lab]["recall"]), 4),
                "f1_score": round(float(report[lab]["f1-score"]), 4),
                "support": int(report[lab]["support"]),
            } for lab in present
        },
        "confusion_matrix_labels": present,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=present).tolist(),
    }


# ------------------------------------------------------------------------- #
# 2. FIGURA CONFUSION MATRIX
# ------------------------------------------------------------------------- #
def plot_confusion_matrices(blocks: list, title: str, filename: str) -> str:
    """
    blocks = lista dict ho kolona 'model', 'confusion_matrix',
    'confusion_matrix_labels'. Dezenha side-by-side.
    """
    blocks = [b for b in blocks if b and b.get("confusion_matrix")]
    if not blocks:
        return ""
    n = len(blocks)
    fig, axes = plt.subplots(1, n, figsize=(4.3 * n, 3.9))
    if n == 1:
        axes = [axes]

    for ax, block in zip(axes, blocks):
        cm = np.array(block["confusion_matrix"])
        labs = block["confusion_matrix_labels"]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(labs)))
        ax.set_xticklabels(labs, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(labs)))
        ax.set_yticklabels(labs, fontsize=8)
        ax.set_xlabel("Predisaun (predicted)")
        ax.set_ylabel("Lolos (actual)")
        acc = block.get("accuracy")
        f1m = block.get("f1_macro")
        ax.set_title(f"{block['model']}\nacc={acc} | F1 macro={f1m}", fontsize=9)
        thresh = cm.max() / 2 if cm.max() else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=9,
                        color="white" if cm[i, j] > thresh else "#111")
        ax.grid(False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    return _save(fig, filename)


# ------------------------------------------------------------------------- #
# 3. FIGURA CLUSTERING
# ------------------------------------------------------------------------- #
def plot_elbow_silhouette(sweep: pd.DataFrame, k_used: int) -> str:
    if sweep is None or sweep.empty:
        return ""
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))

    axes[0].plot(sweep["k"], sweep["inertia"], marker="o", color="#2A9D8F")
    axes[0].axvline(k_used, ls="--", c="#D1495B", lw=1)
    axes[0].set_title("Elbow method (inertia)")
    axes[0].set_xlabel("k (numeru cluster)")
    axes[0].set_ylabel("Inertia (WCSS)")

    axes[1].plot(sweep["k"], sweep["silhouette"], marker="o", color="#6A4C93")
    axes[1].axvline(k_used, ls="--", c="#D1495B", lw=1)
    axes[1].set_title("Silhouette Score tuir k")
    axes[1].set_xlabel("k")
    axes[1].set_ylabel("Silhouette")

    axes[2].plot(sweep["k"], sweep["davies_bouldin"], marker="o",
                 color="#E2A33C", label="Davies-Bouldin (ki'ik = di'ak)")
    axes[2].axvline(k_used, ls="--", c="#D1495B", lw=1)
    axes[2].set_title("Davies-Bouldin Index")
    axes[2].set_xlabel("k")
    axes[2].legend(fontsize=7)

    for ax in axes:
        ax.grid(alpha=0.25)
    fig.suptitle(f"Hili numeru cluster otimu (k uza = {k_used})", fontsize=11)
    fig.tight_layout()
    return _save(fig, "11_elbow_silhouette")


def plot_cluster_scatter(df: pd.DataFrame, cluster_results: dict) -> str:
    """
    VIZUALIZASAUN CLUSTER K-MEANS (pedidu esplisitu).

    3 painel:
      (a) Espasu tomak, kolorida tuir CLUSTER K-Means
      (b) ZOOM ba area densu (p95), tanba review badak sira konsentra besik zero
      (c) Espasu tomak, kolorida tuir KATEGORIA manual -> kompara padraun

    Projesaun: TruncatedSVD(2) linear (haree clustering.py NOTA 2 kona-ba tanba
    sa la uza t-SNE).
    """
    sub = df.dropna(subset=["cluster_x", "cluster_y", "cluster"])
    if sub.empty:
        return ""

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    palette = ["#2A9D8F", "#E76F51", "#6A4C93", "#E2A33C", "#1B6CA8",
               "#D1495B", "#457B9D", "#8AB17D"]

    def _draw_clusters(ax, legend: bool):
        for cid in sorted(sub["cluster"].unique()):
            s = sub[sub["cluster"] == cid]
            terms = cluster_results.get("top_terms", {}).get(str(int(cid)), [])[:3]
            label = f"Cluster {int(cid)} (n={len(s)}): {', '.join(terms)}"
            ax.scatter(s["cluster_x"], s["cluster_y"], s=34, alpha=0.78,
                       color=palette[int(cid) % len(palette)], label=label,
                       edgecolors="white", linewidths=0.5)
        if legend:
            ax.legend(fontsize=6.2, loc="best")

    _draw_clusters(axes[0], legend=True)
    axes[0].set_title("(a) Cluster K-Means — espasu tomak")
    axes[0].set_xlabel("Komponente SVD 1")
    axes[0].set_ylabel("Komponente SVD 2")

    # (b) zoom: review badak sira konsentra besik zero; hasai tail ba klaridade
    _draw_clusters(axes[1], legend=False)
    xmax = float(sub["cluster_x"].quantile(0.85))
    ymin = float(sub["cluster_y"].quantile(0.01))
    ymax = float(sub["cluster_y"].quantile(0.99))
    pad_x, pad_y = xmax * 0.15 or 0.01, (ymax - ymin) * 0.06 or 0.01
    axes[1].set_xlim(-pad_x, xmax + pad_x)
    axes[1].set_ylim(ymin - pad_y, ymax + pad_y)
    n_shown = int(((sub["cluster_x"] <= xmax) & sub["cluster_y"].between(ymin, ymax)).sum())
    axes[1].set_title(f"(b) ZOOM area densu ({n_shown}/{len(sub)} review)")
    axes[1].set_xlabel("Komponente SVD 1")

    for cat in CATEGORIES:
        s = sub[sub["category"].astype(str) == cat]
        if s.empty:
            continue
        axes[2].scatter(s["cluster_x"], s["cluster_y"], s=34, alpha=0.78,
                        color=CATEGORY_COLORS[cat], label=f"{cat} (n={len(s)})",
                        edgecolors="white", linewidths=0.5)
    axes[2].set_title("(c) Espasu hanesan, tuir KATEGORIA final")
    axes[2].set_xlabel("Komponente SVD 1")
    axes[2].legend(fontsize=7, loc="best")

    for ax in axes:
        ax.grid(alpha=0.25)
    sil2d = cluster_results.get("metrics", {}).get("projection", {}).get("silhouette_in_2d")
    extra = f" · silhouette iha 2D = {sil2d}" if sil2d is not None else ""
    fig.suptitle(
        f"Komparasaun: padraun natural (cluster) vs kategoria fixu{extra}", fontsize=11
    )
    fig.tight_layout()
    return _save(fig, "12_cluster_scatter")


def plot_model_comparison(sentiment_res: dict, category_res: dict) -> str:
    """Bar chart kompara F1-macro model hotu ba tarefa rua."""
    rows = []
    for task, res in (("Sentiment", sentiment_res), ("Kategoria", category_res)):
        if not res or res.get("status") != "ok":
            continue
        for model, block in res["models"].items():
            rows.append({"task": task, "model": model,
                         "f1_macro": block["f1_macro"],
                         "accuracy": block["accuracy"]})
    if not rows:
        return ""
    data = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, task in zip(axes, data["task"].unique()):
        d = data[data["task"] == task]
        x = np.arange(len(d))
        ax.bar(x - 0.2, d["accuracy"], width=0.4, label="Accuracy", color="#2A9D8F")
        ax.bar(x + 0.2, d["f1_macro"], width=0.4, label="F1 macro", color="#6A4C93")
        ax.set_xticks(x)
        ax.set_xticklabels(d["model"], rotation=20, ha="right", fontsize=8)
        ax.set_title(f"Tarefa: {task}")
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    axes[0].set_ylabel("Score")
    fig.suptitle("Komparasaun performansia model (holdout test set)", fontsize=11)
    fig.tight_layout()
    return _save(fig, "13_komparasaun_model")


# ------------------------------------------------------------------------- #
# 4. VALIDASAUN MANUAL (Manual seksaun 14: "manual validation se presiza")
# ------------------------------------------------------------------------- #
def sample_for_manual_validation(df: pd.DataFrame, n: int = 30,
                                 random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Foti amostra aleatoriu review ho teksu ba VALIDASAUN MANUAL. Grupu tenke
    prenxe kolona `sentiment_manual` no `category_manual` ho liman iha Excel,
    depois kompara ho predisaun sistema iha relatoriu Chapter 6.
    """
    sub = df[df["has_text"] & df["review_stars"].notna()]
    if sub.empty:
        return pd.DataFrame()
    sample = sub.sample(n=min(n, len(sub)), random_state=random_state)
    out = sample[["review_id", "place_name", "city", "review_text", "review_stars",
                  "sentiment", "sentiment_source", "category",
                  "category_rule_text", "cluster"]].copy()
    out["sentiment_manual"] = ""   # prenxe husi estudante
    out["category_manual"] = ""    # prenxe husi estudante
    out["nota_validador"] = ""
    return out


def validation_sample_preserving_labels(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    """Preserve the selected reviews and human annotations on every rerun."""
    if not path.exists():
        return sample_for_manual_validation(df)
    previous = pd.read_csv(path, keep_default_na=False)
    keys = ["review_id"] if "review_id" in previous else ["place_name", "review_text"]
    human_cols = ["sentiment_manual", "category_manual", "nota_validador"]
    current_cols = ["review_id", "place_name", "city", "review_text", "review_stars",
                    "sentiment", "sentiment_source", "category", "category_rule_text", "cluster"]
    current = df[current_cols].fillna("").drop_duplicates(keys)
    updated = previous[keys + human_cols].merge(current, on=keys, how="left", validate="many_to_one")
    if updated["sentiment"].isna().any():
        # A changed snapshot must not silently destroy earlier validation work.
        return previous
    return updated[current_cols + human_cols]


# ============================================================================
# DEMO DOSEN — MAJORITY BASELINE (PATOKAN MINIMUM, BUKAN MODEL 1/MODEL 2)
# Fungsi ini mencari label terbanyak pada TRAIN, memakai label tersebut untuk
# semua baris TEST, lalu menghitung accuracy, precision, recall, dan F1.
# ============================================================================
def evaluate_majority_baseline(df, target, split_col, labels, task):
    train = df.loc[df[split_col].eq("train"), target].dropna()
    test = df.loc[df[split_col].eq("test"), target].dropna()
    if train.empty or test.empty:
        return {"status": "no_data", "model": f"Baseline - Majority ({task})"}

    # DEMO DOSEN BASELINE 1 — cari kelas yang jumlahnya paling banyak di TRAIN.
    majority = train.value_counts().idxmax()

    # DEMO DOSEN BASELINE 2 — prediksi SEMUA data TEST dengan kelas terbanyak.
    # evaluate_rule_based() kemudian membandingkan prediksi dengan label test
    # untuk menghitung accuracy, precision, recall, dan F1.
    result = evaluate_rule_based(test, np.repeat(majority, len(test)), labels,
                                 f"Baseline - Majority ({task})")
    result["majority_class"] = str(majority)
    result["selection_note"] = "The majority class is selected from training data and evaluated on the same test set as Model 2."
    return result


def evaluate_manual_validation(df, sample):
    """Only score genuine, recognized annotations matched to current reviews."""
    results = {}
    if sample.empty:
        sample = pd.DataFrame(columns=["review_id", "sentiment_manual", "category_manual"])
    keys = ["review_id"] if "review_id" in sample else ["place_name", "review_text"]
    human = sample[keys + ["sentiment_manual", "category_manual"]].copy()
    current = df[keys + ["sentiment", "category"]].drop_duplicates(keys)
    joined = human.merge(current, on=keys, how="left", validate="many_to_one", indicator=True)
    for key, manual_col, labels in (("sentiment", "sentiment_manual", SENTIMENTS),
                                    ("category", "category_manual", CATEGORIES)):
        entered = joined[manual_col].fillna("").astype(str).str.strip()
        recognized = entered.isin(labels)
        matched = joined["_merge"].eq("both")
        valid = recognized & matched
        result = {"status": "pending" if not valid.any() else "partial" if not valid.all() else "complete",
                  "n_valid": int(valid.sum()),
                  "n_invalid": int((entered.ne("") & ~recognized).sum()),
                  "n_unmatched": int((~matched).sum())}
        if valid.any():
            result["metrics"] = evaluate_rule_based(entered[valid], joined.loc[valid, key], labels,
                                                      f"Manual check - final {key}")
        results[key] = result
    return {"n_sample": len(sample), "sentiment_completed": results["sentiment"]["n_valid"],
            "category_completed": results["category"]["n_valid"], "results": results,
            "note": "Human labels only; exploratory sample audit, not an independent model-selection set."}


# ------------------------------------------------------------------------- #
# ORKESTRASAUN
# ------------------------------------------------------------------------- #
def run_evaluation(df: pd.DataFrame, cluster_results: dict,
                   supervised_results: dict, make_figures: bool = True,
                   write_tables: bool = True) -> dict:
    """
    Junta evaluation hotu-hotu ba dict ida + (opsionál) grava figura no tabela.

    write_tables=False uza husi dashboard, atu la sobrescreve tabela ofisial.
    """
    ensure_dirs()

    # --- A. Sentiment: MODEL 1 (lexicon) vs ground truth proxy (rating) -----
    sent_mask = (df["has_text"] & df["sentiment_lexicon"].notna()
                 & df["sentiment_true_proxy"].notna() & df["sentiment_split"].eq("test"))
    lexicon_eval = evaluate_rule_based(
        df.loc[sent_mask, "sentiment_true_proxy"],
        df.loc[sent_mask, "sentiment_lexicon"],
        SENTIMENTS, "Model 1 - Lexicon (sentiment)",
    )
    lexicon_eval["ground_truth_note"] = (
        "Ground truth uses star ratings as proxy labels (>=4 Positive, 3 Neutral, "
        "<=2 Negative), rather than human text annotations. Rating and written "
        "sentiment may occasionally differ."
    )

    # --- B. Kategoria: MODEL 1 (rule-based) vs anotasaun manual -------------
    cat_mask = df["has_text"] & df["category_label"].notna() & df["category_split"].eq("test")
    rule_text_eval = evaluate_rule_based(
        df.loc[cat_mask, "category_label"],
        df.loc[cat_mask, "category_rule_text"],
        CATEGORIES, "Model 1 - Rule-based text (category)",
    )
    rule_ctx_eval = evaluate_rule_based(
        df.loc[cat_mask, "category_label"],
        df.loc[cat_mask, "category_rule_context"],
        CATEGORIES, "Model 1 - Rule-based context (category)",
    )
    rule_ctx_eval["leakage_warning"] = (
        "Context mode uses destination names, which also inform manual annotations. "
        "Its score may be optimistic; use text-only mode for a fair comparison."
    )

    # --- C. Model supervisadu (MODEL 2) ------------------------------------
    sent_ml = supervised_results.get("sentiment", {})
    cat_ml = supervised_results.get("category", {})
    # DEMO DOSEN — jalankan Majority Baseline untuk SENTIMENT.
    sent_majority = evaluate_majority_baseline(df, "sentiment_true_proxy", "sentiment_split", SENTIMENTS, "sentiment")
    # DEMO DOSEN — jalankan Majority Baseline untuk KATEGORI.
    cat_majority = evaluate_majority_baseline(df, "category_label", "category_split", CATEGORIES, "category")

    # --- D. Clustering -----------------------------------------------------
    cluster_metrics = cluster_results.get("metrics", {})
    sweep_df = pd.DataFrame(cluster_results.get("k_sweep", []))

    figures = {}
    if make_figures:
        best_sent_ml = None
        if sent_ml.get("status") == "ok":
            best_sent_ml = dict(sent_ml["models"][sent_ml["best_model"]])
            best_sent_ml["model"] = f"Model 2 - {sent_ml['best_model']} (sentiment)"
        figures["confusion_sentiment"] = plot_confusion_matrices(
            [lexicon_eval, best_sent_ml],
            "Confusion Matrix - Sentiment Classification",
            "09_confusion_matrix_sentiment",
        )

        best_cat_ml = None
        if cat_ml.get("status") == "ok":
            best_cat_ml = dict(cat_ml["models"][cat_ml["best_model"]])
            best_cat_ml["model"] = f"Model 2 - {cat_ml['best_model']} (category)"
        figures["confusion_category"] = plot_confusion_matrices(
            [rule_text_eval, best_cat_ml],
            "Confusion Matrix - Category Classification (4 categories)",
            "10_confusion_matrix_kategoria",
        )

        figures["elbow_silhouette"] = plot_elbow_silhouette(
            sweep_df, cluster_metrics.get("k_used", 4)
        )
        figures["cluster_scatter"] = plot_cluster_scatter(df, cluster_results)
        figures["model_comparison"] = plot_model_comparison(sent_ml, cat_ml)

    # --- E. Tabela tidy ba relatoriu/Power BI ------------------------------
    metric_rows = []
    for block in (lexicon_eval, rule_text_eval, rule_ctx_eval, sent_majority, cat_majority):
        if block.get("status") == "ok":
            metric_rows.append({
                "task": "Sentiment" if "sentiment" in block["model"] else "Category",
                "model": block["model"], "type": "majority-baseline" if "majority_class" in block else "rule-based",
                "accuracy": block["accuracy"],
                "precision_macro": block["precision_macro"],
                "recall_macro": block["recall_macro"],
                "f1_macro": block["f1_macro"],
                "f1_weighted": block["f1_weighted"],
                "n_samples": block["n_samples"],
            })
    for task, res in (("Sentiment", sent_ml), ("Category", cat_ml)):
        if res.get("status") == "ok":
            for name, block in res["models"].items():
                metric_rows.append({
                    "task": task, "model": f"Model 2 - {name}",
                    "type": "supervised",
                    "accuracy": block["accuracy"],
                    "precision_macro": block["precision_macro"],
                    "recall_macro": block["recall_macro"],
                    "f1_macro": block["f1_macro"],
                    "f1_weighted": block["f1_weighted"],
                    "n_samples": block["n_samples"],
                    "cv_f1_macro_mean": block.get("cv_f1_macro_mean"),
                })
    metrics_table = pd.DataFrame(metric_rows)
    validation_path = TABLES_DIR / "sample_validasaun_manual.csv"
    validation_sample = validation_sample_preserving_labels(df, validation_path)
    if write_tables:
        metrics_table.to_csv(TABLES_DIR / "model_metrics.csv", index=False,
                             encoding="utf-8-sig")
        if not validation_sample.empty:
            validation_sample.to_csv(TABLES_DIR / "sample_validasaun_manual.csv",
                                     index=False, encoding="utf-8-sig")

    result = {
        "sentiment": {
            "model_1_lexicon": lexicon_eval,
            "majority_baseline": sent_majority,
            "model_2_supervised": sent_ml,
        },
        "category": {
            "model_1_rule_text": rule_text_eval,
            "model_1_rule_context": rule_ctx_eval,
            "majority_baseline": cat_majority,
            "model_2_supervised": cat_ml,
        },
        "clustering": {
            "metrics": cluster_metrics,
            "k_sweep": cluster_results.get("k_sweep", []),
            "profile": cluster_results.get("profile", []),
            "top_terms": cluster_results.get("top_terms", {}),
            "cluster_x_category": cluster_results.get("cluster_x_category", {}),
            "cluster_x_sentiment": cluster_results.get("cluster_x_sentiment", {}),
            "interpretation_note": (
                "A near-zero silhouette indicates overlapping clusters. Interpret the result "
                "with top terms, cluster profiles, and the documented limitations."
            ),
        },
        "figures": figures,
        "metrics_table": metric_rows,
        "manual_validation": evaluate_manual_validation(df, validation_sample),
        "manual_validation_sample": str(TABLES_DIR / "sample_validasaun_manual.csv")
        if (write_tables and not validation_sample.empty) else None,
    }
    result["interpretation"] = build_interpretation(result)
    return result


if __name__ == "__main__":
    import json

    from category_classifier import apply_category_classification
    from clustering import run_clustering
    from data_collection import collect
    from preprocessing import preprocess
    from sentiment_analysis import apply_sentiment_analysis
    from supervised_models import run_supervised_models

    raw = collect()
    clean, _ = preprocess(raw)
    d = apply_category_classification(clean)
    d = apply_sentiment_analysis(d)
    d, cres = run_clustering(d)
    d, sres = run_supervised_models(d)
    ev = run_evaluation(d, cres, sres)

    print(pd.DataFrame(ev["metrics_table"]).to_string(index=False))
    print("\nFigura evaluation:")
    for k, v in ev["figures"].items():
        print(f"  {k:22s} -> {v}")
