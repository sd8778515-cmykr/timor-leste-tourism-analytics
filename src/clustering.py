"""
clustering.py
=============
LO4 + LO5 | Manual seksaun 13 (Clustering: K-Means, Hierarchical) no
seksaun 14 (Clustering metrics: Silhouette Score, cluster interpretation).

Objetivu clustering iha sistema ida-ne'e DIFERENTE husi kategoria:
  - `category_classifier.py` = kategoria SUPERVISADU/fixu (Historical, Religious,
    Traditional, Modern) - ita hatene ona klase nia naran.
  - `clustering.py`          = deskobre padraun NATURAL (unsupervised) iha
    teksu review. Pergunta investigasaun: "bainhira ita husik dadus koalia
    mesak, review sira agrupa tuir tema hira, no grupu sira-ne'e hanesan ka
    la'os ho 4 kategoria ne'ebe ita defini?"

Komponente ne'ebe implementa:
  1. TF-IDF vectorization (unigram + bigram) husi token ona limpa
  2. K sweep (k = 2..8): inertia (ELBOW) + Silhouette -> hili k otimu
  3. K-Means final (default k=4 atu kompara ho 4 kategoria)
  4. Metrika: Silhouette, Davies-Bouldin, Calinski-Harabasz
  5. Redusaun dimensaun ho TruncatedSVD(2) -> koordenada `cluster_x`/`cluster_y`
     ba SCATTER PLOT (vizualizasaun cluster iha dashboard no Power BI)
  6. Top terms kada cluster (interpretasaun cluster - Manual seksaun 14)
  7. Komparasaun ho Agglomerative (Hierarchical) Clustering
  8. Profiling: cluster x kategoria, cluster x sentiment

TruncatedSVD provides a linear two-dimensional illustration of sparse TF-IDF.
It does not preserve all distances. Read the explained variance alongside the
plot; evaluate clustering in the original TF-IDF space. A larger silhouette
in 2D alone does not prove that the projection preserves the original groups.

"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import K_SWEEP, N_CLUSTERS, RANDOM_STATE

TEXT_COL = "tokens_joined"


def _vectorize(texts, max_features: int = 500):
    vec = TfidfVectorizer(
        max_features=max_features,
        min_df=2,            # ignora liafuan ne'ebe mosu iha review 1 deit (noise)
        max_df=0.85,         # ignora liafuan ne'ebe mosu iha kuaze review hotu
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    X = vec.fit_transform(texts)
    return vec, X


def k_sweep(X, ks=None, random_state: int = RANDOM_STATE) -> pd.DataFrame:
    """
    ELBOW + SILHOUETTE ba k sira. Retorna DataFrame ho kolona:
        k, inertia, silhouette, davies_bouldin, calinski_harabasz
    Uza ba grafiku "hili k otimu" iha dashboard no Power BI.
    """
    ks = list(ks or K_SWEEP)
    n = X.shape[0]
    rows = []
    Xd = X.toarray() if hasattr(X, "toarray") else X
    for k in ks:
        if k >= n:
            continue
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = km.fit_predict(X)
        if len(set(labels)) < 2:
            continue
        rows.append({
            "k": k,
            "inertia": round(float(km.inertia_), 4),
            "silhouette": round(float(silhouette_score(X, labels)), 4),
            "davies_bouldin": round(float(davies_bouldin_score(Xd, labels)), 4),
            "calinski_harabasz": round(float(calinski_harabasz_score(Xd, labels)), 4),
        })
    return pd.DataFrame(rows)


def _top_terms(vec, centers, n_terms: int = 10) -> dict:
    terms = np.array(vec.get_feature_names_out())
    order = centers.argsort()[:, ::-1]
    return {int(i): list(terms[order[i, :n_terms]]) for i in range(centers.shape[0])}


def run_clustering(df: pd.DataFrame, n_clusters: int = N_CLUSTERS,
                   random_state: int = RANDOM_STATE) -> tuple:
    """
    Halao clustering kompletu.

    Retorna (df, results_dict) onde df hetan kolona foun:
        cluster        -> ID cluster K-Means (NaN ba review sem teksu)
        cluster_x/y    -> koordenada TruncatedSVD 2D (ba scatter plot)
        cluster_hier   -> ID cluster Agglomerative (komparasaun)
    """
    df = df.copy()
    df["cluster"] = np.nan
    df["cluster_x"] = np.nan
    df["cluster_y"] = np.nan
    df["cluster_hier"] = np.nan

    mask = df["has_text"] & df[TEXT_COL].fillna("").str.len().gt(0)
    texts = df.loc[mask, TEXT_COL]

    if texts.shape[0] < n_clusters * 2:
        return df, {"status": "insufficient_data", "n_texts": int(texts.shape[0])}

    vec, X = _vectorize(texts)
    k = min(n_clusters, texts.shape[0] - 1)

    # --- 1. K sweep (elbow + silhouette) -----------------------------------
    sweep = k_sweep(X)
    best_k = int(sweep.loc[sweep["silhouette"].idxmax(), "k"]) if not sweep.empty else k

    # --- 2. K-Means final --------------------------------------------------
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    labels = km.fit_predict(X)
    df.loc[mask, "cluster"] = labels

    Xd = X.toarray()
    metrics = {
        "status": "ok",
        "algorithm": "K-Means (TF-IDF, unigram+bigram, sublinear tf)",
        "k_used": int(k),
        "k_best_by_silhouette": best_k,
        "n_texts_clustered": int(texts.shape[0]),
        "n_features": int(X.shape[1]),
        "inertia": round(float(km.inertia_), 4),
        "silhouette_score": round(float(silhouette_score(X, labels)), 4),
        "davies_bouldin_score": round(float(davies_bouldin_score(Xd, labels)), 4),
        "calinski_harabasz_score": round(float(calinski_harabasz_score(Xd, labels)), 4),
    }

    # --- 3. Redusaun dimensaun ba vizualizasaun 2D -------------------------
    # Projesaun LINEAR (TruncatedSVD 2D). Haree NOTA 2 iha docstring: ida-ne'e
    # Vizualizasaun deit; interpretasaun presiza haree variansia retida.
    svd = TruncatedSVD(n_components=2, random_state=random_state)
    coords = svd.fit_transform(X)
    df.loc[mask, "cluster_x"] = coords[:, 0]
    df.loc[mask, "cluster_y"] = coords[:, 1]

    metrics["svd_explained_variance_ratio"] = [
        round(float(v), 4) for v in svd.explained_variance_ratio_
    ]
    metrics["svd_explained_variance_total"] = round(
        float(svd.explained_variance_ratio_.sum()), 4
    )
    # Silhouette iha espasu 2D: se besik ba silhouette orijinal, projesaun ne'e
    # fiel ba partisaun cluster (uza ba justifikasaun iha relatoriu).
    try:
        metrics["projection"] = {
            "method": "TruncatedSVD(2) - linear",
            "silhouette_in_2d": round(float(silhouette_score(coords, labels)), 4),
            "silhouette_original_space": metrics["silhouette_score"],
            "note": (
                "The 2D projection is illustrative. Check retained variance; "
                "the 2D silhouette does not establish fidelity to the TF-IDF space."
            ),
        }
    except ValueError:
        metrics["projection"] = {"method": "TruncatedSVD(2) - linear"}

    # --- 4. Komparasaun ho Hierarchical Clustering -------------------------
    try:
        hier = AgglomerativeClustering(n_clusters=k, linkage="ward")
        hier_labels = hier.fit_predict(Xd)
        df.loc[mask, "cluster_hier"] = hier_labels
        metrics["hierarchical"] = {
            "algorithm": "AgglomerativeClustering (ward linkage)",
            "silhouette_score": round(float(silhouette_score(X, hier_labels)), 4),
            "davies_bouldin_score": round(float(davies_bouldin_score(Xd, hier_labels)), 4),
        }
    except Exception as exc:  # pragma: no cover - defensivu
        metrics["hierarchical"] = {"status": "failed", "error": str(exc)}

    # --- 5. Interpretasaun cluster -----------------------------------------
    top_terms = _top_terms(vec, km.cluster_centers_)

    clustered = df.loc[mask]
    profile = []
    for cid in sorted(clustered["cluster"].dropna().unique()):
        sub = clustered[clustered["cluster"] == cid]
        profile.append({
            "cluster": int(cid),
            "size": int(len(sub)),
            "share_pct": round(float(len(sub) / len(clustered) * 100), 1),
            "avg_rating": round(float(sub["review_stars"].dropna().mean()), 2)
            if sub["review_stars"].notna().any() else None,
            "avg_tokens": round(float(sub["n_tokens"].mean()), 1),
            "dominant_category": sub["category"].astype(str).mode().iat[0]
            if not sub["category"].isna().all() else None,
            "dominant_sentiment": sub["sentiment"].mode().iat[0]
            if sub["sentiment"].notna().any() else None,
            "dominant_language": sub["review_language"].mode().iat[0]
            if sub["review_language"].notna().any() else None,
            "top_terms": top_terms.get(int(cid), []),
        })

    results = {
        "metrics": metrics,
        "k_sweep": sweep.to_dict(orient="records"),
        "top_terms": {str(kk): vv for kk, vv in top_terms.items()},
        "profile": profile,
        "cluster_x_category": pd.crosstab(
            clustered["cluster"], clustered["category"].astype(str)
        ).to_dict(),
        "cluster_x_sentiment": pd.crosstab(
            clustered["cluster"], clustered["sentiment"]
        ).to_dict(),
    }
    return df, results


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
    d, res = run_clustering(d)

    print(json.dumps(res["metrics"], indent=2, ensure_ascii=False))
    print("\nK sweep (elbow + silhouette):")
    print(pd.DataFrame(res["k_sweep"]).to_string(index=False))
    print("\nProfil cluster:")
    for p in res["profile"]:
        print(f"  Cluster {p['cluster']} (n={p['size']}, {p['share_pct']}%) "
              f"| kat={p['dominant_category']} | sent={p['dominant_sentiment']} "
              f"| lingua={p['dominant_language']}")
        print(f"      terms: {', '.join(p['top_terms'])}")
