# Guia Power BI — Sistema Mining & Exploring Turizmu Timor-Leste

> Manual Capstone Project — **Seksaun 15: Dashboard no Visualization**
> Ferramenta: **Power BI Desktop** (gratuita, Windows)

Ficheiru CSV iha folder `powerbi/` ne'e **gera automátikamente** husi pipeline:

```powershell
py -3.11 src\pipeline.py
```

Kada vez ita halao pipeline, CSV sira-ne'e foun fila fali — depois iha Power BI
klik **Refresh** deit.

---

## 1. Import dadus (dala ida deit)

1. Loke **Power BI Desktop**
2. **Home → Get Data → More… → Folder → Connect**
3. Hili folder: `D:\turismo_project (1)\powerbi`
4. Pilih dan import setiap CSV sebagai tabel terpisah. Jangan Combine seluruh folder: schema fact, dimension, dan metrics berbeda.

> **Alternativa liu simples (rekomendadu ba iha-nia dahuluk):**
> **Get Data → Text/CSV** ida-ida ba ficheiru sira ne'ebé presiza:
> `fact_reviews.csv`, `dim_place.csv`, `dim_date.csv`, `dim_category.csv`,
> `dim_sentiment.csv`, `dim_cluster.csv`, `agg_category_sentiment.csv`,
> `cluster_points.csv`, `cluster_ksweep.csv`, `model_metrics.csv`,
> `confusion_matrix.csv`, `meta_pipeline.csv`.
>
> **IMPORTANTE:** iha janela preview, konfirma **File Origin = 65001: Unicode (UTF-8)**
> atu asentu (á, é, í, ó, ú, ç) sai loloos. CSV sira grava ho BOM (utf-8-sig),
> tanba ne'e Power BI normalmente deteta automátikamente.

---

## 2. Modelu dadus (Model view → kria relasaun)

Modelu **estrela**: 1 tabela fatu (`fact_reviews`) + tabela dimensaun.

| Husi (many)                     | Ba (one)                 | Kardinalidade | Direksaun filter |
| ------------------------------- | ------------------------ | ------------- | ---------------- |
| `fact_reviews[place_id]`        | `dim_place[place_id]`    | Many-to-One   | Single           |
| `fact_reviews[date_key]`        | `dim_date[date_key]`     | Many-to-One   | Single           |
| `fact_reviews[category]`        | `dim_category[category]` | Many-to-One   | Single           |
| `fact_reviews[sentiment]`       | `dim_sentiment[sentiment]` | Many-to-One | Single           |
| `fact_reviews[cluster]`         | `dim_cluster[cluster]`   | Many-to-One   | Single           |
| `cluster_points[review_key]`    | `fact_reviews[review_key]` | One-to-One  | Both             |

Depois kria relasaun:

- **`dim_date`** → klik tabela → **Table tools → Mark as date table** → kolona `date`
- **`dim_category`** → hili kolona `category` → **Column tools → Sort by column → `category_sort`**
- **`dim_sentiment`** → hili kolona `sentiment` → **Sort by column → `sentiment_sort`**
- **`dim_place`** → `lat` → **Data category = Latitude**; `lng` → **Longitude**;
  `city` → **City**

---

## 3. Measures DAX (kopia ba tabela `fact_reviews`)

Klik **Home → New measure**, kopia ida-ida:

```dax
Total Review = CALCULATE ( COUNTROWS ( fact_reviews ), fact_reviews[has_review] = TRUE () )

Total Lokal = DISTINCTCOUNT ( fact_reviews[place_id] )

Rating Médiu = AVERAGE ( fact_reviews[review_stars] )

Review Positivu =
CALCULATE ( [Total Review], fact_reviews[sentiment] = "Positive" )

Review Negativu =
CALCULATE ( [Total Review], fact_reviews[sentiment] = "Negative" )

Review ho Sentiment =
CALCULATE ( [Total Review], fact_reviews[sentiment] IN { "Positive", "Neutral", "Negative" } )

% Positivu = DIVIDE ( [Review Positivu], [Review ho Sentiment] )

% Negativu = DIVIDE ( [Review Negativu], [Review ho Sentiment] )

Review ho Teksu =
CALCULATE ( [Total Review], fact_reviews[has_text] = TRUE () )

Sentiment Score Médiu = AVERAGE ( fact_reviews[sentiment_score] )
```

Measure ba KPI "Last updated" (tabela `meta_pipeline`):

```dax
Last Updated =
CALCULATE (
    MAX ( meta_pipeline[metric_value] ),
    meta_pipeline[metric_name] = "last_data_collection_utc"
)

Silhouette Score =
CALCULATE (
    MAX ( meta_pipeline[metric_value] ),
    meta_pipeline[metric_name] = "silhouette_score"
)
```

Formatasaun: `% Positivu` no `% Negativu` → **Column tools → Format → Percentage → 1 decimal**.

---

## 4. Estrutura pájina (tuir Manual seksaun 15)

### Pájina 1 — **Overview**

| Vizual                | Konfigurasaun                                                                     |
| --------------------- | --------------------------------------------------------------------------------- |
| **Card** × 5          | `[Total Lokal]`, `[Total Review]`, `[Rating Médiu]`, `[% Positivu]`, `[% Negativu]` |
| **Card** (text)       | `[Last Updated]` → titlu "Kolesaun dadus ikus"                                     |
| **Stacked column**    | Axis `dim_category[category]` · Legend `dim_sentiment[sentiment]` · Values `[Total Review]` |
| **100% stacked bar**  | Hanesan leten, maibé tipu "100% stacked" → hatudu **% sentiment kada kategoria**    |
| **Donut**             | Legend `dim_sentiment[sentiment]` · Values `[Total Review]`                        |
| **Map**               | Location `dim_place[lat]`,`[lng]` · Legend `category` · Size `[Total Review]`      |
| **Slicer** × 4        | `dim_date[date]` (Between) · `dim_place[city]` · `dim_category[category]` · `dim_sentiment[sentiment]` |

**Kor**: iha **Format → Data colors**, uza kor husi kolona `dim_sentiment[sentiment_color]`
no `dim_category[category_color]` (Conditional formatting → Field value).

| Sentiment | Kor       | Kategoria   | Kor       |
| --------- | --------- | ----------- | --------- |
| Positive  | `#2E9E5B` | Historical     | `#6A4C93` |
| Neutral   | `#E2A33C` | Religious       | `#1B6CA8` |
| Negative  | `#D1495B` | Traditional | `#C1666B` |
| Unknown   | `#8A93A0` | Modern     | `#2A9D8F` |

### Pájina 2 — **Trend & Ranking**

| Vizual              | Konfigurasaun                                                            |
| ------------------- | ------------------------------------------------------------------------ |
| **Line chart**      | Axis `dim_date[year_month]` · Values `[Total Review]` · Legend `sentiment` |
| **Line chart**      | Axis `dim_date[year_month]` · Values `[Rating Médiu]`                     |
| **Bar chart**       | Axis `dim_place[place_name]` · Values `[Total Review]` · Top N = 10       |
| **Bar chart**       | Axis `dim_place[city]` · Values `[% Positivu]`                            |
| **Table**           | `agg_place_summary` (n_reviews, avg_review_stars, pct_positive, pct_negative) |

### Pájina 3 — **Clustering K-Means** ⭐

| Vizual                 | Konfigurasaun                                                                                   |
| ---------------------- | ----------------------------------------------------------------------------------------------- |
| **Scatter chart**      | X = `cluster_points[cluster_x]` · Y = `cluster_points[cluster_y]` · Legend = `dim_cluster[cluster_label]` · Details = `cluster_points[review_key]` |
| **Scatter chart** (2)  | Hanesan, maibé Legend = `cluster_points[category]` → kompara padraun natural vs kategoria manual |
| **Line chart**         | X = `cluster_ksweep[k]` · Y = `cluster_ksweep[inertia]` → **Elbow method**                       |
| **Line chart**         | X = `cluster_ksweep[k]` · Y = `cluster_ksweep[silhouette]` → **Silhouette tuir k**               |
| **Table**              | `dim_cluster` (cluster_label, size, share_pct, avg_rating, dominant_category, dominant_sentiment, top_terms) |
| **Matrix**             | Rows = `cluster` · Columns = `category` · Values = `[Total Review]`                              |
| **Card**               | `[Silhouette Score]`                                                                             |

> **DIKA IMPORTANTE ba scatter:** iha **Format → Marker**, hasai limite "Sampling".
> Iha **Values well**, tau `cluster_points[review_key]` iha **Details** atu kada review
> sai pontu ida (se lae, Power BI sei agrega hotu ba pontu ida deit).

### Pájina 4 — **Model Evaluation** ⭐

| Vizual                | Konfigurasaun                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------ |
| **Table**             | `model_metrics` (task, model, accuracy, precision_macro, recall_macro, f1_macro)            |
| **Clustered column**  | Axis `model_metrics[model]` · Values `accuracy`, `f1_macro` · Small multiples = `task`      |
| **Matrix**            | Rows = `confusion_matrix[actual]` · Columns = `[predicted]` · Values = Sum of `n` · Filter: hili `model` ida |
| **Slicer**            | `confusion_matrix[model]` → troka entre Model 1 no Model 2                                   |
| **Table**             | `model_metrics_per_class` (class, precision, recall, f1_score, support)                      |

**Confusion matrix ho kor:** iha Matrix → **Format → Cell elements → Background color →
Conditional formatting** (bazeia ba `Sum of n`, kor Blues).

### Pájina 5 — **Dadus & Diksionáriu**

| Vizual   | Konfigurasaun                                                     |
| -------- | ----------------------------------------------------------------- |
| **Table**| `fact_reviews` (place, category, sentiment, review_stars, review_text) |
| **Table**| `data_dictionary` (column, data_type, description, unit, missing)  |
| **Table**| `meta_pipeline` (metadata fonte dadus + timestamp)                 |

---

## 5. Refresh depois dadus foun

1. Halao fila fali: `py -3.11 src\pipeline.py`
2. Iha Power BI: **Home → Refresh**

CSV sira substitui automátikamente, tanba ne'e vizual hotu atualiza — **la presiza**
kria fila fali relasaun ka measure.

---

## 6. Ficheiru CSV — referénsia

| Ficheiru                       | Grain / kontéudu                                | Uza ba                                    |
| ------------------------------ | ------------------------------------------------ | ----------------------------------------- |
| `fact_reviews.csv`             | 1 baris = 1 review                               | Tabela fatu prinsipál                     |
| `dim_place.csv`                | 1 baris = 1 lokal turizmu                        | Mapa, ranking lokal                       |
| `dim_date.csv`                 | 1 baris = 1 loron                                | Time intelligence, slicer data            |
| `dim_category.csv`             | 4 kategoria + kor + deskrisaun                   | Slicer, orden, kor                        |
| `dim_sentiment.csv`            | 4 label sentiment + kor                          | Slicer, orden, kor                        |
| `dim_cluster.csv`              | 1 baris = 1 cluster + top terms                  | Legenda scatter, tabela interpretasaun    |
| `agg_category_sentiment.csv`   | Kategoria × Sentiment (+ %)                      | Grafiku pedidu dosente                    |
| `agg_place_summary.csv`        | Rezumu kada lokal                                | Tabela ranking                            |
| `agg_monthly_trend.csv`        | Fulan × kategoria × sentiment                    | Trend chart                               |
| `cluster_points.csv`           | Koordenada 2D kada review                        | **Scatter plot cluster**                  |
| `cluster_ksweep.csv`           | k, inertia, silhouette, davies_bouldin           | **Elbow + Silhouette chart**              |
| `model_metrics.csv`            | Metrika kada model                               | Tabela + bar chart evaluation             |
| `model_metrics_per_class.csv`  | Precision/Recall/F1 kada klase                   | Tabela detallu                            |
| `confusion_matrix.csv`         | model × actual × predicted × n                   | **Matrix confusion**                      |
| `eda_descriptive_stats.csv`    | Estatístika deskritiva                           | Tabela EDA                                |
| `meta_pipeline.csv`            | Metadata (timestamp, fonte, silhouette…)         | KPI card "Last updated"                   |
| `data_dictionary.csv`          | Diksionáriu dadus (Manual seksaun 10)            | Pájina dokumentasaun                      |

---

## 7. Checklist Manual seksaun 15

- [ ] KPI cards (Total Lokal, Total Review, Rating Médiu, % Positivu, % Negativu)
- [ ] Trend chart (review kada fulan)
- [ ] Comparison chart (Kategoria × Sentiment)
- [ ] Map (destinasaun turizmu ho lat/lng)
- [ ] Filter by date / location / category
- [ ] Prediction / Mining result (scatter cluster + tabela evaluation)
- [ ] Last updated timestamp (card `[Last Updated]`)
