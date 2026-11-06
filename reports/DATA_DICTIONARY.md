# Diksionáriu Dadus — Sistema Mining & Exploring Turizmu Timor-Leste

> **Manual Capstone Project seksaun 10:** _"Halo data dictionary: column, data type,
> description, unit no missing values."_

Versaun CSV (ba Power BI / Excel): `powerbi/data_dictionary.csv`

---

## 1. Fonte dadus

| Item                  | Valór                                                                |
| --------------------- | -------------------------------------------------------------------- |
| Fonte                 | Google Maps — Places & Reviews (destinasaun turizmu Timor-Leste)      |
| Métodu asesu          | Apify actor `compass/crawler-google-places` → export JSON             |
| URL                   | https://www.google.com/maps                                          |
| Status API            | **SEIDAUK iha API key** (Google Places API presiza billing account)   |
| Modu kolesaun         | Batch snapshot (re-kolesaun manual)                                  |
| Interval atualizasaun | Rekomenda kada 7 loron (weekly). Sistema **la'ós real-time**.        |
| Kampu timestamp       | `scrapedAt` (nivel lokal)                                            |
| Unidade analiza       | 1 baris = 1 review                                                   |
| Log kolesaun          | `reports/data_collection_log.json` (gera automátikamente)            |

---

## 2. Dataset Layer (Manual seksaun 10)

| Layer         | Ficheiru                              | Deskrisaun                                        | Baris |
| ------------- | ------------------------------------- | ------------------------------------------------- | ----: |
| **Raw**       | `data/raw/dataset_turris.json`        | Dadus orijinal husi scraper, la muda              |    569 lokal |
| **Processed** | `data/processed/reviews_processed.csv`| Depois cleaning, missing-value, text, feature eng. |   3.394 |
| **Feature**   | `data/features/review_features.csv`   | Variavel ne'ebé prepara ba modelu deit            |   3.394 |
| **Output**    | `data/output/reviews_mined.csv`       | Prediction, cluster, sentiment, kategoria         |   3.394 |
| **Anotasaun** | `data/place_category_labels.csv`      | Ground truth kategoria (anotasaun manual) — EDITAVEL, kobre 49/569 lokal | 50 |

---

## 3. Kolona — Identifikasaun lokal

| Kolona                  | Tipu    | Deskrisaun                              | Unidade / valór    | Missing values          |
| ----------------------- | ------- | --------------------------------------- | ------------------ | ----------------------- |
| `place_id`              | text    | ID úniku lokal (Google `placeId`)        | —                  | Nunka mamuk             |
| `place_name`            | text    | Naran lokal turizmu                      | —                  | Nunka mamuk             |
| `place_category_google` | text    | Kategoria husi Google                    | ex: Tourist attraction | Nunka mamuk         |
| `city`                  | text    | Cidade/munisípiu (imputa husi `address`) | —                  | 296 seidauk hetan (359 imputa husi 655 mamuk orijinál) |
| `address`               | text    | Enderesu kompletu                        | —                  | Bele mamuk              |
| `lat`, `lng`            | decimal | Koordenada GPS                           | grau desimál       | 0                       |
| `place_avg_rating`      | decimal | Rating médiu lokal iha Google            | estrela 1–5        | 206 mamuk (lokal sem review) |
| `place_reviews_count`   | integer | Total review iha Google                  | kontajen           | Bele mamuk              |
| `place_url`             | text    | Ligasaun Google Maps                     | URL                | Nunka mamuk             |
| `scraped_at_parsed`     | datetime| Timestamp kolesaun dadus (UTC)           | —                  | Nunka mamuk             |

---

## 4. Kolona — Review orijinal

| Kolona               | Tipu     | Deskrisaun                                  | Unidade / valór | Missing values         |
| -------------------- | -------- | ------------------------------------------- | --------------- | ---------------------- |
| `review_id`          | text     | ID úniku review                             | —               | 206 mamuk (lokal sem review) |
| `reviewer_name`      | text     | Naran reviewer — **uza deit ba dedupe**, la exporta ba dashboard | — | Bele mamuk |
| `review_text`        | text     | Teksu komentáriu orijinal                   | —               | **1.430 mamuk (42.13%)** |
| `review_stars`       | integer  | Rating individual                            | estrela 1–5     | 262 mamuk              |
| `review_date_parsed` | datetime | Data-oras review publika (UTC)              | —               | 206 mamuk              |
| `review_language`    | text     | Kódigu lingua orijinal                       | en/id/pt/tet/…  | `'unknown'` se mamuk   |
| `is_local_guide`     | boolean  | Reviewer mak Google Local Guide ka lae      | true/false      | `false` se mamuk       |
| `has_review`         | boolean  | Baris ne'e iha review ka lokal deit         | true/false      | Nunka mamuk            |

---

## 5. Kolona — Preprocessing (Manual seksaun 11)

| Kolona                   | Tipu    | Deskrisaun                                                    | Unidade / valór   | Missing values   |
| ------------------------ | ------- | ------------------------------------------------------------- | ----------------- | ---------------- |
| `has_text`               | boolean | Review iha teksu ka lae                                        | true/false        | Nunka mamuk      |
| `has_rating`             | boolean | Review iha rating ka lae                                       | true/false        | Nunka mamuk      |
| `text_clean`             | text    | Teksu depois lowercase + hasai URL/emoji/simbolu/numeru        | —                 | `''` se sem teksu |
| `tokens_joined`          | text    | Token depois hasai stopwords (**input ba TF-IDF**)             | —                 | `''` se sem teksu |
| `n_tokens`               | integer | Kuantidade token depois preprocessing                          | token             | 0 se sem teksu   |
| `n_chars`                | integer | Kuantidade karakter teksu orijinal                             | karakter          | 0 se sem teksu   |
| `text_empty_after_clean` | boolean | Iha teksu maibé sai mamuk depois cleaning (emoji deit)         | true/false        | Nunka mamuk (66 true) |
| `is_outlier_length`      | boolean | Flag outlier naruk teksu (**IQR**, limite > 513,5 karakter)    | true/false        | Nunka mamuk (176 true) |
| `is_long_review`         | boolean | Review ho ≥ 8 token                                            | true/false        | Nunka mamuk      |
| `review_year`            | integer | Tinan review                                                   | tinan             | Bele mamuk       |
| `review_month`           | integer | Fulan review                                                   | 1–12              | Bele mamuk       |
| `review_yearmonth`       | text    | Tinan-fulan                                                    | `YYYY-MM`         | `'NaT'` se mamuk |
| `review_weekday`         | text    | Loron semana                                                   | Monday…Sunday     | Bele mamuk       |
| `days_since_review`      | integer | Loron husi review to'o kolesaun dadus                          | loron             | Bele mamuk       |
| `rating_deviation`       | decimal | `review_stars` − `place_avg_rating`                            | estrela           | Bele mamuk       |
| `split`                  | text    | Train/test split ba modelu supervisadu                         | train/test/not_used | Nunka mamuk (1.423/475/1.496) |

---

## 6. Kolona — Kategoria (4 kategoria fixu)

| Kolona                  | Tipu | Deskrisaun                                                        | Unidade / valór                    | Missing values     |
| ----------------------- | ---- | ----------------------------------------------------------------- | ---------------------------------- | ------------------ |
| `category_label`        | text | **GROUND TRUTH** — anotasaun manual nivel-lokal (49/569 lokal deit) | Sejarah/Agama/Tradisional/Moderen  | **3.145 mamuk** (520 lokal seidauk iha anotasaun) |
| `label_confidence`      | text | Nivel konfiansa anotasaun manual                                   | high / medium / low                | Mamuk se `category_label` mamuk |
| `label_evidence`        | text | Justifikasaun anotasaun (sita review ka deskrisaun)                | —                                  | Mamuk se `category_label` mamuk |
| `category`              | text | **Kategoria FINAL uza iha analiza** (= `category_label` se iha, senaun `category_rule_context`) | 4 kategoria | Nunka mamuk (fallback garante valór) |
| `category_source`       | text | Fonte kategoria final                                              | anotasaun_manual (249) / rule_based_fallback (3.145) | Nunka mamuk |
| `category_rule_text`    | text | Predisaun **Model 1** husi teksu review **deit** (evaluation justu)| 4 kategoria                        | Nunka mamuk        |
| `category_rule_context` | text | Predisaun **Model 1** husi teksu + naran lokal + kat. Google       | 4 kategoria                        | Nunka mamuk        |
| `category_rule_match`   | text | Tipu match rule-based                                              | keyword / keyword_tiebreak / fallback | Nunka mamuk     |
| `category_rule_keywords`| text | Keyword sira ne'ebé match (transparénsia)                          | lista separadu ho vírgula          | `''` se fallback   |
| `category_pred_ml`      | text | Predisaun **Model 2** (supervisadu)                                | 4 kategoria                        | Mamuk se la'ós test set |

### Definisaun kategoria

| Kategoria       | Definisaun                                                                        |
| --------------- | --------------------------------------------------------------------------------- |
| **Sejarah**     | Sitiu istóriku: memorial, edifisiu era kolonial, marka guerra/rezisténsia, museu    |
| **Agama**       | Sitiu relijiozu: gruta/santuáriu, igreja, dalan peregrinasaun                       |
| **Tradisional** | Sitiu kultura lisan: uma lulik/lisan, artezanatu, aldeia tradisional, natar         |
| **Moderen**     | Atrasaun natureza, lazer, no fasilidade turizmu kontemporáneu                       |

---

## 7. Kolona — Sentiment

| Kolona                 | Tipu    | Deskrisaun                                                | Unidade / valór                       | Missing values          |
| ---------------------- | ------- | --------------------------------------------------------- | ------------------------------------- | ----------------------- |
| `sentiment`            | text    | **Label FINAL** (teksu → rating → Unknown)                | Positive/Neutral/Negative/Unknown     | Nunka mamuk             |
| `sentiment_source`     | text    | Fonte label final (transparénsia)                          | text (1.898) / rating (1.290) / none (206) | Nunka mamuk        |
| `sentiment_lexicon`    | text    | Predisaun **Model 1** lexicon husi teksu                   | Positive/Neutral/Negative             | Mamuk se sem teksu      |
| `sentiment_true_proxy` | text    | **GROUND TRUTH PROXY** deriva husi rating                  | Positive (≥4) / Neutral (3) / Negative (≤2) | Mamuk se sem rating |
| `sentiment_pred_ml`    | text    | Predisaun **Model 2** (supervisadu)                        | Positive/Neutral/Negative             | Mamuk se la'ós test set |
| `sentiment_score`      | integer | `n_liafuan_positivu − n_liafuan_negativu`                  | kontajen                              | 0 se sem teksu          |
| `sentiment_n_pos`      | integer | Kuantidade liafuan pozitivu ne'ebé match                   | kontajen                              | 0 se sem teksu          |
| `sentiment_n_neg`      | integer | Kuantidade liafuan negativu ne'ebé match                   | kontajen                              | 0 se sem teksu          |

---

## 8. Kolona — Clustering

| Kolona         | Tipu    | Deskrisaun                                              | Unidade / valór | Missing values     |
| -------------- | ------- | ------------------------------------------------------- | --------------- | ------------------ |
| `cluster`      | integer | ID cluster **K-Means**                                   | 0…k−1 (k=4)     | Mamuk se sem teksu (1.496) |
| `cluster_hier` | integer | ID cluster **Agglomerative/Hierarchical** (komparasaun)  | 0…k−1           | Mamuk se sem teksu |
| `cluster_x`    | decimal | Koordenada 1 husi TruncatedSVD (**ba scatter plot**)     | —               | Mamuk se sem teksu |
| `cluster_y`    | decimal | Koordenada 2 husi TruncatedSVD (**ba scatter plot**)     | —               | Mamuk se sem teksu |

---

## 9. Ficheiru CSV Power BI

| Ficheiru                       | Grain                                     |
| ------------------------------ | ----------------------------------------- |
| `fact_reviews.csv`             | 1 baris = 1 review                        |
| `dim_place.csv`                | 1 baris = 1 lokal turizmu                 |
| `dim_date.csv`                 | 1 baris = 1 loron                         |
| `dim_category.csv`             | 1 baris = 1 kategoria (4)                 |
| `dim_sentiment.csv`            | 1 baris = 1 label sentiment (4)           |
| `dim_cluster.csv`              | 1 baris = 1 cluster                       |
| `agg_category_sentiment.csv`   | 1 baris = kategoria × sentiment           |
| `agg_place_summary.csv`        | 1 baris = 1 lokal (rezumu)                |
| `agg_monthly_trend.csv`        | 1 baris = fulan × kategoria × sentiment   |
| `cluster_points.csv`           | 1 baris = 1 review ho koordenada 2D       |
| `cluster_ksweep.csv`           | 1 baris = 1 valór k                       |
| `model_metrics.csv`            | 1 baris = 1 modelu                        |
| `model_metrics_per_class.csv`  | 1 baris = modelu × klase                  |
| `confusion_matrix.csv`         | 1 baris = modelu × actual × predicted     |
| `eda_descriptive_stats.csv`    | 1 baris = 1 variavel numériku             |
| `meta_pipeline.csv`            | 1 baris = 1 metadata (key-value)          |
| `data_dictionary.csv`          | 1 baris = 1 kolona                        |
