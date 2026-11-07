# 🇹🇱 Sistema Mining & Exploring — Turizmu Timor-Leste

**Sentiment Analysis + Category Classification (4 kategoria) + K-Means Clustering**

| Komponente        | Informasaun                                                                     |
| ----------------- | ------------------------------------------------------------------------------- |
| Instituisaun      | Dili Institute of Technology (DIT)                                              |
| Programa Estudu   | S1 Siencia Komputadór                                                           |
| Unidade Kurikulér | Mining and Exploring                                                            |
| Tópiku Manual     | **No. 4 — Data Mining no Rekomendasaun Turizmu**                                |
| Modifikasaun      | **Komponente _Recommendation_ HASAI tuir orientasaun dosente**                   |
| Estudu Kazu       | Timor-Leste (Liquiçá, Maubara, Ainaro, Bobonaro, Covalima, Manufahi, Ermera)     |
| Python            | **3.11 (obrigatóriu)**                                                          |
| Output            | Pipeline Python · Dashboard Streamlit · Export Power BI · Notebook EDA · Figura  |

---

## 1. Problema no Objetivu (LO1)

### Problema

Timor-Leste iha destinasaun turizmu barak (cachoeira, praia, sitiu istóriku, uma
lisan, gruta), maibé **la iha sistema ida ne'ebé analiza sistematikamente saida mak
turista hatete kona-ba destinasaun sira-ne'e**. Review online iha Google Maps mosu
iha lingua oioin (Tetun, Indonésia, Portugés, Inglés) no la organizadu, tanba ne'e
susar ba jestór turizmu atu hatene:

1. Tipu destinasaun (**Historical / Religious / Traditional / Modern**) ida-ne'ebé mak
   simu apresiasaun di'ak liu husi turista?
2. Destinasaun ida-ne'ebé mak presiza atensaun (sentiment negativu makaas)?
3. Iha padraun tema ruma ne'ebé mosu naturalmente iha komentáriu turista?

### Objetivu

| No. | Objetivu                                                                                                  |
| --- | --------------------------------------------------------------------------------------------------------- |
| 1   | Rekolha no organiza review turista husi destinasaun turizmu Timor-Leste                                     |
| 2   | Klasifika kada destinasaun ba **4 kategoria fixu**: Historical, Religious, Traditional, Modern                    |
| 3   | Determina **sentiment** (Positive / Neutral / Negative) husi kada komentáriu                                |
| 4   | Deskobre **padraun natural** iha teksu review ho **K-Means clustering**                                     |
| 5   | Avalia modelu ho metrika formal (Accuracy, Precision, Recall, F1, Confusion Matrix, Silhouette)             |
| 6   | Aprezenta rezultadu iha **dashboard interativu** (Streamlit) no **Power BI**                                |

### Limitasaun (scope)

- Sistema **la halo rekomendasaun** ba turista (hasai tuir orientasaun dosente).
- Dataset = **snapshot batch**, la'ós real-time (haree seksaun 2).
- Dataset atual: **569 lokal, 3.394 baris review** (1.964 ho teksu). Anotasaun
  manual (ground truth 4-kategoria) deit kobre **49 husi 569 lokal** (destinasaun
  orijinál); restu uza fallback rule-based transparente — haree seksaun 5 no 15.

---

## 2. Fonte Dadus (LO2 · Manual seksaun 10)

| Item                | Deskrisaun                                                                                   |
| ------------------- | ---------------------------------------------------------------------------------------------- |
| Fonte               | **Google Maps — Places & Reviews** (destinasaun turizmu Timor-Leste)                           |
| Métodu asesu        | Apify actor `compass/crawler-google-places` → export JSON                                      |
| Ficheiru raw        | `data/raw/dataset_turris.json` (~7,7 MB)                                                       |
| Formatu             | JSON — lista objetu **lokal turizmu**, kada ida iha lista `reviews` aninhadu                    |
| Unidade analiza     | **1 baris = 1 review** (sistema halo _flatten_ husi lokal → review)                            |
| Total lokal         | 569                                                                                            |
| Total baris         | 3.396 raw → **3.394 depois cleaning** (2 duplicate hasai)                                       |
| Review ho teksu     | 1.964 (57,9%) · Review rating-only/sem review: 1.430 (42,1%)                                    |
| Timestamp kolesaun  | Kampu `scrapedAt` → hatudu iha dashboard hanesan "Kolesaun dadus ikus"                          |
| Períodu review      | Setembru 2013 — Setembru 2026                                                                  |

📌 **Nota versaun dataset:** snapshot orijinál (50 lokal, backup iha
`data/raw/dataset_turris_backup_2026-09-05.json`) troka ba snapshot foun ida
(569 lokal) husi crawler `compass/crawler-google-places` nia export
`dataset_crawler-google-places_2026-09-04_06-32-03-783.json` (04 Setembru 2026).
Backup mantein ba auditoria/reproducibility (Manual seksaun 10).

### ⚠️ Status API — importante ba defeza

> **Grupu SEIDAUK iha API key.** Google Places API presiza _billing account_.
> Tanba ne'e, versaun ida-ne'e uza **snapshot JSON** husi scraper, no sistema
> **la reivindika "real-time"** — tuir nota Manual seksaun 5.
>
> - **Modu atual:** batch snapshot (re-kolesaun manual)
> - **Interval rekomendadu:** kada 7 loron (weekly batch)
> - **Migrasaun ba API:** troka **DEIT** funsaun `load_raw_data()` iha
>   `src/data_collection.py` ba _HTTP fetch + polling_ (stub `fetch_from_api()`
>   ona iha). Módulu seluk hotu (preprocessing, EDA, mining, dashboard) **la
>   presiza muda**, tanba sira servisu iha nivel DataFrame.

### Étika no Terms of Service

- Dadus review públiku deit; **la kolekta** email, telefone ka dadus pessoal sensivel.
- Kolona `reviewer_name` uza **deit** ba dedupe teknika, no **la hatudu** iha dashboard.
- Metadata kolesaun grava automátikamente iha `reports/data_collection_log.json`.

---

## 3. Arkitetura Sistema (Manual seksaun 5 · Blok Diagram)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PROBLEMA:  Sentiment no kategoria destinasaun turizmu Timor-Leste          │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  ONLINE DATA SOURCE — Google Maps Places & Reviews (Apify scraper)          │
│  [SEIDAUK iha API key → snapshot batch, la'ós real-time]                    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA COLLECTION            src/data_collection.py                          │
│    · load_raw_data()        le'e JSON                                       │
│    · flatten_reviews()      lokal → nivel review                            │
│    · attach_place_labels()  junta anotasaun manual (ground truth)           │
│    · write_collection_log() metadata fonte + timestamp                      │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA STORAGE (4 layer — Manual seksaun 10)                                 │
│    RAW       data/raw/dataset_turris.json                                   │
│    PROCESSED data/processed/reviews_processed.csv                           │
│    FEATURE   data/features/review_features.csv                              │
│    OUTPUT    data/output/reviews_mined.csv                                  │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  PREPROCESSING              src/preprocessing.py                            │
│  Cleaning → Missing Values → Transformation → Outliers (IQR) →              │
│  Text (lowercase, tokenize, stopwords) → Feature Eng. → Train/Test Split    │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  EDA                        src/eda.py  +  notebooks/01_eda_turismo.ipynb   │
│  Descriptive · Distribution · Trend · Correlation · Ranking · Spatial       │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  DATA MINING                                                                │
│   ┌─────────────────────────────┐  ┌────────────────────────────┐           │
│   │ MODEL 1 (baseline)          │  │ MODEL 2 (machine learning) │           │
│   │ · category_classifier.py    │  │ · supervised_models.py     │           │
│   │   rule-based keyword        │  │   TF-IDF + Naive Bayes     │           │
│   │ · sentiment_analysis.py     │  │   Logistic Regression      │           │
│   │   lexicon multi-lingua      │  │   Linear SVM               │           │
│   │                             │  │   Random Forest            │           │
│   └─────────────────────────────┘  └────────────────────────────┘           │
│   ┌──────────────────────────────────────────────────────────┐              │
│   │ UNSUPERVISED — src/clustering.py                         │              │
│   │ TF-IDF → K-Means (k sweep, elbow) → SVD 2D → top terms   │              │
│   │ + komparasaun Hierarchical (Ward)                        │              │
│   └──────────────────────────────────────────────────────────┘              │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  MODEL EVALUATION           src/model_evaluation.py                         │
│  Accuracy · Precision · Recall · F1 · Confusion Matrix · Cohen's Kappa      │
│  Silhouette · Davies-Bouldin · Calinski-Harabasz · Cross-validation         │
└────────────────────────────────┬────────────────────────────────────────────┘
                                 ▼
┌──────────────────────────────┬──────────────────────────────────────────────┐
│  VISUALIZATION               │  DASHBOARD                                   │
│  reports/figures/*.png (13)  │  dashboard/app.py    (Streamlit, 4 tab)      │
│  src/export_powerbi.py       │  powerbi/*.csv (17)  (Power BI Desktop)      │
└──────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 4. Instalasaun no Oinsá Halao

### Rekezitu

- **Python 3.11** (obrigatóriu — projetu ne'e testadu iha 3.11.6)
- Windows / Linux / macOS
- Power BI Desktop (opsional, ba dashboard Power BI)

### Pasu

```powershell
# 1. Konfirma Python 3.11 iha
py -0p                       # tenke hatudu -V:3.11
py -3.11 --version

# 2. (Rekomendadu) kria virtual environment ho 3.11
py -3.11 -m venv .venv
.\.venv\Scripts\activate     # Linux/macOS: source .venv/bin/activate

# 3. Instala dependénsia
py -3.11 -m pip install -r requirements.txt
```

### Halao sistema

```powershell
# A. PIPELINE KOMPLETU (collection → ... → export Power BI)
py -3.11 src\pipeline.py

# Sem figura PNG (lalais liu)
py -3.11 src\pipeline.py --no-figures

# B. DASHBOARD STREAMLIT
py -3.11 -m streamlit run dashboard\app.py
#    → loke automátikamente iha http://localhost:8501

# C. NOTEBOOK EDA
py -3.11 -m jupyter lab notebooks\01_eda_turismo.ipynb

# D. POWER BI
#    Loke Power BI Desktop → Get Data → Folder → hili folder `powerbi\`
#    Instrusaun detalladu: powerbi\GUIA_POWERBI.md
```

### Halao módulu ida-ida (ba debugging / demonstrasaun)

```powershell
py -3.11 src\annotate_places.py       # rejenera anotasaun manual kategoria
py -3.11 src\data_collection.py       # kolesaun + log
py -3.11 src\preprocessing.py         # preprocessing + meta
py -3.11 src\category_classifier.py   # Model 1 kategoria
py -3.11 src\sentiment_analysis.py    # Model 1 sentiment
py -3.11 src\clustering.py            # K-Means + elbow + profile
py -3.11 src\supervised_models.py     # Model 2 (NB / LR / SVM / RF)
py -3.11 src\eda.py                   # EDA + figura
py -3.11 src\model_evaluation.py      # metrika + confusion matrix
py -3.11 src\export_powerbi.py        # export CSV Power BI
```

---

## 5. Kategoria 4 — Anotasaun Manual (Ground Truth)

Dosente ezije **4 kategoria fixu**: `Historical` · `Religious` · `Traditional` · `Modern`.

Dataset Google Maps **la iha** kolona kategoria sira-ne'e, no Manual seksaun 14
ezije metrika Classification (Accuracy/Precision/Recall/F1/Confusion Matrix) —
metrika sira-ne'e **so bele kalkula bainhira iha ground truth**. Tanba ne'e grupu
halo **anotasaun manual** ba **49 lokal turizmu** (destinasaun orijinál husi
snapshot uluk), bazeia ba:

1. Naran lokal + kategoria Google
2. Deskrisaun ofisial (`description` / `ownerDescription`)
3. **Evidensia husi teksu review rasik** (sita iha kolona `label_evidence`)
4. Konhesimentu kontekstu Timor-Leste

📄 **Ficheiru:** `data/place_category_labels.csv` — **bele edita ho Excel!**
Se dosente ka grupu la konkorda ho label ida, troka deit iha CSV; pipeline le'e
CSV ne'e diretamente (la presiza muda kódigu).

🔁 **Rejenera:** `py -3.11 src\annotate_places.py`
(⚠️ ida-ne'e sobrescreve edisaun manual — no só re-mapeia `EXPERT_LABELS`
ne'ebé eziste; **la** rekomendadu bainhira dataset agora iha 520 lokal foun sem
konhesimentu ekspertu, haree nota kraik)

### ⚠️ Kobertura ground truth (dataset expandidu ba 569 lokal)

Snapshot foun (04 Set 2026) hasae dataset husi 50 ba **569 lokal** (hotel,
kafe, restaurante, muzeu, praia, ns. inklui ona), maibé `place_category_labels.csv`
seidauk atualiza — nia CSV kontinua **50 rezistu (49 mak koinside ho placeId
dataset foun; 1 lokal la mosu ona iha snapshot)**. Grupu **desidiu LA halo
anotasaun automátiku ba 520 lokal foun** ho `annotate_places.py`, tanba nia
diksionáriu `EXPERT_LABELS` bazeia deit ba naran/evidensia husi 50 lokal
orijinál — se aplika direta, 520 lokal foun sei simu fallback "Modern"/`low`
sem evidensia real, ida-ne'e **la lejítimu hanesan ground truth**.

Tanba ne'e, `category_classifier.py` (haree `apply_category_classification()`)
uza dezenhu ida ne'ebé ona iha kódigu: kolona final `category` = anotasaun
manual **bainhira iha** (49 lokal), senaun kai fali ba predisaun rule-based
kontekstu (`category_rule_context`); kolona `category_source` marka
transparentemente `anotasaun_manual` ka `rule_based_fallback` ba kada baris.
Efeitu prátiku:

- **EDA, dashboard, clustering** (unsupervised) — funsiona ba **569 lokal
  hotu**, tanba la presiza ground truth.
- **Model Evaluation ba Classification kategoria** (Manual sek. 14) — treinu/
  teste **DEIT ho 49 lokal ho label lolos** (125 review), atu evitá metrika
  "ground truth" ne'ebé fabrikadu ka la konfiavel — haree seksaun 8 no 15.

### Rezultadu anotasaun (49 lokal ho ground truth)

| Kategoria       | Lokal | Review | Deskrisaun                                                         |
| --------------- | ----: | -----: | ------------------------------------------------------------------ |
| **Historical**     |     4 |     60 | Memorial masakre, edifisiu era kolonial, kompleksu termal Marobo    |
| **Religious**       |     2 |     10 | Gruta Soro (santuáriu Marianu), Flecha (dalan peregrinasaun Ramelau)|
| **Traditional** |     9 |     16 | Uma Lisan, casas tradicionais, artezanatu, aldeia lisan             |
| **Modern**     |    34 |    163 | Cachoeira, praia, lagoa, camping, fasilidade turizmu kontemporáneu  |

Kada label iha **nivel konfiansa**: `high` (28 lokal, evidensia diretu) ka
`medium` (22 lokal, interpretasaun kontekstu — bele diskuti iha relatóriu).
(Ficheiru CSV iha 50 rezistu; 1 la aplika ba dataset atual — haree nota leten.)

**Exemplu evidensia:**

- _Pousada de Maubara_ → **Historical**, review (pt): _"Edifício colonial português
  renovado mas abandonado"_
- _Marobo Hot Spring_ → **Historical** (`medium`), review (en): _"developed by the
  Portuguese, with a resort complex built around the pools, but fell into
  disrepair"_ — lokal ne'e mos iha komponente natureza no lisan, label bele diskuti
- _Flecha (Fleixa)_ → **Religious** (`medium`), review (en): _"This's the start hiking
  point to Mt. Ramelau"_ — dalan peregrinasaun katóliku boot liu iha Timor-Leste

---

## 6. Preprocessing (Manual seksaun 11)

| Etapa Manual            | Implementasaun                                                        | Rezultadu dataset atual                    |
| ----------------------- | --------------------------------------------------------------------- | ------------------------------------------ |
| **Cleaning**            | Hasai duplicate (review_id; lokal+teksu+reviewer) no baris invalidu    | 3.396 → 3.394 baris (2 duplicate)          |
| **Missing Values**      | Tratamentu **esplisitu** (la halo drop): flag `has_text`/`has_rating`; `city` imputa husi `address` | 1.430 review sem teksu (42,1%); 359 `city` imputa (296 seidauk hetan) |
| **Transformation**      | Date parsing UTC, konversaun numériku, normalizasaun teksu             | —                                          |
| **Outliers**            | Metodu **IQR** ba naruk teksu → **FLAG deit, LA HASAI**                | 176 outlier (limite: > 513,5 karakter)     |
| **Text**                | lowercase → hasai URL/emoji/simbolu → tokenize → stopwords (4 lingua)  | 8.049 liafuan úniku; 19,16 token/review     |
| **Feature Engineering** | `n_tokens`, `n_chars`, `is_long_review`, `review_year/month`, `days_since_review`, `rating_deviation`, … | 12 kolona foun |
| **Train/Test Split**    | Stratifikadu tuir `category_label`, `test_size=0.25`, `random_state=42` | 1.423 train / 475 test / 1.496 not_used (nivel review, input ba Model 2 sentiment) |

**Tanba sá outlier flag deit, la hasai?** Review naruk (ex: 1.500 karakter) mak
informasaun valiozu tebes ba text mining, la'ós erru dadus. Hasai sira sei lakon
insight. Flag uza deit atu dokumenta no filtra iha dashboard se presiza.

---

## 7. Data Mining (LO4 · Manual seksaun 13)

### Justifikasaun algoritmu (Manual: _"Projetu tenke justifika tanba sá algoritmu ida apropriadu"_)

| Problema                 | Algoritmu                                         | Justifikasaun                                                                                                                          |
| ------------------------ | ------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| **Category (Model 1)**   | Rule-based keyword lexicon (Tetun/ID/PT/EN)       | Kategoria fixu, defini ona → lójika transparente, audita fasil, la presiza dadus training. Serve hanesan **baseline** ba Model 2.        |
| **Category (Model 2)**   | TF-IDF + **NB / Logistic Regression / SVM / RF**  | Manual seksaun 13 lista algoritmu sira-ne'e ba Classification. Kompara ho baseline → responde "teksu mesak bele preve kategoria ka lae?" |
| **Sentiment (Model 1)**  | Lexicon multi-lingua + tratamentu negasaun        | Manual seksaun 13 lista "lexicon approach" esplisitamente. Dataset la iha label manual → lexicon transparente.                          |
| **Sentiment (Model 2)**  | TF-IDF + NB / LR / SVM / RF                       | Manual seksaun 13 (Sentiment: "NB, LR, SVM, lexicon approach").                                                                          |
| **Clustering**           | **K-Means** (TF-IDF) + **Hierarchical (Ward)**    | Manual seksaun 13 (Clustering: "K-Means, Hierarchical Clustering"). Deskobre padraun natural, valida kategoria 4-fixu.                   |
| **Redusaun dimensaun**   | TruncatedSVD (2 komponente)                       | Matriz TF-IDF mak **sparse**; PCA sklearn la simu sparse diretamente. Uza ba **scatter plot cluster** — haree kaixa iha okos.            |

### 📐 Dezisaun teknika: tanba sá SVD, la'ós t-SNE?

Grupu **testa** projesaun 2D oioin ba scatter plot cluster. Métrika teste:
**silhouette husi label K-Means kalkula iha espasu 2D** — se valór ne'e besik ba
(ka liu aas duke) silhouette iha espasu TF-IDF orijinál, projesaun ne'e **fiel**
ba partisaun ne'ebé K-Means halo lolos. Ho dataset atual (569 lokal, 1.898
review clusterizadu), silhouette orijinál = **0.0734** no silhouette 2D
(TruncatedSVD) = **0.1457**.

| Projesaun                      | Silhouette iha 2D | Dataset                        |
| ------------------------------ | ----------------: | ------------------------------- |
| **TruncatedSVD(2)** ⭐ (atual)  |        **0.1457** | 569 lokal / 1.898 review        |
| SVD(3) komponente 2&3          |           −0.1671 | teste exploratóriu (50 lokal)   |
| SVD(2) baris L2-normalizadu    |           −0.3639 | teste exploratóriu (50 lokal)   |
| SVD(50) → t-SNE (perplexity 30)|           −0.0270 | teste exploratóriu (50 lokal)   |
| SVD(50) → t-SNE (perplexity 10)|           −0.0053 | teste exploratóriu (50 lokal)   |
| t-SNE cosine (perplexity 15)   |           −0.1885 | teste exploratóriu (50 lokal)   |
| MDS cosine                     |           −0.1291 | teste exploratóriu (50 lokal)   |

**Konkluzaun:** só **TruncatedSVD(2)** mak preserva partisaun K-Means. Projesaun
non-linear (t-SNE, MDS) produz "nuvem" pontu sem estrutura — **enganadór**, tanba
fó impresaun katak clustering la funsiona, maski K-Means halo partisaun lolos iha
espasu TF-IDF. Linhas non-linear iha tabela leten mak rezultadu **teste
exploratóriu ida-deit** husi dataset orijinál (50 lokal, dokumentadu iha
`src/clustering.py` NOTA 2) — la re-halao ba dataset foun tanba konkluzaun
kualitativu (SVD manan) la muda; linha TruncatedSVD(2) mak **valór atual**
(re-kalkula kada pipeline run).

### Modu predisaun kategoria (importante ba evaluation honestu)

| Modu                    | Input                                    | Uza ba                                                                       |
| ----------------------- | ---------------------------------------- | ---------------------------------------------------------------------------- |
| `category_rule_text`    | **Teksu review deit**                    | **Evaluation justu** — la iha leakage                                        |
| `category_rule_context` | Teksu + naran lokal + kategoria Google   | Modu produsaun (informasaun barak liu), maibé score otimista (leakage)       |

---

## 8. Model Evaluation (LO5 · Manual seksaun 14)

### Rezultadu (dataset atual)

| Tarefa    | Model                                | Accuracy | Precision (macro) | Recall (macro) | F1 (macro) | CV F1 (macro) |
| --------- | ------------------------------------ | -------: | ----------------: | -------------: | ---------: | ------------: |
| Sentiment | Model 1 — Lexicon                    |   0.6189 |            0.4694 |         0.4457 |     0.4066 |             — |
| Sentiment | Model 2 — Naive Bayes ⚠️_dejeneradu_ |   0.8470 |            0.2823 |         0.3333 |     0.3057 |        0.3093 |
| Sentiment | Model 2 — **Logistic Regression** ⭐  |   0.8082 |            0.4488 |         0.3931 |     0.4067 |    **0.3728** |
| Sentiment | Model 2 — Linear SVM                 |   0.8427 |            0.5620 |         0.3662 |     0.3705 |        0.3334 |
| Sentiment | Model 2 — Random Forest              |   0.8276 |            0.2813 |         0.3257 |     0.3019 |        0.3047 |
| Kategoria | Model 1 — Rule-based (teksu)         |   0.6977 |            0.6965 |         0.3935 |     0.4152 |             — |
| Kategoria | Model 1 — Rule-based (kontekstu)     |   0.6899 |            0.6742 |         0.3864 |     0.4021 |             — |
| Kategoria | Model 2 — Naive Bayes                |   0.4615 |            0.1875 |         0.2343 |     0.1905 |        0.2650 |
| Kategoria | Model 2 — **Logistic Regression** ⭐  |   0.4231 |            0.1898 |         0.2185 |     0.1962 |    **0.3137** |
| Kategoria | Model 2 — Linear SVM                 |   0.3846 |            0.1542 |         0.1958 |     0.1658 |        0.3007 |
| Kategoria | Model 2 — Random Forest              |   0.4615 |            0.1875 |         0.2343 |     0.1905 |        0.2424 |

⭐ = model hili automátikamente husi sistema. **Regra hili:** prefere model ne'ebé
**la'ós dejeneradu**, depois hili tuir **CV F1-macro** (liu estavel duke split ida
deit). Model 2 (Kategoria) treinu/teste **DEIT ho 49 lokal ho ground truth
manual** (125 review, 99 train / 26 test) — haree seksaun 5; ida-ne'e mak razaun
F1 nia kraik liu duke Model 1 rule-based (ne'ebé avalia kontra 129 review husi
mesmu 49 lokal, maibé sem treinu, no F1 aas liu duke Model 2 tanba keyword
lexicon la sofre husi amostra treinu ki'ik).

### Model artifacts

When the full pipeline runs with `write_outputs=True` (the default), it saves
the selected supervised sentiment pipeline to
`Model/sentiment_model.joblib` and, when category training has enough
independent labeled destinations, the selected category pipeline to
`Model/category_model.joblib`. Each artifact contains both the fitted TF-IDF
vectorizer and classifier. The artifacts are trained on the training partition;
the held-out test records remain reserved for evaluation. Clustering objects
and baseline lexicons/rules are not saved as supervised model artifacts.

**Clustering (K-Means, k=4, 1.898 review ho teksu clusterizadu):**

| Metrika                  | Valór   | Interpretasaun          |
| ------------------------ | ------: | ----------------------- |
| Silhouette Score         | 0.0734  | Aas = di'ak (max 1.0)   |
| Davies-Bouldin Index     | 3.6797  | Ki'ik = di'ak           |
| Calinski-Harabasz        | 40.2882 | Boot = di'ak            |
| Hierarchical (Ward) sil. | 0.0960  | Komparasaun algoritmu   |

k=4 uza atu kompara diretamente ho 4 kategoria fixu; sweep k=2..8 hatudu k=5
nia silhouette liu aas (0.0818) — dokumenta hanesan alternativa iha
`reports/tables/mining_cluster_ksweep.csv` (ka `powerbi/cluster_ksweep.csv`).

### ⚠️ Oinsá lee metrika sira-ne'e (importante tebes ba defeza)

1. **⚠️ Modelu dejeneradu — Naive Bayes ba sentiment.** Nia accuracy **0.8470**
   parese di'ak, maibé confusion matrix
   (`reports/figures/09_confusion_matrix_sentiment.png`) hatudu katak nia preve
   **"Positive" ba amostra test hotu-hotu** (393 lolos husi 464, 71 sala — Neutral
   49 no Negative 22 hotu klasifika sala nudar Positive). Ho klase la balansu
   (Positive 1.594 husi 1.842), "sempre hatete Positive" hetan accuracy aas
   **maibé la iha valór prátiku ida** — nia la konsege deteta review negativu
   ida. Sistema deteta ida-ne'e automátikamente (kampu `is_degenerate`) no
   **ESKLUI** modelu dejeneradu husi selesaun "model di'ak liu".

   👉 **Ida-ne'e mak razaun prinsipál tanba sá Manual seksaun 14 ezije
   Precision/Recall/F1 + Confusion Matrix, la'ós accuracy mesak.**

2. **Accuracy aas maibé F1-macro kraik** ba modelu hili (Logistic Regression:
   acc 0.8082, F1-macro 0.4067). Interpretasaun hanesan: modelu di'ak ba klase
   _Positive_ (support 393, F1 0.8954) maibé súsar ba _Neutral_ (support 49, F1
   0.0822) no _Negative_ (support 22, F1 0.2424). **Konkluzaun:** ba dataset la
   balansu, **F1-macro** no **CV F1-macro** mak metrika ne'ebé loloos.

3. **Ground truth sentiment = PROXY husi rating** (≥4 Positive, =3 Neutral,
   ≤2 Negative), la'ós anotasaun manual. Diverjensia normal: turista barak fó
   5 estrela maibé hakerek komentáriu deskritivu neutru ka kritiku badak — haree
   amostra iha `reports/tables/sample_validasaun_manual.csv` ba validasaun.

4. **Silhouette ki'ik (~0.07)** normal ba teksu badak no multi-lingua ho vocabulary
   8.049 liafuan. **Interpretasaun kualitativu** (top terms + profile cluster) mak
   liu importante duke valór absolutu — Manual seksaun 14: _"Silhouette Score,
   cluster interpretation"_.

5. **Validasaun manual:** amostra 42 review disponivel iha
   `reports/tables/sample_validasaun_manual.csv` — prenxe kolona
   `sentiment_manual` no `category_manual` ho liman, depois kompara iha relatóriu
   Chapter 6 (Manual seksaun 14: _"manual validation se presiza"_).

---

## 9. Rezultadu Prinsipál (LO3 + LO4)

### 🎯 Sentiment iha kada kategoria (pedidu esplísitu dosente)

| Kategoria       | Positive | Neutral | Negative | Unknown | **% Positivu** |
| --------------- | -------: | ------: | -------: | ------: | -------------: |
| **Historical**     |      350 |     145 |       34 |      34 |      **62.2%** |
| **Religious**       |      167 |      54 |        8 |       2 |      **72.3%** |
| **Traditional** |      122 |      43 |       10 |      14 |      **64.6%** |
| **Modern**     |    1.618 |     497 |      140 |     156 |      **67.1%** |

(Kolona `Kategoria` = `category` final: anotasaun manual bainhira iha, senaun
fallback rule-based — haree seksaun 5. Total baris kada kategoria inklui
review husi hotu 569 lokal, la'ós so 49 ho ground truth.)

**Insight:** kategoria **Religious** iha proporsaun sentiment positivu aas liu
(72.3%), enkuantu **Historical** kraik liu (62.2%) no iha komentáriu negativu
barak liu (34 review, dominadu husi Marobo Hot Spring 7 no Pousada de Maubara
4) — tema dominante kontinua **kondisaun infraestrutura no manutensaun**
(_"it feels strange and run-down... parking is difficult, the informational
posters are faded"_; _"Tempat bersejarah yang dijaga keasliannya. Sayang kurang
dirawat dengan baik"_ [lokal istóriku ne'ebé mantein orijinalidade, sayang la
kuidadu di'ak]). Ida-ne'e insight aksionavel ba jestaun sitiu heransa.

⚠️ **Nota:** persentajen leten kobre **569 lokal** (via fallback rule-based ba
520 lokal seidauk iha ground truth) — interpreta ho kuidadu, la'ós hanesan
metrika supervisadu validadu. % Positivu baseia deit ba ground truth manual
(49 lokal) sei difere; haree `reports/tables/mining_matriz_kategoria_sentimentu.csv`
ba breakdown kompletu.

### 🧩 Cluster K-Means (k=4, 1.898 review ho teksu clusterizadu)

| Cluster | N (%)         | Rating médiu | Lingua dominante | Termu prinsipál                             | Interpretasaun                                    |
| ------- | ------------- | -----------: | ---------------- | -------------------------------------------- | -------------------------------------------------- |
| 0       | 1.476 (77.8%) |         4.42 | en                | beach, beautiful, place, dili, great, food   | Komentáriu jerál kona-ba praia/lokal, naruk mediu  |
| 1       | 216 (11.4%)   |         4.50 | en                | nice place, good, nice beach, visit          | Komentáriu badak, elogiu jenériku                  |
| 2       | 149 (7.9%)    |         4.75 | en                | timor, leste, history, museum, tour          | Komentáriu kona-ba istória/muzeu/turismu kultural  |
| 3       | 57 (3.0%)     |         4.16 | id                | furak, fatin, fatin furak, suai, spring      | **Komentáriu lingua Tetun/Indonézia** (badak, pozitivu) |

**Konkluzaun importante:** hotu-hotu cluster nia `dominant_category` = **Modern**
— ida-ne'e reflete katak Modern mak klase maioria (71% husi baris, konsekuénsia
husi fallback rule-based ba lokal natureza/lazer), **la'ós** prova katak
clustering funsiona diak ka aat. Cluster sira kontinua separa liu husi **naruk
komentáriu, lingua no tema jerál** (praia/lokal vs istória/muzeu vs elogiu
badak), la'ós husi 4 kategoria manual — hanesan padraun ne'ebé mosu ona iha
dataset orijinál. Ida-ne'e **rezultadu lejítimu** no insight ba diskusaun:
turista ne'ebé hakerek badak iha Tetun/Indonézia ("fatin furak") uza estilu
diferente husi turista internasionál (naruk, deskritivu, foka ba istória/kultura
ka fasilidade).

---

## 10. Dashboard (LO6 · Manual seksaun 15)

### A. Streamlit — `py -3.11 -m streamlit run dashboard\app.py`

| Komponente Manual seksaun 15 | Implementasaun                                                  |
| ---------------------------- | ---------------------------------------------------------------- |
| KPI cards                    | 6 card: Lokal, Review, Rating médiu, % Positivu, % Negativu, Silhouette |
| Trend chart                  | Tab _Explorasaun_ → review kada fulan tuir sentiment             |
| Comparison chart             | Tab _Explorasaun_ → **Kategoria × Sentiment** (absolutu + %)     |
| Map                          | Tab _Explorasaun_ → mapa lat/lng, kor tuir kategoria, size tuir volume |
| Filter date/location/cat     | Sidebar: cidade, kategoria, sentiment, cluster, períodu, ho-teksu-deit |
| Prediction/Mining result     | Tab _Clustering_ (scatter, elbow) + Tab _Model Evaluation_       |
| Last updated timestamp       | Badge iha hero: "Kolesaun dadus ikus: DD-MM-YYYY HH:MM UTC"      |

**4 tab:**

1. **📊 Explorasaun (EDA)** — kategoria × sentiment, distribuisaun, trend, mapa, ranking, lingua
2. **🧩 Clustering K-Means** — **scatter plot 2D interativu** (cluster + kategoria),
   elbow chart, silhouette tuir k, Davies-Bouldin, tabela profile cluster, crosstab
3. **🎯 Model Evaluation** — tabela metrika, **confusion matrix heatmap**, bar
   chart komparasaun model, metrika kada klase
4. **📋 Dadus & Transparénsia** — tabela dadus, download CSV, fonte dadus,
   preprocessing meta, diksionáriu dadus, tabela anotasaun manual

### B. Power BI — `powerbi/*.csv` + `powerbi/GUIA_POWERBI.md`

17 ficheiru CSV ho **modelu estrela** (fact + dimensaun), grava ho encoding
**utf-8-sig** (Power BI presiza BOM ba asentu). Guia kompletu ho:

- Pasu import (Get Data → Folder)
- Tabela relasaun (6 relasaun)
- **Measures DAX** prontu-kopia (9 measure + 2 metadata)
- Layout 5 pájina detalladu (Overview, Trend, **Clustering**, **Model Evaluation**, Dadus)
- Kódigu kor konsistente ho dashboard Streamlit

📄 **Haree:** [`powerbi/GUIA_POWERBI.md`](powerbi/GUIA_POWERBI.md)

---

## 11. Estrutura Folder (Manual seksaun 16)

```
turismo_project/
├── data/
│   ├── raw/
│   │   └── dataset_turris.json              # RAW layer — dadus orijinal
│   ├── processed/
│   │   └── reviews_processed.csv            # PROCESSED layer — limpa + transforma
│   ├── features/
│   │   └── review_features.csv              # FEATURE layer — variavel ba model
│   ├── output/
│   │   └── reviews_mined.csv                # OUTPUT layer — prediction/cluster/sentiment
│   └── place_category_labels.csv            # ⭐ anotasaun manual (ground truth) — EDITAVEL
│
├── src/
│   ├── config.py                            # path, konstante, metadata fonte dadus
│   ├── annotate_places.py                   # gera anotasaun manual kategoria
│   ├── data_collection.py                   # LO2 — kolesaun + flatten + log
│   ├── preprocessing.py                     # Manual sek. 11 — 7 etapa kompletu
│   ├── category_classifier.py               # LO4 — Model 1 kategoria (rule-based)
│   ├── sentiment_analysis.py                # LO4 — Model 1 sentiment (lexicon)
│   ├── supervised_models.py                 # LO4 — Model 2 (NB/LR/SVM/RF)
│   ├── clustering.py                        # LO4 — K-Means + elbow + SVD + Hierarchical
│   ├── eda.py                               # LO3 — Manual sek. 12 kompletu
│   ├── model_evaluation.py                  # LO5 — Manual sek. 14 kompletu
│   ├── export_powerbi.py                    # LO6 — export star schema CSV
│   └── pipeline.py                          # ⭐ orkestrador (halao ne'e!)
│
├── notebooks/
│   └── 01_eda_turismo.ipynb                 # notebook EDA (25 sel)
│
├── dashboard/
│   └── app.py                               # dashboard Streamlit (4 tab)
│
├── powerbi/
│   ├── GUIA_POWERBI.md                      # ⭐ guia pasu-ba-pasu + DAX
│   └── *.csv                                # 17 ficheiru (fact + dim + agg + eval)
│
├── reports/
│   ├── figures/*.png                        # 13 figura ba relatóriu/PPT
│   ├── tables/*.csv                         # tabela tidy + amostra validasaun manual
│   ├── eda_summary.json                     # rezumu EDA + preprocessing
│   ├── model_evaluation.json                # metrika kompletu
│   ├── data_collection_log.json             # metadata kolesaun (Manual sek. 10)
│   ├── DATA_DICTIONARY.md                   # diksionáriu dadus
│   └── TEMPLATE_SUBMISSION.md               # template Manual sek. 23 + outline relatóriu
│
├── README.md
└── requirements.txt
```

---

## 12. Figura ne'ebé Sistema Produz (ba Relatóriu no PowerPoint)

| Ficheiru                                | Kontéudu                                                       | Uza iha PPT slide |
| --------------------------------------- | -------------------------------------------------------------- | ----------------- |
| `01_distribuisaun_rating.png`           | Histogram rating + boxplot rating tuir kategoria                | 10 (EDA Findings) |
| `02_distribuisaun_naruk_teksu.png`      | Histogram + density naruk teksu review                          | 10                |
| `03_kategoria_vs_sentiment.png`         | ⭐ **Sentiment kada kategoria** (absolutu + %)                   | 10                |
| `04_trend_tempu.png`                    | Volume review + trend sentiment kada fulan                      | 10                |
| `05_korelasaun_scatter.png`             | Matriz korelasaun + scatter naruk vs rating                     | 10                |
| `06_ranking_kategoria_lokal.png`        | Frekuensia kategoria + top 10 lokal + top 10 cidade             | 10                |
| `07_mapa_jeografiku.png`                | Distribuisaun jeografika lokal turizmu                          | 10                |
| `08_lingua_review.png`                  | Lingua orijinal review                                          | 10                |
| `09_confusion_matrix_sentiment.png`     | ⭐ Confusion matrix Model 1 vs Model 2 (sentiment)               | 12 (Evaluation)   |
| `10_confusion_matrix_kategoria.png`     | ⭐ Confusion matrix Model 1 vs Model 2 (kategoria)               | 12                |
| `11_elbow_silhouette.png`               | ⭐ Elbow + Silhouette + Davies-Bouldin tuir k                    | 11 (Mining Model) |
| `12_cluster_scatter.png`                | ⭐ **Scatter plot cluster K-Means** (3 painel: tomak · zoom · kategoria) | 11        |
| `13_komparasaun_model.png`              | Bar chart komparasaun performansia model                        | 12                |

---

## 13. Diksionáriu Dadus (Manual seksaun 10)

Diksionáriu kompletu (kolona, tipu, deskrisaun, unidade, missing values):

- 📄 Markdown: [`reports/DATA_DICTIONARY.md`](reports/DATA_DICTIONARY.md)
- 📊 CSV (ba Power BI): `powerbi/data_dictionary.csv`
- 🖥️ Iha dashboard: tab **📋 Dadus & Transparénsia**

---

## 14. Roadmap — Migrasaun ba API (bainhira grupu hetan API key)

```python
# src/data_collection.py — troka DEIT funsaun ida-ne'e:

def fetch_from_api(poll_interval_seconds: int = 300):
    """Near-real-time: polling kada 5 minutu (Manual seksaun 5 nota)."""
    while True:
        resp = requests.get(PLACES_API_URL, params={...}, timeout=30)
        yield resp.json()          # estrutura HANESAN ho load_raw_data()
        time.sleep(poll_interval_seconds)
```

Depois ne'e:

1. Muda `collect()` atu bolu `fetch_from_api()` fali `load_raw_data()`
2. Atualiza `DATA_SOURCE["collection_mode"]` iha `config.py` ba `"near-real-time (polling 5 min)"`
3. **Módulu seluk hotu la presiza muda** — preprocessing, EDA, mining, evaluation,
   dashboard hotu servisu iha nivel DataFrame

---

## 15. Limitasaun no Trabalhu Futuru

### Limitasaun

1. **Kobertura ground truth limitadu** — dataset agora iha 569 lokal (3.394
   review, 1.964 ho teksu), maibé anotasaun manual (ground truth 4-kategoria)
   deit kobre **49 lokal orijinál** (125 review ho teksu, treinu/teste Model 2
   kategoria). Ba **520 lokal foun**, kolona `category` final uza fallback
   rule-based (`category_source = rule_based_fallback`, haree seksaun 5) —
   sira **la** hetan validasaun manual, tanba ne'e EDA/dashboard ne'ebé mostra
   569 lokal tenke interpreta ho kuidadu ba parte "kategoria".
2. **La'ós real-time** — snapshot batch tanba seidauk iha API key.
3. **Ground truth sentiment = proxy rating**, la'ós anotasaun manual kompletu.
4. **Lexicon rule-based** la kaer sarkasmu, ironia, ka polisemia entre lingua hat.
5. **Cluster silhouette kraik (~0.07)** — natural ba teksu badak multi-lingua,
   maibé signifika separasaun cluster la forte; `dominant_category` kada
   cluster hotu-hotu "Modern" reflete dezekilibriu klase (fallback rule-based),
   la'ós padraun cluster nia rasik.
6. **Anotasaun manual `medium` confidence** (22 husi 49 lokal ho ground truth
   iha dataset atual) bele diskuti — ex: Marobo Hot Spring (Historical vs Modern
   vs Traditional).
7. **Ekstensaun ground truth seidauk halao** — `annotate_places.py` nia
   diksionáriu `EXPERT_LABELS` seidauk atualiza ba 520 lokal foun (presiza
   revizaun manual husi grupu/dosente atu evita fallback "Modern"/`low`
   sem evidensia — haree seksaun 5).

### Trabalhu futuru

- ✅ ~~Aumenta dataset (liu 1.000 review)~~ — **kompletu**: dataset agora iha
  3.394 review (569 lokal); tarefa tuir mai mak **anotasaun manual kategoria**
  ba 520 lokal foun (haree Limitasaun no. 7)
- Anotasaun manual sentiment ba amostra 200+ review → ground truth loloos
- Uza modelu multilingual embedding (ex: sentence-transformers) fali TF-IDF
- Hatama analiza aspect-based sentiment (asesu, fasilidade, prezu, servisu)
- Halo API oficial (Google Places billing) atu troka snapshot manual ba
  near-real-time polling — haree seksaun 14

---

## 16. Ligasaun Submisaun

| Item | Ligasaun |
| --- | --- |
| GitHub Repository | [Source code](https://github.com/sd8778515-cmykr/timor-leste-tourism-analytics) |
| Dashboard (deploy) | [Open live dashboard](https://timor-leste-tourism-analytics-r4vsccrruqoajcvpcql2d7.streamlit.app/) |

---

## 17. Referénsia

1. Manual Capstone Project — Unidade Kurikulér _Mining and Exploring_, S1 Siencia
   Komputadór, Dili Institute of Technology, 2026.
2. Pedregosa, F. et al. (2011). _Scikit-learn: Machine Learning in Python_.
   Journal of Machine Learning Research, 12, 2825–2830.
3. Rousseeuw, P. J. (1987). _Silhouettes: a graphical aid to the interpretation
   and validation of cluster analysis_. Journal of Computational and Applied
   Mathematics, 20, 53–65.
4. Davies, D. L. & Bouldin, D. W. (1979). _A Cluster Separation Measure_.
   IEEE Transactions on Pattern Analysis and Machine Intelligence, 1(2), 224–227.
5. Manning, C. D., Raghavan, P., & Schütze, H. (2008). _Introduction to
   Information Retrieval_. Cambridge University Press. (TF-IDF, text clustering)

---

_Sistema ida-ne'e dezenvolve tuir Manual Capstone Project — Mining and Exploring,
DIT 2026. Python 3.11._
