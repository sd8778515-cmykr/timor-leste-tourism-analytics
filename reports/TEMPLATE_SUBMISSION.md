# Template Submisaun Final + Outline Relatóriu

> Baze: **Manual Capstone Project — Mining and Exploring**, seksaun 8, 17, 18, 20, 22 no 23.
> Prenxe kampu `______` husi grupu.

---

## A. Template Final Submission (Manual seksaun 23)

| Field                   | Preenche husi Estudante                                                    |
| ----------------------- | -------------------------------------------------------------------------- |
| **Project Title**       | Data Mining Sentiment no Kategoria Destinasaun Turizmu Timor-Leste          |
| **Case Study Location** | Timor-Leste (Liquiçá, Maubara, Ainaro, Bobonaro, Covalima, Manufahi, Ermera)|
| **Student/Group**       | ______                                                                     |
| **Student ID**          | ______                                                                     |
| **Data Source/API**     | Google Maps Places & Reviews (Apify `compass/crawler-google-places`) — snapshot batch, seidauk iha API key |
| **Mining Method**       | Rule-based keyword lexicon · Lexicon sentiment · K-Means + Hierarchical clustering · Naive Bayes / Logistic Regression / Linear SVM / Random Forest |
| **Dashboard Tool**      | Streamlit (interativu) + Power BI Desktop                                   |
| **GitHub Link**         | ______                                                                     |
| **YouTube Link**        | ______                                                                     |
| **Submission Date**     | ______                                                                     |
| **Lecturer/Assessor**   | ______                                                                     |

---

## B. Lista Submisaun Final (Manual seksaun 20)

| No. | Item                          | Status | Onde iha projetu ne'e                                |
| --- | ----------------------------- | ------ | ---------------------------------------------------- |
| 1   | Project Proposal              | ☐      | _(dokumentu separadu — uza outline seksaun C)_       |
| 2   | Final Report (.docx / PDF)    | ☐      | _(uza outline seksaun D)_                            |
| 3   | Source Code / GitHub Link     | ☐      | `src/`, `dashboard/`, `notebooks/`                   |
| 4   | Dataset / data access instr.  | ✅     | `data/raw/dataset_turris.json` + README seksaun 2    |
| 5   | Data Dictionary               | ✅     | `reports/DATA_DICTIONARY.md`                         |
| 6   | System Architecture           | ✅     | README seksaun 3 (blok diagram)                      |
| 7   | EDA Results                   | ✅     | `reports/figures/01–08*.png`, `reports/eda_summary.json`, `notebooks/01_eda_turismo.ipynb` |
| 8   | Model Evaluation              | ✅     | `reports/model_evaluation.json`, `reports/figures/09–13*.png`, `reports/tables/model_metrics.csv` |
| 9   | Dashboard Link/Screenshots    | ☐      | `dashboard/app.py` + `powerbi/GUIA_POWERBI.md`       |
| 10  | PowerPoint                    | ☐      | _(uza outline seksaun E)_                            |
| 11  | YouTube Video Link            | ☐      | ______                                               |
| 12  | Final Submission Form         | ☐      | Tabela A iha leten                                   |

---

## C. Outline Project Proposal (Manual seksaun 8)

| No. | Seksaun                 | Kontéudu ba projetu ne'e                                                                 |
| --- | ----------------------- | ----------------------------------------------------------------------------------------- |
| 1   | Cover                   | Títulu, naran grupu, ID, Unidade Kurikulér, semestre                                      |
| 2   | Antesedente             | Turizmu Timor-Leste sei kresimentu; review online la organizadu; la iha analiza sistemátiku |
| 3   | Problema                | Tipu destinasaun ida-ne'ebé simu apresiasaun di'ak liu? Ne'ebé presiza atensaun?           |
| 4   | Objetivu                | README seksaun 1 (6 objetivu)                                                              |
| 5   | Limitasaun              | Ground truth kategoria kobre 49/569 lokal deit, snapshot batch (la'ós real-time), ground truth sentiment = proxy, **la halo rekomendasaun** |
| 6   | Fontes Dadus            | README seksaun 2                                                                           |
| 7   | Metodologia             | README seksaun 3 + 6 + 7                                                                   |
| 8   | Blok Diagram Sistema    | README seksaun 3 (kopia diagram ASCII ba diagram vizuál)                                   |
| 9   | Rezultadu Espetativa    | Dashboard, klasifikasaun 4 kategoria, sentiment, cluster, metrika evaluation               |
| 10  | Time-Line               | Plano 16 semana (Manual seksaun 9)                                                         |
| 11  | Tools                   | Python 3.11, pandas, scikit-learn, matplotlib, plotly, Streamlit, Power BI, GitHub         |
| 12  | Referénsia              | README seksaun 17                                                                          |

---

## D. Outline Final Report (Manual seksaun 17)

| No. | Seksaun                                | Fonte materiál iha projetu                                                   |
| --- | -------------------------------------- | ----------------------------------------------------------------------------- |
| 1   | Cover Page                             | —                                                                             |
| 2   | Approval/Declaration                   | —                                                                             |
| 3   | Abstract (Tetun/English)               | Rezumu husi README seksaun 1, 8, 9                                            |
| 4   | Acknowledgement                        | —                                                                             |
| 5   | Table of Contents                      | —                                                                             |
| 6   | **Chapter 1: Introduction**            | README seksaun 1 (problema, objetivu, limitasaun)                             |
| 7   | **Chapter 2: Literature Review**       | README seksaun 17 (referénsia: TF-IDF, K-Means, Silhouette, lexicon sentiment) |
| 8   | **Chapter 3: Methodology**             | README seksaun 3 (arkitetura) + seksaun 7 (justifikasaun algoritmu)           |
| 9   | **Chapter 4: Data Collection & Preprocessing** | README seksaun 2 + 5 + 6; `reports/data_collection_log.json`; `reports/DATA_DICTIONARY.md` |
| 10  | **Chapter 5: EDA & Data Mining Results**| README seksaun 9; figura `01–08*.png` no `11–12*.png`; notebook EDA           |
| 11  | **Chapter 6: Evaluation & Discussion** | README seksaun 8 (inklui "Oinsá lee metrika"); figura `09`, `10`, `13`; `reports/tables/sample_validasaun_manual.csv` |
| 12  | **Chapter 7: Conclusion & Recommendation** | README seksaun 15 (limitasaun + trabalhu futuru)                          |
| 13  | References                             | README seksaun 17                                                             |
| 14  | Appendices                             | Kódigu (`src/`), screenshot dashboard, diksionáriu dadus, ligasaun YouTube    |

### 💡 Sujestaun kontéudu Chapter 6 (Evaluation & Discussion)

Iha kapítulu ne'e, esplika **tanba sá accuracy aas maibé F1-macro kraik**:

1. Hatudu tabela metrika (README seksaun 8)
2. Hatudu confusion matrix (`reports/figures/09` no `10`)
3. Esplika klase la balansu: Positive 1.594 vs Neutral 155 vs Negative 93
4. Esplika ground truth = proxy rating, ho ezemplu konkretu review 5★ ho teksu neutru
5. Kompara Model 1 (rule-based/lexicon) vs Model 2 (ML) — saida mak manan, tanba sá
6. Diskute silhouette kraik no interpretasaun kualitativu cluster
7. Prenxe no diskute `reports/tables/sample_validasaun_manual.csv` (validasaun manual)

---

## E. Outline PowerPoint (Manual seksaun 18)

| Slide | Kontéudu                | Materiál husi projetu                                            |
| ----- | ----------------------- | ---------------------------------------------------------------- |
| 1     | Title & Team            | —                                                                |
| 2     | Background              | README seksaun 1                                                 |
| 3     | Problem Statement       | README seksaun 1                                                 |
| 4     | Objectives              | README seksaun 1 (6 objetivu)                                    |
| 5     | Data Source             | README seksaun 2 (**inklui slide "seidauk iha API key"**)        |
| 6     | Methodology             | README seksaun 7 (tabela justifikasaun algoritmu)                |
| 7     | System Architecture     | README seksaun 3 (blok diagram)                                  |
| 8     | Dataset                 | 569 lokal, 3.394 review (49 lokal ho ground truth manual); tabela anotasaun manual (README seksaun 5) |
| 9     | Preprocessing           | README seksaun 6 (tabela 7 etapa)                                |
| 10    | EDA Findings            | Figura `03_kategoria_vs_sentiment.png` ⭐ + `04`, `06`, `07`      |
| 11    | Data Mining Model       | Figura `11_elbow_silhouette.png` + `12_cluster_scatter.png` ⭐    |
| 12    | Evaluation              | Figura `09`, `10` (confusion matrix) + `13` (komparasaun model)  |
| 13    | Dashboard               | Screenshot Streamlit (4 tab) + Power BI                          |
| 14    | Live Demo / Screenshots | Demo direto                                                      |
| 15    | Conclusion & Recommendation | README seksaun 15                                            |
| 16    | YouTube + GitHub Links  | —                                                                |

---

## F. Struktura Vídeo YouTube (Manual seksaun 20)

| Parte | Kontéudu                              | Tempu    | Saida mak hatudu                                          |
| ----- | ------------------------------------- | -------- | --------------------------------------------------------- |
| 1     | Introduction: títulu, grupu, problema | 1–2 min  | Slide 1–3                                                 |
| 2     | Objectives & Case Study               | 1–2 min  | Slide 4 + mapa Timor-Leste                                |
| 3     | Data Source & Architecture            | 2 min    | Slide 5–7; hatudu `data/raw/` no blok diagram             |
| 4     | Collection, Preprocessing & EDA       | 2–4 min  | Halao `py -3.11 src\pipeline.py` iha terminal (pasu 1–2, 6) |
| 5     | Data Mining & Evaluation              | 2–3 min  | Terminal pasu 3–5, 7; figura `11`, `12`, `09`, `10`       |
| 6     | Live Dashboard/System Demo            | 3–5 min  | Streamlit 4 tab + Power BI                                |
| 7     | Conclusion                            | 1–2 min  | Slide 15                                                  |

**Formatu títulu:**
`MINING AND EXPLORING CAPSTONE – Data Mining Sentiment no Kategoria Turizmu Timor-Leste – [Grupu] – S1 Computer Science – DIT`

**Rekezitu:** 10–20 minutu · MP4 1080p · Public ka Unlisted · screen recording
obrigatóriu · live demo obrigatóriu · dashboard demo obrigatóriu.

---

## G. Checklist Final (Manual seksaun 22)

| Item                          | Status | Nota                                                       |
| ----------------------------- | ------ | ---------------------------------------------------------- |
| Tema no problema klaru        | ✅     | README seksaun 1                                           |
| Data source dokumentadu       | ✅     | README seksaun 2 + `reports/data_collection_log.json`      |
| Data collection reproducible  | ✅     | `py -3.11 src\pipeline.py` (deterministiku, `random_state=42`) |
| Raw no processed data organize| ✅     | 4 layer: raw / processed / features / output               |
| Data preprocessing dokumentadu| ✅     | README seksaun 6 + `reports/eda_summary.json`              |
| EDA kompletu                  | ✅     | 7 pontu Manual seksaun 12 + 8 figura + notebook            |
| Model/algorithm implemented   | ✅     | Model 1 (2) + Model 2 (4) + Clustering (2)                 |
| Evaluation metrics included   | ✅     | Accuracy/Precision/Recall/F1/Confusion/Kappa/Silhouette/DB/CH |
| Insight no discussion included| ✅     | README seksaun 8 + 9                                       |
| Dashboard functional          | ✅     | Streamlit (testadu) + 17 CSV Power BI + guia               |
| Source code organized         | ✅     | 12 módulu iha `src/`, kada ida iha docstring ho referénsia Manual |
| README complete               | ✅     | 17 seksaun                                                 |
| Final report complete         | ☐      | Uza outline seksaun D                                      |
| PowerPoint ready              | ☐      | Uza outline seksaun E                                      |
| YouTube video uploaded        | ☐      | Uza struktura seksaun F                                    |
| YouTube link tested           | ☐      | ______                                                     |
| GitHub link tested            | ☐      | ______                                                     |

---

## H. Rubrika Avaliasaun (Manual seksaun 21) — auto-avaliasaun

| No. | Kritériu                                   | Peso | Evidensia iha projetu                                              |
| --- | ------------------------------------------ | ---- | ------------------------------------------------------------------ |
| 1   | Problem Statement & Objectives             | 10%  | README seksaun 1                                                   |
| 2   | Data Source & Data Collection              | 10%  | README seksaun 2, `src/data_collection.py`, log JSON               |
| 3   | Data Preprocessing & Data Management       | 10%  | README seksaun 6, `src/preprocessing.py` (7 etapa), 4 dataset layer|
| 4   | Exploratory Data Analysis (EDA)            | 15%  | `src/eda.py` (7 pontu Manual), 8 figura, notebook                  |
| 5   | Data Mining Implementation                 | 20%  | Model 1 + Model 2 + K-Means + Hierarchical, ho justifikasaun       |
| 6   | Model Evaluation & Discussion              | 10%  | `src/model_evaluation.py`, 5 figura, README seksaun 8              |
| 7   | Dashboard & Visualization                  | 10%  | Streamlit 4 tab + Power BI 17 CSV + guia DAX                       |
| 8   | Source Code, Documentation & Reproducibility | 5% | README 17 seksaun, docstring, `requirements.txt`, `random_state=42`|
| 9   | YouTube Demonstration                      | 5%   | ☐ _(halao husi grupu)_                                             |
| 10  | Final Presentation                         | 5%   | ☐ _(halao husi grupu)_                                             |
