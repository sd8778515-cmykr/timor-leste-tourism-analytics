# Evaluasi dan interpretasi sistem

Evaluation uses the full dataset and remains fixed when dashboard filters change.

## Sentiment

Logistic Regression was selected using cross-validation on training data only. On 748 test records, accuracy=0.8516 and macro F1=0.4763. This macro F1 is higher than the majority-class baseline (0.3097; difference +0.1666). This comparison is descriptive and does not establish statistical significance.

- Weakest class represented in the test set: Neutral, F1=0.2075, support=61.
- Macro F1 includes every configured class. A zero score for a class with no test support does not demonstrate performance on that class.
- Reference labels come from star ratings rather than human text annotations. Evaluation therefore measures agreement with a rating proxy. The same destination may occur in training and test data.

Kesalahan terbanyak pada model terpilih:

- Neutral → Positive: 46 review.
- Positive → Neutral: 27 review.
- Negative → Positive: 23 review.
- Negative → Neutral: 7 review.
- Neutral → Negative: 4 review.

## Category

Linear SVM was selected using cross-validation on training data only. On 57 test records, accuracy=0.2982 and macro F1=0.1181. This macro F1 is higher than the majority-class baseline (0.1149; difference +0.0032). This comparison is descriptive and does not establish statistical significance.

- Weakest class represented in the test set: Historical, F1=0.0000, support=38.
- Macro F1 includes every configured class. A zero score for a class with no test support does not demonstrate performance on that class.
- Training and test sets are separated by destination. Reference labels are limited, so rare-class results are sensitive to the split.
- Not represented in the independent test set: Religious.

Kesalahan terbanyak pada model terpilih:

- Historical → Modern: 36 review.
- Historical → Religious: 2 review.
- Traditional → Modern: 2 review.

## Clustering

K-Means with k=4 produced silhouette=0.0373, indicating weak separation with substantial overlap. The highest silhouette in the tested range occurred at k=8. The configured k is an analytical choice rather than an automatically optimal value. The 2D SVD projection retains 4.07% of variance; the scatter plot is illustrative. Text clusters are not equivalent to tourism categories.

## Validasi manusia

- sentiment: pending; 0 label valid dari 30 sampel; 0 label tidak dikenali, 0 review tidak cocok dengan dataset.
- category: pending; 0 label valid dari 30 sampel; 0 label tidak dikenali, 0 review tidak cocok dengan dataset.

Isi `reports/tables/sample_validasaun_manual.csv` berdasarkan penilaian manusia. Label yang kosong tidak diisi otomatis. Jalankan pipeline ulang untuk menghitung hasil validasi.
