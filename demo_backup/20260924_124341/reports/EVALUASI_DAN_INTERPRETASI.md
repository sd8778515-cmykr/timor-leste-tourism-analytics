# Evaluasi dan interpretasi sistem

Evaluation uses the full dataset and remains fixed when dashboard filters change.

## Sentiment

Logistic Regression was selected using cross-validation on training data only. On 461 test records, accuracy=0.8156 and macro F1=0.3788. This macro F1 is higher than the majority-class baseline (0.3093; difference +0.0695). This comparison is descriptive and does not establish statistical significance.

- Weakest class represented in the test set: Negative, F1=0.0741, support=23.
- Macro F1 includes every configured class. A zero score for a class with no test support does not demonstrate performance on that class.
- Reference labels come from star ratings rather than human text annotations. Evaluation therefore measures agreement with a rating proxy. The same destination may occur in training and test data.

Kesalahan terbanyak pada model terpilih:

- Neutral → Positive: 32 review.
- Positive → Neutral: 28 review.
- Negative → Positive: 22 review.
- Positive → Negative: 2 review.
- Neutral → Negative: 1 review.

## Category

Naive Bayes was selected using cross-validation on training data only. On 22 test records, accuracy=0.6818 and macro F1=0.2027. This macro F1 is lower than the majority-class baseline (0.2179; difference -0.0152). This comparison is descriptive and does not establish statistical significance.

- Weakest class represented in the test set: Historical, F1=0.0000, support=3.
- Macro F1 includes every configured class. A zero score for a class with no test support does not demonstrate performance on that class.
- Training and test sets are separated by destination. Reference labels are limited, so rare-class results are sensitive to the split.
- Not represented in the independent test set: Religious.

Kesalahan terbanyak pada model terpilih:

- Historical → Modern: 3 review.
- Modern → Historical: 2 review.
- Traditional → Modern: 2 review.

## Clustering

K-Means with k=4 produced silhouette=0.0758, indicating weak separation with substantial overlap. The highest silhouette in the tested range occurred at k=7. The configured k is an analytical choice rather than an automatically optimal value. The 2D SVD projection retains 4.45% of variance; the scatter plot is illustrative. Text clusters are not equivalent to tourism categories.

## Validasi manusia

- sentiment: pending; 0 label valid dari 30 sampel; 0 label tidak dikenali, 0 review tidak cocok dengan dataset.
- category: pending; 0 label valid dari 30 sampel; 0 label tidak dikenali, 0 review tidak cocok dengan dataset.

Isi `reports/tables/sample_validasaun_manual.csv` berdasarkan penilaian manusia. Label yang kosong tidak diisi otomatis. Jalankan pipeline ulang untuk menghitung hasil validasi.
