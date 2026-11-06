# Panduan Demo Sistem Tahap per Tahap

Jalankan semua perintah dari terminal PowerShell pada folder proyek:

```powershell
cd "D:\turismo_project (1)"
```

## 1. Aktifkan lingkungan Python

```powershell
.\.venv\Scripts\Activate.ps1
```

Pastikan interpreter benar:

```powershell
python --version
```

Versi yang diharapkan adalah Python 3.11.

## 2. Tunjukkan dataset mentah

```powershell
python src\check_dataset.py
```

Penjelasan: dataset merupakan snapshot dari Google Maps Places. Dataset mentah
tidak diubah oleh pipeline dan berisi 569 destinasi.

## 3. Jalankan pipeline tahap per tahap

```powershell
python src\pipeline.py --step-by-step
```

Pipeline berhenti setelah setiap tahap selesai. Jangan langsung menekan
`Enter`. Baca hasil di terminal, jelaskan kepada dosen, lalu tekan `Enter` untuk
melanjutkan. Mode biasa tanpa jeda menggunakan `python src\pipeline.py`.

Terminal akan menampilkan tahapan secara berurutan. Berhenti sejenak pada setiap
judul `PASU` untuk menjelaskan:

### Jeda 1 — Data source dan collection (Manual: database/data storage)

Kode yang dipanggil: `collect()` dalam `src/data_collection.py`.

Proses: membaca `data/raw/dataset_turris.json`, mengambil informasi destinasi
dan review, mengubah review bertingkat menjadi satu baris per review, menjaga
destinasi tanpa review, lalu menggabungkan label kategori manual berdasarkan
`place_id`.

Hasil yang diharapkan: 569 destinasi dan 3.396 baris sebelum cleaning.

Kalimat presentasi: "Kami menggunakan snapshot Google Maps karena tidak
memiliki API key. Data asli tidak diubah. Sistem meratakan destinasi dan review
menjadi tabel tingkat review serta tetap menjaga destinasi tanpa review."

### Jeda 2 — Preprocessing (Manual: data cleaning dan preprocessing)

Kode yang dipanggil: `preprocess()` dalam `src/preprocessing.py`.

Proses: menghapus duplikat, memeriksa nilai kosong, mengisi kota dari alamat
jika memungkinkan, mengubah tanggal dan rating, membersihkan teks, menghapus
stopword, membuat token, menandai outlier panjang, membuat fitur, dan membagi
data train/test.

Hasil yang diharapkan: 3.396 menjadi 3.394 baris karena 2 duplikat dihapus;
1.964 review memiliki teks; 176 review panjang ditandai sebagai outlier dan
tidak dihapus; 1.423 baris train, 475 test, dan 1.496 tidak dipakai pada split
awal preprocessing.

File yang dibuat: `data/processed/reviews_processed.csv`.

Kalimat presentasi: "Outlier tidak kami hapus karena review panjang masih
memiliki informasi. Data tanpa teks juga tetap disimpan untuk statistik rating,
informasi destinasi, dan peta."

### Jeda 3 — Model 1 (Manual: implementasi model 1)

Kode yang dipanggil: `apply_category_classification()` dan
`apply_sentiment_analysis()`.

Proses: kategori ditentukan dari anotasi manual atau aturan kata kunci pada
tingkat destinasi. Sentiment teks dihitung dengan kamus positif dan negatif;
review tanpa teks tetapi memiliki bintang menggunakan rating sebagai fallback.

Hasil yang diharapkan: empat kategori Historical, Religious, Traditional, dan
Modern; sentiment Positive, Neutral, Negative, dan Unknown. Unknown adalah 206
destinasi tanpa bukti review.

### Jeda 4 — Clustering (Manual: pendekatan alternatif/unsupervised)

Kode yang dipanggil: `run_clustering()` dalam `src/clustering.py`.

Proses: 1.898 teks yang masih memiliki token diubah menjadi TF-IDF, diuji untuk
`k=2` sampai `k=8`, dikelompokkan dengan K-Means, dibandingkan dengan
Hierarchical Clustering, lalu dicari kata dominan tiap cluster.

Hasil yang diharapkan: sistem menggunakan 4 cluster dan silhouette sekitar
0,0758. Nilai yang dekat nol berarti cluster masih bertumpang tindih, sehingga
hasil dipakai sebagai eksplorasi pola, bukan pemisahan yang kuat.

### Jeda 5 — Model 2 (Manual: implementasi dan perbandingan model 2)

Kode yang dipanggil: `run_supervised_models()` dalam
`src/supervised_models.py`.

Proses: membandingkan Naive Bayes, Logistic Regression, Linear SVM, dan Random
Forest. TF-IDF hanya dipelajari dari train. Pemilihan model menggunakan
cross-validation F1-macro pada train; test tidak digunakan untuk memilih model.

Hasil yang diharapkan: Logistic Regression terpilih untuk sentiment dan Naive
Bayes untuk kategori berdasarkan hasil pipeline terakhir. Jelaskan hasil yang
muncul di terminal jika nilainya berubah setelah pipeline dijalankan ulang.

### Jeda 6 — EDA (Manual: statistics, distribution, visualization)

Kode yang dipanggil: `run_eda()` dalam `src/eda.py`.

Proses: menghitung jumlah destinasi/review, mean, median, distribusi rating,
panjang teks, bahasa, tren waktu, kategori dibanding sentiment, dan koordinat
peta.

Hasil yang diharapkan: 569 destinasi, 3.394 baris, 3.188 review valid, 206
destinasi tanpa review, dan rata-rata rating sekitar 4,40.

### Jeda 7 — Model evaluation (Manual: evaluation dan interpretation)

Kode yang dipanggil: `run_evaluation()` dalam `src/model_evaluation.py`.

Proses: menghitung accuracy, precision, recall, F1-macro, F1-weighted,
confusion matrix, cross-validation, dan metrik clustering. Macro F1 digunakan
karena distribusi kelas tidak seimbang.

Kalimat presentasi: "Accuracy tidak cukup. Model yang selalu memilih kelas
Positive bisa mempunyai accuracy tinggi, sehingga kami juga menggunakan
F1-macro dan cross-validation."

### Jeda 8 — Storage dan export (Manual: integration dan dashboard)

Pipeline menyimpan:

- `data/features/review_features.csv` untuk fitur model;
- `data/output/reviews_mined.csv` untuk kategori, sentiment, cluster, dan prediksi;
- `reports/eda_summary.json` untuk ringkasan EDA;
- `reports/model_evaluation.json` untuk evaluasi;
- `reports/figures/` untuk grafik;
- `powerbi/` untuk tabel ekspor.

Tunggu sampai terminal menampilkan `PIPELINE KOMPLETU` sebelum membuka dashboard.

Pipeline dapat memerlukan beberapa menit. Tunggu sampai muncul `PIPELINE KOMPLETU`.

## 4. Jalankan pengujian integrasi

```powershell
python -m unittest tests.test_integrity tests.test_evaluation
```

Penjelasan: pengujian memastikan jumlah data, train/test split, kategori, metrik
evaluasi, dan hubungan antar-output tetap konsisten.

## 5. Jalankan dashboard

```powershell
python -m streamlit run dashboard\app.py --server.address 127.0.0.1
```

Buka `http://localhost:8501`. Jelaskan KPI, filter, EDA, peta, clustering,
perbandingan model, confusion matrix, dan tabel data. Hentikan server dengan
`Ctrl+C` setelah demo.

## 6. Reset untuk latihan berikutnya

```powershell
powershell -ExecutionPolicy Bypass -File scripts\demo_reset.ps1
```

Skrip memindahkan hasil pipeline ke `demo_backup` sehingga dapat dipulihkan.
Dataset mentah dan source code tidak dihapus.
