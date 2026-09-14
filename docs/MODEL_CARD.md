# Model card — semester1-v1.0.0

## Intended use

Demonstrasi pendampingan akademik berdasarkan informasi sampai akhir semester 1. Model memprediksi status pada akhir durasi normal program, bukan waktu dropout. Tidak untuk sanksi, penolakan, atau keputusan akademik otomatis.

## Model dan protokol

- Terpilih: `random_forest` dari kandidat Logistic Regression, Random Forest, HistGradientBoosting.
- Baseline: DummyClassifier prior; seluruh kandidat dibandingkan pada development 5-fold CV.
- Preprocessing: StandardScaler numerik + OneHotEncoder nominal, selalu di dalam pipeline training.
- Kalibrasi: sigmoid 3-fold, ensemble tiga model.
- Fitur: 14; target tiga kelas string; tidak memakai encoder target terpisah.
- Split: 2.654 development, 885 policy validation, 885 historical holdout.
- Threshold: **0.19**, dipilih dengan F2 pada policy validation. Tie memilih threshold lebih tinggi.
- Tidak dilakukan refit pada seluruh data setelah holdout. Artefak sama dengan yang dievaluasi.

Pemilihan model memakai macro F1; kandidat lain dapat memiliki Brier/AP lebih baik. Prosedur tidak memilih ulang model menggunakan hasil holdout.

## Hasil holdout historis

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Dropout | 0.715 | 0.655 | 0.684 | 284 |
| Enrolled | 0.426 | 0.270 | 0.331 | 159 |
| Graduate | 0.760 | 0.900 | 0.824 | 442 |

Macro F1 **0.6129**, accuracy **70.85%**, weighted F1 **0.6904**. Dropout Brier **0.1338**, average precision **0.7788**.

Kebijakan peninjauan: recall **88.03%**, precision **54.82%**, review rate **51.53%**. TP=250, FN=34, FP=206, TN=395. Performa ini berbeda dari confusion matrix argmax multiclass.

## Interval 95% bootstrap

| Ukuran | Batas bawah | Batas atas |
|---|---:|---:|
| Macro F1 | 0.5785 | 0.6469 |
| Recall kebijakan | 0.8422 | 0.9138 |
| Precision kebijakan | 0.5017 | 0.6012 |

500 resampling baris untuk model tetap. Tidak mencakup variasi training/seleksi atau pergeseran distribusi. Jangan menafsirkan interval sebagai jaminan pada institusi baru.

## Error kelompok

Hasil di `reports/subgroup_metrics.csv` mengukur gender, usia, dan status beasiswa hanya untuk diagnosis. Misalnya, kelompok penerima beasiswa dalam holdout memiliki hanya 19 kejadian Dropout; recall lebih tidak stabil. Gender/beasiswa tidak masuk fitur utama, tetapi error kelompok tetap berbeda. Tidak ada klaim fairness atau mitigasi bias yang sudah terbukti.

## Keterbatasan utama

1. Holdout telah diperiksa pada proyek lama; bukan validation eksternal yang pristine.
2. Tidak ada tanggal kejadian; sebagian mahasiswa mungkin sudah dropout sebelum akhir semester 1.
3. Enrolled belum outcome akhir yang tuntas; recall kelas ini terbatas.
4. Data Portugal tidak otomatis sesuai konteks Indonesia.
5. Review rate sekitar setengah data dapat melampaui kapasitas tim; tidak ada kapasitas nyata yang ditetapkan.
6. Kalibrasi dinilai secara internal; probabilitas tetap dapat meleset pada populasi baru.
7. Permutation importance global bukan efek kausal atau penjelasan individual.
8. Tidak ada bukti dampak intervensi atau manfaat operasional nyata.

## Reproduksi dan pembaruan

Manifest mencatat checksum model/data, daftar fitur, threshold, versi library, dan code hashes. Untuk pembaruan, tetapkan protokol baru sebelum eksperimen, dokumentasikan perubahan, dan evaluasi data baru bila tersedia. Jangan memindahkan threshold berdasarkan hasil holdout lama untuk mempercantik skor.
