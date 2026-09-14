# Figures — panduan membaca grafik

[Kembali ke README utama](../../README.md) · [Peta repository](../../docs/REPOSITORY_GUIDE.md)

Grafik di folder ini berasal dari data dan hasil eksperimen yang tercatat. Setiap nama memiliki dua format dengan isi analisis yang sama: PNG untuk tampilan umum dan SVG untuk gambar yang tetap tajam saat diperbesar.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [status_distribution.png](status_distribution.png) | Komposisi Dropout, Enrolled, dan Graduate pada seluruh snapshot. | Lihat jumlah dan proporsi; bukan estimasi angka dropout kampus lain. |
| [academic_patterns.png](academic_patterns.png) | Sebaran unit lulus dan nilai semester 1 menurut status pada seluruh snapshot. | Bandingkan pola kelompok; asosiasi tidak membuktikan penyebab. Outlier tidak ditampilkan pada boxplot. |
| [model_selection.png](model_selection.png) | Macro F1 rata-rata ± simpangan baku 5 fold pada development. | Nilai lebih tinggi berarti keseimbangan klasifikasi tiga kelas lebih baik dalam eksperimen ini. |
| [confusion_matrix.png](confusion_matrix.png) | Jumlah prediksi benar dan keliru untuk tiga kelas pada holdout. | Baris adalah status aktual, kolom adalah prediksi; diagonal menunjukkan klasifikasi benar. |
| [precision_recall.png](precision_recall.png) | Hubungan recall dan precision Dropout pada holdout serta titik ambang validation. | Menunjukkan manfaat dan beban peninjauan; grafik holdout bukan tempat memilih ulang ambang. |
| [calibration.png](calibration.png) | Reliability curve dan histogram probabilitas Dropout pada holdout. | Kedekatan ke diagonal menunjukkan kesesuaian probabilitas dengan proporsi kejadian pada kelompok data ini. |
| [threshold_selection.png](threshold_selection.png) | Recall, precision, review rate, dan F2 di berbagai ambang policy validation. | Garis vertikal menandai ambang terpilih; penurunan ambang biasanya memperbanyak profil yang ditinjau. |
| [feature_importance.png](feature_importance.png) | Sepuluh fitur teratas berdasarkan penurunan macro F1 ketika diacak pada holdout. | Penurunan lebih besar menunjukkan ketergantungan model lebih besar; bukan efek kausal atau penjelasan satu mahasiswa. |
| [status_distribution.svg](status_distribution.svg) | Versi vektor grafik status_distribution. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |
| [academic_patterns.svg](academic_patterns.svg) | Versi vektor grafik academic_patterns. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |
| [model_selection.svg](model_selection.svg) | Versi vektor grafik model_selection. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |
| [confusion_matrix.svg](confusion_matrix.svg) | Versi vektor grafik confusion_matrix. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |
| [precision_recall.svg](precision_recall.svg) | Versi vektor grafik precision_recall. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |
| [calibration.svg](calibration.svg) | Versi vektor grafik calibration. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |
| [threshold_selection.svg](threshold_selection.svg) | Versi vektor grafik threshold_selection. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |
| [feature_importance.svg](feature_importance.svg) | Versi vektor grafik feature_importance. | Ekspor untuk presentasi atau penyuntingan tata letak; makna sama dengan PNG di atas. |


## Menghasilkan ulang

Dari root repository, dengan dependensi terpasang:

```bash
python -m student_success.visuals
```

Perintah ini membaca dataset dan laporan yang sudah ada, lalu menulis ulang seluruh PNG/SVG. Training juga memanggil pembuat grafik ini. Jika model atau ambang berubah, perbarui laporan melalui training terlebih dahulu agar figur konsisten.

Jangan mengganti angka atau bentuk grafik secara manual untuk mengubah kesimpulan. Gambar dapat dipakai dalam README/presentasi dengan tetap menyertakan konteks dataset, metrik, dan atribusi.
