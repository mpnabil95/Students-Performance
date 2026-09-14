# Reports — bukti hasil eksperimen

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Folder ini adalah kumpulan bukti yang mendukung kesimpulan project. Untuk pembaca umum, mulai dari bagian hasil pada README utama dan grafik. CSV/JSON disediakan agar reviewer dapat menelusuri dan menghitung ulang hasil.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [metrics.json](metrics.json) | Ringkasan skor validation/holdout, baseline, hasil pemilihan model, dan interval bootstrap. | Mencari angka sumber pada README atau model card. |
| [protocol.json](protocol.json) | Keputusan eksperimen: skenario, split, seed, seleksi, kalibrasi, dan kebijakan threshold. | Memeriksa aturan eksperimen sebelum menafsirkan skor. |
| [data_quality.json](data_quality.json) | Jumlah baris/kolom, label, null, duplikat, kode Unknown, validasi schema, dan checksum data. | Memeriksa kualitas serta identitas snapshot. |
| [split_assignments.csv](split_assignments.csv) | Pembagian setiap baris sumber ke development, policy validation, atau historical holdout. | Menelusuri data mana yang dipakai pada masing-masing tahap. |
| [cv_folds.csv](cv_folds.csv) | Skor tiap kandidat/fold, termasuk eksperimen fitur pendaftaran sebagai pembanding. | Melihat kestabilan antar-fold dan rincian eksperimen. |
| [model_comparison.csv](model_comparison.csv) | Rata-rata dan variasi skor kandidat utama pada cross-validation. | Memahami alasan kandidat utama terpilih. |
| [threshold_analysis.csv](threshold_analysis.csv) | Precision, recall, F2, beban peninjauan, dan jumlah kesalahan pada tiap ambang validation. | Memahami konsekuensi pemilihan ambang peninjauan. |
| [policy_validation_predictions.csv](policy_validation_predictions.csv) | Label aktual, prediksi, probabilitas, dan tanda review untuk baris policy validation. | Menelusuri data yang digunakan memilih threshold. |
| [historical_holdout_predictions.csv](historical_holdout_predictions.csv) | Label aktual, prediksi, probabilitas, dan tanda review untuk holdout historis. | Memeriksa contoh kesalahan serta menghitung ulang evaluasi. |
| [subgroup_metrics.csv](subgroup_metrics.csv) | Metrik kebijakan menurut gender, kelompok usia, dan status beasiswa pada holdout. | Diagnosis perbedaan error antarkelompok setelah model dikunci. |
| [permutation_importance.csv](permutation_importance.csv) | Perubahan macro F1 saat setiap fitur diacak, beserta simpangan bakunya. | Melihat ketergantungan prediksi terhadap fitur secara global. |
| [verification.json](verification.json) | Catatan pemeriksaan lokal terakhir oleh verify_package, termasuk tes yang dilewati. | Melihat hasil mesin untuk verifikasi lokal, bukan status Actions terbaru. |
| [figures/](figures/README.md) | Delapan grafik, masing-masing dalam PNG dan SVG. | Membaca hasil secara visual atau menggunakan gambar pada presentasi. |


## Urutan membaca yang disarankan

1. [Model card](../docs/MODEL_CARD.md): penjelasan hasil dan batasnya.
2. [model_comparison.csv](model_comparison.csv): alasan pemilihan model.
3. [threshold_analysis.csv](threshold_analysis.csv): alasan pemilihan ambang.
4. [metrics.json](metrics.json): hasil akhir dan ketidakpastiannya.
5. Prediksi per baris atau diagnosis kelompok bila perlu menelusuri lebih jauh.

## Arti kolom penting

| Kolom/istilah | Arti |
|---|---|
| `source_row` | Nomor baris data mulai dari 1 pada `data/raw/students.csv`; tidak termasuk header |
| `actual` / `target` | Label yang benar-benar tercatat pada dataset |
| `predicted` | Kelas dengan probabilitas paling tinggi |
| `prob_dropout`, `prob_enrolled`, `prob_graduate` | Perkiraan probabilitas kelas dalam rentang 0–1 |
| `review` | True jika probabilitas Dropout mencapai ambang peninjauan |
| `tp` / `fn` | Kasus Dropout yang ditandai / yang terlewat |
| `fp` / `tn` | Kasus selain Dropout yang ditandai / yang tidak ditandai |
| `review_rate` | Proporsi seluruh baris yang ditandai untuk peninjauan |
| `feature_set` | Kelompok fitur yang dipakai eksperimen terkait |

Nomor baris di laporan merujuk dataset sumber. Nomor baris dalam unduhan aplikasi merujuk file yang diunggah pengguna; keduanya tidak otomatis sama. File prediksi evaluasi juga memakai `predicted` dan `review`, sedangkan unduhan aplikasi memakai `predicted_status` dan `action`.

## Asal dan aturan perubahan

Training menghasilkan laporan di atas dan grafik, kecuali `verification.json` yang ditulis oleh `python scripts/verify_package.py`. Jangan mengedit angka hasil secara manual. Ulangi proses yang sesuai dan periksa konsistensi dengan artefak, notebook, serta dokumentasi.

Policy validation digunakan memilih ambang, sedangkan holdout digunakan melaporkan hasil setelah pilihan dikunci. Holdout ini pernah diperiksa pada submission lama dan bukan validasi eksternal independen. Subgroup metrics tidak membuktikan model bebas bias; permutation importance tidak membuktikan sebab-akibat.
