# v1.0.0 — Student Success Portfolio Edition

Versi portofolio pertama mengembangkan baseline submission Dicoding menjadi studi kasus prediksi status studi dengan informasi sampai akhir semester 1.

## Perubahan utama

- Skenario dan kontrak 14 fitur ditetapkan; seluruh fitur semester 2 dikeluarkan.
- Seleksi model melalui cross-validation dan kalibrasi di dalam training.
- Pemilihan threshold pada policy validation yang terpisah.
- Model, schema, dan manifest konsisten untuk training serta inference.
- Dashboard historis, prediksi individu/batch, dan halaman kinerja dalam satu aplikasi.
- Notebook sudah dijalankan; laporan metrik, error kelompok, importance, dan figur disertakan.
- Validasi domain, relasi akademik, preset, dan kesetaraan batch/individu diuji.

## Hasil

Model terpilih `random_forest`. Macro F1 holdout historis 0.6129. Pada threshold 0.19, recall peninjauan Dropout 88.03%, precision 54.82%, dan review rate 51.53%.

Hasil bersifat retrospektif pada holdout yang pernah dilihat dalam proyek lama. Tidak ada klaim validasi eksternal atau keberhasilan intervensi.

## Menjalankan

Gunakan Python 3.12, instal `requirements.txt`, lalu jalankan `streamlit run app.py`. Artefak sudah tersedia; panduan lengkap di README.

## Verifikasi sebelum publikasi release

Periksa `docs/VALIDATION.md`, jalankan CI dan tes Streamlit pada environment lengkap, serta tambahkan tautan deployment hanya setelah berhasil. Teks ini tidak mengklaim pengujian yang belum dijalankan.

Arsip submission tetap pada `dicoding-submission-v1.0.0`.
