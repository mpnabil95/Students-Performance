# Student success — kode inti project

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Folder ini adalah mesin di balik notebook dan aplikasi. Aturan input, training, serta prediksi dipakai bersama sehingga kedua cara penggunaan mengikuti kontrak yang sama. Pembaca umum tidak perlu membaca seluruh modul.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [__init__.py](__init__.py) | Menandai direktori sebagai paket Python dan menyimpan versi paket. | Memeriksa identitas paket. |
| [config.py](config.py) | Lokasi file, urutan kelas, seed, dan protokol eksperimen. | Memahami konfigurasi serta batas eksperimen. |
| [schema.py](schema.py) | Definisi 14 fitur, kode kategori, validasi, pembacaan CSV, dan profil contoh. | Mengubah atau memeriksa aturan input. |
| [modeling.py](modeling.py) | Pembuatan kandidat pipeline, penyelarasan probabilitas, metrik, dan bootstrap. | Memahami cara model dibangun dan dievaluasi. |
| [train.py](train.py) | Orkestrasi split, seleksi, kalibrasi, ambang, penyimpanan model, dan laporan. | Mengulang eksperimen atau menelusuri urutan training. |
| [inference.py](inference.py) | Pemuatan model, prediksi, kategori tindakan, peringatan rentang, dan saran pendampingan. | Memahami bagaimana input pengguna menjadi keluaran aplikasi. |
| [visuals.py](visuals.py) | Pembuatan grafik dari data dan laporan yang sudah dihitung. | Memperbarui visualisasi atau memahami sumber suatu grafik. |


## Urutan membaca kode

Mulai dari `config.py` dan `schema.py`. Untuk alur prediksi, lanjut ke `inference.py` lalu `app.py` di root. Untuk eksperimen, lanjut ke `modeling.py`, `train.py`, lalu `visuals.py`.

Training terpisah, dari root repository:

```bash
python -m student_success.train
```

Perintah ini menulis artefak, laporan, grafik, dan template contoh. Bila ingin notebook ikut diperbarui, gunakan builder notebook sebagai jalur lengkap; builder sudah menjalankan training.

## Aturan perubahan

Manifest menyimpan hash modul Python dalam folder ini. Perubahan kode inti perlu diperiksa dampaknya terhadap artefak serta reproduksi; jangan mengubah hash manifest secara manual untuk meloloskan pemeriksaan. README ini adalah dokumentasi dan tidak masuk hash kode model.

[Kontrak fitur](../docs/FEATURE_DICTIONARY.md) · [Reproduksi](../docs/REPRODUCIBILITY.md) · [Tes](../tests/README.md)
