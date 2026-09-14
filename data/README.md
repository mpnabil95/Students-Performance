# Data — sumber eksperimen

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Folder ini berisi data yang digunakan untuk analisis dan training. Data sumber dipisahkan dari laporan hasil eksperimen agar asal setiap hasil dapat ditelusuri.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [raw/](raw/README.md) | Snapshot asli 4.424 mahasiswa dari distribusi Dicoding/UCI. | Menelusuri sumber, memeriksa kolom asli, atau mengulang analisis. |


## Mulai membaca

Baca [data card](../docs/DATA_CARD.md) untuk konteks Portugal, lisensi, kualitas data, dan keterbatasannya. Baca [kamus fitur](../docs/FEATURE_DICTIONARY.md) untuk 14 kolom yang benar-benar masuk model. Dataset mentah tetap memiliki 37 kolom, termasuk `Status`; tidak semua kolom digunakan saat memprediksi.

Data baru yang ingin diprediksi **tidak perlu dimasukkan ke folder ini**. Gunakan halaman Prediksi Batch dan format pada [examples/](../examples/README.md). Hasil evaluasi model tersimpan di [reports/](../reports/README.md), sedangkan file hasil unggahan diunduh dari aplikasi ke perangkat pengguna.

## Aturan perubahan

Pertahankan snapshot mentah. Perubahan isinya memerlukan pembaruan sumber, checksum, protokol, dan evaluasi. Training akan menolak checksum yang berbeda dari snapshot yang ditetapkan.
