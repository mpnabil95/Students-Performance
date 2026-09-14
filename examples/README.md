# Examples — mencoba prediksi dengan data sintetis

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Gunakan folder ini untuk mencoba aplikasi tanpa memasukkan data mahasiswa nyata. Contoh dibuat secara sintetis untuk demonstrasi input, bukan sampel mahasiswa dari dataset penelitian.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [students_template.csv](students_template.csv) | 14 kolom fitur dan tiga profil sintetis: umum, perlu dukungan akademik, dan akademik kuat. | Percobaan pertama Prediksi Batch atau acuan header CSV baru. |


## Percobaan pertama

1. Jalankan aplikasi, lalu buka **Prediksi Batch**.
2. Unggah `students_template.csv` atau unduh template melalui tombol dalam aplikasi.
3. Klik **Validasi dan prediksi**.
4. Baca jumlah profil yang perlu ditinjau dan unduh seluruh hasil.

Nama profil sintetis tidak menjadi label kebenaran; prediksi model tidak wajib mengikuti nama contoh. File template tidak menyertakan `Status` atau identitas mahasiswa.

## Membuat input sendiri

Salin template ke file baru di perangkatmu. Pertahankan nama 14 kolom, gunakan kode kategori yang tercantum dalam [kamus fitur](../docs/FEATURE_DICTIONARY.md), serta simpan CSV UTF-8 dengan koma atau titik koma. Desimal menggunakan titik. Nilai pendaftaran memakai skala 0–200, nilai semester 0–20, dan jumlah unit bukan otomatis SKS Indonesia.

Kolom ekstra tidak masuk model dan tidak ikut diekspor. Nomor `source_row` pada unduhan menunjukkan urutan baris dalam unggahan; simpan pemetaan identitas secara terpisah bila diperlukan. Input kosong, tidak valid, atau lebih dari 10.000 baris ditolak. Batas file 10 MB.

Template ini dihasilkan ulang saat training. Untuk demo pribadi yang ingin dipertahankan, simpan sebagai file terpisah. [Panduan aplikasi](../docs/APP_GUIDE.md) menjelaskan seluruh keluaran.
