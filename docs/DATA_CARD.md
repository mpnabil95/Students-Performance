# Data card

## Identitas dan sumber

- Nama asli: **Predict Students' Dropout and Academic Success**.
- Kreator: Valentim Realinho, Mónica Vieira Martins, Jorge Machado, Luís Baptista (2021).
- [UCI dan DOI](https://doi.org/10.24432/C5MC89).
- [Distribusi yang digunakan: Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance).
- Konteks sumber: pendidikan tinggi di Portugal, bukan data aktual Jaya Jaya Institut.
- Lisensi dataset yang dinyatakan UCI: **CC BY 4.0**. Atribusi dataset tetap berlaku; LICENSE kode tidak mengubah lisensi data.

## Snapshot

File `data/raw/students.csv` mempertahankan data yang digunakan original submission: 4.424 baris × 37 kolom. Setiap baris merepresentasikan mahasiswa; target adalah Status. Tidak tersedia pengenal mahasiswa nyata, timestamp per mahasiswa, tanggal dropout, atau tahun cohort yang eksplisit pada file.

SHA-256:

`a37dbda5555089a8d39b8be6f1a242f68403963a1f81423115c9676c8c7100e9`

Jangan mengedit file mentah. Untuk mengganti snapshot, perbarui provenance, checksum, protokol, dan evaluasi secara eksplisit.

## Target

| Label | Arti |
|---|---|
| Dropout | Status keluar pada titik pelabelan dataset |
| Enrolled | Masih terdaftar pada akhir durasi normal program; outcome belum selesai |
| Graduate | Lulus pada titik pelabelan |

## Kualitas dan semantik

Null eksplisit: 0. Duplikat baris penuh: 0. Kode Unknown=34 pada kualifikasi ibu: 130 baris; ayah: 112 baris. Kedua fitur tidak dipakai model utama tetapi kekurangan informasi tetap dicatat.

Nilai nol akademik tidak diubah otomatis menjadi null. Kode kategori diperlakukan sesuai makna nominal. Nilai masuk memakai skala 0–200, nilai semester 0–20. Curricular units tidak diterjemahkan otomatis sebagai SKS.

Kamus fitur utama ada di `FEATURE_DICTIONARY.md` dan `artifacts/feature_schema.json`. Batas usia/count pada schema merupakan batas operasi demo; rentang data pengembangan yang sebenarnya tersimpan dalam manifest. Input valid di luar rentang pengembangan memperoleh peringatan.

## Fitur yang tidak masuk model

Seluruh fitur semester 2; Debtor, Tuition_fees_up_to_date, Scholarship_holder; Unemployment_rate, Inflation_rate, GDP; Marital_status, Nacionality, Mothers_qualification, Fathers_qualification, Mothers_occupation, Fathers_occupation, Displaced, Educational_special_needs, Gender, International.

Sebagian atribut dipakai **hanya untuk evaluasi kelompok setelah model dikunci**, bukan saat training/inference. Data mentah menyimpan seluruh kolom untuk provenance dan analisis historis.

## Pembatasan generalisasi

Timing detail variabel dan kejadian dropout belum tersedia. Tidak ada validasi cohort temporal atau institusi eksternal. Program studi dan jalur masuk mengikuti sumber Portugal; input Indonesia memerlukan pemetaan makna, bukan sekadar mengganti label tampilan.

Unggahan batch aplikasi tidak ditulis ke file proyek. Pengguna tetap perlu menjaga file unduhan di perangkatnya; demo publik sebaiknya memakai contoh sintetis yang tersedia.
