# Penyelesaian temuan audit baseline

Audit awal mengacu pada commit `1e94b40ed9ff28b29bb04f410f001db268b38579`. Tabel ini membedakan perbaikan implementasi dari batas data dan validasi yang masih terbuka.

| ID | Temuan awal | Tindakan versi portofolio | Status |
|---|---|---|---|
| A01 | Waktu prediksi ambigu | Kontrak semester 1; enam fitur semester 2 serta finansial dengan timing tidak jelas dikeluarkan | Implementasi selesai; tanggal dropout tetap tidak tersedia |
| A02 | Test set dipakai memilih model | CV hanya pada development; threshold di policy validation; holdout setelah freeze | Diperbaiki; keterpaparan historis holdout diakui |
| A03 | Banner dan rekomendasi berbeda | Satu `action_label`, threshold dari validation, kalibrasi dan trade-off dilaporkan | Diperbaiki dan diuji pada logika bersama |
| A04 | Batch hanya memeriksa nama kolom | Tipe, finite, null, kategori, batas, relasi, file kosong/ukuran, header duplikat | Diperbaiki dan diuji |
| A05 | Preset 7 lulus dari 6 diambil | Preset sintetis koheren; validator bersama | Diperbaiki dan diuji |
| A06 | Heuristik cardinality menentukan input | Schema eksplisit dengan label kategori dan domain | Diperbaiki |
| A07 | Logistic tanpa scaling dan warning tersembunyi | Pipeline scaling/OHE; convergence warning menjadi error | Training lulus tanpa warning konvergensi |
| A08 | Weighted F1 dianggap adil antar kelas | Macro F1 untuk seleksi; per-class, policy metrics, AP, Brier dan workload | Diperbaiki |
| A09 | Korelasi >0,90 salah | Notebook menghitung ulang 0,769 dan 0,703 untuk audit historis | Dikoreksi |
| A10 | Klaim asosiasi menjadi sebab-akibat | Narasi membedakan pola, prediksi, dan usulan pilot; importance bukan kausal | Diperbaiki pada dokumentasi baru |
| A11 | Nol null dianggap lengkap | Unknown pendidikan orang tua dicatat; audit domain dan relasi | Diperbaiki |
| A12 | Jalur ekspor berbeda dari aplikasi | Semua artefak di `artifacts/`; checksum dan prediksi ulang diuji | Diperbaiki |
| A13 | Versi dan data tidak terkunci | Snapshot checksum, pin versi inti, manifest environment dan code hashes | Diperbaiki pada inti model; instalasi UI perlu environment lengkap |
| A14 | Model terbaik hardcoded | Pemilihan terprogram dari mean macro F1 CV, tie deterministik | Diperbaiki |
| A15 | Dashboard kurang operasional | Dashboard historis dengan filter, denominator; batch review terpisah | Diperbaiki sesuai data; bukan real-time monitoring |
| A16 | Metabase memerlukan langkah manual | Dashboard portofolio dibangun dari CSV dalam aplikasi; Metabase tetap di arsip | Diganti pada main; pemulihan Metabase lama tidak diklaim sudah diuji |
| A17 | Jalur kegagalan berbeda | Semua halaman prediksi memerlukan loader yang sama; domain tidak tergantung min/max dataset | Inti diperbaiki; integrasi UI perlu tes Streamlit |
| A18 | Judul/narasi/dokumentasi tidak konsisten | Notebook, README, business/data/model card ditulis ulang | Diperbaiki |
| A19 | Tidak ada kontrak test/error kelompok | Unit tests, real AppTest script, CI, laporan kelompok dan interval | Core tests lulus; real AppTest belum berjalan di runtime pembuat paket |
| A20 | Release HTML dan Markdown ganda | `DICODING_RELEASE_CLEAN.md` disediakan | Perubahan body release tetap dilakukan pemilik |

## Peningkatan yang tidak diklaim

- Tidak ada klaim bahwa data telah divalidasi pada institusi nyata atau cohort masa depan.
- Tidak ada klaim perbaikan accuracy dibanding model submission, karena fitur dan skenario berbeda.
- Tidak ada klaim fairness, manfaat intervensi, atau kesiapan produksi yang sudah terbukti.
- Tidak ada perubahan terhadap GitHub, release, atau deployment oleh pembuat paket.

## Pemeriksaan penerimaan sebelum demo publik

Jalankan tes Streamlit pada environment lengkap, lihat seluruh halaman/form/batch secara langsung, konfirmasi deployment memakai branch dan versi Python yang benar, lalu masukkan tautan demo aktual. Struktur, model, reports, notebook, tests, dan dokumentasi sudah disediakan untuk tahap tersebut.
