# Business case dan keputusan desain

## Masalah

Jaya Jaya Institut adalah institusi fiktif dalam studi kasus Dicoding. Tim akademik ingin mengenali mahasiswa yang mungkin membutuhkan pendampingan. Data publik yang tersedia berisi riwayat pendaftaran, performa dua semester, dan status pada akhir durasi normal program.

## Skenario yang dipilih

Versi portofolio menggunakan **informasi sampai akhir semester 1** untuk memprediksi status **Dropout / Enrolled / Graduate** pada titik pelabelan dataset. Pengguna demonstrasi adalah dosen wali yang meninjau hasil sebelum menawarkan dukungan.

Ini merupakan prediksi retrospektif. Tidak ada tanggal dropout atau bukti bahwa setiap mahasiswa masih aktif pada akhir semester 1. Model tidak diklaim mengukur waktu menuju dropout atau mengidentifikasi seluruh kasus sebelum kejadian.

## Mengapa tetap multiclass

Enrolled merupakan outcome yang belum selesai pada titik pelabelan. Memetakan Enrolled menjadi “tidak dropout/aman” akan menyederhanakan target secara tidak tepat. Multiclass mempertahankan makna label sumber, sedangkan probabilitas Dropout dipakai untuk membentuk kebijakan peninjauan.

## Mengapa 14 fitur

- Delapan fitur pendaftaran memberi informasi program, jalur masuk, nilai, dan usia.
- Enam fitur akademik semester 1 memberi indikator capaian awal.
- Seluruh semester 2 dikeluarkan agar sesuai waktu informasi yang dipilih.
- Debtor, Tuition_fees_up_to_date, Scholarship_holder dan fitur makroekonomi tidak digunakan karena waktu operasional pencatatannya belum cukup jelas.
- Atribut demografis langsung dan latar keluarga tidak diperlukan untuk formulir utama. Usia tetap digunakan sebagai bagian profil masuk; evaluasi error menurut umur tersedia.

Jumlah fitur ini merupakan keputusan desain awal berdasarkan konteks, bukan hasil memilih subset yang memaksimalkan skor holdout. Mengeluarkan fitur dapat mengurangi accuracy, tetapi membuat skenario dan formulir lebih jelas. Tidak ada klaim bahwa eksklusi fitur menjamin fairness.

## Keputusan produk

| Aspek | Implementasi |
|---|---|
| Klasifikasi | Kelas dengan probabilitas tertinggi |
| Tindakan | Perlu peninjauan jika P(Dropout) ≥ threshold; selain itu pemantauan rutin |
| Pemilihan threshold | Maksimalkan F2 pada 885 baris policy validation |
| Biaya kesalahan | False negative melewatkan kebutuhan bantuan; false positive menambah beban peninjauan |
| Kapasitas tim | Belum diketahui; review rate wajib ditampilkan dan tidak dibatasi secara fiktif |
| Pendampingan | Aturan transparan berdasarkan unit lulus, unit tanpa evaluasi, dan hasil peninjauan |
| Dashboard | Analisis data historis dengan filter dan denominator yang jelas |

F2 merupakan asumsi desain untuk demo yang mengutamakan recall. Sebelum pilot, institusi harus menetapkan kapasitas dan konsekuensi kesalahan; ambang perlu dievaluasi kembali melalui protokol baru jika kebutuhannya berubah.

## Hasil bisnis yang belum terbukti

Belum ada bukti pengurangan dropout, dampak beasiswa, peningkatan nilai akibat mentoring, atau efisiensi kerja dosen wali. Rekomendasi berikut merupakan rancangan pilot.

| Tindakan | Calon pelaksana | Bukti yang perlu dikumpulkan |
|---|---|---|
| Tinjau profil yang ditandai | Dosen wali | Jumlah ditinjau, waktu proses, kebutuhan bantuan yang dikonfirmasi |
| Klarifikasi unit tanpa evaluasi | Akademik/administrasi | Kelengkapan data dan hambatan partisipasi |
| Tawarkan tutoring sukarela | Tim pendampingan | Partisipasi dan perkembangan akademik berikutnya |
| Evaluasi cohort baru | Tim data | Precision, recall, beban peninjauan, kalibrasi, error kelompok |

Jangan mengubah prediksi menjadi sanksi, penolakan beasiswa, atau keputusan administratif otomatis.
