# Kamus fitur utama

Sumber kategori: UCI dataset 697 / kamus Dicoding. Urutan di bawah adalah urutan input model.

| Feature | Label | Tipe | Domain | Keterangan |
|---|---|---|---|---|
| `Course` | Program studi | category | Kategori: 33, 171, 8014, 9003, 9070, 9085, 9119, 9130, 9147, 9238, 9254, 9500, 9556, 9670, 9773, 9853, 9991 | Kode program sesuai sumber Portugal; bukan kode prodi Indonesia. |
| `Application_mode` | Jalur pendaftaran | category | Kategori: 1, 2, 5, 7, 10, 15, 16, 17, 18, 26, 27, 39, 42, 43, 44, 51, 53, 57 | Kategori jalur masuk menurut kamus data sumber. |
| `Application_order` | Urutan pilihan | integer | 0–9 | Urutan pilihan program, 0 sampai 9. |
| `Daytime_evening_attendance` | Jadwal kuliah | category | Kategori: 0, 1 | Jadwal yang dipilih saat pendaftaran. |
| `Previous_qualification` | Pendidikan sebelumnya | category | Kategori: 1, 2, 3, 4, 5, 6, 9, 10, 12, 14, 15, 19, 38, 39, 40, 42, 43 | Kategori kualifikasi sebelum masuk. |
| `Previous_qualification_grade` | Nilai pendidikan sebelumnya | continuous | 0–200 | Skala sumber 0–200; jangan memasukkan IPK 0–4. |
| `Admission_grade` | Nilai penerimaan | continuous | 0–200 | Skala sumber 0–200. |
| `Age_at_enrollment` | Usia saat masuk | integer | 16–100 | Batas operasional demo 16–100; rentang historis diperiksa terpisah. |
| `Curricular_units_1st_sem_credited` | Unit yang diakui / transfer | integer | 0–60 | Jumlah curricular units yang dikreditkan pada semester 1; bukan otomatis SKS. |
| `Curricular_units_1st_sem_enrolled` | Unit yang diambil | integer | 0–60 | Jumlah curricular units yang didaftarkan pada semester 1. |
| `Curricular_units_1st_sem_evaluations` | Jumlah evaluasi | integer | 0–120 | Jumlah evaluasi; boleh melebihi unit karena evaluasi berulang. |
| `Curricular_units_1st_sem_approved` | Unit yang lulus | integer | 0–60 | Jumlah unit yang diselesaikan; tidak boleh melebihi unit yang diambil. |
| `Curricular_units_1st_sem_grade` | Rata-rata nilai semester 1 | continuous | 0–20 | Skala sumber 0–20. Nol dapat mencerminkan tidak adanya nilai. |
| `Curricular_units_1st_sem_without_evaluations` | Unit tanpa evaluasi | integer | 0–60 | Tidak boleh melebihi jumlah unit yang diambil. |

## Program studi

| Kode | Arti |
|---|---|
| 33 | Biofuel Production Technologies |
| 171 | Animation and Multimedia Design |
| 8014 | Social Service (evening) |
| 9003 | Agronomy |
| 9070 | Communication Design |
| 9085 | Veterinary Nursing |
| 9119 | Informatics Engineering |
| 9130 | Equinculture |
| 9147 | Management |
| 9238 | Social Service |
| 9254 | Tourism |
| 9500 | Nursing |
| 9556 | Oral Hygiene |
| 9670 | Advertising and Marketing Management |
| 9773 | Journalism and Communication |
| 9853 | Basic Education |
| 9991 | Management (evening) |

## Jalur pendaftaran

| Kode | Arti |
|---|---|
| 1 | 1st phase - general |
| 2 | Ordinance 612/93 |
| 5 | 1st phase - Azores |
| 7 | Holders of other higher courses |
| 10 | Ordinance 854-B/99 |
| 15 | International bachelor applicant |
| 16 | 1st phase - Madeira |
| 17 | 2nd phase - general |
| 18 | 3rd phase - general |
| 26 | Ordinance 533-A/99 b2 |
| 27 | Ordinance 533-A/99 b3 |
| 39 | Over 23 years old |
| 42 | Transfer |
| 43 | Change of course |
| 44 | Technological specialization |
| 51 | Change of institution/course |
| 53 | Short cycle diploma |
| 57 | Change of institution/course - international |

## Jadwal kuliah

| Kode | Arti |
|---|---|
| 0 | Malam |
| 1 | Siang |

## Pendidikan sebelumnya

| Kode | Arti |
|---|---|
| 1 | Secondary education |
| 2 | Higher education - bachelor's degree |
| 3 | Higher education - degree |
| 4 | Higher education - master's |
| 5 | Higher education - doctorate |
| 6 | Attendance in higher education |
| 9 | 12th year - not completed |
| 10 | 11th year - not completed |
| 12 | Other - 11th year |
| 14 | 10th year |
| 15 | 10th year - not completed |
| 19 | Basic education 3rd cycle |
| 38 | Basic education 2nd cycle |
| 39 | Technological specialization |
| 40 | Higher education - degree (1st cycle) |
| 42 | Professional higher technical course |
| 43 | Higher education - master (2nd cycle) |

## Validasi bersama

- Harus finite, tidak kosong, dan bertipe numerik yang sesuai.
- Kategori harus ada di kamus; jumlah unit harus bilangan bulat.
- Approved dan without evaluations tidak boleh melebihi enrolled.
- Evaluations boleh melebihi enrolled karena penilaian berulang.
- Extra columns di CSV tidak digunakan untuk prediksi dan tidak diekspor.
- Batas domain demo tidak ditentukan dari min/max dataset. Nilai yang valid tetapi di luar rentang development diberi peringatan.
