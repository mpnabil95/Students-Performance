"""Refresh metric-driven documentation after training (does not retrain)."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from student_success.schema import FIELDS
m=json.loads((ROOT/'reports/metrics.json').read_text());manifest=json.loads((ROOT/'artifacts/manifest.json').read_text())
h=m['historical_holdout'];p=h['policy'];ci=m['bootstrap_95ci']
def write(path,text):(ROOT/path).write_text(text.strip()+'\n',encoding='utf-8')

write('README.md',f'''
# Student Success — Semester-One Outcome Prediction

Studi kasus Data Science untuk memahami status studi mahasiswa dan membantu prioritas peninjauan menggunakan informasi sampai **akhir semester 1**.

**14 fitur · 3 kelas · validasi terpisah · probabilitas terkalibrasi · dashboard Streamlit**

Proyek ini berkembang dari submission Dicoding dengan konteks Jaya Jaya Institut (fiktif). Versi portofolio mempertajam waktu prediksi, memperbaiki evaluasi dan validasi input, serta menyatukan analisis dan prediksi dalam satu aplikasi.

> Model merupakan demonstrasi prediksi retrospektif. Dataset tidak memiliki tanggal dropout per mahasiswa; hasil tidak membuktikan bahwa setiap prediksi dibuat sebelum kejadian. Enrolled bukan jaminan lulus atau bebas risiko.

![Distribusi status studi](reports/figures/status_distribution.png)

## Apa yang dapat dilakukan

- Menjelajahi data historis dengan filter program studi dan usia.
- Memasukkan profil semester 1 dengan label kategori dan skala yang jelas.
- Memvalidasi serta memprediksi CSV secara batch; mengunduh hasil dengan source_row.
- Melihat probabilitas tiga kelas dan satu kategori peninjauan yang konsisten.
- Memeriksa performa, calibration curve, trade-off peninjauan, dan keterbatasan.

## Desain studi kasus

| Aspek | Keputusan |
|---|---|
| Target | Dropout / Enrolled / Graduate pada akhir durasi normal program |
| Skenario | Fitur pendaftaran + hasil semester 1 |
| Fitur dikeluarkan | Semester 2, status finansial/makro yang timing-nya belum jelas, gender/kebangsaan dan atribut keluarga; usia tetap digunakan |
| Pemilihan model | Mean macro F1 pada 5-fold CV, hanya data development |
| Kalibrasi | Sigmoid 3-fold di dalam training |
| Threshold | Maksimalkan F2 pada policy validation, terpisah dari seleksi model |
| Output tindakan | Perlu peninjauan / Pemantauan rutin; ditentukan oleh P(Dropout) |
| Penggunaan | Pendampingan oleh manusia, bukan keputusan akademik otomatis |

Detail: [Business case](docs/BUSINESS_CASE.md) · [Data card](docs/DATA_CARD.md) · [Kamus fitur](docs/FEATURE_DICTIONARY.md).

## Hasil yang diperoleh

Model terpilih: **{m['selected_candidate']}**, dengan threshold **{m['threshold']:.2f}**.

| Metrik holdout historis (n = {h['n']}) | Nilai |
|---|---:|
| Accuracy multiclass | {h['accuracy']:.2%} |
| Macro F1 | {h['macro_f1']:.4f} |
| Weighted F1 | {h['weighted_f1']:.4f} |
| Recall Dropout pada kebijakan peninjauan | {p['recall']:.2%} |
| Precision pada kebijakan peninjauan | {p['precision']:.2%} |
| Proporsi profil yang ditandai | {p['review_rate']:.2%} |
| Dropout average precision | {h['dropout_average_precision']:.4f} |

Kebijakan mengenali **{p['tp']} dari {p['tp']+p['fn']}** kasus Dropout, melewatkan **{p['fn']}**, dan menghasilkan **{p['fp']}** false positive. Total **{p['review_count']} profil** perlu ditinjau. Recall tinggi disertai beban peninjauan besar; kapasitas institusi nyata belum ditetapkan.

**Evaluasi ini memakai holdout historis yang pernah dilihat pada submission.** Hasil bukan validasi eksternal independen. Skor juga tidak dibandingkan langsung sebagai peningkatan terhadap model lama yang memakai fitur semester 2 dan finansial.

![Seleksi model](reports/figures/model_selection.png)
![Trade-off peninjauan](reports/figures/precision_recall.png)

[Model card](docs/MODEL_CARD.md) memuat interval, kelemahan per kelas, error kelompok, dan batas penggunaan. Hasil terstruktur tersedia pada `reports/metrics.json`.

## Mulai dalam lingkungan lokal

Gunakan **Python 3.12**. Di root repository:

```bash
python -m venv .venv
```

Aktifkan environment dengan `.venv\\Scripts\\activate.bat` (Windows Command Prompt), `.\\.venv\\Scripts\\Activate.ps1` (PowerShell), atau `source .venv/bin/activate` (Linux/macOS), kemudian:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Model terlatih sudah disertakan. Contoh CSV sintetis ada di `examples/students_template.csv`. Instal versi dependensi yang sesuai karena model memeriksa versi scikit-learn saat dimuat.

## Training, notebook, dan pengujian

```bash
python -m student_success.train
python scripts/build_notebook.py
python -m unittest discover -s tests -v
```

Builder notebook menjalankan training kembali dan menghasilkan `notebook.ipynb` dengan output nyata. Untuk Jupyter:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebook.ipynb
```

Notebook yang disertakan sudah dieksekusi: **41 sel, 28 sel kode**. [Protokol reproduksi](docs/REPRODUCIBILITY.md) menjelaskan alur, file keluaran, dan batas environment. [Catatan verifikasi](docs/VALIDATION.md) membedakan pemeriksaan yang lulus dan pengujian UI yang masih perlu dijalankan pada environment dengan Streamlit.

## Struktur repository

| Path | Peran |
|---|---|
| `app.py` | Aplikasi Streamlit: dashboard, individu, batch, kinerja |
| `notebook.ipynb` | Narasi analisis dan alur training yang sudah dijalankan |
| `student_success/` | Schema, pipeline, training, inference, figur |
| `data/raw/` | Snapshot dataset asli, checksum tetap |
| `artifacts/` | Model terkalibrasi, manifest, schema |
| `reports/` | Split, CV, evaluasi, prediksi, diagnostik dan figur |
| `examples/` | CSV sintetis untuk demonstrasi |
| `tests/` | Kontrak data/artefak dan pengujian Streamlit |
| `docs/` | Business/data/model card, migrasi, verifikasi |
| `scripts/` | Builder notebook dan dokumentasi |
| `.github/workflows/` | CI pada main dan pull request |

## Deployment dan arsip

Versi portofolio menggunakan entrypoint `app.py` dan Python 3.12 pada Streamlit Community Cloud. Tautan demo baru ditambahkan setelah deployment berhasil; paket ini tidak mengubah deployment lama.

Original submission: [branch dicoding-submission](https://github.com/mpnabil95/Students-Performance/tree/dicoding-submission) · [release arsip](https://github.com/mpnabil95/Students-Performance/releases/tag/dicoding-submission-v1.0.0).

Ikuti [panduan migrasi](docs/MIGRATION.md) untuk mengganti isi main tanpa mengubah arsip. [Penyelesaian temuan audit](docs/AUDIT_REMEDIATION.md) menjelaskan perubahan dari baseline.

## Sumber, lisensi, dan atribusi

- Konteks pembelajaran: Dicoding, Penerapan Data Science — Menyelesaikan Permasalahan Institusi Pendidikan.
- Dataset: [Dicoding Academy](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance), bersumber dari [UCI](https://doi.org/10.24432/C5MC89).
- Realinho, V., Vieira Martins, M., Machado, J., & Baptista, L. (2021). *Predict Students' Dropout and Academic Success*. UCI Machine Learning Repository.
- Kode: [MIT License](LICENSE). Dataset: CC BY 4.0 sesuai sumber UCI; atribusi data tetap berlaku.
- Pengembang: **Muhammad Pangeran Nabil**.
''')

per_class='\n'.join(f"| {c} | {h['classification_report'][c]['precision']:.3f} | {h['classification_report'][c]['recall']:.3f} | {h['classification_report'][c]['f1-score']:.3f} | {h['classification_report'][c]['support']:.0f} |" for c in ['Dropout','Enrolled','Graduate'])
write('docs/MODEL_CARD.md',f'''
# Model card — semester1-v1.0.0

## Intended use

Demonstrasi pendampingan akademik berdasarkan informasi sampai akhir semester 1. Model memprediksi status pada akhir durasi normal program, bukan waktu dropout. Tidak untuk sanksi, penolakan, atau keputusan akademik otomatis.

## Model dan protokol

- Terpilih: `{m['selected_candidate']}` dari kandidat Logistic Regression, Random Forest, HistGradientBoosting.
- Baseline: DummyClassifier prior; seluruh kandidat dibandingkan pada development 5-fold CV.
- Preprocessing: StandardScaler numerik + OneHotEncoder nominal, selalu di dalam pipeline training.
- Kalibrasi: sigmoid 3-fold, ensemble tiga model.
- Fitur: 14; target tiga kelas string; tidak memakai encoder target terpisah.
- Split: 2.654 development, 885 policy validation, 885 historical holdout.
- Threshold: **{m['threshold']:.2f}**, dipilih dengan F2 pada policy validation. Tie memilih threshold lebih tinggi.
- Tidak dilakukan refit pada seluruh data setelah holdout. Artefak sama dengan yang dievaluasi.

Pemilihan model memakai macro F1; kandidat lain dapat memiliki Brier/AP lebih baik. Prosedur tidak memilih ulang model menggunakan hasil holdout.

## Hasil holdout historis

| Kelas | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
{per_class}

Macro F1 **{h['macro_f1']:.4f}**, accuracy **{h['accuracy']:.2%}**, weighted F1 **{h['weighted_f1']:.4f}**. Dropout Brier **{h['dropout_brier']:.4f}**, average precision **{h['dropout_average_precision']:.4f}**.

Kebijakan peninjauan: recall **{p['recall']:.2%}**, precision **{p['precision']:.2%}**, review rate **{p['review_rate']:.2%}**. TP={p['tp']}, FN={p['fn']}, FP={p['fp']}, TN={p['tn']}. Performa ini berbeda dari confusion matrix argmax multiclass.

## Interval 95% bootstrap

| Ukuran | Batas bawah | Batas atas |
|---|---:|---:|
| Macro F1 | {ci['macro_f1']['lower']:.4f} | {ci['macro_f1']['upper']:.4f} |
| Recall kebijakan | {ci['dropout_recall_policy']['lower']:.4f} | {ci['dropout_recall_policy']['upper']:.4f} |
| Precision kebijakan | {ci['dropout_precision_policy']['lower']:.4f} | {ci['dropout_precision_policy']['upper']:.4f} |

500 resampling baris untuk model tetap. Tidak mencakup variasi training/seleksi atau pergeseran distribusi. Jangan menafsirkan interval sebagai jaminan pada institusi baru.

## Error kelompok

Hasil di `reports/subgroup_metrics.csv` mengukur gender, usia, dan status beasiswa hanya untuk diagnosis. Misalnya, kelompok penerima beasiswa dalam holdout memiliki hanya 19 kejadian Dropout; recall lebih tidak stabil. Gender/beasiswa tidak masuk fitur utama, tetapi error kelompok tetap berbeda. Tidak ada klaim fairness atau mitigasi bias yang sudah terbukti.

## Keterbatasan utama

1. Holdout telah diperiksa pada proyek lama; bukan validation eksternal yang pristine.
2. Tidak ada tanggal kejadian; sebagian mahasiswa mungkin sudah dropout sebelum akhir semester 1.
3. Enrolled belum outcome akhir yang tuntas; recall kelas ini terbatas.
4. Data Portugal tidak otomatis sesuai konteks Indonesia.
5. Review rate sekitar setengah data dapat melampaui kapasitas tim; tidak ada kapasitas nyata yang ditetapkan.
6. Kalibrasi dinilai secara internal; probabilitas tetap dapat meleset pada populasi baru.
7. Permutation importance global bukan efek kausal atau penjelasan individual.
8. Tidak ada bukti dampak intervensi atau manfaat operasional nyata.

## Reproduksi dan pembaruan

Manifest mencatat checksum model/data, daftar fitur, threshold, versi library, dan code hashes. Untuk pembaruan, tetapkan protokol baru sebelum eksperimen, dokumentasikan perubahan, dan evaluasi data baru bila tersedia. Jangan memindahkan threshold berdasarkan hasil holdout lama untuk mempercantik skor.
''')

lines=['# Kamus fitur utama','', 'Sumber kategori: UCI dataset 697 / kamus Dicoding. Urutan di bawah adalah urutan input model.', '', '| Feature | Label | Tipe | Domain | Keterangan |','|---|---|---|---|---|']
for key,v in FIELDS.items():
    domain='Kategori: '+', '.join(str(k) for k in v.options) if v.options else f'{v.low}–{v.high}'
    lines.append(f'| `{key}` | {v.label} | {v.kind} | {domain} | {v.description} |')
for key,v in FIELDS.items():
    if v.options:
        lines.extend(['',f'## {v.label}','','| Kode | Arti |','|---|---|'])
        lines.extend(f'| {k} | {value} |' for k,value in v.options.items())
lines.extend(['','## Validasi bersama','','- Harus finite, tidak kosong, dan bertipe numerik yang sesuai.','- Kategori harus ada di kamus; jumlah unit harus bilangan bulat.','- Approved dan without evaluations tidak boleh melebihi enrolled.','- Evaluations boleh melebihi enrolled karena penilaian berulang.','- Extra columns di CSV tidak digunakan untuk prediksi dan tidak diekspor.','- Batas domain demo tidak ditentukan dari min/max dataset. Nilai yang valid tetapi di luar rentang development diberi peringatan.'])
write('docs/FEATURE_DICTIONARY.md','\n'.join(lines))

write('docs/PORTFOLIO_RELEASE.md',f'''
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

Model terpilih `{m['selected_candidate']}`. Macro F1 holdout historis {h['macro_f1']:.4f}. Pada threshold {m['threshold']:.2f}, recall peninjauan Dropout {p['recall']:.2%}, precision {p['precision']:.2%}, dan review rate {p['review_rate']:.2%}.

Hasil bersifat retrospektif pada holdout yang pernah dilihat dalam proyek lama. Tidak ada klaim validasi eksternal atau keberhasilan intervensi.

## Menjalankan

Gunakan Python 3.12, instal `requirements.txt`, lalu jalankan `streamlit run app.py`. Artefak sudah tersedia; panduan lengkap di README.

## Verifikasi sebelum publikasi release

Periksa `docs/VALIDATION.md`, jalankan CI dan tes Streamlit pada environment lengkap, serta tambahkan tautan deployment hanya setelah berhasil. Teks ini tidak mengklaim pengujian yang belum dijalankan.

Arsip submission tetap pada `dicoding-submission-v1.0.0`.
''')
print('Refreshed README, model card, feature dictionary, and release notes.')
