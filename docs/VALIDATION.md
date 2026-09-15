# Catatan verifikasi paket

Training, notebook, dan figur di bawah berasal dari pembangunan model sebelumnya. Pada pembaruan UI 14 September 2026, artefak tersebut dipertahankan dan diverifikasi; tidak ada training ulang. Suite lokal dijalankan kembali dengan 23 tes. [Catatan UI](UI_CHANGELOG.md) merinci perubahan dan batas pemeriksaan visual.

## Pemeriksaan yang benar-benar dijalankan

- Training CV, kalibrasi, threshold selection, holdout evaluation, dan diagnostik selesai.
- Notebook dibangun dan dieksekusi: 41 sel total, 28 sel kode, tanpa output error.
- Seluruh kode Python berhasil diparse.
- Unit/integration suite: 23 tes ditemukan, 23 lulus, 0 dilewati, tanpa failure/error.
- Core tests mencakup domain invalid, file kosong, kategori tidak sah, relasi akademik, preset, threshold boundary, urutan fitur, CSV delimiter, pemuatan artefak, integritas split, dan kesetaraan batch/individu.
- Model yang dimuat kembali menghasilkan probabilitas yang sama dengan file prediksi holdout.
- Checksum dataset/model serta hash sumber modul sesuai manifest.
- Figur data dan evaluasi dibangun dari hasil aktual; sebagian figur diperiksa secara visual.

## Batas verifikasi

- Tes Streamlit/AppTest: **sudah dijalankan**.
- Instalasi dependensi tidak diaudit oleh script ini. Ketersediaan Streamlit dicatat pada hasil lokal; lihat bukti CI secara terpisah.
- Pemeriksa ini membaca struktur JSON dan sintaks sel; tidak menjalankan validasi nbformat atau eksekusi kernel Jupyter. Builder notebook menggunakan eksekusi Python biasa. Validasi format nbformat memiliki langkah tersendiri di CI.
- Tampilan browser, alur unggah-unduh melalui browser, dan deployment Streamlit belum diuji langsung.
- Database Metabase lama tidak dipulihkan; dashboard portofolio menggunakan CSV dan Streamlit.
- Holdout historis sudah dilihat pada proyek lama; tidak ada validasi institusi eksternal.

## Pengujian pada lingkungan lengkap

```bash
python -m pip install -r requirements-notebook.txt
python -m unittest discover -s tests -v
python -c "import nbformat; nbformat.validate(nbformat.read('notebook.ipynb', as_version=4))"
streamlit run app.py
```

Catatan di atas merekam lingkungan verifikasi lokal dan tidak menunjukkan status GitHub Actions terbaru. Workflow `.github/workflows/quality.yml` menjalankan pembangunan notebook/dokumen, tests, serta validasi format notebook. Bukti run CI historis yang diperiksa tersedia pada [README tests](../tests/README.md#bukti-ci-historis).

Hasil mesin: `reports/verification.json`. Jalankan `python scripts/verify_package.py` untuk memperbarui catatan pemeriksaan lokal setelah menyiapkan environment lengkap.
