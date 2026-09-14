# Catatan verifikasi paket awal

Dokumen ini merekam lingkungan pembuatan paket awal. Pemeriksaan berikutnya pada GitHub Actions untuk commit `6a5c888` telah berhasil, termasuk 16 tes tanpa skip dan validasi nbformat. [Bukti CI historis dan batas cakupannya](../tests/README.md#bukti-ci-historis) menjelaskan perbedaannya. Catatan lokal awal di bawah dipertahankan sebagai riwayat.

## Pemeriksaan yang benar-benar dijalankan

- Training CV, kalibrasi, threshold selection, holdout evaluation, dan diagnostik selesai.
- Notebook dibangun dan dieksekusi: 41 sel total, 28 sel kode, tanpa output error.
- Seluruh kode Python berhasil diparse.
- Unit/integration suite: 16 tes ditemukan, 15 lulus, 1 dilewati, tanpa failure/error.
- Core tests mencakup domain invalid, file kosong, kategori tidak sah, relasi akademik, preset, threshold boundary, urutan fitur, CSV delimiter, pemuatan artefak, integritas split, dan kesetaraan batch/individu.
- Model yang dimuat kembali menghasilkan probabilitas yang sama dengan file prediksi holdout.
- Checksum dataset/model serta hash sumber modul sesuai manifest.
- Figur data dan evaluasi dibangun dari hasil aktual; sebagian figur diperiksa secara visual.

## Batas verifikasi

- Tes Streamlit/AppTest: **belum dapat dijalankan: Streamlit tidak tersedia pada runtime pembuat paket**.
- Pemasangan Streamlit dan dependensi notebook melalui pip tidak berhasil pada lingkungan pembuatan. Ini tidak membuktikan paket tersebut tidak tersedia pada komputer pengguna.
- Builder notebook menggunakan eksekusi Python biasa; validasi native nbformat/Jupyter serta antarmuka interaktif belum dijalankan pada runtime ini.
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

Pada saat paket awal dibangun, CI belum dijalankan. Run berikutnya telah berhasil; lihat [README tests](../tests/README.md#bukti-ci-historis). `reports/verification.json` tetap merekam pemeriksaan lokal awal dan tidak otomatis diperbarui oleh CI.

Hasil mesin: `reports/verification.json`. Jalankan `python scripts/verify_package.py` untuk memperbarui catatan pemeriksaan lokal setelah menyiapkan environment lengkap.
