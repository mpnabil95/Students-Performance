# Workflows — Project quality

[Kembali ke README utama](../../README.md) · [Peta repository](../../docs/REPOSITORY_GUIDE.md)

Workflow adalah urutan perintah yang dijalankan oleh GitHub Actions pada lingkungan sementara.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [quality.yml](quality.yml) | CI Python 3.12: instalasi dependensi, import, builder notebook/dokumen, tests, dan validasi nbformat. | Memeriksa pemicu, lingkungan, atau tahapan otomatisasi. |


## Pemicu dan alur

Workflow berjalan pada push ke `main`, pull request menuju `main`, atau pemicu manual. Job `contracts-and-app` memakai Ubuntu dan Python 3.12 dengan izin repository `contents: read`.

1. Mengambil isi repository dan menyiapkan Python.
2. Memasang `requirements-notebook.txt` serta memeriksa import Streamlit/nbformat.
3. Membangun notebook; tahap ini menjalankan training kembali.
4. Menyegarkan dokumentasi dari hasil training di lingkungan CI.
5. Menjalankan unit/integration tests.
6. Memvalidasi struktur notebook melalui nbformat.

Keluaran training dan perubahan dokumen pada runner bersifat sementara. Workflow ini **tidak melakukan commit/push, mengunggah artefak hasil run, atau deployment**. Karena itu metrik baru pada runner tidak otomatis mengganti laporan yang tersimpan di repository.

Jika gagal, buka langkah merah di Actions dan baca error pertama yang relevan. Cakupan dan bukti run terdahulu ada di [README tests](../../tests/README.md).
