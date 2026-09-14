# Mengganti main dengan versi portofolio

Paket ini berisi isi root repository yang baru. Arsip Dicoding tetap pada branch `dicoding-submission` dan tag `dicoding-submission-v1.0.0` di commit `1e94b40ed9ff28b29bb04f410f001db268b38579`.

## 1. Siapkan repository lokal

Di terminal dalam folder repository lokal:

```bash
git switch main
git pull --ff-only
git status --short
```

Pastikan perubahan lokal lain telah disimpan sebelum menyalin paket. Jika belum punya clone lokal, gunakan `git clone https://github.com/mpnabil95/Students-Performance.git` lalu masuk ke folder hasil clone.

## 2. Salin isi paket

Ekstrak ZIP ke folder terpisah. Salin **isi folder `students-performance-portfolio`** ke root repository lokal, termasuk `.github`, `.streamlit`, `.gitignore`, `.gitattributes`, dan `.python-version`. Timpa README, notebook, app, dan requirements lama. Jangan membuat atau mengganti folder `.git`.

Hapus file versi lama berikut **dari main saja**, karena alur baru tidak memakainya:

```bash
git rm -r -- model jaya-institute_database
git rm -- data.csv data_cleaned.csv metabase.db.mv.db pangeran_nabil-dashboard.png
```

Jika salah satu path sudah tidak ada, hapus hanya path lain yang masih dilacak. File tersebut tetap dapat diakses melalui tag/branch arsip; tidak perlu dihapus dari riwayat atau memindahkan tag.

## 3. Periksa paket dan jalankan

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Lalu:

```bash
python -m pip install -r requirements-notebook.txt
python -m unittest discover -s tests -v
streamlit run app.py
```

Artefak dan notebook sudah disertakan; tidak wajib melatih ulang sebelum commit. Tes Streamlit yang belum dijalankan di runtime pembuat paket akan dijalankan setelah dependensi tersedia dan juga oleh GitHub Actions.

## 4. Commit sebagai pemilik repository

```bash
git add .
git diff --cached --stat
git status
git commit -m "Build semester-one student success portfolio project"
git push origin main
```

Periksa daftar file sebelum commit. Paket tidak mempunyai riwayat git, commit, atau co-author trailer. Commit dibuat oleh konfigurasi identitas Git milikmu sendiri.

## 5. Deployment

Jika aplikasi submission lama harus tetap tersedia, pastikan deployment lama memakai `dicoding-submission`. Untuk versi portofolio, buat deployment Streamlit Community Cloud dengan repository ini, branch `main`, entrypoint `app.py`, dan Python 3.12. Paket tidak mengubah deployment yang ada.

Jangan menambahkan URL demo baru pada README sebelum deployment berhasil dan tautannya diuji. Setelah berhasil, tambahkan tautan aktual melalui perubahan dokumentasi kecil.

## Release berikutnya

Gunakan judul `v1.0.0 — Student Success Portfolio Edition` untuk release portofolio bila ingin memisahkannya dari tag `dicoding-submission-v1.0.0`. Template catatan tersedia di `PORTFOLIO_RELEASE.md`. Rilis setelah CI dan pemeriksaan aplikasi lulus; jangan mengklaim test UI telah lulus sebelum dijalankan.

Deskripsi release arsip sebelumnya mengandung salinan HTML dan Markdown. Versi Markdown bersih disediakan di `DICODING_RELEASE_CLEAN.md` jika ingin merapikan body release. Tag dan kode arsip tidak perlu berubah.
