# Tests — apa yang diperiksa otomatis

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Tes otomatis memeriksa perilaku tertentu dengan input yang hasilnya dapat diperiksa. Tes membantu menemukan regresi, tetapi tidak membuktikan bahwa prediksi akan berhasil di semua kampus atau bahwa seluruh interaksi browser sudah diuji.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [test_contracts.py](test_contracts.py) | 15 tes data/artefak: input invalid, domain, relasi akademik, CSV, preset, threshold, split, dan konsistensi inference. | Memeriksa aturan inti atau menambahkan tes untuk perubahan perilaku model/input. |
| [test_streamlit.py](test_streamlit.py) | 8 tes AppTest: navigasi, filter kosong/reset, kesetaraan inference, penolakan input, pergantian preset, pemulihan profil, batch sintetis/filter, dan pergantian sumber batch. | Memeriksa aplikasi dapat dimuat dan alur individu dasar berjalan. |


## Menjalankan

Dari root repository, dengan environment aktif:

```bash
python -m pip install -r requirements-notebook.txt
python -m unittest discover -s tests -v
```

Di Windows tanpa aktivasi, gunakan `.\.venv\Scripts\python.exe` sebagai pengganti `python`. Jika Streamlit tidak terpasang, tes AppTest **dilewati**; periksa jumlah skipped, jangan hanya melihat kata OK.

## Bukti CI historis

[Project quality — run 34809204916](https://github.com/mpnabil95/student-success-prediction/actions/runs/34809204916), commit `6a5c888dea0b6719f248ed0454c61ee6da68c5d6`, diperiksa pada 14 September 2026:

| Pemeriksaan | Bukti |
|---|---|
| Instalasi dependensi notebook dan import Streamlit/nbformat | Langkah workflow berhasil |
| Pembangunan dan eksekusi notebook melalui builder | Langkah workflow berhasil |
| Pembangunan dokumen | Langkah workflow berhasil |
| Unit/integration tests | Log menunjukkan 16 tes, OK; AppTest dijalankan dan tidak dilewati |
| Validasi struktur notebook dengan nbformat | Langkah workflow berhasil |

Bukti di atas berlaku untuk commit tersebut. Status terbaru dapat diperiksa di [Actions](https://github.com/mpnabil95/student-success-prediction/actions). Berkas `reports/verification.json` yang masih tersimpan merekam pemeriksaan lokal saat paket dibangun; itu tidak otomatis diperbarui oleh workflow.

## Yang belum dicakup

Tes AppTest memeriksa batch sintetis dan perubahan state, tetapi belum menguji pemilih file unggahan dan unduhan dari browser sesungguhnya, tata letak pada berbagai ukuran layar, atau deployment publik. Builder menjalankan kode notebook melalui Python; validasi nbformat memeriksa format dan bukan eksekusi ulang melalui kernel Jupyter terpisah. CI juga belum menjalankan audit kerentanan dependensi.

[Workflow](../.github/workflows/README.md) · [Catatan paket awal](../docs/VALIDATION.md) · [Batas model](../docs/MODEL_CARD.md)

## Verifikasi pembaruan antarmuka

Pada 14 September 2026, suite lokal terdiri dari **23 tes: 15 kontrak data/model dan 8 tes antarmuka**. Seluruhnya lulus tanpa skip. Keberhasilan CI 16 tes pada bagian historis tetap berlaku untuk commit lamanya; perubahan antarmuka ini perlu melewati CI lagi setelah di-commit. [Catatan UI](../docs/UI_CHANGELOG.md) memuat cakupan dan batas verifikasi.
