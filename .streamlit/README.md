# Streamlit — konfigurasi aplikasi

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Folder ini mengatur tampilan dan perilaku dasar Streamlit. Logika prediksi berada di paket `student_success`, sedangkan halaman aplikasi berada di `app.py`.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [config.toml](config.toml) | Tema terang, warna, font, batas unggahan 10 MB, dan pengaturan statistik penggunaan. | Menyesuaikan tampilan aplikasi atau konfigurasi server. |


## Catatan penyuntingan

`config.toml` dapat disunting manual. Mengubah batas unggahan di sini saja tidak mengubah batas parser CSV atau batas jumlah baris di `student_success/schema.py`; aturan perlu diselaraskan bila kebutuhan berubah.

`gatherUsageStats = false` menonaktifkan pengumpulan statistik penggunaan oleh Streamlit melalui pengaturan tersebut. Itu bukan jaminan privasi menyeluruh untuk layanan hosting.

Aplikasi saat ini tidak membutuhkan API key. Jika kelak memakai secrets, jangan commit `secrets.toml`; path tersebut sudah dikecualikan oleh `.gitignore`. [Panduan aplikasi](../docs/APP_GUIDE.md) menjelaskan alur penggunaan.
