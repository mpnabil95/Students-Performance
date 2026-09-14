# GitHub — otomatisasi repository

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Folder ini berisi konfigurasi khusus GitHub. Otomatisasi menjalankan pemeriksaan ketika kode diperbarui sehingga hasilnya dapat diperiksa melalui tab Actions.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [workflows/](workflows/README.md) | Definisi workflow Project quality. | Memahami kapan dan bagaimana GitHub menguji project. |


## Untuk pembaca umum

Buka [Actions](https://github.com/mpnabil95/student-success-prediction/actions) untuk melihat hasil. Tanda berhasil berarti langkah workflow pada commit tersebut selesai; baca [cakupan tes](../tests/README.md) untuk mengetahui batas pemeriksaannya.

Tidak ada workflow deployment pada versi ini. Memperbarui konfigurasi di folder ini tidak melatih model yang sedang berjalan pada layanan eksternal atau menerbitkan release secara otomatis.
