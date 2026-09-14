# Raw data — snapshot asli

[Kembali ke README utama](../../README.md) · [Peta repository](../../docs/REPOSITORY_GUIDE.md)

Ini adalah titik awal eksperimen. Satu baris merepresentasikan satu mahasiswa dalam dataset historis; nomor baris bukan identitas mahasiswa nyata.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [students.csv](students.csv) | 4.424 baris × 37 kolom, termasuk label `Status`; pemisah titik koma (`;`). | Membaca data mentah atau memeriksa asal sebuah baris evaluasi. |


## Cara membuka

GitHub atau editor teks dapat dipakai untuk melihat header. Di spreadsheet, impor sebagai UTF-8 dengan pemisah titik koma dan hindari menyimpan ulang file sumber. Dengan pandas:

```python
import pandas as pd

data = pd.read_csv("data/raw/students.csv", sep=";")
print(data.shape)
```

Jalankan contoh dari root repository. Data ini berbeda dari template batch: dataset memuat target dan kolom lain, sementara template hanya berisi 14 fitur masukan.

## Penelusuran dan integritas

`source_row` pada laporan eksperimen menghitung baris data mulai dari **1**, tidak termasuk header. Misalnya, `source_row = 1` berarti mahasiswa pada baris data pertama di file ini. Indeks pandas untuk baris tersebut adalah `0`.

Checksum SHA-256 yang diharapkan:

```text
a37dbda5555089a8d39b8be6f1a242f68403963a1f81423115c9676c8c7100e9
```

Aturan akhir baris CSV dijaga oleh `.gitattributes`. Jangan mengedit atau memformat ulang CSV ini hanya untuk merapikan tampilannya. [Data card](../../docs/DATA_CARD.md) memuat sumber dan lisensi CC BY 4.0.
