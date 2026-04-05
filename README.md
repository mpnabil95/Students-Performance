# Proyek Akhir: Menyelesaikan Permasalahan "Jaya Jaya Institut"

## Business Understanding

Jaya Jaya Institut merupakan sebuah institusi pendidikan tinggi yang telah berdiri sejak tahun 2000 dan memiliki reputasi yang baik dalam menghasilkan lulusan berkualitas. Meskipun demikian, institusi ini masih menghadapi tantangan serius berupa tingginya jumlah mahasiswa yang tidak menyelesaikan studi atau mengalami dropout.

Tingginya angka dropout dapat memberikan dampak negatif bagi institusi, baik dari sisi akademik, operasional, maupun reputasi. Oleh karena itu, pihak institusi membutuhkan solusi berbasis data untuk mendeteksi mahasiswa yang berpotensi dropout sedini mungkin agar dapat diberikan intervensi yang tepat.

Dalam proyek ini, pendekatan data science digunakan untuk memahami faktor-faktor yang berkaitan dengan status akhir mahasiswa, membangun model machine learning untuk memprediksi status mahasiswa, serta menyediakan dashboard interaktif agar pihak institusi dapat memantau performa mahasiswa secara lebih efektif.

### Permasalahan Bisnis

Jaya Jaya Institut ingin menjawab beberapa permasalahan utama berikut:  

1. Faktor apa saja yang paling berkaitan dengan status akhir mahasiswa.
2. Bagaimana cara mengidentifikasi mahasiswa yang berisiko dropout lebih awal.
3. Bagaimana menyediakan sarana monitoring yang mudah dipahami oleh pihak institusi.

### Cakupan Proyek

Proyek ini mencakup beberapa tahapan utama berikut:

1. Melakukan data understanding dan exploratory data analysis untuk memahami pola dalam data mahasiswa.
2. Mengidentifikasi faktor-faktor utama yang berhubungan dengan status mahasiswa.
3. Membangun model machine learning multiclass classification untuk memprediksi status mahasiswa ke dalam tiga kategori: Dropout, Enrolled, dan Graduate.
4. Membuat dashboard interaktif menggunakan Metabase untuk membantu monitoring performa mahasiswa.
5. Mengembangkan prototype sistem prediksi berbasis Streamlit agar solusi machine learning dapat digunakan secara praktis.

### Persiapan

Sumber data: https://raw.githubusercontent.com/mpnabil95/Students-Performance/main/data.csv 


1. Setup Environment

- Membuat virtual environment
    ```
    python -m venv venv
    ```

- Setup environment:  
    ```
    python -m venv venv  
    source venv/bin/activate  # Untuk Linux/Mac  
    venv\Scripts\activate     # Untuk Windows
    ```

- Menginstal seluruh library yang dibutuhkan:  
    ```
    pip install -r requirements.txt
    ```


2. Cara Mengakses Dashboard Metabase

Proyek ini menggunakan Metabase versi v0.59.4. Untuk menjalankan dashboard menggunakan file database yang telah diekspor (metabase.db.mv.db), ikuti langkah-langkah berikut:

- Pastikan Docker sudah terinstal dan berjalan di sistem Anda.

- Buka terminal/Command Prompt dan arahkan ke dalam direktori folder submission ini (tempat file ```metabase.db.mv.db``` berada).

- Jalankan perintah Docker berikut untuk menjalankan container Metabase dan menghubungkannya dengan database lokal:  
    ```
    docker run -d -p 3000:3000 -v "%cd%":/metabase-data -e MB_DB_FILE=/metabase-data/metabase.db metabase/metabase:v0.59.4  
    ```  
  
  Catatan: Untuk pengguna Linux/Mac, ganti ```"%cd%"``` dengan ```$(pwd)```

- Setelah container Docker berhasil berjalan, Anda harus mengimpor database dashboard terlebih dahulu dengan menjalankan langkah berikut:
  - Pertama, klik container yang telah dijalankan tadi di Docker
  - Klik kolom ```Files```
  - Arahkan kursor ke folder ```app```, kemuadian klik kanan pada mouse
  - Klik bagian ```Import```

  <div align="center">
    <img width="400" height="285" alt="Image1" src="https://i.ibb.co.com/5g1YH4dm/Cuplikan-layar-2026-04-03-161500.png" />
  </div>
    
  - Pilih folder ```jaya-institute_database```

  <div align="center">
    <img width="400" height="245" alt="Image2" src="https://i.ibb.co.com/4nhr6Xz1/Cuplikan-layar-2026-04-03-164725.png" />
  </div>

  - Database dashboard kita sudah terimpor jika sudah terlihat seperti gambar berikut 

  <div align="center">
    <img width="400" height="330" alt="Image3" src="https://i.ibb.co.com/G4xdvZ9N/Cuplikan-layar-2026-04-03-161522.png" />
  </div>


- Setelah database diimpor, Anda dapat langsung melihat dashboard tanpa perlu melakukan login dengan membuka tautan localhost publik berikut di browser:
http://localhost:3000/public/dashboard/883b448e-07b2-4366-adec-4100575bb77b

### Cara Menjalankan Prototype Secara Lokal

1. Pastikan seluruh dependency telah terinstal.
2. Jalankan perintah berikut pada terminal:
    ```
    streamlit run app.py
    ```
3. Buka browser pada alamat lokal yang ditampilkan oleh Streamlit.

### Link Deployment Streamlit

- Link aplikasi: https://students-performance-cqhuunms2dwehvf5ymiwpf.streamlit.app


## Business Dashboard

Dashboard dibuat menggunakan Metabase untuk membantu pihak institusi memahami pola performa mahasiswa dan memonitor faktor-faktor yang berkaitan dengan dropout.

Dashboard utama menampilkan beberapa komponen penting berikut:

1. KPI total mahasiswa.
2. KPI jumlah mahasiswa Dropout.
3. KPI jumlah mahasiswa Enrolled.
4. KPI jumlah mahasiswa Graduate.
5. Distribusi status mahasiswa.
6. Perbandingan status mahasiswa berdasarkan tuition fees up to date.
7. Perbandingan status mahasiswa berdasarkan debtor.
8. Perbandingan status mahasiswa berdasarkan scholarship holder.
9. Rata-rata approved units semester 1 per status.
10. Rata-rata grade semester 1 per status.
11. Rata-rata approved units semester 2 per status.
12. Rata-rata grade semester 2 per status.

Melalui dashboard tersebut, pihak institusi dapat dengan cepat melihat:

- proporsi mahasiswa berdasarkan status akhir,
- hubungan antara kondisi finansial dengan risiko dropout,
- perbedaan performa akademik antar kelompok status mahasiswa,
- serta indikator utama yang dapat digunakan sebagai sinyal awal untuk intervensi.


## Conclusion

Berdasarkan hasil analisis data, terdapat beberapa faktor yang paling berkaitan dengan status akhir mahasiswa.

- Faktor akademik menjadi indikator yang sangat kuat, terutama jumlah mata kuliah yang lulus dan nilai rata-rata pada semester 1 dan semester 2. Mahasiswa dengan jumlah mata kuliah lulus yang lebih rendah serta nilai akademik yang lebih rendah cenderung memiliki risiko dropout yang lebih tinggi.

- Selain faktor akademik, faktor finansial juga menunjukkan hubungan yang jelas dengan risiko dropout. Mahasiswa yang memiliki status debtor atau belum up to date dalam pembayaran biaya kuliah cenderung lebih banyak berada pada kelompok dropout dibandingkan mahasiswa yang kondisi finansialnya lebih stabil.

- Model machine learning yang dibangun pada proyek ini mampu memprediksi status mahasiswa ke dalam tiga kategori, yaitu Dropout, Enrolled, dan Graduate. Dengan dukungan dashboard Metabase dan prototype Streamlit, solusi yang dihasilkan tidak hanya memberikan insight analitis, tetapi juga dapat digunakan sebagai sistem peringatan dini untuk membantu institusi mengambil tindakan yang lebih cepat dan tepat.

## Rekomendasi Action Items

Berdasarkan hasil proyek ini, berikut beberapa rekomendasi yang dapat diterapkan oleh Jaya Jaya Institut:

1. Membuat sistem monitoring rutin untuk mahasiswa dengan jumlah mata kuliah lulus yang rendah pada semester 1 dan semester 2.
2. Menyediakan program pendampingan akademik bagi mahasiswa yang menunjukkan penurunan nilai atau performa belajar.
3. Menjalankan intervensi finansial bagi mahasiswa dengan status debtor atau pembayaran biaya kuliah yang belum up to date.
4. Memanfaatkan prototype machine learning sebagai alat bantu untuk mengidentifikasi mahasiswa berisiko tinggi secara lebih dini.
5. Mengintegrasikan dashboard monitoring ke proses evaluasi akademik berkala agar keputusan yang diambil lebih berbasis data.
6. Melakukan evaluasi lanjutan secara periodik terhadap performa model agar sistem prediksi tetap relevan jika terdapat perubahan pola mahasiswa di masa depan.