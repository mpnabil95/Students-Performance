# Artifacts — model yang siap dipakai

[Kembali ke README utama](../README.md) · [Peta repository](../docs/REPOSITORY_GUIDE.md)

Folder ini menyimpan hasil training yang diperlukan aplikasi untuk membuat prediksi. Bayangkan model sebagai alat yang sudah dilatih, manifest sebagai kartu identitasnya, dan schema sebagai petunjuk masukan. Mulai dari `manifest.json` bila hanya ingin mengetahui model yang digunakan.

## Isi folder

| File atau folder | Fungsi | Kapan dibuka |
|---|---|---|
| [model.joblib](model.joblib) | Model terlatih yang mencakup preprocessing, classifier, dan kalibrasi probabilitas. | Dibaca otomatis oleh aplikasi; bukan dokumen untuk dibuka sebagai teks. |
| [manifest.json](manifest.json) | Identitas model: versi, fitur, kelas, threshold, versi library, checksum, split, dan rentang development. | Memeriksa model, lingkungan yang sesuai, atau penyebab ketidakcocokan artefak. |
| [feature_schema.json](feature_schema.json) | Salinan aturan 14 fitur: label, tipe, domain, dan kode kategori. | Integrasi atau audit input; versi ramah pembaca ada di kamus fitur. |


## Hubungan antar-file

Ketiga file dibentuk oleh `python -m student_success.train` atau training yang dijalankan builder notebook. `student_success/inference.py` memuat model dan manifest, lalu memeriksa checksum model, daftar fitur, kelas, dan versi scikit-learn. Form dan validasi utama memakai `student_success/schema.py`; `feature_schema.json` adalah snapshot yang dapat dibaca alat lain.

**Jangan menyunting model, threshold dalam manifest, atau schema keluaran secara terpisah.** Untuk mengubah perilaku model, ikuti [protokol pembaruan](../docs/REPRODUCIBILITY.md), lalu hasilkan dan commit keluaran yang konsisten. File `.joblib` hanya boleh dimuat dari sumber tepercaya karena format ini dapat mengeksekusi kode saat dibaca.

- Pengguna aplikasi: tidak perlu membuka folder ini.
- Reviewer: mulai dari [model card](../docs/MODEL_CARD.md).
- Pengembang: lanjut ke [kode inference](../student_success/inference.py).
