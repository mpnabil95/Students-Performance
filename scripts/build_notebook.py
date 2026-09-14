"""Generate AND execute the notebook with a small Python executor (no network).

All cells use ordinary Python. In Jupyter they run normally with the Python 3
kernel. This builder records real stdout, DataFrame results and matplotlib
figures; it does not fabricate execution outputs. Failure stops the build.
"""
import ast
import base64
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0,str(ROOT))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')
os.environ.setdefault('MPLBACKEND','Agg')

cells=[]
def md(text):cells.append({'cell_type':'markdown','metadata':{},'source':text.strip().splitlines(keepends=True),'id':f'cell-{len(cells)+1:03}'})
def code(text):cells.append({'cell_type':'code','metadata':{},'source':text.strip().splitlines(keepends=True),'id':f'cell-{len(cells)+1:03}','execution_count':None,'outputs':[]})

md('''# Student Success — Semester-One Review Support

**Muhammad Pangeran Nabil · Portfolio edition v1.0.0**

Studi kasus prediksi retrospektif status mahasiswa dengan informasi sampai akhir semester 1.
Konteks bisnis: Jaya Jaya Institut (fiktif). Sumber: dataset UCI melalui Dicoding.

Notebook ini menjalankan training, seleksi, kalibrasi, evaluasi, dan pemuatan ulang artefak dari kode yang sama dengan aplikasi.
Original submission tersimpan pada tag `dicoding-submission-v1.0.0`.

**Navigasi:** 1. Business understanding · 2. Data & quality · 3. Protocol · 4. Modeling ·
5. Evaluation · 6. Interpretation · 7. Deployment · 8. Conclusions & actions.''')
md('''## 1. Business understanding

Tujuan: membantu dosen wali menyusun prioritas peninjauan setelah semester 1. Prediksi tidak menjadi keputusan akademik otomatis.

| Keputusan desain | Pilihan |
|---|---|
| Unit analisis | Satu mahasiswa dalam dataset historis |
| Input utama | 14 fitur pendaftaran dan akademik semester 1 |
| Target | Dropout / Enrolled / Graduate pada akhir durasi normal program |
| Pengguna | Dosen wali / tim pendampingan dalam demonstrasi |
| Output | Distribusi probabilitas, kelas paling mungkin, kategori peninjauan |
| Seleksi model | Mean macro F1 pada 5-fold CV |
| Kebijakan peninjauan | Threshold yang memaksimalkan Dropout F2 pada policy validation |

**Batas temporal:** tanggal dropout tidak tersedia. “Akhir semester 1” menjelaskan himpunan fitur, bukan bukti bahwa setiap prediksi dibuat sebelum dropout. Enrolled bukan label aman. Keberhasilan intervensi belum diukur.

F2 dipakai sebagai asumsi desain demo yang memberi perhatian lebih besar pada recall. Precision dan beban peninjauan tetap dilaporkan; belum ada kapasitas tim yang disepakati institusi nyata.''')
code('''import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(key, '1')
import sys
from pathlib import Path
ROOT = Path.cwd()
if not (ROOT / 'student_success').is_dir():
    raise RuntimeError('Jalankan notebook dari root repository.')
sys.path.insert(0, str(ROOT))
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from student_success.config import DATA_PATH, ARTIFACT_DIR, REPORT_DIR, PROTOCOL, CLASS_NAMES
from student_success.schema import FEATURES, FIELDS, schema_records, validate_features, defaults
from student_success.inference import load_bundle, predict_frame, sha256_file
from student_success.train import EXPECTED_DATA_SHA256
print('Python:', sys.version.split()[0])
print('Project:', ROOT.name)''')
md('''## 2. Data understanding dan pemeriksaan kualitas

Data asli dipertahankan pada `data/raw/students.csv`, delimiter titik koma, checksum dikunci.
Sumber awal menggabungkan data pendaftaran dengan hasil semester 1 dan 2; target mencatat outcome pada akhir durasi normal program.

Referensi primer: [UCI dataset 697](https://doi.org/10.24432/C5MC89).
Kolom kategori berkode angka tidak otomatis bermakna kontinu. Unknown pada pendidikan orang tua merupakan ketidaklengkapan semantik meskipun tidak berupa null.''')
code('''data = pd.read_csv(DATA_PATH, sep=';')
assert sha256_file(DATA_PATH) == EXPECTED_DATA_SHA256
X_checked = validate_features(data)
print('Raw shape:', data.shape)
print('Selected features:', X_checked.shape)
print('Null eksplisit:', int(data.isna().sum().sum()))
print('Duplikat penuh:', int(data.duplicated().sum()))
data.head()''')
code('''quality = pd.DataFrame({
    'dtype': data.dtypes.astype(str),
    'unique': data.nunique(),
    'nulls': data.isna().sum(),
})
quality''')
code('''counts = data.Status.value_counts().reindex(CLASS_NAMES)
pd.DataFrame({'count': counts, 'proportion': counts / len(data)})''')
code('''unknowns = {c: int(data[c].eq(34).sum()) for c in ['Mothers_qualification', 'Fathers_qualification']}
print('Kode Unknown (34):', unknowns)
print('Kolom tersebut tidak digunakan model utama; tetap dicatat pada kualitas sumber.')''')
md('''### Kontrak fitur dan alasan eksklusi

Fitur kategori diberi mapping eksplisit. Nilai sumber memakai skala 0–200 untuk nilai masuk, 0–20 untuk nilai semester; curricular units tidak otomatis identik dengan SKS Indonesia.

Enam fitur semester 2 dikeluarkan. Debtor, Tuition_fees_up_to_date, Scholarship_holder, dan indikator makroekonomi dikeluarkan karena timing operasional tidak cukup jelas untuk kontrak prediksi ini. Gender, kebangsaan, marital status, kebutuhan khusus, pekerjaan/kualifikasi orang tua, displaced dan international tidak diperlukan dalam formulir utama. Eksklusi atribut langsung tidak menjamin tidak ada bias melalui fitur lain.

Batas usia 16–100 dan batas jumlah unit/evaluasi merupakan kontrak operasional demo, bukan klaim batas universal institusi. Nilai valid yang di luar rentang pengembangan diberi peringatan.''')
code('''schema_table = pd.DataFrame(schema_records())
schema_table[['feature', 'label', 'kind', 'low', 'high', 'group']]''')
code('''excluded = [c for c in data if c not in FEATURES and c != 'Status']
print('Fitur terpilih:', len(FEATURES))
print('Fitur tidak dipakai:', len(excluded))
pd.DataFrame({'excluded_feature': excluded})''')
md('''## 3. Desain evaluasi sebelum pemodelan

| Bagian | Jumlah | Fungsi |
|---|---:|---|
| Model development | 2.654 | Seleksi 5-fold CV, kemudian fit model terpilih |
| Policy validation | 885 | Memilih threshold peninjauan |
| Historical holdout | 885 | Pelaporan akhir setelah model dan threshold dikunci |

Holdout memakai pembagian historis 20% dari submission. Data tersebut telah dilihat sebelumnya, sehingga hasil tidak disebut evaluasi eksternal yang independen.

Preprocessing dan kalibrasi 3-fold berada **di dalam** setiap fold training. Pemilihan tiga kandidat menggunakan macro F1; Dummy merupakan baseline. Setelah pemilihan, model difit pada 60% data, threshold ditetapkan dari validation, kemudian holdout dibuka. Tidak dilakukan refit seluruh data setelah melihat holdout.

Logistic admission-only merupakan pembanding tetap untuk melihat manfaat fitur akademik awal, bukan kandidat tambahan yang dipilih setelah melihat holdout.''')
code('''pd.DataFrame({'setting': list(PROTOCOL), 'value': [str(v) for v in PROTOCOL.values()]})''')
md('''## 4. Training, seleksi model, dan kalibrasi

Kode implementasi berada di `student_success/modeling.py` dan `student_success/train.py`.
Logistic Regression memakai scaling dan encoding; model pohon menggunakan pipeline encoding yang sama agar kategori nominal tidak dianggap sebagai kode ordinal.

Semua kandidat memakai kalibrasi sigmoid melalui cross-validation. Warning konvergensi diperlakukan sebagai kegagalan yang harus diperbaiki, bukan disembunyikan.

Sel berikut benar-benar menjalankan pipeline. Set `RETRAIN=False` hanya bila ingin membaca hasil paket tanpa melatih ulang; build notebook yang dikirim menjalankan `RETRAIN=True`.''')
code('''RETRAIN = True
if RETRAIN:
    import io
    import contextlib
    from student_success.train import main as train_project
    training_output = io.StringIO()
    with contextlib.redirect_stdout(training_output):
        train_project()
    print('Pipeline selesai: CV, kalibrasi, threshold validation, holdout, diagnostik, dan artefak.')
else:
    print('Mode membaca artefak yang sudah disediakan.')
metrics = json.loads((REPORT_DIR / 'metrics.json').read_text())
manifest = json.loads((ARTIFACT_DIR / 'manifest.json').read_text())
print('Model terpilih:', metrics['selected_candidate'])
print('Threshold peninjauan:', metrics['threshold'])''')
code('''comparison = pd.read_csv(REPORT_DIR / 'model_comparison.csv')
comparison.sort_values('macro_f1_mean', ascending=False)''')
code('''def show_plot(name):
    image = plt.imread(REPORT_DIR / 'figures' / f'{name}.png')
    height, width = image.shape[:2]
    fig, ax = plt.subplots(figsize=(11, 11 * height / width))
    ax.imshow(image)
    ax.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    plt.close(fig)

show_plot('model_selection')''')
code('''print('Admission-only Logistic CV macro F1:', round(metrics['admission_logistic_cv_macro_f1'], 4))
print('Semester-1 Logistic CV macro F1:', round(float(comparison.loc[comparison.candidate == 'logistic', 'macro_f1_mean'].iloc[0]), 4))
print('Perbandingan ini memakai algoritma/prosedur yang sama dan subset fitur yang berbeda.')''')
md('''### Pemilihan threshold

Kategori “Perlu peninjauan” memakai satu kondisi: probabilitas Dropout ≥ threshold terpilih. Kelas argmax tidak mengubah aturan ini. Saran akademik tambahan merupakan aturan pendampingan yang terpisah dari model.

Pemaksimalan F2 tidak membatasi jumlah kasus yang ditandai. Karena itu, review rate menjadi bagian hasil wajib; jika kapasitas institusi berbeda, threshold harus ditentukan ulang pada validasi sesuai kapasitas, lalu diuji dengan protokol baru.''')
code('''thresholds = pd.read_csv(REPORT_DIR / 'threshold_analysis.csv')
thresholds.sort_values(['f2', 'threshold'], ascending=False).head(10)''')
code("show_plot('threshold_selection')")
md('''## 5. Evaluasi pada holdout historis

Hasil di bawah dihitung setelah model dan threshold dikunci. Accuracy/macro F1 mengukur klasifikasi tiga kelas; precision/recall kebijakan mengukur peninjauan Dropout berdasarkan threshold. Keduanya tidak boleh disamakan.''')
code('''holdout = metrics['historical_holdout']
summary = {k: holdout[k] for k in ['n', 'accuracy', 'macro_f1', 'weighted_f1', 'log_loss', 'dropout_brier', 'dropout_average_precision']}
pd.DataFrame.from_dict(summary, orient='index', columns=['value'])''')
code('''pd.DataFrame(holdout['classification_report']).T''')
code("show_plot('confusion_matrix')")
code('''policy = holdout['policy']
print(f"Terdeteksi: {policy['tp']} dari {policy['tp'] + policy['fn']} Dropout")
print(f"Terlewat: {policy['fn']} | False positive: {policy['fp']}")
print(f"Perlu peninjauan: {policy['review_count']} dari {holdout['n']} ({policy['review_rate']:.1%})")
pd.DataFrame.from_dict(policy, orient='index', columns=['value'])''')
code("show_plot('precision_recall')")
code("show_plot('calibration')")
md('''Kalibrasi tidak membuat probabilitas menjadi pasti. Reliability curve serta Brier score memberi bukti kesesuaian probabilitas dengan frekuensi pada data evaluasi. Dataset eksternal masih diperlukan.

Interval berikut menggunakan bootstrap baris untuk **model tetap**, 500 resampling. Interval tidak mencakup ketidakpastian training, pemilihan model, threshold, atau pergeseran institusi.''')
code('''pd.DataFrame(metrics['bootstrap_95ci']).T''')
md('''## 6. Interpretasi, error kelompok, dan pola data

Permutation importance mengukur perubahan macro F1 saat satu fitur diacak. Nilai ini bukan efek kausal dan bukan penjelasan individual; korelasi antarfitur dapat membagi kontribusi. Diagnostik dilakukan setelah model dibekukan dan tidak digunakan untuk mengubah model dalam versi ini.''')
code("show_plot('feature_importance')")
code('''groups = pd.read_csv(REPORT_DIR / 'subgroup_metrics.csv')
groups[['attribute','group','n','actual_dropout','precision','recall','review_rate','fn']]''')
md('''Perbedaan error antar kelompok perlu diperhatikan sebelum penggunaan nyata. Gender dan scholarship tidak masuk model, tetapi perbedaan masih dapat muncul melalui distribusi fitur lain. Pada kelompok dengan sedikit kejadian Dropout, estimasi recall lebih tidak stabil. Tabel ini belum merupakan sertifikasi fairness.

Grafik berikut menggambarkan seluruh dataset historis. Grafik digunakan sebagai konteks deskriptif, bukan sebagai bukti kausal atau dasar memilih parameter model setelah evaluasi.''')
code("show_plot('status_distribution')")
code("show_plot('academic_patterns')")
code('''correlations = {
    sem: data[f'Curricular_units_{sem}_sem_enrolled'].corr(data[f'Curricular_units_{sem}_sem_approved'])
    for sem in ['1st','2nd']
}
print('Korelasi enrolled–approved (audit ulang):', correlations)
print('Semester 2 ditampilkan hanya untuk koreksi dokumentasi historis, tidak masuk model utama.')''')
md('''## 7. Deployment dan kesetaraan training–inference

Satu model terkalibrasi dan manifest dipakai oleh prediksi individu maupun batch. Aplikasi memverifikasi checksum, versi scikit-learn, urutan fitur, serta kelas. Pengguna hanya mengunggah CSV, tidak mengunggah pickle.

Artefak disimpan langsung ke `artifacts/`; tidak ada langkah memindahkan model dari root secara manual.
Dashboard historis berada dalam aplikasi Streamlit yang sama sehingga tidak memerlukan database Metabase untuk versi portofolio.''')
code('''model, bundle = load_bundle()
examples = pd.read_csv(ROOT / 'examples' / 'students_template.csv')
results = predict_frame(examples, model, bundle)
results[['source_row','predicted_status','prob_dropout','prob_enrolled','prob_graduate','action']]''')
code('''single = predict_frame(examples.iloc[[0]], model, bundle)
np.testing.assert_allclose(single[['prob_dropout','prob_enrolled','prob_graduate']].to_numpy(), results[['prob_dropout','prob_enrolled','prob_graduate']].iloc[[0]].to_numpy())
print('PASS: prediksi individu sama dengan baris yang sesuai pada batch.')
print('Model checksum:', bundle['model_sha256'])
pd.DataFrame.from_dict(bundle['environment'], orient='index', columns=['version'])''')
md('''## 8. Kesimpulan dan rekomendasi tindakan

Proyek menyediakan baseline semester-1 dengan seleksi model, kalibrasi, threshold validation, dan inference yang konsisten. Hasil menunjukkan trade-off nyata antara mahasiswa Dropout yang terdeteksi dan jumlah profil yang perlu ditinjau.

**Interpretasi hasil:** model semester-1 tidak boleh diklaim mengungguli atau kalah secara setara terhadap model lama yang memakai informasi semester-2/finansial. Tujuan, fitur, dan proses evaluasi berbeda.

| Usulan tindakan | Penanggung jawab calon | Ukuran evaluasi yang perlu dikumpulkan |
|---|---|---|
| Meninjau profil yang ditandai | Dosen wali | Waktu peninjauan, proporsi kasus yang dikonfirmasi memerlukan bantuan |
| Mengklarifikasi unit tanpa evaluasi | Dosen wali dan administrasi | Kelengkapan catatan, alasan ketidakhadiran |
| Menawarkan pendampingan akademik | Tim akademik | Partisipasi, capaian pada evaluasi berikutnya |
| Menguji pilot pada cohort baru | Tim data dan institusi | Recall, precision, workload, kalibrasi, error kelompok |

Semua tindakan merupakan rancangan pilot, bukan hasil intervensi yang sudah terbukti. Manfaat aktual harus diukur setelah memperoleh data dan persetujuan institusi terkait.

**Menjalankan aplikasi:** `streamlit run app.py` setelah instalasi `requirements.txt`.
**Pengujian:** `python -m unittest discover -s tests -v`.
**Dokumentasi:** README, `docs/BUSINESS_CASE.md`, `docs/MODEL_CARD.md`, `docs/DATA_CARD.md`, dan `docs/MIGRATION.md`.

Referensi: [UCI](https://doi.org/10.24432/C5MC89), [Dicoding dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/students_performance), [scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html), [calibration](https://scikit-learn.org/stable/modules/calibration.html).
''')

namespace={'__name__':'__notebook__'}
import matplotlib.pyplot as plt
count=0
for cell_index,cell in enumerate(cells,1):
    if cell['cell_type']!='code':continue
    count+=1;cell['execution_count']=count;outputs=[];stdout=io.StringIO();cursor=0
    def flush_stream():
        global cursor
        value=stdout.getvalue()[cursor:]
        if value:outputs.append({'output_type':'stream','name':'stdout','text':value})
        cursor=stdout.tell()
    def capture_show(*args,**kwargs):
        flush_stream()
        for number in plt.get_fignums():
            buffer=io.BytesIO();plt.figure(number).savefig(buffer,format='png',dpi=130,bbox_inches='tight')
            outputs.append({'output_type':'display_data','metadata':{},'data':{'image/png':base64.b64encode(buffer.getvalue()).decode(),'text/plain':'<matplotlib figure>'}})
    old_show=plt.show;plt.show=capture_show
    source=''.join(cell['source']);tree=ast.parse(source)
    tail=tree.body.pop() if tree.body and isinstance(tree.body[-1],ast.Expr) else None
    try:
        with contextlib.redirect_stdout(stdout),contextlib.redirect_stderr(stdout):
            exec(compile(tree,f'<notebook-cell-{cell_index}>','exec'),namespace)
            if tail is not None:
                value=eval(compile(ast.Expression(tail.value),f'<notebook-cell-{cell_index}>','eval'),namespace)
                flush_stream()
                if value is not None:
                    mime={'text/plain':repr(value)}
                    if hasattr(value,'to_html'):mime['text/html']=value.to_html()
                    outputs.append({'output_type':'execute_result','execution_count':count,'metadata':{},'data':mime})
            flush_stream()
    finally:plt.show=old_show
    cell['outputs']=outputs
    print(f'Executed code cell {count} (notebook cell {cell_index})',flush=True)

notebook={'nbformat':4,'nbformat_minor':5,'cells':cells,'metadata':{
    'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},
    'language_info':{'name':'python','version':sys.version.split()[0]},
    'execution':{'method':'scripts/build_notebook.py — real sequential Python execution','retrained':True},
}}
(ROOT/'notebook.ipynb').write_text(json.dumps(notebook,indent=1,ensure_ascii=False)+'\n',encoding='utf-8')
print(f'Saved notebook: {len(cells)} total cells; {count} code cells executed.',flush=True)
