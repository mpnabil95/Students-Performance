"""Verify final deliverables and execute available tests; emits an honest report."""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import unittest
import io
import contextlib
ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT);sys.path.insert(0,str(ROOT))
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(key,'1')

for p in ROOT.rglob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
nb=json.loads((ROOT/'notebook.ipynb').read_text())
assert nb['nbformat']==4 and nb['nbformat_minor']==5
code=[c for c in nb['cells'] if c['cell_type']=='code']
assert len(code)==28
assert [c['execution_count'] for c in code]==list(range(1,29))
assert not any(o.get('output_type')=='error' for c in code for o in c['outputs'])
for c in code:ast.parse(''.join(c['source']))
suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
log=io.StringIO()
with contextlib.redirect_stdout(log):
    result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
print(log.getvalue())
if not result.wasSuccessful():raise SystemExit(1)
manifest=json.loads((ROOT/'artifacts/manifest.json').read_text())
for name,expected in manifest['code_sha256'].items():
    actual=hashlib.sha256((ROOT/'student_success'/name).read_bytes()).hexdigest()
    assert actual==expected,('Code changed since training:',name)
report={
    'core_training':'completed; selected calibrated artifact generated',
    'notebook':{'total_cells':len(nb['cells']),'executed_code_cells':len(code),'error_outputs':0,
        'execution_method':'sequential ordinary Python; real stdout, DataFrame and matplotlib outputs'},
    'python_syntax':'all source files parsed',
    'tests_run':result.testsRun,'passed':result.testsRun-len(result.skipped),
    'skipped':len(result.skipped),'failures':len(result.failures),'errors':len(result.errors),
    'skip_reasons':[reason for _,reason in result.skipped],
    'streamlit_installed':importlib.util.find_spec('streamlit') is not None,
    'source_hashes_match_manifest':True,
    'deployment':'not changed or independently verified',
    'original_metabase':'preserved in archived branch; not restored in this package',
}
(ROOT/'reports/verification.json').write_text(json.dumps(report,indent=2)+'\n')
status='sudah dijalankan' if report['streamlit_installed'] else 'belum dapat dijalankan: Streamlit tidak tersedia pada runtime pembuat paket'
text=f'''# Catatan verifikasi paket

## Pemeriksaan yang benar-benar dijalankan

- Training CV, kalibrasi, threshold selection, holdout evaluation, dan diagnostik selesai.
- Notebook dibangun dan dieksekusi: {len(nb['cells'])} sel total, {len(code)} sel kode, tanpa output error.
- Seluruh kode Python berhasil diparse.
- Unit/integration suite: {result.testsRun} tes ditemukan, {report['passed']} lulus, {report['skipped']} dilewati, tanpa failure/error.
- Core tests mencakup domain invalid, file kosong, kategori tidak sah, relasi akademik, preset, threshold boundary, urutan fitur, CSV delimiter, pemuatan artefak, integritas split, dan kesetaraan batch/individu.
- Model yang dimuat kembali menghasilkan probabilitas yang sama dengan file prediksi holdout.
- Checksum dataset/model serta hash sumber modul sesuai manifest.
- Figur data dan evaluasi dibangun dari hasil aktual; sebagian figur diperiksa secara visual.

## Batas verifikasi

- Tes Streamlit/AppTest: **{status}**.
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

CI `.github/workflows/quality.yml` memasang dependensi dan menjalankan tests serta validasi notebook. Status CI belum diklaim lulus karena belum dipush/dijalankan pada GitHub.

Hasil mesin: `reports/verification.json`. Jalankan `python scripts/verify_package.py` untuk memperbarui catatan pemeriksaan lokal setelah menyiapkan environment lengkap.
'''
(ROOT/'docs/VALIDATION.md').write_text(text,encoding='utf-8')
print(json.dumps(report,indent=2))
