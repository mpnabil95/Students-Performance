"""Shared prediction and action policy for individual and batch flows."""
import hashlib
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import sklearn
from .config import ARTIFACT_DIR, CLASS_NAMES
from .schema import FEATURES, validate_features
from .modeling import align_probabilities

def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def load_bundle(directory=ARTIFACT_DIR):
    directory=Path(directory)
    manifest=json.loads((directory/'manifest.json').read_text(encoding='utf-8'))
    model_path=directory/'model.joblib'
    if sha256_file(model_path)!=manifest['model_sha256']:
        raise ValueError('Checksum model berbeda. Pulihkan artefak atau jalankan training kembali.')
    if manifest['features']!=FEATURES:
        raise ValueError('Schema aplikasi berbeda dengan model. Gunakan paket versi yang sama.')
    if sklearn.__version__!=manifest['environment']['scikit-learn']:
        raise ValueError('Versi scikit-learn berbeda dari training. Instal requirements.txt paket ini.')
    # Only loads the repository's trusted, locally generated artifact; never user uploads.
    model=joblib.load(model_path)
    if list(model.classes_)!=CLASS_NAMES:raise ValueError('Kelas model tidak sesuai manifest.')
    return model,manifest

def action_label(probability, threshold):
    return 'Perlu peninjauan' if float(probability)>=float(threshold) else 'Pemantauan rutin'

def predict_frame(frame,model,manifest):
    x=validate_features(frame)
    p=align_probabilities(model,x)
    if not np.isfinite(p).all() or not np.allclose(p.sum(axis=1),1,atol=1e-6):
        raise ValueError('Output probabilitas model tidak valid.')
    out=x.copy();out.insert(0,'source_row',np.arange(1,len(x)+1))
    out['predicted_status']=np.asarray(CLASS_NAMES)[p.argmax(axis=1)]
    for i,c in enumerate(CLASS_NAMES):out['prob_'+c.lower()]=p[:,i]
    out['action']=[action_label(v,manifest['threshold']) for v in p[:,0]]
    out['model_version']=manifest['model_version']
    return out

def reference_warnings(frame,manifest):
    x=validate_features(frame);rows=[]
    for c,bounds in manifest['training_ranges'].items():
        mask=(x[c]<bounds['min'])|(x[c]>bounds['max'])
        for i in np.flatnonzero(mask):
            rows.append({'row':int(i)+1,'column':c,'message':'Di luar rentang data pengembangan; interpretasikan dengan hati-hati.'})
    return pd.DataFrame(rows,columns=['row','column','message'])

def recommendations(row):
    """Transparent support suggestions, not causal explanations or automatic decisions."""
    notes=[]
    if row['action']=='Perlu peninjauan':
        notes.append('Tinjau profil bersama dosen wali dan konfirmasi kebutuhan mahasiswa sebelum menentukan pendampingan.')
    else:
        notes.append('Lanjutkan pemantauan rutin. Hasil ini tidak menjamin mahasiswa bebas risiko dropout.')
    enrolled=float(row['Curricular_units_1st_sem_enrolled'])
    approved=float(row['Curricular_units_1st_sem_approved'])
    if enrolled>0 and approved/enrolled<.5:
        notes.append('Kurang dari separuh unit yang diambil berhasil diselesaikan; diskusikan hambatan akademik dan opsi tutoring.')
    if float(row['Curricular_units_1st_sem_without_evaluations'])>0:
        notes.append('Ada unit tanpa evaluasi; konfirmasi partisipasi dan kendala mengikuti penilaian.')
    if enrolled==0:
        notes.append('Tidak ada unit yang terdaftar pada semester 1; periksa kelengkapan catatan akademik.')
    return notes
