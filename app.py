"""Streamlit entry point: streamlit run app.py."""
import json
import hashlib
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from student_success.config import DATA_PATH,ARTIFACT_DIR,REPORT_DIR,CLASS_NAMES
from student_success.schema import FIELDS,FEATURES,COURSES,defaults,parse_csv,ValidationError
from student_success.inference import load_bundle,predict_frame,recommendations,reference_warnings

st.set_page_config(page_title='Student Success | Semester 1',page_icon='🎓',layout='wide')

@st.cache_resource
def resources():return load_bundle()

@st.cache_data
def historical_data():return pd.read_csv(DATA_PATH,sep=';')

@st.cache_data
def evaluation():return json.loads((REPORT_DIR/'metrics.json').read_text(encoding='utf-8'))

def apply_preset():
    for field,value in defaults(st.session_state['preset']).items():
        st.session_state['input_'+field]=float(value) if FIELDS[field].kind=='continuous' else int(value)
    st.session_state.pop('individual_result',None)
    st.session_state.pop('individual_warnings',None)

def show_validation(err):
    st.error('Data belum dapat diprediksi. Perbaiki nilai berikut terlebih dahulu.')
    st.dataframe(err.issues.rename(columns={'row':'Baris data','column':'Kolom','message':'Keterangan'}),hide_index=True,use_container_width=True)
    st.caption('Nomor baris dimulai dari 1 setelah header; baris 0 berarti masalah file atau schema.')

def introduction():
    st.markdown('''<style>
      .block-container{padding-top:2rem;max-width:1280px}
      [data-testid="stMetric"]{padding:18px;background:#f0f6f6;border-radius:12px;border:1px solid #d8e7e7}
      .eyebrow{font-size:.75rem;letter-spacing:.15em;font-weight:700;color:#19857b}
      .hero{padding:8px 0 24px}.hero h1{font-size:2.8rem;letter-spacing:-.04em;color:#142b3b;margin-bottom:.4rem}
      .hero p{max-width:720px;font-size:1.05rem;color:#526272;line-height:1.65}
      .footnote{color:#667085;font-size:.84rem;line-height:1.6}
    </style>''',unsafe_allow_html=True)
    st.markdown('''<div class="hero"><div class="eyebrow">STUDENT SUCCESS / SEMESTER 1</div>
      <h1>Memahami progres.<br>Mendukung langkah berikutnya.</h1>
      <p>Eksplorasi pola studi dan gunakan data akhir semester pertama untuk membantu
      menentukan profil yang perlu ditinjau lebih lanjut.</p></div>''',unsafe_allow_html=True)

def overview():
    st.subheader('Gambaran data historis')
    st.caption('Dataset publik UCI yang digunakan dalam studi kasus Dicoding. Bukan catatan mahasiswa aktif atau monitoring real-time.')
    data=historical_data()
    left,right=st.columns([2,1])
    with left: chosen=st.multiselect('Program studi',options=sorted(COURSES),format_func=lambda k:COURSES[k],help='Kosong berarti semua program studi.')
    with right:age=st.slider('Usia saat masuk',min_value=17,max_value=70,value=(17,70))
    mask=data.Age_at_enrollment.between(*age)
    if chosen:mask &= data.Course.isin(chosen)
    subset=data.loc[mask].copy()
    if subset.empty:st.info('Tidak ada data untuk kombinasi filter ini.');return
    counts=subset.Status.value_counts().reindex(CLASS_NAMES,fill_value=0)
    cols=st.columns(4)
    cols[0].metric('Mahasiswa dalam filter',f'{len(subset):,}')
    for col,status in zip(cols[1:],CLASS_NAMES):col.metric(status,f'{counts[status]:,}',f'{counts[status]/len(subset):.1%} dari filter',delta_color='off')
    a,b=st.columns(2)
    with a:
        st.markdown('**Komposisi status**')
        st.bar_chart(pd.DataFrame({'Jumlah':counts}),color='#19857b',horizontal=True)
    with b:
        st.markdown('**Capaian semester 1 menurut status**')
        means=subset.groupby('Status')[['Curricular_units_1st_sem_approved','Curricular_units_1st_sem_grade']].mean().reindex(CLASS_NAMES)
        measure=st.radio('Ukuran akademik',['Unit lulus','Rata-rata nilai'],horizontal=True)
        field='Curricular_units_1st_sem_approved' if measure=='Unit lulus' else 'Curricular_units_1st_sem_grade'
        st.bar_chart(means[[field]].rename(columns={field:measure}),color='#458499')
    st.markdown('**Proporsi Dropout per program studi**')
    groups=subset.groupby('Course').agg(jumlah=('Status','size'),dropout=('Status',lambda s:s.eq('Dropout').sum()))
    groups['proporsi_dropout']=groups.dropout/groups.jumlah
    groups['Program studi']=[COURSES[c] for c in groups.index]
    show=groups.reset_index(drop=True)[['Program studi','jumlah','dropout','proporsi_dropout']].sort_values('proporsi_dropout',ascending=False)
    st.dataframe(show,hide_index=True,use_container_width=True,column_config={'proporsi_dropout':st.column_config.NumberColumn('Proporsi Dropout',format='%.3f')})
    st.caption('Denominator setiap proporsi adalah jumlah mahasiswa pada program tersebut setelah filter. Kelompok kecil lebih tidak stabil; perbedaan bukan bukti sebab-akibat.')
    with st.expander('Arti status dan batas data'):
        st.write('Dropout, Enrolled, dan Graduate adalah status pada akhir durasi normal program dalam dataset. Enrolled berarti belum selesai pada titik pelabelan, bukan jaminan lulus di masa depan.')
        st.write('Data semester 2 dan informasi finansial tidak digunakan oleh model utama. Dataset tidak memiliki waktu dropout per mahasiswa; aplikasi ini merupakan demonstrasi prediksi retrospektif.')

def individual(model,manifest):
    st.subheader('Prediksi individu')
    st.write('Masukkan data sesuai skala sumber pada akhir semester 1. Hasil membantu peninjauan oleh manusia.')
    p1,p2=st.columns([3,1])
    with p1:st.selectbox('Contoh input sintetis',['Contoh umum','Perlu dukungan akademik','Akademik kuat'],key='preset')
    with p2:st.write('');st.button('Terapkan contoh',on_click=apply_preset,use_container_width=True)
    for field,spec in FIELDS.items():st.session_state.setdefault('input_'+field,float(spec.default) if spec.kind=='continuous' else int(spec.default))
    with st.form('student_form'):
        values={}
        for group in ['Pendaftaran','Semester 1']:
            st.markdown('**'+group+'**')
            columns=st.columns(2)
            for i,(field,spec) in enumerate((k,v) for k,v in FIELDS.items() if v.group==group):
                with columns[i%2]:
                    if spec.options:
                        values[field]=st.selectbox(spec.label,list(spec.options),format_func=lambda x,m=spec.options:m[x],key='input_'+field,help=spec.description)
                    elif spec.kind=='integer':
                        values[field]=st.number_input(spec.label,min_value=int(spec.low),max_value=int(spec.high),step=1,key='input_'+field,help=spec.description)
                    else:
                        values[field]=st.number_input(spec.label,min_value=float(spec.low),max_value=float(spec.high),step=.1,key='input_'+field,help=spec.description)
        submitted=st.form_submit_button('Lihat hasil peninjauan',use_container_width=True)
    if submitted:
        st.session_state.pop('individual_result',None)
        try:
            frame=pd.DataFrame([values])
            st.session_state['individual_result']=predict_frame(frame,model,manifest)
            st.session_state['individual_warnings']=reference_warnings(frame,manifest)
        except ValidationError as err:show_validation(err)
        except Exception:st.error('Prediksi gagal diproses. Periksa bahwa artefak dan aplikasi berasal dari versi proyek yang sama.')
    result=st.session_state.get('individual_result')
    if result is not None:
        row=result.iloc[0];st.divider()
        if row.action=='Perlu peninjauan':st.warning('Perlu peninjauan oleh dosen wali')
        else:st.info('Pemantauan rutin — tetap perhatikan perkembangan mahasiswa')
        a,b,c=st.columns(3)
        a.metric('Status paling mungkin',row.predicted_status)
        b.metric('Probabilitas Dropout',f'{row.prob_dropout:.1%}')
        c.metric('Ambang peninjauan',f'{manifest["threshold"]:.0%}')
        st.caption('Kategori tindakan mengikuti probabilitas Dropout dan satu ambang validasi; dapat berbeda dari kelas dengan probabilitas tertinggi.')
        st.bar_chart(pd.DataFrame({'Probabilitas':[row['prob_'+s.lower()] for s in CLASS_NAMES]},index=CLASS_NAMES),color='#19857b',horizontal=True)
        st.markdown('**Tindak lanjut yang disarankan**')
        for note in recommendations(row):st.write('• '+note)
        st.caption('Saran berasal dari aturan pendampingan yang transparan; bukan penjelasan kausal model atau keputusan akademik otomatis.')
        issues=st.session_state.get('individual_warnings')
        if issues is not None and not issues.empty:st.warning('Sebagian nilai berada di luar rentang data pengembangan.');st.dataframe(issues,hide_index=True)
        with st.expander('Input yang menghasilkan prediksi ini'):st.dataframe(result[FEATURES],hide_index=True,use_container_width=True)
        st.download_button('Unduh hasil individu',result.to_csv(index=False).encode('utf-8'),file_name='student_prediction.csv',mime='text/csv')

def batch(model,manifest):
    st.subheader('Prediksi batch')
    st.write('Unggah CSV UTF-8 dengan 14 kolom fitur. Koma dan titik koma didukung; maksimal 10 MB atau 10.000 baris.')
    st.caption('Unggahan diproses dalam sesi aplikasi dan tidak ditulis ke file proyek. Kolom tambahan diabaikan; source_row pada hasil menjaga hubungan dengan baris unggahan.')
    template=pd.DataFrame([defaults(n) for n in ['Contoh umum','Perlu dukungan akademik','Akademik kuat']])
    st.download_button('Unduh template dan 3 contoh sintetis',template.to_csv(index=False).encode('utf-8'),file_name='students_template.csv',mime='text/csv')
    upload=st.file_uploader('CSV mahasiswa',type=['csv'])
    upload_hash=hashlib.sha256(upload.getvalue()).hexdigest() if upload is not None else None
    if st.session_state.get('batch_upload_hash')!=upload_hash:
        st.session_state.pop('batch_result',None)
        st.session_state.pop('batch_warnings',None)
        st.session_state['batch_upload_hash']=upload_hash
    if upload is not None and st.button('Validasi dan prediksi',type='primary'):
        st.session_state.pop('batch_result',None)
        try:
            frame=parse_csv(upload.getvalue());result=predict_frame(frame,model,manifest)
            notes=reference_warnings(frame,manifest)
            st.session_state['batch_result']=result
            st.session_state['batch_warnings']=notes
        except ValidationError as err:show_validation(err)
        except Exception:st.error('File belum dapat diproses. Pastikan struktur CSV valid dan artefak model cocok dengan versi aplikasi.')
    result=st.session_state.get('batch_result')
    if result is not None:
        notes=st.session_state['batch_warnings']
        a,b,c=st.columns(3);a.metric('Baris diproses',len(result));b.metric('Perlu peninjauan',int(result.action.eq('Perlu peninjauan').sum()));c.metric('Ambang peninjauan',f'{manifest["threshold"]:.0%}')
        st.dataframe(result[['source_row','predicted_status','prob_dropout','action']],hide_index=True,use_container_width=True)
        if not notes.empty:
            st.warning(f'{notes.row.nunique()} baris memuat nilai di luar rentang pengembangan.')
            with st.expander('Rincian nilai di luar rentang'):st.dataframe(notes,hide_index=True)
        st.download_button('Unduh seluruh hasil',result.to_csv(index=False).encode('utf-8'),file_name='batch_predictions.csv',mime='text/csv')

def performance(manifest):
    metrics=evaluation();m=metrics['historical_holdout'];p=m['policy']
    st.subheader('Kinerja dan batas penggunaan')
    st.info('Evaluasi menggunakan holdout historis yang pernah dilihat pada proyek submission. Ini belum merupakan validasi independen pada kampus atau cohort baru.')
    a,b,c,d=st.columns(4)
    a.metric('Macro F1',f'{m["macro_f1"]:.3f}');b.metric('Recall peninjauan Dropout',f'{p["recall"]:.1%}');c.metric('Precision peninjauan',f'{p["precision"]:.1%}');d.metric('Profil yang ditandai',f'{p["review_rate"]:.1%}')
    st.caption('Recall dan precision peninjauan dihitung dengan ambang tindakan, bukan dengan kelas argmax tiga kategori.')
    a,b=st.columns(2)
    with a:st.image(str(REPORT_DIR/'figures'/'confusion_matrix.png'),use_container_width=True)
    with b:st.image(str(REPORT_DIR/'figures'/'precision_recall.png'),use_container_width=True)
    st.write(f'Model: **{manifest["candidate"]}**, dengan kalibrasi sigmoid. Pemilihan berdasarkan macro F1 lima fold pada data pengembangan; ambang {manifest["threshold"]:.2f} dipilih pada data validasi kebijakan dengan F2.')
    st.image(str(REPORT_DIR/'figures'/'calibration.png'),use_container_width=True)
    st.markdown('**Batas penggunaan**')
    for line in [
        'Model memakai data pendaftaran dan akhir semester 1. Tidak ada fitur semester 2.',
        'Tidak tersedia tanggal dropout per mahasiswa; model tidak membuktikan bahwa seluruh prediksi mendahului kejadian dropout.',
        'Enrolled adalah outcome belum selesai, bukan label aman atau lulus.',
        'Data berasal dari konteks Portugal. Penerapan pada kampus Indonesia memerlukan pemetaan data dan validasi baru.',
        'Probabilitas dan rekomendasi digunakan untuk pendampingan. Jangan gunakan untuk sanksi, penolakan, atau keputusan otomatis.',
        'Mengecualikan fitur demografis langsung tidak menjamin fairness; evaluasi kelompok tersedia dalam laporan proyek.',
    ]:st.write('• '+line)

def main():
    introduction()
    with st.sidebar:
        st.markdown('### Student Success')
        page=st.radio('Jelajahi',['Gambaran Data','Prediksi Individu','Prediksi Batch','Kinerja Model'])
        st.divider();st.caption('Studi kasus portofolio\n\nData sampai semester 1 · 14 fitur\n\nJaya Jaya Institut (fiktif)')
        st.markdown('[Sumber dataset UCI](https://doi.org/10.24432/C5MC89)')
    if st.session_state.get('last_page')!=page:
        st.session_state.pop('individual_result',None)
        st.session_state['last_page']=page
    if page=='Gambaran Data':overview()
    else:
        try:model,manifest=resources()
        except Exception as err:
            st.error('Model belum siap digunakan. Instal dependensi proyek dan pastikan artefak model lengkap.')
            with st.expander('Rincian untuk menjalankan aplikasi'):st.code(str(err))
            st.stop()
        if page=='Prediksi Individu':individual(model,manifest)
        elif page=='Prediksi Batch':batch(model,manifest)
        else:performance(manifest)
    st.divider();st.caption('Student Success · Prediksi retrospektif untuk studi kasus portofolio · Muhammad Pangeran Nabil')

if __name__=='__main__':main()
