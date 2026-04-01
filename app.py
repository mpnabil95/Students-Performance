import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "model", "label_encoder.pkl")
DATA_PATH = os.path.join(BASE_DIR, "data.csv")

st.set_page_config(
    page_title="Prediksi Status Mahasiswa",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = [
    "Marital_status",
    "Application_mode",
    "Application_order",
    "Course",
    "Daytime_evening_attendance",
    "Previous_qualification",
    "Previous_qualification_grade",
    "Nacionality",
    "Mothers_qualification",
    "Fathers_qualification",
    "Mothers_occupation",
    "Fathers_occupation",
    "Admission_grade",
    "Displaced",
    "Educational_special_needs",
    "Debtor",
    "Tuition_fees_up_to_date",
    "Gender",
    "Scholarship_holder",
    "Age_at_enrollment",
    "International",
    "Curricular_units_1st_sem_credited",
    "Curricular_units_1st_sem_enrolled",
    "Curricular_units_1st_sem_evaluations",
    "Curricular_units_1st_sem_approved",
    "Curricular_units_1st_sem_grade",
    "Curricular_units_1st_sem_without_evaluations",
    "Curricular_units_2nd_sem_credited",
    "Curricular_units_2nd_sem_enrolled",
    "Curricular_units_2nd_sem_evaluations",
    "Curricular_units_2nd_sem_approved",
    "Curricular_units_2nd_sem_grade",
    "Curricular_units_2nd_sem_without_evaluations",
    "Unemployment_rate",
    "Inflation_rate",
    "GDP",
]

SECTION_GROUPS = {
    "Profil dan Latar Belakang": [
        "Marital_status",
        "Application_mode",
        "Application_order",
        "Course",
        "Daytime_evening_attendance",
        "Previous_qualification",
        "Previous_qualification_grade",
        "Nacionality",
        "Age_at_enrollment",
        "International",
        "Gender",
    ],
    "Keluarga dan Dukungan Sosial": [
        "Mothers_qualification",
        "Fathers_qualification",
        "Mothers_occupation",
        "Fathers_occupation",
        "Displaced",
        "Educational_special_needs",
        "Scholarship_holder",
    ],
    "Kondisi Finansial": [
        "Admission_grade",
        "Debtor",
        "Tuition_fees_up_to_date",
        "Unemployment_rate",
        "Inflation_rate",
        "GDP",
    ],
    "Performa Akademik Semester 1": [
        "Curricular_units_1st_sem_credited",
        "Curricular_units_1st_sem_enrolled",
        "Curricular_units_1st_sem_evaluations",
        "Curricular_units_1st_sem_approved",
        "Curricular_units_1st_sem_grade",
        "Curricular_units_1st_sem_without_evaluations",
    ],
    "Performa Akademik Semester 2": [
        "Curricular_units_2nd_sem_credited",
        "Curricular_units_2nd_sem_enrolled",
        "Curricular_units_2nd_sem_evaluations",
        "Curricular_units_2nd_sem_approved",
        "Curricular_units_2nd_sem_grade",
        "Curricular_units_2nd_sem_without_evaluations",
    ],
}

DISPLAY_NAMES = {
    "Marital_status": "Marital Status",
    "Application_mode": "Application Mode",
    "Application_order": "Application Order",
    "Course": "Course",
    "Daytime_evening_attendance": "Daytime / Evening Attendance",
    "Previous_qualification": "Previous Qualification",
    "Previous_qualification_grade": "Previous Qualification Grade",
    "Nacionality": "Nationality",
    "Mothers_qualification": "Mother's Qualification",
    "Fathers_qualification": "Father's Qualification",
    "Mothers_occupation": "Mother's Occupation",
    "Fathers_occupation": "Father's Occupation",
    "Admission_grade": "Admission Grade",
    "Displaced": "Displaced",
    "Educational_special_needs": "Educational Special Needs",
    "Debtor": "Debtor",
    "Tuition_fees_up_to_date": "Tuition Fees Up to Date",
    "Gender": "Gender",
    "Scholarship_holder": "Scholarship Holder",
    "Age_at_enrollment": "Age at Enrollment",
    "International": "International Student",
    "Curricular_units_1st_sem_credited": "1st Sem Credited Units",
    "Curricular_units_1st_sem_enrolled": "1st Sem Enrolled Units",
    "Curricular_units_1st_sem_evaluations": "1st Sem Evaluations",
    "Curricular_units_1st_sem_approved": "1st Sem Approved Units",
    "Curricular_units_1st_sem_grade": "1st Sem Average Grade",
    "Curricular_units_1st_sem_without_evaluations": "1st Sem Without Evaluations",
    "Curricular_units_2nd_sem_credited": "2nd Sem Credited Units",
    "Curricular_units_2nd_sem_enrolled": "2nd Sem Enrolled Units",
    "Curricular_units_2nd_sem_evaluations": "2nd Sem Evaluations",
    "Curricular_units_2nd_sem_approved": "2nd Sem Approved Units",
    "Curricular_units_2nd_sem_grade": "2nd Sem Average Grade",
    "Curricular_units_2nd_sem_without_evaluations": "2nd Sem Without Evaluations",
    "Unemployment_rate": "Unemployment Rate",
    "Inflation_rate": "Inflation Rate",
    "GDP": "GDP",
}

YES_NO_FIELDS = {
    "Daytime_evening_attendance": {0: "Evening", 1: "Daytime"},
    "Displaced": {0: "No", 1: "Yes"},
    "Educational_special_needs": {0: "No", 1: "Yes"},
    "Debtor": {0: "No", 1: "Yes"},
    "Tuition_fees_up_to_date": {0: "No", 1: "Yes"},
    "Gender": {0: "Female", 1: "Male"},
    "Scholarship_holder": {0: "No", 1: "Yes"},
    "International": {0: "No", 1: "Yes"},
}

FALLBACK_DEFAULTS = {
    "Marital_status": 1,
    "Application_mode": 17,
    "Application_order": 1,
    "Course": 9238,
    "Daytime_evening_attendance": 1,
    "Previous_qualification": 1,
    "Previous_qualification_grade": 133.1,
    "Nacionality": 1,
    "Mothers_qualification": 19,
    "Fathers_qualification": 19,
    "Mothers_occupation": 5,
    "Fathers_occupation": 7,
    "Admission_grade": 126.1,
    "Displaced": 1,
    "Educational_special_needs": 0,
    "Debtor": 0,
    "Tuition_fees_up_to_date": 1,
    "Gender": 0,
    "Scholarship_holder": 0,
    "Age_at_enrollment": 20,
    "International": 0,
    "Curricular_units_1st_sem_credited": 0,
    "Curricular_units_1st_sem_enrolled": 6,
    "Curricular_units_1st_sem_evaluations": 8,
    "Curricular_units_1st_sem_approved": 5,
    "Curricular_units_1st_sem_grade": 12.29,
    "Curricular_units_1st_sem_without_evaluations": 0,
    "Curricular_units_2nd_sem_credited": 0,
    "Curricular_units_2nd_sem_enrolled": 6,
    "Curricular_units_2nd_sem_evaluations": 8,
    "Curricular_units_2nd_sem_approved": 5,
    "Curricular_units_2nd_sem_grade": 12.2,
    "Curricular_units_2nd_sem_without_evaluations": 0,
    "Unemployment_rate": 11.1,
    "Inflation_rate": 1.4,
    "GDP": 0.32,
}


def inject_css():
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.6rem; padding-bottom: 1rem;}
        .hero {
            background: linear-gradient(135deg, #0f172a 0%, #1d4ed8 100%);
            padding: 1.5rem 1.7rem;
            border-radius: 20px;
            color: white;
            margin-bottom: 1rem;
            box-shadow: 0 12px 30px rgba(15, 23, 42, 0.28);
        }
        .hero h1 {margin: 0; font-size: 2rem; line-height: 1.2;}
        .hero p {margin: 0.6rem 0 0 0; color: #e2e8f0;}
        .mini-card {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            padding: 1rem 1.1rem;
            border-radius: 18px;
            min-height: 110px;
        }
        .mini-card h4 {margin: 0 0 0.35rem 0; font-size: 0.95rem; color: #334155;}
        .mini-card .big-number {font-size: 2rem; font-weight: 700; color: #0f172a;}
        .result-box {
            padding: 1.2rem 1.3rem;
            border-radius: 18px;
            color: white;
            margin-bottom: 1rem;
        }
        .risk-high {background: linear-gradient(135deg, #b91c1c 0%, #ef4444 100%);}
        .risk-medium {background: linear-gradient(135deg, #b45309 0%, #f59e0b 100%);}
        .risk-low {background: linear-gradient(135deg, #166534 0%, #22c55e 100%);}
        .tip-box {
            background: #f8fafc;
            border-left: 5px solid #2563eb;
            border-radius: 10px;
            padding: 1rem 1rem 0.8rem 1rem;
            margin-top: 0.8rem;
        }
        div[data-testid="stMetric"] {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            padding: 0.75rem 0.9rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model tidak ditemukan di {MODEL_PATH}")
    if not os.path.exists(ENCODER_PATH):
        raise FileNotFoundError(f"Label encoder tidak ditemukan di {ENCODER_PATH}")

    model = joblib.load(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    return model, label_encoder


@st.cache_data
def load_reference_data():
    if not os.path.exists(DATA_PATH):
        return None
    for sep in [";", ","]:
        try:
            df = pd.read_csv(DATA_PATH, sep=sep)
            if all(col in df.columns for col in FEATURE_COLUMNS):
                return df
        except Exception:
            continue
    return None


def get_feature_specs(df_ref):
    specs = {}
    if df_ref is None:
        for col in FEATURE_COLUMNS:
            default_value = FALLBACK_DEFAULTS[col]
            default_num = float(default_value)
            upper = max(10.0, abs(default_num) * 2)
            specs[col] = {
                "default": default_value,
                "min": 0 if default_num >= 0 else -10,
                "max": int(upper) if isinstance(default_value, int) else float(upper),
                "options": list(YES_NO_FIELDS[col].keys()) if col in YES_NO_FIELDS else None,
                "is_int": isinstance(default_value, int),
            }
        return specs

    for col in FEATURE_COLUMNS:
        series = df_ref[col]
        is_int = pd.api.types.is_integer_dtype(series)
        default_val = int(series.median()) if is_int else float(series.median())
        options = sorted(series.dropna().unique().tolist()) if series.nunique() <= 20 else None
        specs[col] = {
            "default": default_val,
            "min": int(series.min()) if is_int else float(series.min()),
            "max": int(series.max()) if is_int else float(series.max()),
            "options": options,
            "is_int": is_int,
        }
    return specs


def build_preset(name, specs):
    base = {col: specs[col]["default"] for col in FEATURE_COLUMNS}

    if name == "Median Dataset":
        return base

    if name == "Risiko Dropout Tinggi":
        base.update(
            {
                "Debtor": 1,
                "Tuition_fees_up_to_date": 0,
                "Scholarship_holder": 0,
                "Age_at_enrollment": 28,
                "Curricular_units_1st_sem_approved": 2,
                "Curricular_units_1st_sem_grade": 9.8,
                "Curricular_units_2nd_sem_approved": 1,
                "Curricular_units_2nd_sem_grade": 8.7,
                "Curricular_units_1st_sem_without_evaluations": 2,
                "Curricular_units_2nd_sem_without_evaluations": 2,
            }
        )
        return base

    if name == "Potensi Graduate Tinggi":
        base.update(
            {
                "Debtor": 0,
                "Tuition_fees_up_to_date": 1,
                "Scholarship_holder": 1,
                "Age_at_enrollment": 19,
                "Curricular_units_1st_sem_approved": 7,
                "Curricular_units_1st_sem_grade": 14.5,
                "Curricular_units_2nd_sem_approved": 7,
                "Curricular_units_2nd_sem_grade": 14.7,
                "Curricular_units_1st_sem_without_evaluations": 0,
                "Curricular_units_2nd_sem_without_evaluations": 0,
            }
        )
        return base

    return base


def render_field(field_name, specs, preset_values):
    spec = specs[field_name]
    label = DISPLAY_NAMES.get(field_name, field_name)
    default_value = preset_values[field_name]

    if field_name in YES_NO_FIELDS:
        option_map = YES_NO_FIELDS[field_name]
        options = list(option_map.keys())
        default_key = int(default_value)
        default_index = options.index(default_key) if default_key in options else 0
        chosen = st.selectbox(
            label,
            options=options,
            index=default_index,
            format_func=lambda x: f"{x} - {option_map[x]}",
            key=f"field_{field_name}",
        )
        return int(chosen)

    if spec["options"] is not None:
        options = spec["options"]
        if spec["is_int"]:
            options = [int(x) for x in options]
            default_cast = int(default_value)
        else:
            options = [float(x) for x in options]
            default_cast = float(default_value)

        default_index = options.index(default_cast) if default_cast in options else 0
        chosen = st.selectbox(label, options=options, index=default_index, key=f"field_{field_name}")
        return int(chosen) if spec["is_int"] else float(chosen)

    if spec["is_int"]:
        return int(
            st.number_input(
                label,
                min_value=int(spec["min"]),
                max_value=int(spec["max"]),
                value=int(default_value),
                step=1,
                key=f"field_{field_name}",
            )
        )

    return float(
        st.number_input(
            label,
            min_value=float(spec["min"]),
            max_value=float(spec["max"]),
            value=float(default_value),
            step=0.1,
            format="%.2f",
            key=f"field_{field_name}",
        )
    )


def safe_label_names(label_encoder, n_classes=None):
    if hasattr(label_encoder, "classes_"):
        return list(label_encoder.classes_)
    if n_classes is not None:
        return [str(i) for i in range(n_classes)]
    return ["Dropout", "Enrolled", "Graduate"]


def make_prediction(model, label_encoder, input_dict):
    input_df = pd.DataFrame([input_dict])[FEATURE_COLUMNS]
    predicted_idx = model.predict(input_df)[0]
    try:
        predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    except Exception:
        predicted_label = str(predicted_idx)

    probabilities = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_df)[0]
        try:
            classes = label_encoder.inverse_transform(np.arange(len(proba)))
        except Exception:
            classes = safe_label_names(label_encoder, len(proba))
        probabilities = pd.DataFrame({"Status": classes, "Probabilitas": proba}).sort_values(
            by="Probabilitas", ascending=False
        )

    return predicted_label, probabilities, input_df


def get_dropout_probability(probabilities):
    if probabilities is None:
        return None
    row = probabilities.loc[probabilities["Status"].astype(str) == "Dropout", "Probabilitas"]
    if row.empty:
        return None
    return float(row.iloc[0])


def recommendation_text(input_data, dropout_prob):
    recommendations = []

    if dropout_prob is not None:
        if dropout_prob >= 0.60:
            recommendations.append("Prioritaskan intervensi segera melalui pendampingan akademik dan konseling.")
        elif dropout_prob >= 0.35:
            recommendations.append("Masukkan mahasiswa ke dalam daftar monitoring intensif jangka pendek.")
        else:
            recommendations.append("Pertahankan pemantauan rutin dan evaluasi berkala.")

    if input_data["Tuition_fees_up_to_date"] == 0 or input_data["Debtor"] == 1:
        recommendations.append("Pertimbangkan bantuan finansial atau pengaturan ulang pembayaran biaya kuliah.")

    if input_data["Curricular_units_1st_sem_approved"] <= 3 or input_data["Curricular_units_2nd_sem_approved"] <= 3:
        recommendations.append("Berikan pendampingan akademik karena jumlah mata kuliah lulus masih rendah.")

    if input_data["Curricular_units_1st_sem_grade"] < 10.5 or input_data["Curricular_units_2nd_sem_grade"] < 10.5:
        recommendations.append("Lakukan evaluasi performa belajar dan tawarkan tutoring atau mentoring.")

    if input_data["Scholarship_holder"] == 0 and (dropout_prob or 0) >= 0.35:
        recommendations.append("Evaluasi kelayakan dukungan beasiswa atau program bantuan retensi.")

    if not recommendations:
        recommendations.append("Profil mahasiswa terlihat stabil. Lanjutkan pemantauan rutin.")

    return recommendations


def render_result(predicted_label, probabilities, input_data):
    dropout_prob = get_dropout_probability(probabilities)

    if predicted_label == "Dropout" or (dropout_prob is not None and dropout_prob >= 0.60):
        css_class = "risk-high"
        title = "Risiko Dropout Tinggi"
        subtitle = "Mahasiswa sangat disarankan masuk prioritas intervensi."
    elif dropout_prob is not None and dropout_prob >= 0.35:
        css_class = "risk-medium"
        title = "Perlu Monitoring"
        subtitle = "Mahasiswa menunjukkan sinyal risiko yang perlu dipantau lebih dekat."
    else:
        css_class = "risk-low"
        title = "Profil Relatif Stabil"
        subtitle = "Mahasiswa cenderung berada pada kondisi yang lebih aman."

    st.markdown(
        f"""
        <div class="result-box {css_class}">
            <h2 style="margin-top:0;">Prediksi Utama: {predicted_label}</h2>
            <p style="font-size:1.05rem; margin-bottom:0.3rem;"><strong>{title}</strong></p>
            <p style="margin-bottom:0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(3)
    with metric_cols[0]:
        st.metric("Prediksi Model", predicted_label)
    with metric_cols[1]:
        st.metric("Probabilitas Dropout", f"{dropout_prob:.1%}" if dropout_prob is not None else "N/A")
    with metric_cols[2]:
        if probabilities is not None:
            st.metric("Confidence Tertinggi", f"{float(probabilities.iloc[0]['Probabilitas']):.1%}")
        else:
            st.metric("Confidence Tertinggi", "N/A")

    if probabilities is not None:
        st.subheader("Probabilitas Setiap Status")
        chart_df = probabilities.set_index("Status")
        st.bar_chart(chart_df)

        display_df = probabilities.copy()
        display_df["Probabilitas"] = display_df["Probabilitas"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.subheader("Rekomendasi Tindak Lanjut")
    for idx, rec in enumerate(recommendation_text(input_data, dropout_prob), start=1):
        st.markdown(f'<div class="tip-box"><strong>{idx}.</strong> {rec}</div>', unsafe_allow_html=True)


def validate_batch_file(batch_df):
    missing_cols = [col for col in FEATURE_COLUMNS if col not in batch_df.columns]
    return missing_cols


def read_uploaded_csv(uploaded_file):
    raw_bytes = uploaded_file.getvalue()
    for sep in [",", ";"]:
        try:
            df = pd.read_csv(pd.io.common.BytesIO(raw_bytes), sep=sep)
            if len(df.columns) > 1:
                return df
        except Exception:
            continue
    raise ValueError("File CSV tidak dapat dibaca. Pastikan file valid dan delimiter sesuai.")


def main():
    inject_css()

    st.markdown(
        """
        <div class="hero">
            <h1>🎓 Prediksi Status Mahasiswa</h1>
            <p>
                Prototype ini membantu institusi memprediksi status mahasiswa
                (<strong>Dropout</strong>, <strong>Enrolled</strong>, atau <strong>Graduate</strong>)
                dan menampilkan sinyal risiko dropout secara lebih mudah dipahami.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.title("⚙️ Pengaturan")
        st.caption("Gunakan preset agar pengisian form lebih cepat.")
        preset_name = st.radio(
            "Preset input",
            ["Median Dataset", "Risiko Dropout Tinggi", "Potensi Graduate Tinggi"],
            index=0,
        )
        st.divider()
        st.markdown("### Catatan")
        st.info(
            "Aplikasi ini menggunakan model machine learning yang telah dilatih pada dataset students' performance."
        )
        st.markdown(
            """
            **Tips penggunaan**
            - Gunakan **Preset input** untuk contoh cepat.
            - Sesuaikan field penting seperti **Debtor**, **Tuition Fees Up to Date**,
              dan performa akademik semester 1–2.
            - Untuk evaluasi banyak data sekaligus, buka tab **Prediksi Batch**.
            """
        )

    model, label_encoder = None, None
    artifact_error = None
    try:
        model, label_encoder = load_artifacts()
    except Exception as err:
        artifact_error = err
        st.error(f"Gagal memuat artefak model: {err}")
        st.info("Pastikan dependency model sudah terinstal dan file model tersedia. Jika model Anda dibuat dengan XGBoost, instal package xgboost lalu jalankan ulang aplikasi.")

    df_ref = load_reference_data()
    specs = get_feature_specs(df_ref)
    preset_values = build_preset(preset_name, specs)

    tab1, tab2, tab3 = st.tabs(["Prediksi Individu", "Prediksi Batch", "Tentang Aplikasi"])

    with tab1:
        st.subheader("Form Prediksi Individu")
        st.caption("Isi data mahasiswa berikut, lalu klik tombol prediksi.")

        if model is None or label_encoder is None:
            st.warning("Prediksi individu belum dapat digunakan karena artefak model belum berhasil dimuat.")
        else:
            with st.form("prediction_form", clear_on_submit=False):
                input_data = {}
                for section_name, fields in SECTION_GROUPS.items():
                    with st.expander(section_name, expanded=section_name == "Profil dan Latar Belakang"):
                        cols = st.columns(3)
                        for idx, field in enumerate(fields):
                            with cols[idx % 3]:
                                input_data[field] = render_field(field, specs, preset_values)

                submitted = st.form_submit_button("🔍 Prediksi Sekarang", use_container_width=True)

            st.markdown("### Ringkasan Input Penting")
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown(
                    f'<div class="mini-card"><h4>Tuition Fees</h4><div class="big-number">{"Yes" if input_data["Tuition_fees_up_to_date"] == 1 else "No"}</div></div>',
                    unsafe_allow_html=True,
                )
            with c2:
                st.markdown(
                    f'<div class="mini-card"><h4>Debtor</h4><div class="big-number">{"Yes" if input_data["Debtor"] == 1 else "No"}</div></div>',
                    unsafe_allow_html=True,
                )
            with c3:
                st.markdown(
                    f'<div class="mini-card"><h4>Approved 1st Sem</h4><div class="big-number">{input_data["Curricular_units_1st_sem_approved"]}</div></div>',
                    unsafe_allow_html=True,
                )
            with c4:
                st.markdown(
                    f'<div class="mini-card"><h4>Approved 2nd Sem</h4><div class="big-number">{input_data["Curricular_units_2nd_sem_approved"]}</div></div>',
                    unsafe_allow_html=True,
                )

            if submitted:
                predicted_label, probabilities, _ = make_prediction(model, label_encoder, input_data)
                st.divider()
                render_result(predicted_label, probabilities, input_data)
    with tab2:
        st.subheader("Prediksi Batch")
        st.caption("Unggah file CSV yang memiliki kolom fitur yang sama dengan data training.")

        uploaded_file = st.file_uploader("Upload file CSV", type=["csv"], key="batch_file")

        if uploaded_file is not None:
            try:
                batch_df = read_uploaded_csv(uploaded_file)
                missing_cols = validate_batch_file(batch_df)

                if missing_cols:
                    st.error("Kolom berikut belum ada di file Anda:")
                    st.write(missing_cols)
                else:
                    batch_input = batch_df[FEATURE_COLUMNS].copy()
                    preds = model.predict(batch_input)
                    try:
                        batch_df["Prediksi_Status"] = label_encoder.inverse_transform(preds)
                    except Exception:
                        batch_df["Prediksi_Status"] = preds.astype(str)

                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(batch_input)
                        try:
                            classes = label_encoder.inverse_transform(np.arange(probas.shape[1]))
                        except Exception:
                            classes = safe_label_names(label_encoder, probas.shape[1])
                        for idx, cls in enumerate(classes):
                            batch_df[f"Prob_{cls}"] = probas[:, idx]

                    st.success("Prediksi batch berhasil dibuat.")
                    st.dataframe(batch_df.head(20), use_container_width=True)

                    csv_bytes = batch_df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Hasil Prediksi",
                        data=csv_bytes,
                        file_name="hasil_prediksi_mahasiswa.csv",
                        mime="text/csv",
                    )
            except Exception as err:
                st.error(f"Gagal memproses file batch: {err}")

        st.markdown("#### Format file yang disarankan")
        example_df = pd.DataFrame([{col: FALLBACK_DEFAULTS[col] for col in FEATURE_COLUMNS}])
        st.dataframe(example_df, use_container_width=True, hide_index=True)

        template_bytes = example_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Template CSV",
            data=template_bytes,
            file_name="template_prediksi_mahasiswa.csv",
            mime="text/csv",
        )

    with tab3:
        st.subheader("Tentang Aplikasi")
        st.markdown(
            """
            Aplikasi ini dirancang sebagai **prototype early warning system** untuk membantu institusi pendidikan
            mengidentifikasi mahasiswa yang berpotensi mengalami dropout lebih cepat.

            **Fitur utama aplikasi**
            - Prediksi individu dengan antarmuka yang mudah digunakan.
            - Probabilitas untuk setiap kelas status mahasiswa.
            - Rekomendasi tindak lanjut berbasis hasil prediksi.
            - Prediksi batch untuk banyak mahasiswa sekaligus.
            """
        )

        st.markdown("#### Daftar fitur model")
        feature_df = pd.DataFrame(
            [
                {"Kelompok": group, "Feature": DISPLAY_NAMES.get(feature, feature)}
                for group, features in SECTION_GROUPS.items()
                for feature in features
            ]
        )
        st.dataframe(feature_df, use_container_width=True, hide_index=True)

        st.markdown("#### Kelas prediksi")
        st.write(safe_label_names(label_encoder))
        if artifact_error is not None:
            st.caption("Catatan: daftar kelas di atas menggunakan fallback default karena artefak model belum berhasil dimuat.")

        if df_ref is not None:
            st.markdown("#### Referensi data")
            st.write(
                f"Dataset referensi berhasil dimuat dengan bentuk: **{df_ref.shape[0]} baris × {df_ref.shape[1]} kolom**"
            )
        else:
            st.warning("File data.csv tidak ditemukan. Aplikasi tetap berjalan menggunakan fallback default value.")


if __name__ == "__main__":
    main()
