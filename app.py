# ============================================================
# STREAMLIT APP FOR MUGILIDAE FISH CLASSIFICATION
# (FINAL FIX - Memastikan Format Data Tepat)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Mugilidae Fish Classifier",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("---")

# ============================================================
# 1. MUAT SEMUA FAIL MODEL
# ============================================================

@st.cache_resource
def load_all_models():
    """Muat semua fail yang diperlukan dan pulangkan dalam satu tuple."""
    required_files = [
        'scaler.pkl',
        'label_encoder.pkl',
        'feature_names.pkl',
        'ann_model_balanced.pkl',
        'pso_model_balanced.pkl',
        'ga_model_balanced.pkl',
        'gwo_model_balanced.pkl'
    ]

    # Semak kewujudan fail
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        st.error(f"❌ Fail berikut tidak dijumpai: {', '.join(missing)}")
        st.stop()

    try:
        with open('scaler.pkl', 'rb') as f: scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f: label_encoder = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f: feature_names = pickle.load(f)
        with open('ann_model_balanced.pkl', 'rb') as f: ann_model = pickle.load(f)
        with open('pso_model_balanced.pkl', 'rb') as f: pso_model = pickle.load(f)
        with open('ga_model_balanced.pkl', 'rb') as f: ga_model = pickle.load(f)
        with open('gwo_model_balanced.pkl', 'rb') as f: gwo_model = pickle.load(f)
        
        st.sidebar.success("✅ Semua model berjaya dimuatkan.")
        return scaler, label_encoder, feature_names, ann_model, pso_model, ga_model, gwo_model
    
    except Exception as e:
        st.error(f"❌ Ralat memuatkan model: {e}")
        st.stop()

scaler, label_encoder, feature_names, ann_model, pso_model, ga_model, gwo_model = load_all_models()

# ============================================================
# 2. PAPARAN STATUS DI SIDEBAR
# ============================================================

st.sidebar.header("🔍 Status Sistem")
st.sidebar.write(f"**Feature Names:** {len(feature_names)} features loaded")
st.sidebar.write(f"**Species Classes:** {list(label_encoder.classes_)}")

# ============================================================
# 3. PILIHAN MODEL DI SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Tetapan Model")

model_dict = {
    "ANN": ann_model,
    "ANN-PSO": pso_model,
    "ANN-GA": ga_model,
    "ANN-GWO": gwo_model
}

selected_model_name = st.sidebar.selectbox(
    "Pilih Model untuk Ramalan",
    options=list(model_dict.keys()),
    index=3  # GWO sebagai lalai
)
selected_model = model_dict[selected_model_name]

# ============================================================
# 4. BORANG INPUT (Sama seperti sebelumnya)
# ============================================================

st.markdown("### 📝 Masukkan 15 Ukuran Ikan")

# Susun dalam 3 column
col1, col2, col3 = st.columns(3)

input_values = []

with col1:
    st.subheader("📏 Meristik (6)")
    nd1 = st.number_input("ND1_Total", min_value=0.0, max_value=50.0, value=6.0, step=1.0, key="nd1")
    nd2 = st.number_input("ND2_Total", min_value=0.0, max_value=50.0, value=7.0, step=1.0, key="nd2")
    np_val = st.number_input("NP", min_value=0.0, max_value=50.0, value=14.0, step=1.0, key="np")
    nc_val = st.number_input("NC", min_value=0.0, max_value=50.0, value=14.0, step=1.0, key="nc")
    nv = st.number_input("NV_Total", min_value=0.0, max_value=50.0, value=6.0, step=1.0, key="nv")
    na = st.number_input("NA_Total", min_value=0.0, max_value=50.0, value=10.0, step=1.0, key="na")
    input_values.extend([nd1, nd2, np_val, nc_val, nv, na])

with col2:
    st.subheader("📐 Morfometrik (4)")
    sl = st.number_input("SL (cm)", min_value=0.0, max_value=500.0, value=150.0, step=5.0, key="sl")
    pl = st.number_input("PL (cm)", min_value=0.0, max_value=300.0, value=40.0, step=1.0, key="pl")
    bh = st.number_input("BH (cm)", min_value=0.0, max_value=300.0, value=45.0, step=1.0, key="bh")
    hl = st.number_input("HL (cm)", min_value=0.0, max_value=300.0, value=35.0, step=1.0, key="hl")
    input_values.extend([sl, pl, bh, hl])

with col3:
    st.subheader("📐 Rangkaian Truss (5)")
    head = st.number_input("Head_Truss (mm)", min_value=0.0, max_value=300.0, value=60.0, step=1.0, key="head")
    ant = st.number_input("Anterior_Truss (mm)", min_value=0.0, max_value=300.0, value=60.0, step=1.0, key="ant")
    mid = st.number_input("Mid_Truss (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0, key="mid")
    post = st.number_input("Posterior_Truss (mm)", min_value=0.0, max_value=500.0, value=80.0, step=1.0, key="post")
    tail = st.number_input("Tail_Truss (mm)", min_value=0.0, max_value=300.0, value=70.0, step=1.0, key="tail")
    input_values.extend([head, ant, mid, post, tail])

# ============================================================
# 5. FUNGSI RAMALAN YANG DIPERBAIKI (PUNCA UTAMA)
# ============================================================

def predict_species(input_data, model, scaler, encoder):
    """
    Fungsi yang DIPERBAIKI untuk memastikan format data betul.
    """
    try:
        # --- LANGKAH PENTING 1: Tukar ke numpy array dan pastikan bentuk 2D ---
        input_array = np.array(input_data, dtype=np.float64).reshape(1, -1)
        
        # --- LANGKAH PENTING 2: Piawaikan data ---
        input_scaled = scaler.transform(input_array)
        
        # --- LANGKAH PENTING 3: Ramal kebarangkalian ---
        # Pastikan model menerima input dalam bentuk 2D
        probabilities = model.predict_proba(input_scaled)
        
        # Dapatkan kelas dengan kebarangkalian tertinggi
        predicted_class_index = np.argmax(probabilities, axis=1)[0]
        confidence_score = np.max(probabilities, axis=1)[0]
        
        # Tukar indeks kepada nama spesies
        predicted_species = encoder.inverse_transform([predicted_class_index])[0]
        
        return predicted_species, confidence_score, probabilities[0]
    
    except Exception as e:
        st.error(f"⚠️ Ralat dalam fungsi ramalan: {e}")
        return None, None, None

# ============================================================
# 6. BUTANG RAMALAN
# ============================================================

if st.button("🔍 Ramal Spesies", type="primary"):
    with st.spinner("Sedang meramal..."):
        species, confidence, probs = predict_species(
            input_values, selected_model, scaler, label_encoder
        )
    
    if species:
        # Paparan keputusan
        st.markdown("---")
        st.markdown("### 🎯 Keputusan Ramalan")

        # Paparan kad
        col_a, col_b, col_c = st.columns([1, 2, 1])
        with col_b:
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #e8f4f8; border-radius: 15px; border: 2px solid #1e90ff;">
                <h1 style="color: #1e90ff;">🐟 {species}</h1>
                <h3>Keyakinan: {confidence*100:.2f}%</h3>
                <p>Model: <b>{selected_model_name}</b></p>
                <p>Data: <b>Balanced (200/species)</b></p>
            </div>
            """, unsafe_allow_html=True)

        # --- LANGKAH PENTING 4: PAPARAN UNTUK DEBUG ---
        # Ini akan membantu kita lihat apa yang model terima
        with st.expander("🔧 Lihat Data untuk Debug (Klik untuk buka)"):
            st.write("**Input Data (sebelum piawaian):**", input_values)
            st.write("**Bentuk Array (sebelum piawaian):**", np.array(input_values).shape)
            # Kira dan paparkan data selepas piawaian untuk semakan
            input_array = np.array(input_values, dtype=np.float64).reshape(1, -1)
            input_scaled = scaler.transform(input_array)
            st.write("**Data Selepas Piawaian (5 nilai pertama):**", input_scaled[0][:5])
            st.write("**Kebarangkalian untuk setiap kelas:**")
            proba_df = pd.DataFrame({
                'Spesies': label_encoder.classes_,
                'Kebarangkalian (%)': probs * 100
            })
            st.dataframe(proba_df)

        # Graf kebarangkalian
        st.markdown("#### 📊 Taburan Kebarangkalian")
        species_names = label_encoder.classes_
        proba_df_plot = pd.DataFrame({
            'Spesies': species_names,
            'Kebarangkalian (%)': probs * 100
        })

        fig = px.bar(
            proba_df_plot,
            x='Spesies',
            y='Kebarangkalian (%)',
            color='Spesies',
            color_discrete_sequence=px.colors.qualitative.Set2,
            title='Kebarangkalian Ramalan untuk Setiap Spesies'
        )
        fig.update_layout(height=400, showlegend=False, yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# 7. KAKI
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    🐟 Sistem Klasifikasi Ikan Mugilidae | Pendekatan Hibrid ANN-Metaheuristik
    <br> Dibangunkan oleh Nur Zulaifah binti Mohamad Zaini | Universiti Malaysia Terengganu
</div>
""", unsafe_allow_html=True)
