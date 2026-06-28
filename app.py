# ============================================================
# STREAMLIT APP FOR MUGILIDAE FISH CLASSIFICATION
# Hybrid ANN-Metaheuristic Models (Balanced Data)
# ============================================================

import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Mugilidae Fish Classifier",
    page_icon="🐟",
    layout="wide"
)

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("---")

# ============================================================
# LOAD MODELS
# ============================================================

@st.cache_resource
def load_models():
    try:
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('feature_names.pkl', 'rb') as f:
            feature_names = pickle.load(f)
        with open('ann_model_balanced.pkl', 'rb') as f:
            ann_model = pickle.load(f)
        with open('pso_model_balanced.pkl', 'rb') as f:
            pso_model = pickle.load(f)
        with open('ga_model_balanced.pkl', 'rb') as f:
            ga_model = pickle.load(f)
        with open('gwo_model_balanced.pkl', 'rb') as f:
            gwo_model = pickle.load(f)
        return scaler, label_encoder, feature_names, ann_model, pso_model, ga_model, gwo_model
    except FileNotFoundError:
        st.error("❌ Model files not found. Please ensure all .pkl files are in the same directory.")
        return None, None, None, None, None, None, None

scaler, label_encoder, feature_names, ann_model, pso_model, ga_model, gwo_model = load_models()

# ============================================================
# SIDEBAR: MODE & MODEL SELECTION
# ============================================================

st.sidebar.header("⚙️ Settings")

# Model selection
model_options = {
    "ANN": ann_model,
    "ANN-PSO": pso_model,
    "ANN-GA": ga_model,
    "ANN-GWO": gwo_model
}

selected_model_name = st.sidebar.selectbox(
    "Select Model",
    options=list(model_options.keys()),
    index=3  # Default to GWO
)

selected_model = model_options[selected_model_name]

# Mode selection (just for display, both use same balanced models)
mode = st.sidebar.radio(
    "Dataset Mode",
    options=["Balanced Data (200/species)"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**📊 Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis
""")

# ============================================================
# MAIN CONTENT: INPUT FORM
# ============================================================

st.markdown("### 📝 Enter Fish Measurements")

if feature_names is None:
    st.error("Feature names not loaded. Please check files.")
    st.stop()

# Split features into 3 columns for better layout
col1, col2, col3 = st.columns(3)

input_values = []

with col1:
    st.subheader("📏 Meristic (6)")
    nd1_total = st.number_input("ND1 Total", min_value=0.0, max_value=50.0, value=6.0, step=1.0)
    nd2_total = st.number_input("ND2 Total", min_value=0.0, max_value=50.0, value=7.0, step=1.0)
    np_count = st.number_input("NP", min_value=0.0, max_value=50.0, value=14.0, step=1.0)
    nc_count = st.number_input("NC", min_value=0.0, max_value=50.0, value=14.0, step=1.0)
    nv_total = st.number_input("NV Total", min_value=0.0, max_value=50.0, value=6.0, step=1.0)
    na_total = st.number_input("NA Total", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
    input_values.extend([nd1_total, nd2_total, np_count, nc_count, nv_total, na_total])

with col2:
    st.subheader("📐 Morphometric (4)")
    sl = st.number_input("SL (cm)", min_value=0.0, max_value=500.0, value=150.0, step=5.0)
    pl = st.number_input("PL (cm)", min_value=0.0, max_value=300.0, value=40.0, step=1.0)
    bh = st.number_input("BH (cm)", min_value=0.0, max_value=300.0, value=45.0, step=1.0)
    hl = st.number_input("HL (cm)", min_value=0.0, max_value=300.0, value=35.0, step=1.0)
    input_values.extend([sl, pl, bh, hl])

with col3:
    st.subheader("📐 Truss Network (5)")
    head_truss = st.number_input("Head Truss (mm)", min_value=0.0, max_value=300.0, value=60.0, step=1.0)
    anterior_truss = st.number_input("Anterior Truss (mm)", min_value=0.0, max_value=300.0, value=60.0, step=1.0)
    mid_truss = st.number_input("Mid Truss (mm)", min_value=0.0, max_value=500.0, value=100.0, step=1.0)
    posterior_truss = st.number_input("Posterior Truss (mm)", min_value=0.0, max_value=500.0, value=80.0, step=1.0)
    tail_truss = st.number_input("Tail Truss (mm)", min_value=0.0, max_value=300.0, value=70.0, step=1.0)
    input_values.extend([head_truss, anterior_truss, mid_truss, posterior_truss, tail_truss])

# ============================================================
# PREDICT BUTTON
# ============================================================

if st.button("🔍 Predict Species", type="primary"):
    if scaler is None or selected_model is None or label_encoder is None:
        st.error("❌ Model files not loaded properly. Please check files.")
    else:
        try:
            # Convert input to numpy array
            input_array = np.array(input_values).reshape(1, -1)
            
            # Standardize
            input_scaled = scaler.transform(input_array)
            
            # Predict
            prediction_proba = selected_model.predict_proba(input_scaled)
            prediction_class = np.argmax(prediction_proba, axis=1)[0]
            confidence = np.max(prediction_proba, axis=1)[0]
            
            # Get species name
            species = label_encoder.inverse_transform([prediction_class])[0]
            
            # ============================================================
            # DISPLAY RESULTS
            # ============================================================
            
            st.markdown("---")
            st.markdown("### 🎯 Prediction Results")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                # Display species with confidence
                st.markdown(f"""
                <div style="text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px;">
                    <h1 style="color: #1e90ff;">🐟 {species}</h1>
                    <h3>Confidence: {confidence*100:.1f}%</h3>
                    <p>Model: <b>{selected_model_name}</b></p>
                    <p>Mode: <b>{mode}</b></p>
                </div>
                """, unsafe_allow_html=True)
            
            # ============================================================
            # PROBABILITY CHART
            # ============================================================
            
            st.markdown("#### 📊 Probability Distribution")
            
            species_names = label_encoder.classes_
            proba_df = pd.DataFrame({
                'Species': species_names,
                'Probability': prediction_proba[0] * 100
            })
            
            fig = px.bar(
                proba_df,
                x='Species',
                y='Probability',
                color='Species',
                color_discrete_sequence=px.colors.qualitative.Set2,
                title='Prediction Probability for Each Species',
                labels={'Probability': 'Probability (%)', 'Species': ''}
            )
            fig.update_layout(
                height=400,
                showlegend=False,
                yaxis_range=[0, 100]
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # ============================================================
            # CONFIDENCE GAUGE
            # ============================================================
            
            st.markdown("#### 🎯 Confidence Level")
            
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=confidence*100,
                title={'text': "Confidence Score (%)"},
                domain={'x': [0, 1], 'y': [0, 1]},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "#1e90ff"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 75], 'color': "gray"},
                        {'range': [75, 100], 'color': "darkgray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=250)
            st.plotly_chart(fig_gauge, use_container_width=True)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("Please check that all input values are valid numbers.")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    🐟 Mugilidae Fish Classification System | Hybrid ANN-Metaheuristic Approach
    <br> Developed by Nur Zulaifah binti Mohamad Zaini | Universiti Malaysia Terengganu
</div>
""", unsafe_allow_html=True)
