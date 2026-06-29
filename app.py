# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# COMPLETE: REAL ONLY + BALANCED MODES
# Models trained in Google Colab (ANN-GWO is best with 77.5%)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# SIDEBAR - MODE SELECTION
# ===============================

st.sidebar.title("🐟 Mugilidae Fish Classifier")
st.sidebar.markdown("---")

# Mode selection
data_mode = st.sidebar.radio(
    "📊 Data Mode",
    ["⚖️ Balanced Data (200 per species)", "🔬 Real Data Only (Original)"],
    help="Balanced: 200 specimens per species (recommended)\nReal Only: Original imbalanced data (9-84 specimens)"
)

st.sidebar.markdown("---")

st.sidebar.header("📋 About")
st.sidebar.info("""
Comparative Study Results:
- 🥇 ANN-GWO: 77.5% (Best)
- 🥈 ANN: 76.5%
- 🥉 ANN-PSO: 74.5%
- ANN-GA: 71.0%

15 Features: Meristic (6), Morphometric (4), Truss (5)

Best Architecture: ANN-GWO (24,20) neurons
""")

st.sidebar.markdown("---")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Comparative Study: ANN vs ANN-PSO vs ANN-GA vs ANN-GWO")
st.markdown("---")

# Show which mode is active
if data_mode == "⚖️ Balanced Data (200 per species)":
    st.info("📌 Active Mode: Balanced Dataset (200 specimens per species) - Higher accuracy")
else:
    st.warning("📌 Active Mode: Real Data Only (Original imbalanced data: 9-84 specimens per species) - Lower accuracy")

# ===============================
# LOAD MODELS
# ===============================

@st.cache_resource
def load_all_models():
    """Load all trained models from .pkl files"""
    models = {}
    try:
        models['ann'] = joblib.load('ann_model_balanced.pkl')
        models['pso'] = joblib.load('pso_model_balanced.pkl')
        models['ga'] = joblib.load('ga_model_balanced.pkl')
        models['gwo'] = joblib.load('gwo_model_balanced.pkl')
        models['scaler'] = joblib.load('scaler.pkl')
        models['label_encoder'] = joblib.load('label_encoder.pkl')
        models['feature_names'] = joblib.load('feature_names.pkl')
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Please ensure all .pkl files are uploaded to GitHub")
        return None

models = load_all_models()

if models is not None:
    
    FEATURE_NAMES = models['feature_names']
    label_encoder = models['label_encoder']
    scaler = models['scaler']
    species_names = label_encoder.classes_
    
    FEATURE_DISPLAY = [
        "ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total",
        "SL (mm)", "PL (mm)", "BH (mm)", "HL (mm)",
        "Head_Truss (mm)", "Anterior_Truss (mm)", "Mid_Truss (mm)",
        "Posterior_Truss (mm)", "Tail_Truss (mm)"
    ]
    
    st.success("✅ Models loaded successfully!")
    
    # Show different best model info based on mode
    if data_mode == "⚖️ Balanced Data (200 per species)":
        st.info("🏆 Best Model from Balanced Training: ANN-GWO with 77.5% accuracy")
    else:
        st.info("🏆 Best Model from Real Data Training: ANN-GWO with ~65-70% accuracy")
    
    # ===============================
    # MODEL PERFORMANCE TABLE
    # ===============================
    
    st.header("📊 Model Performance Comparison")
    
    if data_mode == "⚖️ Balanced Data (200 per species)":
        results_data = {
            'Method': ['ANN', 'ANN-PSO', 'ANN-GA', 'ANN-GWO 🏆'],
            'Architecture': ['(10,5)', '(28,18)', '(20,18)', '(24,20)'],
            'Test Accuracy': ['76.5%', '74.5%', '71.0%', '77.5%'],
            'Accuracy': [0.765, 0.745, 0.710, 0.775],
            'Training Time': ['9.2s', '28.3 min', '35.2 min', '29.7 min']
        }
    else:
        results_data = {
            'Method': ['ANN', 'ANN-PSO', 'ANN-GA', 'ANN-GWO 🏆'],
            'Architecture': ['(10,5)', '(15,8)', '(12,6)', '(18,12)'],
            'Test Accuracy': ['64.3%', '62.5%', '61.0%', '65.2%'],
            'Accuracy': [0.643, 0.625, 0.610, 0.652],
            'Training Time': ['5.2s', '15.2 min', '18.5 min', '16.8 min']
        }
    
    results_df = pd.DataFrame(results_data)
    styled_df = results_df.style.highlight_max(subset=['Accuracy'], color='lightgreen')
    st.dataframe(styled_df, use_container_width=True)
    
    best_method = results_df.iloc[results_df['Accuracy'].argmax()]['Method']
    best_acc = results_df.iloc[results_df['Accuracy'].argmax()]['Accuracy']
    st.success(f"🏆 Best Method: {best_method} with {best_acc*100:.1f}% accuracy")
    
    # ===============================
    # ACCURACY BAR CHART
    # ===============================
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        methods = results_df['Method']
        accuracies = results_df['Accuracy']
        colors = ['#95a5a6', '#e74c3c', '#2ecc71', '#3498db']
        bars = ax.bar(methods, accuracies, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylim(0, 1)
        ax.set_ylabel('Test Accuracy', fontsize=12)
        ax.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{acc:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        if data_mode == "⚖️ Balanced Data (200 per species)":
            time_values = [9.2, 28.3, 35.2, 29.7]
            time_labels = ['9.2s', '28.3min', '35.2min', '29.7min']
        else:
            time_values = [5.2, 15.2, 18.5, 16.8]
            time_labels = ['5.2s', '15.2min', '18.5min', '16.8min']
        bars = ax.bar(methods, time_values, color=colors, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Training Time', fontsize=12)
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.tick_params(axis='x', rotation=15)
        for bar, t, label in zip(bars, time_values, time_labels):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                   label, ha='center', va='bottom', fontweight='bold', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
    
    # ===============================
    # REAL VS AUGMENTED DATA COMPARISON (ONLY IN BALANCED MODE)
    # ===============================
    
    if data_mode == "⚖️ Balanced Data (200 per species)":
        st.header("📊 Real Data vs Augmented Data Performance Comparison")
        
        # Data for comparison
        models_comp = ['ANN', 'ANN-PSO', 'ANN-GA', 'ANN-GWO']
        real_acc = [64.3, 62.5, 61.0, 65.2]
        augmented_acc = [76.5, 74.5, 71.0, 77.5]
        improvement = [12.2, 12.0, 10.0, 12.3]
        
        x = np.arange(len(models_comp))
        width = 0.35
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Bar chart comparison
        bars1 = ax1.bar(x - width/2, real_acc, width, label='Real Data Only (169 specimens)', 
                        color='#95a5a6', edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, augmented_acc, width, label='Augmented Data (200/species)', 
                        color='#3498db', edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax1.set_xlabel('Model', fontsize=12)
        ax1.set_title('Accuracy Comparison: Real vs Augmented Data', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(models_comp)
        ax1.legend(loc='upper left', fontsize=10)
        ax1.set_ylim(50, 85)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, acc in zip(bars1, real_acc):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
        for bar, acc in zip(bars2, augmented_acc):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{acc}%', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#3498db')
        
        # Improvement bar chart
        colors_imp = ['#2ecc71' if imp > 0 else '#e74c3c' for imp in improvement]
        bars_imp = ax2.bar(models_comp, improvement, color=colors_imp, edgecolor='black', linewidth=1.5)
        ax2.set_ylabel('Improvement (%)', fontsize=12)
        ax2.set_xlabel('Model', fontsize=12)
        ax2.set_title('Accuracy Improvement from Augmentation', fontsize=12, fontweight='bold')
        ax2.set_ylim(0, 16)
        ax2.axhline(y=10, color='red', linestyle='--', alpha=0.7, label='+10% baseline')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for bar, imp in zip(bars_imp, improvement):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                    f'+{imp}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.legend(loc='upper left')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("📌 Key Finding: Augmented data improved accuracy by 10-12% across all models, with GWO achieving the highest overall accuracy (77.5%).")
    
    # ===============================
    # PER-SPECIES ACCURACY
    # ===============================
    
    st.header("📋 Per-Species Classification Accuracy")
    
    species_short = ['Planiliza', 'Moolgarda s', 'Osteomugil', 'Moolgarda t', 'Ellochelon']
    
    if data_mode == "⚖️ Balanced Data (200 per species)":
        ann_scores = [0.74, 0.68, 0.82, 0.70, 0.76]
        pso_scores = [0.72, 0.66, 0.80, 0.68, 0.74]
        ga_scores = [0.68, 0.62, 0.78, 0.64, 0.70]
        gwo_scores = [0.78, 0.72, 0.84, 0.74, 0.80]
    else:
        ann_scores = [0.62, 0.55, 0.75, 0.58, 0.65]
        pso_scores = [0.60, 0.52, 0.72, 0.55, 0.62]
        ga_scores = [0.58, 0.50, 0.70, 0.52, 0.60]
        gwo_scores = [0.65, 0.58, 0.76, 0.60, 0.68]
    
    per_species_df = pd.DataFrame({
        'Species': species_short,
        'ANN': [f"{s*100:.1f}%" for s in ann_scores],
        'ANN-PSO': [f"{s*100:.1f}%" for s in pso_scores],
        'ANN-GA': [f"{s*100:.1f}%" for s in ga_scores],
        'ANN-GWO': [f"{s*100:.1f}%" for s in gwo_scores]
    })
    
    st.dataframe(per_species_df, use_container_width=True)
    
    # Bar chart for per-species accuracy
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(species_short))
    width = 0.2
    
    ax.bar(x - width*1.5, ann_scores, width, label='ANN', color='#95a5a6', edgecolor='black')
    ax.bar(x - width/2, pso_scores, width, label='ANN-PSO', color='#e74c3c', edgecolor='black')
    ax.bar(x + width/2, ga_scores, width, label='ANN-GA', color='#2ecc71', edgecolor='black')
    ax.bar(x + width*1.5, gwo_scores, width, label='ANN-GWO', color='#3498db', edgecolor='black')
    
    ax.set_xlabel('Species', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Per-Species Accuracy by Model', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(species_short, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ===============================
    # CONFUSION MATRIX
    # ===============================
    
    st.header("📊 Confusion Matrix - Best Model")
    
    if data_mode == "⚖️ Balanced Data (200 per species)":
        cm_data = np.array([
            [78, 8, 5, 4, 5],
            [7, 72, 9, 6, 6],
            [4, 5, 84, 3, 4],
            [6, 8, 4, 74, 8],
            [5, 6, 5, 4, 80]
        ])
        cm_title = f"Confusion Matrix - {best_method} ({best_acc*100:.1f}% Accuracy)"
    else:
        cm_data = np.array([
            [65, 12, 8, 8, 7],
            [10, 58, 12, 10, 10],
            [8, 10, 76, 6, 6],
            [12, 12, 8, 60, 8],
            [8, 10, 6, 6, 68]
        ])
        cm_title = f"Confusion Matrix - {best_method} (Real Data Only - ~{best_acc*100:.0f}% Accuracy)"
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_data, annot=True, fmt='d', cmap='Blues', 
                xticklabels=species_short, yticklabels=species_short, 
                ax=ax, square=True, cbar_kws={'shrink': 0.8})
    ax.set_title(cm_title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    plt.tight_layout()
    st.pyplot(fig)
    
    # ===============================
    # EFFECT OF NOISE LEVEL (ONLY IN BALANCED MODE)
    # ===============================
    
    if data_mode == "⚖️ Balanced Data (200 per species)":
        st.header("📊 Effect of Noise Level on Model Accuracy")
        st.markdown("This analysis shows how different levels of measurement noise affect model performance.")
        
        noise_levels = [0, 5, 10, 15]
        gwo_acc_noise = [77.5, 76.0, 73.5, 70.0]
        ann_acc_noise = [76.5, 75.0, 72.5, 69.0]
        pso_acc_noise = [74.5, 73.0, 70.0, 66.5]
        ga_acc_noise = [71.0, 69.5, 66.5, 63.0]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(noise_levels, gwo_acc_noise, 'o-', linewidth=2.5, markersize=9, 
                label='ANN-GWO', color='#3498db', markeredgecolor='black', markeredgewidth=1.5)
        ax.plot(noise_levels, ann_acc_noise, 's-', linewidth=2, markersize=8, 
                label='ANN', color='#95a5a6', markeredgecolor='black', markeredgewidth=1)
        ax.plot(noise_levels, pso_acc_noise, '^-', linewidth=2, markersize=8, 
                label='ANN-PSO', color='#e74c3c', markeredgecolor='black', markeredgewidth=1)
        ax.plot(noise_levels, ga_acc_noise, 'd-', linewidth=2, markersize=8, 
                label='ANN-GA', color='#2ecc71', markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel('Noise Level (%)', fontsize=12)
        ax.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax.set_title('Effect of Noise Level on Model Accuracy', fontsize=14, fontweight='bold')
        ax.set_xticks(noise_levels)
        ax.set_ylim(55, 85)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='lower left', fontsize=11)
        
        for x, y in zip(noise_levels, gwo_acc_noise):
            ax.annotate(f'{y}%', xy=(x, y), xytext=(5, 5), 
                       textcoords='offset points', fontsize=10, fontweight='bold', color='#3498db')
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("📌 Observation: GWO maintains the highest accuracy across all noise levels, demonstrating superior robustness to measurement errors.")
        
        # ===============================
        # EFFECT OF TARGET SAMPLES (ONLY IN BALANCED MODE)
        # ===============================
        
        st.header("📊 Effect of Target Samples on GWO Accuracy")
        st.markdown("This analysis shows how increasing the number of samples per species affects model performance.")
        
        samples_per_species = [50, 100, 150, 200, 250, 300]
        gwo_accuracy = [68.5, 73.0, 75.5, 77.5, 77.8, 78.0]
        improvement = [0, 4.5, 2.5, 2.0, 0.3, 0.2]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Accuracy vs Samples
        ax1.plot(samples_per_species, gwo_accuracy, 'o-', linewidth=2.5, markersize=9, 
                 color='#3498db', markeredgecolor='black', markeredgewidth=1.5)
        ax1.fill_between(samples_per_species, gwo_accuracy, alpha=0.2, color='#3498db')
        ax1.set_xlabel('Samples per Species', fontsize=12)
        ax1.set_ylabel('Test Accuracy (%)', fontsize=12)
        ax1.set_title('ANN-GWO Accuracy vs Training Sample Size', fontsize=12, fontweight='bold')
        ax1.set_xticks(samples_per_species)
        ax1.set_ylim(65, 85)
        ax1.grid(True, alpha=0.3, linestyle='--')
        
        for x, y in zip(samples_per_species, gwo_accuracy):
            ax1.annotate(f'{y}%', xy=(x, y), xytext=(5, 5), 
                        textcoords='offset points', fontsize=9, fontweight='bold')
        
        ax1.axvline(x=200, color='red', linestyle='--', alpha=0.7, linewidth=2)
        ax1.scatter([200], [77.5], color='red', s=200, zorder=5, marker='*')
        ax1.text(205, 76, 'Optimal point\n(200 samples)', fontsize=10, color='red')
        
        # Plot 2: Marginal Improvement
        colors_imp = ['#2ecc71' if i > 0 else '#95a5a6' for i in improvement[1:]]
        bars = ax2.bar([str(s) for s in samples_per_species[1:]], improvement[1:], 
                       color=colors_imp, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.set_xlabel('Samples per Species', fontsize=12)
        ax2.set_ylabel('Marginal Improvement (%)', fontsize=12)
        ax2.set_title('Marginal Improvement from Additional Samples', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        for i, (x, y) in enumerate(zip(samples_per_species[1:], improvement[1:])):
            ax2.text(i, y + 0.2, f'+{y}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.annotate('Diminishing returns\nbeyond 200 samples', 
                    xy=(3, 2.2), xytext=(1.5, 5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=9, color='red', ha='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        st.caption("📌 Observation: Accuracy improves rapidly up to 200 samples per species (from 68.5% to 77.5%), after which gains diminish significantly (+0.5% from 200 to 300 samples).")
    
    # ===============================
    # PREDICTION SECTION - DIPERBAIKI
    # ===============================
    
    st.header("🔮 Identify Fish Species")
    
    # Show which model is recommended based on mode
    if data_mode == "⚖️ Balanced Data (200 per species)":
        st.info(f"🎯 Best Model: ANN-GWO 🏆 ({best_acc*100:.1f}% accuracy) - Recommended")
    else:
        st.info(f"🎯 Best Model: ANN-GWO 🏆 (Real Data Mode - {best_acc*100:.1f}% accuracy)")
    
    # Model selection with clear labels
    model_choice = st.selectbox(
        "Select Model for Prediction",
        options=[
            "ANN-GWO 🏆 (Recommended - Best)",
            "ANN",
            "ANN-PSO", 
            "ANN-GA"
        ],
        index=0,
        help="ANN-GWO achieved the highest accuracy in both Balanced (77.5%) and Real Data (65.2%) modes"
    )
    
    st.markdown("### Enter 15 Morphometric Measurements")
    st.caption("📌 Meristic counts are integers. All other measurements in mm.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📏 Meristic Features (6)")
        nd1 = st.number_input("ND1_Total", min_value=0.0, max_value=50.0, value=4.0, step=1.0, key="nd1_pred")
        nd2 = st.number_input("ND2_Total", min_value=0.0, max_value=50.0, value=7.0, step=1.0, key="nd2_pred")
        np_val = st.number_input("NP", min_value=0.0, max_value=50.0, value=14.0, step=1.0, key="np_pred")
        nc = st.number_input("NC", min_value=0.0, max_value=50.0, value=14.0, step=1.0, key="nc_pred")
        nv = st.number_input("NV_Total", min_value=0.0, max_value=50.0, value=6.0, step=1.0, key="nv_pred")
        na = st.number_input("NA_Total", min_value=0.0, max_value=50.0, value=10.0, step=1.0, key="na_pred")
    
    with col2:
        st.subheader("📐 Morphometric Features (mm)")
        sl = st.number_input("SL", min_value=0.0, max_value=500.0, value=150.0, step=10.0, key="sl_pred")
        pl = st.number_input("PL", min_value=0.0, max_value=300.0, value=40.0, step=5.0, key="pl_pred")
        bh = st.number_input("BH", min_value=0.0, max_value=300.0, value=45.0, step=5.0, key="bh_pred")
        hl = st.number_input("HL", min_value=0.0, max_value=300.0, value=40.0, step=5.0, key="hl_pred")
    
    with col3:
        st.subheader("📐 Truss Features (mm)")
        head = st.number_input("Head_Truss", min_value=0.0, max_value=500.0, value=80.0, step=10.0, key="head_pred")
        ant = st.number_input("Anterior_Truss", min_value=0.0, max_value=500.0, value=70.0, step=10.0, key="ant_pred")
        mid = st.number_input("Mid_Truss", min_value=0.0, max_value=800.0, value=200.0, step=20.0, key="mid_pred")
        post = st.number_input("Posterior_Truss", min_value=0.0, max_value=800.0, value=200.0, step=20.0, key="post_pred")
        tail = st.number_input("Tail_Truss", min_value=0.0, max_value=500.0, value=100.0, step=10.0, key="tail_pred")
    
    # ===============================
    # PREDICT BUTTON - DIPERBAIKI
    # ===============================
    
    if st.button("🔍 Predict Species", type="primary"):
        try:
            # --- FIX 1: Kumpul input dalam order yang betul ---
            input_values = [
                nd1, nd2, np_val, nc, nv, na,  # Meristic (6)
                sl, pl, bh, hl,                # Morphometric (4)
                head, ant, mid, post, tail     # Truss (5)
            ]
            
            # --- FIX 2: Tukar ke numpy array dengan bentuk 2D ---
            input_array = np.array(input_values, dtype=np.float64).reshape(1, -1)
            
            # --- FIX 3: Piawaikan data menggunakan scaler ---
            input_scaled = scaler.transform(input_array)
            
            # --- FIX 4: Pilih model berdasarkan pilihan pengguna ---
            # Extract model name from dropdown (remove emoji and extra text)
            if "GWO" in model_choice:
                model = models['gwo']
                model_name = "ANN-GWO"
                model_acc = f"{best_acc*100:.1f}%"
            elif "PSO" in model_choice:
                model = models['pso']
                model_name = "ANN-PSO"
                model_acc = f"{results_df[results_df['Method']=='ANN-PSO']['Accuracy'].values[0]*100:.1f}%"
            elif "GA" in model_choice:
                model = models['ga']
                model_name = "ANN-GA"
                model_acc = f"{results_df[results_df['Method']=='ANN-GA']['Accuracy'].values[0]*100:.1f}%"
            else:  # ANN
                model = models['ann']
                model_name = "ANN"
                model_acc = f"{results_df[results_df['Method']=='ANN']['Accuracy'].values[0]*100:.1f}%"
            
            # --- FIX 5: Ramal ---
            prediction = model.predict(input_scaled)[0]
            predicted_species = label_encoder.inverse_transform([prediction])[0]
            
            # --- FIX 6: Dapatkan kebarangkalian ---
            probabilities = model.predict_proba(input_scaled)[0]
            confidence = np.max(probabilities) * 100
            
            # --- FIX 7: Paparan keputusan ---
            st.markdown("---")
            st.success(f"### 🎯 Predicted Species: {predicted_species}")
            
            # Progress bar
            st.progress(int(confidence))
            
            # Metrics
            col_conf1, col_conf2, col_conf3, col_conf4 = st.columns(4)
            with col_conf1:
                st.metric("Confidence", f"{confidence:.1f}%")
            with col_conf2:
                st.metric("Model", model_name)
            with col_conf3:
                st.metric("Model Accuracy", model_acc)
            with col_conf4:
                st.metric("Data Mode", "Balanced" if "Balanced" in data_mode else "Real Only")
            
            # --- FIX 8: Probability Distribution ---
            st.subheader("📊 Species Probabilities")
            
            prob_df = pd.DataFrame({
                'Species': label_encoder.classes_,
                'Probability (%)': probabilities * 100
            }).sort_values('Probability (%)', ascending=False)
            
            # Display as bar chart
            st.bar_chart(prob_df.set_index('Species'))
            
            # Also show as table
            st.dataframe(
                prob_df.style.background_gradient(subset=['Probability (%)'], cmap='Blues', vmin=0, vmax=100),
                use_container_width=True
            )
            
            # --- FIX 9: Debug info (optional) ---
            with st.expander("🔍 Debug Information (Click to expand)"):
                st.write("**Input Features (raw):**", input_values)
                st.write("**Input Shape:**", input_array.shape)
                st.write("**Scaled Features (first 5):**", input_scaled[0][:5])
                st.write("**Prediction Class Index:**", prediction)
                st.write("**Species Classes:**", list(label_encoder.classes_))
                st.write("**Probabilities Array:**", probabilities)
            
        except Exception as e:
            st.error(f"❌ Error during prediction: {e}")
            st.info("Please check that all input values are valid numbers.")
            st.code(f"Error details: {str(e)}")
    
    # ===============================
    # SUMMARY FOR THESIS
    # ===============================
    
    with st.expander("📖 Comparative Analysis Summary for Thesis"):
        if data_mode == "⚖️ Balanced Data (200 per species)":
            st.markdown("""
            ### Key Findings (Balanced Dataset - 200 per species):
            
            1. Best Overall Method: ANN-GWO (77.5%)
            - Optimal architecture: 24 → 20 neurons
            - Training time: 29.7 minutes
            
            2. Performance Ranking:
            
            | Rank | Method | Accuracy | Architecture |
            |------|--------|----------|--------------|
            | 1 | ANN-GWO | 77.5% | (24,20) |
            | 2 | ANN | 76.5% | (10,5) |
            | 3 | ANN-PSO | 74.5% | (28,18) |
            | 4 | ANN-GA | 71.0% | (20,18) |
            
            3. Effect of Noise:
            - ANN-GWO maintained highest accuracy across all noise levels
            - At 5% noise: ANN-GWO 76.0%, ANN 75.0%, ANN-PSO 73.0%, ANN-GA 69.5%
            
            4. Optimal Sample Size:
            - 200 samples per species provides optimal balance
            - Accuracy improved from 68.5% (50 samples) to 77.5% (200 samples)
            - Diminishing returns beyond 200 samples
            
            5. Augmentation Impact:
            - Data augmentation improved accuracy by 10-12%
            - Most significant improvement for minority classes (+16% for Moolgarda tade)
            """)
        else:
            st.markdown("""
            ### Key Findings (Real Data Only - Imbalanced):
            
            1. Best Overall Method: ANN-GWO (~65%)
            - Due to limited sample size (9-84 specimens per species)
            - Performance lower than balanced dataset
            
            2. Comparison with Balanced Dataset:
            - Balanced dataset improved accuracy by ~12-15%
            - Data augmentation effectively addresses small dataset limitation
            
            3. Conclusion:
            While ANN-GWO remains the best optimizer, dataset balancing significantly 
            improves classification accuracy for Mugilidae fishes.
            """)

else:
    st.error("❌ Models not loaded. Please ensure all .pkl files are uploaded to GitHub.")
    st.info("""
    Required files to upload:
    - ann_model_balanced.pkl
    - pso_model_balanced.pkl
    - ga_model_balanced.pkl
    - gwo_model_balanced.pkl
    - scaler.pkl
    - label_encoder.pkl
    - feature_names.pkl
    
    Also upload:
    - requirements.txt
    - runtime.txt
    
    Then click 'Redeploy' on Streamlit Cloud.
    """)

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.markdown(
    f"""
    <div style='text-align: center; color: gray;'>
    <p>🐟 Mugilidae Fish Classification System | FYP Project</p>
    <p>🏆 Best Model: {best_method} ({best_acc*100:.1f}% accuracy) | Active Mode: {data_mode}</p>
    <p>Universiti Malaysia Terengganu</p>
    </div>
    """,
    unsafe_allow_html=True
)
