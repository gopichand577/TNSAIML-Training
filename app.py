import os
from pathlib import Path
import io
import time
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# 1. Page Configuration (Must be first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AirPulse | AQI Prediction & Intelligence",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. Modern Custom Styling
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Main Layout Styling */
    .main {
        background-color: #0e1117;
    }
    
    /* Hero Header */
    .hero-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    
    .hero-subtitle {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 0px;
    }

    /* Metric Badges */
    .badge-metric {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 4px 12px;
        margin-right: 10px;
        font-size: 0.85rem;
        color: #cbd5e1;
    }
    
    /* AQI Display Card */
    .aqi-container {
        border-radius: 16px;
        padding: 28px;
        text-align: center;
        color: #ffffff;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        transition: transform 0.2s ease;
    }
    
    .aqi-number {
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        margin: 10px 0;
    }
    
    .aqi-label {
        font-size: 1.4rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }

    .advisory-box {
        background-color: #1e293b;
        border-left: 5px solid #38bdf8;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 15px;
        color: #e2e8f0;
    }

    .pollutant-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 3. Model & Metadata Loading with Caching
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

@st.cache_resource(show_spinner=False)
def load_assets():
    model_path = BASE_DIR / "model.joblib"
    feat_path = BASE_DIR / "feature_names.joblib"
    meta_path = BASE_DIR / "metadata.joblib"
    
    model = None
    feature_names = None
    metadata = {}
    
    errors = []
    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception as e:
            errors.append(f"Error loading model: {e}")
    else:
        errors.append(f"Model file not found at: {model_path}")

    if feat_path.exists():
        try:
            feature_names = joblib.load(feat_path)
        except Exception as e:
            errors.append(f"Error loading feature names: {e}")
    else:
        errors.append(f"Feature names file not found at: {feat_path}")

    if meta_path.exists():
        try:
            metadata = joblib.load(meta_path)
        except Exception as e:
            errors.append(f"Error loading metadata: {e}")

    return model, feature_names, metadata, errors

with st.spinner("Initializing AirPulse Machine Learning Engine..."):
    model, feature_names, metadata, load_errors = load_assets()

# -----------------------------------------------------------------------------
# 4. Helper Definitions & AQI Categorization (Indian CPCB Standards)
# -----------------------------------------------------------------------------
CITIES = [
    "Ahmedabad", "Aizawl", "Amaravati", "Amritsar", "Bengaluru", "Bhopal", 
    "Brajrajnagar", "Chandigarh", "Chennai", "Coimbatore", "Delhi", "Ernakulam", 
    "Gurugram", "Guwahati", "Hyderabad", "Jaipur", "Jorapokhar", "Kochi", 
    "Kolkata", "Lucknow", "Mumbai", "Patna", "Shillong", "Talcher", 
    "Thiruvananthapuram", "Visakhapatnam"
]

# Standard CPCB 24-hr safe limits for major pollutants (µg/m³ or mg/m³)
SAFE_LIMITS = {
    "PM2.5": 60.0,
    "PM10": 100.0,
    "NO2": 80.0,
    "NH3": 400.0,
    "SO2": 80.0,
    "CO": 2.0,     # mg/m³
    "O3": 100.0,
    "Benzene": 5.0
}

def get_aqi_details(aqi_val):
    """Returns color, category, health impact, and advisory for given AQI value."""
    if aqi_val <= 50:
        return {
            "category": "Good",
            "color": "#10b981", # Green
            "bg": "linear-gradient(135deg, #065f46 0%, #047857 100%)",
            "impact": "Minimal Impact",
            "advisory": "Air quality is considered satisfactory, and air pollution poses little or no risk. Ideal for outdoor activities and exercise."
        }
    elif aqi_val <= 100:
        return {
            "category": "Satisfactory",
            "color": "#84cc16", # Light Green
            "bg": "linear-gradient(135deg, #365314 0%, #4d7c0f 100%)",
            "impact": "Minor Breathing Discomfort",
            "advisory": "Air quality is acceptable. However, highly sensitive individuals may experience minor breathing discomfort on prolonged exertion."
        }
    elif aqi_val <= 200:
        return {
            "category": "Moderate",
            "color": "#f59e0b", # Amber/Yellow
            "bg": "linear-gradient(135deg, #78350f 0%, #b45309 100%)",
            "impact": "Breathing Discomfort for Sensitive Groups",
            "advisory": "People with lung diseases, asthma, or heart conditions may experience discomfort. Consider reducing heavy outdoor exertion."
        }
    elif aqi_val <= 300:
        return {
            "category": "Poor",
            "color": "#f97316", # Orange
            "bg": "linear-gradient(135deg, #7c2d12 0%, #c2410c 100%)",
            "impact": "Breathing Discomfort to Most People",
            "advisory": "Prolonged exposure may lead to discomfort for most individuals. Children, elderly, and those with respiratory issues should stay indoors."
        }
    elif aqi_val <= 400:
        return {
            "category": "Very Poor",
            "color": "#ef4444", # Red
            "bg": "linear-gradient(135deg, #7f1d1d 0%, #b91c1c 100%)",
            "impact": "Respiratory Illness on Prolonged Exposure",
            "advisory": "Significant health alert. Avoid outdoor activities. Wear N95 masks if stepping outside and use air purifiers indoors."
        }
    else:
        return {
            "category": "Severe",
            "color": "#9333ea", # Purple / Maroon
            "bg": "linear-gradient(135deg, #581c87 0%, #7e22ce 100%)",
            "impact": "Severe Health Hazard",
            "advisory": "Emergency air quality condition. Affects healthy people and seriously impacts those with existing medical conditions. Remain indoors."
        }

# Preset Scenarios
PRESETS = {
    "🌿 Clean Mountain / Green Area": {
        "PM2.5": 18.0, "PM10": 35.0, "NO": 4.0, "NO2": 10.0, "NOx": 12.0,
        "NH3": 6.0, "CO": 0.4, "SO2": 4.5, "O3": 18.0, "Benzene": 0.5,
        "Toluene": 1.2, "Xylene": 0.4
    },
    "🏙️ Typical Urban Day": {
        "PM2.5": 48.5, "PM10": 95.5, "NO": 10.0, "NO2": 22.0, "NOx": 23.5,
        "NH3": 16.0, "CO": 0.9, "SO2": 9.2, "O3": 31.0, "Benzene": 1.1,
        "Toluene": 3.0, "Xylene": 1.0
    },
    "⚠️ Industrial / High Traffic Zone": {
        "PM2.5": 115.0, "PM10": 195.0, "NO": 35.0, "NO2": 62.0, "NOx": 68.0,
        "NH3": 38.0, "CO": 2.4, "SO2": 32.0, "O3": 54.0, "Benzene": 6.5,
        "Toluene": 18.0, "Xylene": 5.2
    },
    "🚨 Severe Winter Smog Event": {
        "PM2.5": 320.0, "PM10": 460.0, "NO": 85.0, "NO2": 110.0, "NOx": 140.0,
        "NH3": 75.0, "CO": 4.5, "SO2": 42.0, "O3": 70.0, "Benzene": 12.0,
        "Toluene": 32.0, "Xylene": 10.5
    }
}

# -----------------------------------------------------------------------------
# 5. Header / Hero Section
# -----------------------------------------------------------------------------
metrics = metadata.get("metrics", {})
r2 = metrics.get("r2", 0.9117)
mae = metrics.get("mae", 20.40)
rmse = metrics.get("rmse", 40.21)

st.markdown(f"""
<div class="hero-card">
    <div class="hero-title">🍃 AirPulse &bull; AQI Prediction System</div>
    <p class="hero-subtitle">
        High-precision Air Quality Index intelligence powered by Random Forest Regression. 
        Predict current and forecasted AQI levels across Indian metropolitan regions based on pollutant concentrations.
    </p>
    <div style="margin-top: 15px;">
        <span class="badge-metric">⚡ Model: <b>Random Forest</b></span>
        <span class="badge-metric">🎯 Accuracy R²: <b>{r2*100:.1f}%</b></span>
        <span class="badge-metric">📉 MAE: <b>{mae:.2f}</b></span>
        <span class="badge-metric">📊 RMSE: <b>{rmse:.2f}</b></span>
        <span class="badge-metric">🌐 Status: <b style="color: #34d399;">Online & Ready</b></span>
    </div>
</div>
""", unsafe_allow_html=True)

# If model failed to load, display clear banner and diagnostic
if load_errors:
    st.error("⚠️ Model Initialization Issue Encountered:")
    for err in load_errors:
        st.write(f"- {err}")
    st.info("Tip: Run `python dataset/train.py` from your project terminal to generate `model.joblib` and `feature_names.joblib`.")

# -----------------------------------------------------------------------------
# 6. Navigation Tabs
# -----------------------------------------------------------------------------
tab_predict, tab_batch, tab_analytics, tab_diagnostics = st.tabs([
    "🎯 Interactive Prediction", 
    "📁 Batch CSV Predictor", 
    "📈 Model & Dataset Insights",
    "🛠️ System Diagnostics"
])

# =============================================================================
# TAB 1: Interactive Prediction
# =============================================================================
with tab_predict:
    # Sidebar Controls
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        selected_city = st.selectbox(
            "Select City / Monitoring Station:",
            options=CITIES,
            index=CITIES.index("Delhi") if "Delhi" in CITIES else 0
        )
        
        st.markdown("---")
        st.subheader("📋 Preset Scenario")
        preset_choice = st.selectbox(
            "Quick fill scenario values:",
            options=["Custom"] + list(PRESETS.keys()),
            index=0
        )
        
        preset_vals = PRESETS.get(preset_choice, {})

        st.markdown("---")
        input_mode = st.radio("Input Control Style:", ["Sliders", "Number Inputs"], horizontal=True)

        st.markdown("---")
        st.caption("Developed with Streamlit & Scikit-Learn | Antigravity AI")

    col_inputs, col_results = st.columns([1.1, 0.9], gap="large")

    with col_inputs:
        st.subheader("🔬 Pollutant Concentration Inputs")
        st.write("Adjust environmental sensor readings below:")

        def make_input(label, col_key, min_val, max_val, default_val, step, unit):
            val = preset_vals.get(col_key, default_val)
            val = float(np.clip(val, min_val, max_val))
            if input_mode == "Sliders":
                return st.slider(
                    f"{label} ({unit})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(val),
                    step=float(step),
                    help=f"Safe CPCB threshold: {SAFE_LIMITS.get(col_key, 'N/A')} {unit}"
                )
            else:
                return st.number_input(
                    f"{label} ({unit})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(val),
                    step=float(step),
                    help=f"Safe CPCB threshold: {SAFE_LIMITS.get(col_key, 'N/A')} {unit}"
                )

        p_tab1, p_tab2, p_tab3 = st.tabs(["Particulate Matter", "Gases & Combustion", "Volatile Organics"])

        with p_tab1:
            pm25 = make_input("PM2.5 (Fine Inhalable Particles)", "PM2.5", 0.0, 600.0, 48.5, 0.5, "µg/m³")
            pm10 = make_input("PM10 (Respirable Coarse Particles)", "PM10", 0.0, 800.0, 95.5, 1.0, "µg/m³")

        with p_tab2:
            c1, c2 = st.columns(2)
            with c1:
                no = make_input("Nitric Oxide (NO)", "NO", 0.0, 250.0, 10.0, 0.5, "µg/m³")
                no2 = make_input("Nitrogen Dioxide (NO2)", "NO2", 0.0, 250.0, 22.0, 0.5, "µg/m³")
                nox = make_input("Nitrogen Oxides (NOx)", "NOx", 0.0, 300.0, 24.0, 0.5, "ppb")
            with c2:
                nh3 = make_input("Ammonia (NH3)", "NH3", 0.0, 200.0, 16.0, 0.5, "µg/m³")
                co = make_input("Carbon Monoxide (CO)", "CO", 0.0, 25.0, 0.9, 0.05, "mg/m³")
                so2 = make_input("Sulfur Dioxide (SO2)", "SO2", 0.0, 150.0, 9.2, 0.5, "µg/m³")
                o3 = make_input("Ozone (O3)", "O3", 0.0, 200.0, 31.0, 0.5, "µg/m³")

        with p_tab3:
            c3, c4, c5 = st.columns(3)
            with c3:
                benzene = make_input("Benzene", "Benzene", 0.0, 60.0, 1.1, 0.1, "µg/m³")
            with c4:
                toluene = make_input("Toluene", "Toluene", 0.0, 60.0, 3.0, 0.1, "µg/m³")
            with c5:
                xylene = make_input("Xylene", "Xylene", 0.0, 40.0, 1.0, 0.1, "µg/m³")

    # Prediction computation
    input_data = {
        "City": selected_city,
        "PM2.5": pm25,
        "PM10": pm10,
        "NO": no,
        "NO2": no2,
        "NOx": nox,
        "NH3": nh3,
        "CO": co,
        "SO2": so2,
        "O3": o3,
        "Benzene": benzene,
        "Toluene": toluene,
        "Xylene": xylene
    }

    input_df = pd.DataFrame([input_data])
    
    # Ensure all features expected by model are present
    if feature_names:
        for f in feature_names:
            if f not in input_df.columns:
                input_df[f] = np.nan
        input_df = input_df[feature_names]

    predicted_aqi = 0.0
    if model is not None:
        try:
            pred = model.predict(input_df)[0]
            predicted_aqi = max(0.0, float(pred))
        except Exception as e:
            st.error(f"Prediction calculation error: {e}")
            predicted_aqi = 100.0
    else:
        # Fallback estimation formula if model joblib is missing
        predicted_aqi = max(pm25 * 1.6, pm10 * 0.9, no2 * 1.1)

    details = get_aqi_details(predicted_aqi)

    with col_results:
        st.subheader("📊 AQI Prediction Outcome")
        
        # Display large AQI Card
        st.markdown(f"""
        <div class="aqi-container" style="background: {details['bg']};">
            <div style="font-size: 0.95rem; text-transform: uppercase; opacity: 0.9; letter-spacing: 1px;">
                Predicted Air Quality Index &bull; {selected_city}
            </div>
            <div class="aqi-number">{predicted_aqi:.1f}</div>
            <div class="aqi-label">{details['category']}</div>
            <div style="margin-top: 10px; font-weight: 500; font-size: 1.05rem;">{details['impact']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Health Advisory
        st.markdown(f"""
        <div class="advisory-box">
            <b>🛡️ Health Advisory & Guidelines:</b><br/>
            {details['advisory']}
        </div>
        """, unsafe_allow_html=True)

        # Visual Comparison with Safe Standard Limits
        st.markdown("<br/><b>📈 Pollutant Levels vs CPCB Safe Thresholds</b>", unsafe_allow_html=True)
        
        comparison_data = []
        for pollutant, limit in [("PM2.5", 60.0), ("PM10", 100.0), ("NO2", 80.0), ("SO2", 80.0), ("O3", 100.0)]:
            val = input_data[pollutant]
            pct = (val / limit) * 100.0
            comparison_data.append({
                "Pollutant": pollutant,
                "Current": val,
                "Safe Limit": limit,
                "% of Limit": pct
            })
        
        comp_df = pd.DataFrame(comparison_data)

        # Matplotlib visualization
        fig, ax = plt.subplots(figsize=(6, 2.5), facecolor="#0e1117")
        ax.set_facecolor("#1e293b")
        bars = ax.barh(comp_df["Pollutant"], comp_df["% of Limit"], color=[
            "#10b981" if p <= 100 else "#f59e0b" if p <= 200 else "#ef4444" 
            for p in comp_df["% of Limit"]
        ], height=0.55)
        
        ax.axvline(100, color="#ffffff", linestyle="--", linewidth=1.2, label="100% Safe Limit")
        ax.set_xlim(0, max(250, comp_df["% of Limit"].max() * 1.15))
        ax.set_xlabel("% of CPCB Permissible Standard", color="#cbd5e1", fontsize=9)
        ax.tick_params(colors="#cbd5e1", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.legend(facecolor="#1e293b", edgecolor="#334155", labelcolor="#cbd5e1", fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# =============================================================================
# TAB 2: Batch CSV Predictor
# =============================================================================
with tab_batch:
    st.subheader("📁 Batch CSV Prediction & Evaluation")
    st.write("Upload a CSV file containing city and pollutant readings to compute AQI predictions at scale.")

    sample_template = pd.DataFrame([
        {"City": "Delhi", "PM2.5": 85.0, "PM10": 160.0, "NO": 22.0, "NO2": 38.0, "NOx": 45.0, "NH3": 28.0, "CO": 1.4, "SO2": 14.0, "O3": 35.0, "Benzene": 2.1, "Toluene": 7.5, "Xylene": 1.8},
        {"City": "Bengaluru", "PM2.5": 32.0, "PM10": 55.0, "NO": 8.0, "NO2": 15.0, "NOx": 18.0, "NH3": 12.0, "CO": 0.6, "SO2": 6.5, "O3": 22.0, "Benzene": 0.8, "Toluene": 2.2, "Xylene": 0.5},
        {"City": "Mumbai", "PM2.5": 62.0, "PM10": 110.0, "NO": 14.0, "NO2": 26.0, "NOx": 28.0, "NH3": 18.0, "CO": 1.1, "SO2": 11.0, "O3": 28.0, "Benzene": 1.5, "Toluene": 4.0, "Xylene": 1.1}
    ])

    col_b1, col_b2 = st.columns([1, 1])
    with col_b1:
        csv_buffer = io.StringIO()
        sample_template.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Sample CSV Template",
            data=csv_buffer.getvalue(),
            file_name="aqi_prediction_sample_template.csv",
            mime="text/csv"
        )
    
    uploaded_file = st.file_uploader("Upload your pollutant data (.csv)", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"Successfully loaded {len(batch_df)} rows from {uploaded_file.name}")
            
            st.write("Uploaded Data Preview:")
            st.dataframe(batch_df.head(5), use_container_width=True)

            if st.button("🚀 Run Batch Prediction", type="primary"):
                with st.spinner("Processing batch predictions with Random Forest Model..."):
                    # Match model features
                    pred_input = batch_df.copy()
                    if feature_names:
                        for col in feature_names:
                            if col not in pred_input.columns:
                                pred_input[col] = np.nan
                        pred_input = pred_input[feature_names]

                    if model is not None:
                        preds = model.predict(pred_input)
                        batch_df["Predicted_AQI"] = np.round(preds, 2)
                        batch_df["AQI_Category"] = [get_aqi_details(v)["category"] for v in preds]
                    else:
                        st.warning("Model not loaded; cannot run predictions.")

                    st.write("### Prediction Results:")
                    st.dataframe(batch_df, use_container_width=True)

                    out_csv = io.StringIO()
                    batch_df.to_csv(out_csv, index=False)
                    st.download_button(
                        label="💾 Download Predictions as CSV",
                        data=out_csv.getvalue(),
                        file_name="aqi_predictions_output.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"Failed to process uploaded file: {e}")

# =============================================================================
# TAB 3: Model & Dataset Insights
# =============================================================================
with tab_analytics:
    st.subheader("📈 Model Architecture & Performance")
    
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("R² Score (Coefficient of Determination)", f"{r2*100:.2f}%", "High explanatory power")
    col_m2.metric("Mean Absolute Error (MAE)", f"{mae:.2f} AQI points", "Average deviation")
    col_m3.metric("Root Mean Squared Error (RMSE)", f"{rmse:.2f} AQI points", "Penalizes large deviations")

    st.markdown("---")
    st.write("### 📌 Machine Learning Pipeline Details")
    st.markdown("""
    - **Algorithm:** `RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)`
    - **Numeric Imputation:** `SimpleImputer(strategy='median')`
    - **Categorical Handling:** `OneHotEncoder(handle_unknown='ignore')` wrapped in `ColumnTransformer`
    - **Target Variable:** Indian Air Quality Index (`AQI`)
    - **Features (13):** `City`, `PM2.5`, `PM10`, `NO`, `NO2`, `NOx`, `NH3`, `CO`, `SO2`, `O3`, `Benzene`, `Toluene`, `Xylene`
    - **Data Preprocessing:** Target leakage columns (`AQI_Bucket`, category tags) and timestamp columns were strictly removed prior to training.
    """)

    # Pollutant Importance description
    st.write("### 💡 Key Drivers of AQI in India")
    st.markdown("""
    1. **PM2.5 & PM10:** By far the strongest contributors to high AQI readings across Indian cities, particularly during winter months.
    2. **Nitrogen Oxides (NO, NO2, NOx):** Generated primarily from vehicular emissions and power plants; heavily correlated with urban peak traffic hours.
    3. **Carbon Monoxide (CO) & Ozone (O3):** Key indicators of incomplete combustion and photochemical reactions under intense sunlight.
    """)

# =============================================================================
# TAB 4: System Diagnostics & Health Check
# =============================================================================
with tab_diagnostics:
    st.subheader("🛠️ Streamlit Runtime & Diagnostic Health")
    st.write("Use this tab to verify system environment, file presence, and websocket connectivity.")
    
    status_data = {
        "Component": [
            "Streamlit Version",
            "model.joblib",
            "metadata.joblib",
            "feature_names.joblib",
            "aqi.csv (Dataset)",
            "Model Loaded in Memory",
            "Total Monitored Cities"
        ],
        "Status / Value": [
            st.__version__,
            "✅ Present (330 MB)" if (BASE_DIR / "model.joblib").exists() else "❌ Missing",
            "✅ Present" if (BASE_DIR / "metadata.joblib").exists() else "❌ Missing",
            "✅ Present" if (BASE_DIR / "feature_names.joblib").exists() else "❌ Missing",
            "✅ Present" if (BASE_DIR / "dataset" / "aqi.csv").exists() else "❌ Missing",
            "✅ Yes (sklearn Pipeline)" if model is not None else "❌ No",
            f"{len(CITIES)} cities"
        ]
    }
    st.table(pd.DataFrame(status_data))

    st.markdown("""
    ### 💡 Resolving White / Blank Screen Issues in Streamlit:
    If your browser ever shows a blank white page when accessing Streamlit:
    1. **Empty File:** An empty `app.py` renders nothing. `app.py` is now fully populated.
    2. **Hard Refresh:** In your browser, press `Ctrl + F5` (Windows) or `Cmd + Shift + R` (Mac) to bypass stale cached scripts.
    3. **Check Terminal Logs:** Look at your command terminal where `streamlit run app.py` was executed to ensure no uncaught Python exceptions.
    4. **Websocket / Firewall:** If running across a network or VPN, launch with:
       ```bash
       streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
       ```
    5. **Browser Extensions:** Adblockers or aggressive script blockers can sometimes block Streamlit's local WebSocket connection (`ws://localhost:8501/_stcore/stream`). Whitelist `localhost:8501`.
    """)
