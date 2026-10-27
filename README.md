# TNSAIML-Training

## 🍃 Air Quality Index (AQI) Prediction & Intelligence System

A machine learning and interactive web application designed to forecast and analyze the Air Quality Index (AQI) across major Indian metropolitan cities based on ambient pollutant concentrations.

---

### 📌 Overview

The system uses a **Random Forest Regressor** trained on historical air quality records across Indian cities. It predicts the continuous AQI score and categorizes air quality according to the **Central Pollution Control Board (CPCB)** standards (Good, Satisfactory, Moderate, Poor, Very Poor, Severe), providing health advisories and threshold warnings.

### 📊 Model Performance

- **Algorithm:** Random Forest Regressor (`n_estimators=200`, `random_state=42`)
- **Coefficient of Determination ($R^2$):** **91.17%**
- **Mean Absolute Error (MAE):** **20.40** AQI points
- **Root Mean Squared Error (RMSE):** **40.21** AQI points

---

### 🔬 Features & Pollutants Tracked

| Feature Category | Parameters |
|---|---|
| **Location** | City / Monitoring Station (26 Indian cities) |
| **Particulate Matter** | PM2.5, PM10 |
| **Nitrogen Compounds** | NO, NO2, NOx, NH3 |
| **Gases & Combustion** | CO, SO2, O3 |
| **Volatile Organics** | Benzene, Toluene, Xylene |

---

### 🚀 Getting Started

#### 1. Clone the Repository
```bash
git clone https://github.com/gopichand577/TNSAIML-Training.git
cd TNSAIML-Training
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Train the Model (Optional / If generating model.joblib)
Because the pre-trained `model.joblib` file is ~315 MB (exceeding standard GitHub storage limits), you can train or regenerate it locally in seconds:
```bash
python dataset/train.py
```
This will automatically generate `model.joblib`, `feature_names.joblib`, and `metadata.joblib`.

#### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Then navigate to `http://localhost:8501` in your browser.

---

### 🖥️ Dashboard Features

1. **Interactive Predictor:** Adjust pollutant sliders or select quick real-world presets (*Clean Mountain Air*, *Typical Urban Day*, *Industrial Zone*, *Winter Smog Event*).
2. **Instant AQI Category & Advisory:** Live gauge, health risks, and precautionary guidance.
3. **CPCB Standard Limits Comparison:** Visual chart comparing current pollutant readings against 24-hour safe permissible limits.
4. **Batch CSV Predictor:** Upload multi-row environmental readings and download predictions with categorized outcomes.
5. **System Health & Diagnostics:** Live status of pipeline artifacts, model status, and runtime environment.

---

### 📂 Project Structure

```text
├── dataset/
│   ├── aqi.csv                 # Kaggle historical air quality dataset
│   └── train.py                # Model training, preprocessing, & evaluation pipeline
├── app.py                      # Streamlit interactive web application
├── feature_names.joblib        # Stored feature columns
├── metadata.joblib             # Model metrics, target variable, & column types
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules (excludes >100MB model.joblib)
└── README.md                   # Project documentation
```

---

### 📜 License
This project is developed as part of TNSAIML-Training.
