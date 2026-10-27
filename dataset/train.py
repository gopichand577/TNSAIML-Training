
# Step 1: Import required libraries
from pathlib import Path
import warnings
import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")


# Step 2: Define file paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = Path(__file__).resolve().parent / "aqi.csv"

MODEL_PATH = BASE_DIR / "model.joblib"
FEATURE_PATH = BASE_DIR / "feature_names.joblib"

# Step 3: Load dataset
if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_PATH}\n"
        "Place your Kaggle CSV at dataset/aqi.csv"
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

print("\n========== AQI DATASET ==========")
print("Dataset shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nDataset information:")
df.info()
print("\nStatistical summary:")
print(df.describe(include="all").T)
print("\nMissing values:")
print(df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())

# Step 4: Find the AQI target column
target_candidates = [
    "AQI",
    "AQI Value",
    "AQI_Value",
    "Air Quality Index",
    "Air_Quality_Index",
    "Air Quality Index (AQI)"
]

normalized_columns = {
    str(col).strip().lower().replace("_", " "): col
    for col in df.columns
}

target = None

for candidate in target_candidates:
    key = candidate.lower().replace("_", " ")
    if key in normalized_columns:
        target = normalized_columns[key]
        break

if target is None:
    raise ValueError(
        "Could not identify the numerical AQI target column.\n"
        f"Available columns: {df.columns.tolist()}\n"
        "Edit target_candidates in train.py to match your CSV."
    )

print("\nTarget column:", target)

# Step 5: Clean target values
df[target] = pd.to_numeric(df[target], errors="coerce")
df = df.dropna(subset=[target]).copy()

if df.empty:
    raise ValueError("No valid numeric AQI target values were found.")

# Step 6: Remove columns that cause target leakage
# AQI category/bucket columns reveal the target indirectly.
leakage_columns = [
    col for col in df.columns
    if col != target and any(
        word in col.lower().replace("_", " ")
        for word in ["aqi bucket", "aqi category",
                     "air quality category", "air quality bucket"]
    )
]

# Dates and unique record identifiers are not used as model features.
date_columns = [
    col for col in df.columns
    if col != target and (
        col.lower().strip() in ["date", "datetime", "timestamp"]
        or col.lower().strip().endswith("_id")
        or col.lower().strip() == "id"
    )
]

drop_columns = leakage_columns + date_columns
X = df.drop(columns=[target] + drop_columns, errors="ignore")
y = df[target]

# Remove columns with no usable values or a single unique value.
remove_features = [
    col for col in X.columns
    if X[col].dropna().nunique() <= 1
]

X = X.drop(columns=remove_features, errors="ignore")

if X.shape[1] == 0:
    raise ValueError("No usable feature columns remain in the dataset.")

# Convert numeric-looking object columns to numbers when possible.
for col in X.columns:
    if X[col].dtype == "object":
        converted = pd.to_numeric(X[col], errors="coerce")
        if converted.notna().mean() >= 0.90:
            X[col] = converted

# Normalize categorical values for reliable encoding.
for col in X.select_dtypes(include=["object", "category"]).columns:
    X[col] = X[col].apply(
        lambda value: str(value) if pd.notna(value) else np.nan
    )

feature_names = X.columns.tolist()

print("\nFeatures used:", feature_names)
print("Rows available for training:", len(X))

if len(X) < 10:
    raise ValueError(
        "At least 10 usable rows are required for this training example."
    )

# Step 7: Identify numeric and categorical features
numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()

categorical_features = [
    col for col in X.columns
    if col not in numeric_features
]

# Step 8: Create preprocessing pipelines
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

transformers = []

if numeric_features:
    transformers.append(
        ("numeric", numeric_pipeline, numeric_features)
    )

if categorical_features:
    transformers.append(
        ("categorical", categorical_pipeline, categorical_features)
    )

preprocessor = ColumnTransformer(transformers=transformers)

# Step 9: Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# Step 10: Train the Random Forest regression model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(X_train, y_train)

# Step 11: Evaluate using unseen test data
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R2): {r2:.4f}")

# Step 12: Save model and feature names
joblib.dump(model, MODEL_PATH)
joblib.dump(feature_names, FEATURE_PATH)

# Save metadata for the prediction application.
metadata = {
    "target": target,
    "numeric_features": numeric_features,
    "categorical_features": categorical_features,
    "metrics": {
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2)
    }
}

joblib.dump(metadata, BASE_DIR / "metadata.joblib")

print("\nModel saved:", MODEL_PATH)
print("Feature names saved:", FEATURE_PATH)
print("Metadata saved:", BASE_DIR / "metadata.joblib")
print("\nTraining completed successfully.")
