# scripts/model_comparison.py

import pandas as pd
import numpy as np
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==== 1. Load Data ====
df = pd.read_csv("data/processed/preprocessed_v3.csv")
with open("models/feature_columns.json", "r") as f:
    features = json.load(f)

X = df[features]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==== 2. Define Models ====
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

models = {
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(),
    "Lasso": Lasso(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "SVR": SVR(),
    "XGBoost": XGBRegressor(objective='reg:squarederror'),
    "LightGBM": LGBMRegressor(),
}

# ==== 3. Train & Evaluate ====
results = []

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    })

    print(f"Model {name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2: {r2:.4f}")

# ==== 4. Pilih model terbaik (berdasarkan R2) dan simpan ====
best_model = max(results, key=lambda x: x["R2"])
final_model = models[best_model["model"]]
final_model.fit(X, y)
joblib.dump(final_model, "models/model_best_regression.pkl")

print("\nModel terbaik:", best_model["model"])
print("Model disimpan ke models/model_best_regression.pkl")

# ==== 5. Simpan hasil evaluasi ====
pd.DataFrame(results).to_csv("output/model_comparison_result.csv", index=False)
print("Hasil evaluasi disimpan ke output/model_comparison_result.csv ✅")
