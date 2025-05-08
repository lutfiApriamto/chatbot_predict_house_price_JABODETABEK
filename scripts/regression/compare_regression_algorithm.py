# scripts/regression/model_comparison.py

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==== 1. Define Models ====
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

# ==== 2. Load Data ====
df = pd.read_csv("data/processed/preprocessed_final.csv")
with open("models/feature_columns.json", "r") as f:
    features = json.load(f)

X = df[features]
y = df["log_price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    "LinearRegression": LinearRegression(),
    "DecisionTree": DecisionTreeRegressor(),
    "RandomForest": RandomForestRegressor(),
    "GradientBoosting": GradientBoostingRegressor(),
    "XGBoost": XGBRegressor(objective='reg:squarederror'),
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

# ==== 4. Pilih model terbaik (berdasarkan R2) ====
best_model = max(results, key=lambda x: x["R2"])
print("\nModel terbaik berdasarkan R2 Score adalah:", best_model["model"])

# ==== 5. Simpan hasil evaluasi ====
pd.DataFrame(results).to_csv("output/model_comparison_result.csv", index=False)
print("Hasil evaluasi disimpan ke output/model_comparison_result.csv ✅")

# Konversi results ke DataFrame
df_results = pd.DataFrame(results)

# 1. Histogram MAE
plt.figure(figsize=(10, 6))
sns.barplot(x="model", y="MAE", data=df_results, palette="Blues_d")
plt.title("Perbandingan MAE antar Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Histogram RMSE
plt.figure(figsize=(10, 6))
sns.barplot(x="model", y="RMSE", data=df_results, palette="Greens_d")
plt.title("Perbandingan RMSE antar Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Histogram R2
plt.figure(figsize=(10, 6))
sns.barplot(x="model", y="R2", data=df_results, palette="Purples_d")
plt.title("Perbandingan R2 Score antar Model")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()