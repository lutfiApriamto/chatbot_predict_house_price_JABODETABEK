# scripts/regression/train_regression_final.py

import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np

# 1. Load Data
df = pd.read_csv("data/processed/preprocessed_final.csv")
with open("models/feature_columns.json", "r") as f:
    features = json.load(f)

X = df[features]
y = df["log_price"]

# 2. Train Model dengan eval_set untuk rekam metrik pelatihan
eval_set = [(X, y)]  # gunakan data yang sama karena kita tidak split
model = XGBRegressor(objective='reg:squarederror', n_estimators=100, eval_metric="mae")
model.fit(X, y, eval_set=eval_set, verbose=False)

# 3. Simpan model ke file
joblib.dump(model, "models/model_best_regression.pkl")
print("✅ Model XGBoost berhasil dilatih dan disimpan.")

# 4. Prediksi dan hitung metrik performa
y_pred = model.predict(X)
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(((y - y_pred) ** 2).mean())
r2 = r2_score(y, y_pred)

print(f"\n🎯 Evaluasi Model XGBoost:")
print(f"MAE  : {mae:.4f}")
print(f"RMSE : {rmse:.4f}")
print(f"R2   : {r2:.4f}")

# 5. Visualisasi 1: Learning Curve (MAE)
results = model.evals_result()
plt.figure(figsize=(10, 5))
plt.plot(results['validation_0']['mae'], label='MAE')
plt.title("Learning Curve: MAE selama Pelatihan")
plt.xlabel("Iterasi")
plt.ylabel("MAE")
plt.legend()
plt.tight_layout()
plt.show()

# 6. Visualisasi 3: Histogram Residuals (Opsional)
residuals = y - y_pred
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=40, color="gray", edgecolor="black")
plt.title("Distribusi Residuals (Aktual - Prediksi)")
plt.xlabel("Residual")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.show()
