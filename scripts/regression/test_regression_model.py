# scripts/regression/test_regression_model.py

import pandas as pd
import joblib
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

# 1. Load model dan data
model = joblib.load("models/model_best_regression.pkl")
df = pd.read_csv("data/processed/preprocessed_final.csv")

with open("models/feature_columns.json", "r") as f:
    features = json.load(f)

X = df[features]
y_actual = df["price_in_rp"]  # ambil target asli (dalam rupiah)
y_log = df["log_price"]

# 2. Prediksi log_price → kembalikan ke skala rupiah
y_pred_log = model.predict(X)
y_pred = np.expm1(y_pred_log)

# 3. Evaluasi akurasi
mae = mean_absolute_error(y_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
r2 = r2_score(y_actual, y_pred)

print("\n📊 Hasil Uji Coba Model Regresi terhadap Data Aktual:")
print(f"MAE  : Rp {mae:,.0f}")
print(f"RMSE : Rp {rmse:,.0f}")
print(f"R2   : {r2:.4f}")

# 4. Buat DataFrame perbandingan
df_result = pd.DataFrame({
    "Harga Aktual (Rp)": y_actual,
    "Harga Prediksi (Rp)": y_pred,
    "Residual": y_actual - y_pred
})

# 5. Visualisasi 1: Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Harga Aktual (Rp)", y="Harga Prediksi (Rp)", data=df_result, alpha=0.5)
plt.plot([y_actual.min(), y_actual.max()], [y_actual.min(), y_actual.max()], '--', color='red')
plt.title("📈 Scatter Plot: Harga Aktual vs Prediksi")
plt.xlabel("Harga Aktual (Rp)")
plt.ylabel("Harga Prediksi (Rp)")
plt.tight_layout()
plt.show()

