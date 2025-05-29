import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# 1. Load data dan model
df = pd.read_csv("data/processed/preprocessed_final.csv")
model = joblib.load("models/model_zone_classifier.pkl")

with open("models/feature_columns_zone.json", "r") as f:
    features = json.load(f)

X = df[features]
y_actual = df["price_category"]

# 2. Prediksi zona
y_pred = model.predict(X)

# 3. Evaluasi metrik
acc = accuracy_score(y_actual, y_pred)
prec = precision_score(y_actual, y_pred, average='macro', zero_division=0)
rec = recall_score(y_actual, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_actual, y_pred, average='macro', zero_division=0)

print("📊 Evaluasi Model Klasifikasi Zona:")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1-Score : {f1:.4f}")
print("\n📋 Classification Report:")
print(classification_report(y_actual, y_pred))

# 4. Confusion Matrix
conf_matrix = confusion_matrix(y_actual, y_pred, labels=sorted(y_actual.unique()))

plt.figure(figsize=(6, 5))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=sorted(y_actual.unique()), yticklabels=sorted(y_actual.unique()))
plt.title("Confusion Matrix - Model Klasifikasi Zona")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# 5. Visualisasi: Bar Plot Hasil Prediksi vs Aktual
df_result = pd.DataFrame({
    "Actual": y_actual,
    "Predicted": y_pred
})

# Hitung frekuensi
actual_counts = df_result['Actual'].value_counts().sort_index()
predicted_counts = df_result['Predicted'].value_counts().sort_index()

# Gabungkan ke dalam satu DataFrame
compare_df = pd.DataFrame({
    "Aktual": actual_counts,
    "Prediksi": predicted_counts
}).fillna(0)

# Plot side-by-side bar
compare_df.plot(kind="bar", figsize=(8, 5), color=["#FF7F0E", "#1F77B4"])
plt.title("Perbandingan Jumlah Zona: Aktual vs Prediksi")
plt.xlabel("Zona Harga")
plt.ylabel("Jumlah")
plt.xticks(rotation=0)
plt.legend(title="Label")
plt.tight_layout()
plt.show()

