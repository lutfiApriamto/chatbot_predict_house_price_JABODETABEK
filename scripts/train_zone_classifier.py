# scripts/train_zone_classifier.py

import pandas as pd
import joblib
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# === Load data ===
print("📦 Membaca data...")
df = pd.read_csv("data/processed/preprocessed_v2.csv")

# Hapus baris tanpa label kategori harga
df = df.dropna(subset=["price_category"])

# Pisahkan fitur dan target
X = df.drop(columns=["log_price", "price_category"])
y = df["price_category"]

# Simpan nama kolom fitur
feature_columns = X.columns.tolist()
with open("models/feature_columns_zone.json", "w") as f:
    json.dump(feature_columns, f)

# Bagi data
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Inisialisasi dan latih model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "models/model_zone_classifier.pkl")
print("✅ Model klasifikasi zona berhasil disimpan ke models/model_zone_classifier.pkl")

# Evaluasi
y_pred = model.predict(X_test)
print("\n📊 Evaluasi Model:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
