# scripts/zone_price_estimator.py

import pandas as pd
import joblib
import numpy as np
import json
import random

# === Load model dan data ===
print("\n📦 Load model dan data...")

# Model untuk klasifikasi zona dan regresi harga
model_zone = joblib.load("models/model_zone_classifier.pkl")
model_regression = joblib.load("models/model_best_regression.pkl")

# Gunakan fitur yang benar untuk masing-masing model
with open("models/feature_columns_zone.json") as f:
    zone_features = json.load(f)

with open("models/feature_columns.json") as f:
    regression_features = json.load(f)

# Dataset hasil preprocessing
df = pd.read_csv("data/processed/preprocessed_final.csv")
df_zone_raw = pd.read_csv("data/processed/zone_label.csv")

# === Contoh input spesifikasi ===
sample_input = {
    "bedrooms": 3,
    "bathrooms": 2,
    "land_area": 120,
    "building_area": 90,
    "carports": 1,
    "floors": 1,
    "garages": 0,
    "lat": df["lat"].median(),
    "long": df["long"].median()
}

# Fitur tambahan
sample_input["building_ratio"] = sample_input["building_area"] / (sample_input["land_area"] + 1)
sample_input["room_ratio"] = sample_input["bedrooms"] / (sample_input["bathrooms"] + 1)
sample_input["luas_per_kamar"] = sample_input["building_area"] / (sample_input["bedrooms"] + 1)
sample_input["kamar_mandi_per_kamar"] = sample_input["bathrooms"] / (sample_input["bedrooms"] + 1)

# Lengkapi dummy kolom untuk model zona
for col in zone_features:
    if col not in sample_input:
        sample_input[col] = 0

X_zone = pd.DataFrame([sample_input])[zone_features]

# === Prediksi zona harga ===
predicted_zone = model_zone.predict(X_zone)[0]
print(f"\n📍 Zona harga yang diprediksi: {predicted_zone.upper()}")

# === Ambil sampel data dari zona tersebut ===
df_zone = df[df["price_category"] == predicted_zone]

if df_zone.empty:
    print("❌ Tidak ada data tersedia untuk zona ini.")
else:
    samples = df_zone.sample(n=min(2, len(df_zone)), random_state=42)
    for i, row in samples.iterrows():
        input_row = row[regression_features]
        harga_log = model_regression.predict([input_row])[0]
        harga_rupiah = round(np.expm1(harga_log), -4)

        # Cari nama distrik asli (jika tersedia)
        matching_district = [col for col in row.index if col.startswith("district_") and row[col] == 1]
        nama_distrik = "(tidak diketahui)"
        if matching_district:
            nama_distrik = matching_district[0].replace("district_", "").replace("_", " ").title()

        print(f"\n📦 Zona harga: {predicted_zone.upper()}")
        print(f"📍 Contoh lokasi: {nama_distrik}")
        print(f"💰 Estimasi harga: Rp {harga_rupiah:,.0f}")
