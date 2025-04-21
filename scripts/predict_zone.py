# scripts/predict_zone.py

import pandas as pd
import numpy as np
import joblib
import json

# === 1. Load model dan fitur ===
with open("models/feature_columns_zone.json", "r") as f:
    feature_columns = json.load(f)

model = joblib.load("models/model_zone_classifier.pkl")

# === 2. Contoh input spesifikasi ===
input_spec = {
    "bedrooms": 3,
    "bathrooms": 2,
    "land_area": 100,
    "building_area": 80,
    "carports": 1,
    "floors": 1,
    "garages": 0,
    "lat": -6.3,
    "long": 106.8
}

# === 3. Bangun dataframe input ===
row = {
    "building_ratio": input_spec["building_area"] / (input_spec["land_area"] + 1),
    "room_ratio": input_spec["bedrooms"] / (input_spec["bathrooms"] + 1),
    "luas_per_kamar": input_spec["building_area"] / (input_spec["bedrooms"] + 1),
    "kamar_mandi_per_kamar": input_spec["bathrooms"] / (input_spec["bedrooms"] + 1),
}
row.update(input_spec)

input_df = pd.DataFrame([row])
input_df = input_df.reindex(columns=feature_columns, fill_value=0)

# === 4. Prediksi zona ===
predicted_zone = model.predict(input_df)[0]
print(f"\n📍 Zona harga yang diprediksi: {predicted_zone.upper()}")

# === 5. Cari lokasi contoh dari data asli ===
df_zone = pd.read_csv("data/processed/zone_label.csv")

# Pastikan kolom district dan lainnya bersih
df_zone["district"] = df_zone["district"].astype(str).str.strip().str.lower()
df_zone.dropna(subset=["land_area", "bedrooms", "bathrooms"], inplace=True)

# Langsung gunakan df_zone tanpa merge
df_zone_full = df_zone.copy()

# Filter zona yang sama & spesifikasi mirip
mask = (
    (df_zone_full["price_category"] == predicted_zone) &
    (abs(df_zone_full["land_area"] - input_spec["land_area"]) <= 20) &
    (abs(df_zone_full["bedrooms"] - input_spec["bedrooms"]) <= 1) &
    (abs(df_zone_full["bathrooms"] - input_spec["bathrooms"]) <= 1)
)

matching_rows = df_zone_full[mask]
sample_districts = matching_rows["district"].value_counts().head(3).index.tolist()

if sample_districts:
    print(f"📌 Contoh lokasi di zona ini: {', '.join([d.title() for d in sample_districts])}")
else:
    print("⚠️ Tidak ditemukan contoh lokasi yang cocok dalam zona ini.")
