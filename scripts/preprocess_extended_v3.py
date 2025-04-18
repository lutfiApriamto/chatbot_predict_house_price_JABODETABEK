# scripts/preprocess_extended_v3.py

import pandas as pd
import numpy as np
import os
import json

# ==== 1. Load data ====
df = pd.read_csv("data/raw/jabodetabek_house_price.csv")
print("Jumlah data awal:", df.shape)

# ==== 2. Normalisasi kolom kategori ====
for col in ["city", "district", "property_type"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

# ==== 3. Deteksi wilayah Jakarta dari district/address ====
def classify_jakarta_region(row):
    combined = f"{row['address']} {row['district']}".lower()
    for wilayah in ["selatan", "utara", "timur", "barat", "pusat"]:
        if f"jakarta {wilayah}" in combined:
            return f"jakarta {wilayah}"
    return row['city']

df['city'] = df.apply(classify_jakarta_region, axis=1)

# ==== 4. Hapus data tanpa harga ====
df = df.dropna(subset=["price_in_rp"])

# ==== 5. Konversi & isi missing values numerik ====
num_cols = ["bedrooms", "bathrooms", "land_area", "building_area", "carports", "floors", "garages"]
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col].fillna(df[col].median(), inplace=True)

# ==== 6. Lat Long ====
df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
df["long"] = pd.to_numeric(df["long"], errors="coerce")
df["lat"].fillna(df["lat"].median(), inplace=True)
df["long"].fillna(df["long"].median(), inplace=True)

num_cols += ["lat", "long"]

# ==== 7. Isi missing value kategori ====
df["city"].fillna("unknown", inplace=True)
df["district"].fillna("unknown", inplace=True)
df["property_type"].fillna("unknown", inplace=True)

# ==== 8. Hapus outlier harga (IQR) ====
Q1 = df["price_in_rp"].quantile(0.25)
Q3 = df["price_in_rp"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df["price_in_rp"] >= lower) & (df["price_in_rp"] <= upper)]

# ==== 9. Fitur tambahan ====
df["building_ratio"] = df["building_area"] / (df["land_area"] + 1)
df["room_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
df["luas_per_kamar"] = df["building_area"] / (df["bedrooms"] + 1)
df["kamar_mandi_per_kamar"] = df["bathrooms"] / (df["bedrooms"] + 1)

# ==== 10. Transformasi harga ====
df["log_price"] = np.log1p(df["price_in_rp"])

# ==== 11. Label zona harga berdasarkan kota ====
def label_per_kota(sub_df):
    p33 = sub_df["price_in_rp"].quantile(0.33)
    p66 = sub_df["price_in_rp"].quantile(0.66)
    def zone(price):
        if price < p33:
            return "murah"
        elif price < p66:
            return "sedang"
        else:
            return "mahal"
    return sub_df["price_in_rp"].apply(zone)

df["price_category"] = df.groupby("city", group_keys=False).apply(label_per_kota)

# Simpan city dan district asli sebelum one-hot
zone_label_df = df[[
    "city", "district", "price_in_rp", "price_category",
    "land_area", "bedrooms", "bathrooms"
]].copy()

# ==== 12. One-hot encoding ====
df = pd.get_dummies(df, columns=["city", "district", "property_type"], drop_first=True)

# ==== 13. Simpan ke file ====
features = num_cols + [
    "building_ratio", "room_ratio", "luas_per_kamar", "kamar_mandi_per_kamar"
] + [col for col in df.columns if col.startswith("city_") or col.startswith("property_type_") or col.startswith("district_")]

df_model = df[features + ["log_price", "price_category"]]

# Simpan kolom fitur
with open("models/feature_columns.json", "w") as f:
    json.dump(features, f)

# Simpan label zona harga yang bisa digunakan untuk chatbot
zone_label_df.to_csv("data/processed/zone_label.csv", index=False)

# Simpan dataset final untuk modeling
df_model.to_csv("data/processed/preprocessed_v3.csv", index=False)

print("Jumlah data akhir:", df_model.shape)
print("Data berhasil diproses dan disimpan ✅")
