# scripts/preprocessing/preprocess_final.py

import pandas as pd
import numpy as np
import json
import os

# ==== 1. Load data ====
df = pd.read_csv("data/raw/jabodetabek_house_price.csv")
print("Jumlah data awal:", df.shape)

# ==== 2. Normalisasi kolom kategori ====
for col in ["city", "district", "property_type"]:
    df[col] = df[col].astype(str).str.strip().str.lower()

# === Jumlah city unik setelah normalisasi ===
print("Jumlah city unik setelah normalisasi:", df["city"].nunique())

# ==== 3. Deteksi wilayah Jakarta dari address/district ====
def classify_jakarta_region(row):
    combined = f"{row.get('address', '')} {row['district']}".lower()
    for wilayah in ["selatan", "utara", "timur", "barat", "pusat"]:
        if f"jakarta {wilayah}" in combined:
            return f"jakarta {wilayah}"
    return row["city"]

df["city"] = df.apply(classify_jakarta_region, axis=1)

# === Menampilkan Ke terminal untuk ringkasan data perkota ===
print("\n--- 📊 Ringkasan Jumlah Data per Kota (Sebelum Dibersihkan) ---")
print("Total data awal:", len(df))
for kota in [
    "bekasi", "tangerang", "depok", "bogor", 
    "jakarta utara", "jakarta timur", "jakarta pusat", 
    "jakarta selatan", "jakarta barat"
]:
    print(f"- Jumlah data di kota {kota.title():<18}: {len(df[df['city'] == kota])}")

# ==== 4. Hapus data tanpa harga ====
df = df.dropna(subset=["price_in_rp"])

# === Jumlah data setelah hapus yang tidak ada harga ===
print("Jumlah data setelah hapus yang tidak ada harga:", df.shape)

# ==== 5. Konversi & isi missing values numerik ====
num_cols = ["bedrooms", "bathrooms", "land_area", "building_area", "carports", "floors", "garages"]

# === Jumlah missing value numerik sebelum imputasi ===
missing_before = df[num_cols].isna().sum().sum()


for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    df[col].fillna(df[col].median(), inplace=True)

# === Jumlah missing value numerik sebelum imputasi ===
print(f"Jumlah missing value numerik sebelum imputasi: {missing_before} → berhasil diisi.")

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

# === Menampilkan Ke terminal Proses IQR ===
print("\n--- 🔎 Proses Deteksi Outlier dengan IQR ---")
print(f"Q1  (25% terendah)     : Rp {Q1:,.0f}")
print(f"Q3  (75% terendah)     : Rp {Q3:,.0f}")
print(f"IQR (Q3 - Q1)          : Rp {IQR:,.0f}")
print(f"Lower Bound (Q1 - 1.5*IQR): Rp {lower:,.0f}")
print(f"Upper Bound (Q3 + 1.5*IQR): Rp {upper:,.0f}")

# (Opsional) tampilkan jumlah data sebelum dan sesudah
print("\nJumlah data setelah filter outlier:", len(df))

# ==== 9. Fitur tambahan ====
df["building_ratio"] = df["building_area"] / (df["land_area"] + 1)
df["room_ratio"] = df["bedrooms"] / (df["bathrooms"] + 1)
df["luas_per_kamar"] = df["building_area"] / (df["bedrooms"] + 1)
df["kamar_mandi_per_kamar"] = df["bathrooms"] / (df["bedrooms"] + 1)

# === Fitur tambahan berhasil dibuat ===
print("Fitur tambahan berhasil dibuat: building_ratio, room_ratio, luas_per_kamar, kamar_mandi_per_kamar.")

# ==== 10. Transformasi harga ====
df["log_price"] = np.log1p(df["price_in_rp"])

# === Transformasi harga selesai === 
print("Transformasi harga (log) selesai.")

# ==== 11. Label zona harga global (v2-style) ====
p33 = df["price_in_rp"].quantile(0.33)
p66 = df["price_in_rp"].quantile(0.66)

def global_zone_label(price):
    if price < p33:
        return "murah"
    elif price < p66:
        return "sedang"
    else:
        return "mahal"

df["price_category"] = df["price_in_rp"].apply(global_zone_label)

# === Label zona harga selesai (jumlah murah, sedang, mahal) ===
print("Label zona harga selesai:")
print(df["price_category"].value_counts().to_string())

# === Menampilkan kelayar Data seteleh dilakukan Preprocessing ===
print("\n--- 📊 Ringkasan Jumlah Data per Kota (Setelah Dibersihkan) ---")
print("Total data akhir:", len(df))
for kota in [
    "bekasi", "tangerang", "depok", "bogor", 
    "jakarta utara", "jakarta timur", "jakarta pusat", 
    "jakarta selatan", "jakarta barat"
]:
    print(f"- Jumlah data di kota {kota.title():<18}: {len(df[df['city'] == kota])}")

# ==== 12. One-hot encoding ====
df = pd.get_dummies(df, columns=["city", "district", "property_type"], drop_first=True)

# === One-hot encoding selesai (jumlah kolom fitur final) ===
print(f"One-hot encoding selesai: total {df.shape[1]} kolom fitur.")

# ==== 13. Simpan kolom fitur utama ====
features = num_cols + [
    "building_ratio", "room_ratio", "luas_per_kamar", "kamar_mandi_per_kamar"
] + [col for col in df.columns if col.startswith("city_") or col.startswith("property_type_") or col.startswith("district_")]

# === One-hot encoding selesai (jumlah kolom fitur yang digunakan oleh model) ===
print(f"One-hot encoding selesai: total {len(features)} kolom fitur digunakan oleh model.")

with open("models/feature_columns.json", "w") as f:
    json.dump(features, f)

# ==== 14. Simpan data model ====
df_model = df[features + ["log_price", "price_category", "price_in_rp"]]
df_model.to_csv("data/processed/preprocessed_final.csv", index=False)

# ==== 15. Simpan data untuk chatbot (zona berdasarkan kota) ====
# (gunakan data sebelum one-hot)
df_chatbot = pd.read_csv("data/raw/jabodetabek_house_price.csv")
df_chatbot = df_chatbot.dropna(subset=["price_in_rp"])
df_chatbot["city"] = df_chatbot.apply(classify_jakarta_region, axis=1)
df_chatbot["price_category"] = df_chatbot.groupby("city", group_keys=False)["price_in_rp"].transform(
    lambda x: pd.qcut(x, q=[0, 0.33, 0.66, 1.0], labels=["murah", "sedang", "mahal"])
)
df_chatbot = df_chatbot[["city", "district", "price_in_rp", "price_category", "land_area", "bedrooms", "bathrooms"]]
df_chatbot.to_csv("data/processed/zone_label.csv", index=False)

print("Jumlah data akhir:", df_model.shape)
print("Data berhasil diproses dan disimpan ✅")
