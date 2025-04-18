# chatbot/chatbot_helpers.py

import joblib
import numpy as np
import pandas as pd
import random
import json

# Load model & metadata
zone_model = joblib.load("models/model_zone_classifier.pkl")
price_model = joblib.load("models/model_best_regression.pkl")

with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

with open("models/feature_columns_zone.json", "r") as f:
    zone_feature_columns = json.load(f)

zone_df = pd.read_csv("data/processed/zone_label.csv")
raw_df = pd.read_csv("data/raw/jabodetabek_house_price.csv")

def predict_zone_from_features(input_row):
    # Buat dummy kosong untuk semua fitur zona
    input_zone = input_row.copy()

    # Tambahkan kolom one-hot kosong kalau belum ada
    for col in zone_feature_columns:
        if col not in input_zone.columns:
            input_zone[col] = 0

    # Hanya ambil kolom yang dipakai model
    input_zone = input_zone[zone_feature_columns]

    return zone_model.predict(input_zone)[0]


def estimate_price_in_zone(input_row, zone_label, sample_n=2):
    filtered_df = zone_df[zone_df["price_category"] == zone_label]
    if filtered_df.empty:
        return []

    samples = filtered_df.sample(n=min(sample_n, len(filtered_df)), random_state=42)

    prices = []
    for _, row in samples.iterrows():
        modified_input = input_row.copy()
        modified_input["log_price"] = row["log_price"]  # optional, bisa diabaikan
        y_pred_log = price_model.predict(modified_input[feature_columns])[0]
        y_pred = round(np.expm1(y_pred_log))
        prices.append((row["price_in_rp"], y_pred))

    return prices

def get_district_examples(zone_label, kota=None, n=2):
    df = raw_df.copy()
    df["city"] = df["city"].str.strip().str.lower()  # Pindahkan ini ke awal
    df["price_category"] = df["price_in_rp"].apply(lambda x:
        "murah" if x < zone_df["price_in_rp"].quantile(0.33) else
        "sedang" if x < zone_df["price_in_rp"].quantile(0.66) else "mahal"
    )
    filtered = df[df["price_category"] == zone_label]

    if kota:
        kota = kota.strip().lower()
        filtered = filtered[filtered["city"].str.contains(kota)]

    districts = filtered["district"].dropna().unique().tolist()

    if not districts:
        return ["Belum tersedia"] * n

    return random.sample(districts, min(n, len(districts)))


def build_zone_price_response(input_row, kota=None):
    zone = predict_zone_from_features(input_row)
    locations = get_district_examples(zone, kota)
    prices = estimate_price_in_zone(input_row, zone)

    response = f"📍 Zona harga yang diprediksi: {zone.upper()}\n"
    for i, (real_price, est_price) in enumerate(prices):
        district = locations[i] if i < len(locations) and locations[i] else "Belum tersedia"
        response += (
            f"\n📦 Zona harga: {zone.upper()}\n"
            f"📍 Contoh lokasi: {district.title()}\n"
            f"💰 Estimasi harga: Rp {est_price:,.0f}\n"
        )
    return response.strip()

def get_spec_from_budget(budget_rp, kota=None):
    df = raw_df.copy()

    if kota:
        kota = kota.strip().lower()
        df["city"] = df["city"].astype(str).str.strip().str.lower()
        df = df[df["city"].str.contains(kota)]

    # Log debugging
    print("\n[DEBUG] === get_spec_from_budget ===")
    print(f"Kota      : {kota}")
    print(f"Budget    : Rp {budget_rp:,.0f}")
    print(f"Data cocok: {len(df)} baris")

    # Filter sesuai budget
    df = df[df["price_in_rp"] <= budget_rp].sort_values("price_in_rp", ascending=False)

    # Jika tidak ditemukan, berikan saran rumah terdekat (fallback)
    if df.empty:
        print("[DEBUG] Tidak ada rumah di bawah budget.")
        closest_df = raw_df.copy()
        if kota:
            closest_df["city"] = closest_df["city"].astype(str).str.strip().str.lower()
            closest_df = closest_df[closest_df["city"].str.contains(kota)]

        if closest_df.empty:
            return f"Maaf, kami belum menemukan rumah di {kota.title()}."

        closest_df["selisih"] = abs(closest_df["price_in_rp"] - budget_rp)
        closest_df = closest_df.sort_values("selisih")
        top = closest_df.iloc[0]

        return (
            f"Maaf, tidak ada rumah di {kota.title()} untuk budget tersebut. "
            f"Tapi Anda bisa mempertimbangkan rumah di {top['district'].title()} seharga "
            f"Rp {top['price_in_rp']:,.0f} dengan spesifikasi:\n"
            f"- Luas tanah: {top['land_area']} m2\n"
            f"- Luas bangunan: {top['building_area']} m2\n"
            f"- {top['bedrooms']} kamar tidur\n"
            f"- {top['bathrooms']} kamar mandi"
        )

    # Jika ditemukan rumah yang sesuai
    top = df.iloc[0]
    return (
        f"📊 Dengan budget sekitar Rp {budget_rp:,.0f}, Anda bisa mendapatkan rumah di {top['district'].title()} dengan spesifikasi:\n"
        f"- Luas tanah: {top['land_area']} m2\n"
        f"- Luas bangunan: {top['building_area']} m2\n"
        f"- {top['bedrooms']} kamar tidur\n"
        f"- {top['bathrooms']} kamar mandi"
    )
