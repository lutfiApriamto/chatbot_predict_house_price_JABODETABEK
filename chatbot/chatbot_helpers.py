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
    input_zone = input_row.copy()
    for col in zone_feature_columns:
        if col not in input_zone.columns:
            input_zone[col] = 0
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
        modified_input["log_price"] = np.log1p(row["price_in_rp"])
        y_pred_log = price_model.predict(modified_input[feature_columns])[0]
        y_pred = round(np.expm1(y_pred_log))
        prices.append((row["price_in_rp"], y_pred))
    return prices


def get_district_examples(zone_label, kota=None, n=2):
    df = zone_df.copy()  # gunakan hasil preprocessing final
    df["city"] = df["city"].astype(str).str.strip().str.lower()
    df["district"] = df["district"].astype(str).str.strip().str.lower()
    
    # Filter berdasarkan zona
    df = df[df["price_category"] == zone_label]
    
    # Filter berdasarkan kota jika ada
    if kota:
        kota = kota.strip().lower()
        df = df[df["city"].str.contains(kota, case=False, na=False)]

    # Ambil daftar distrik unik
    districts = df["district"].dropna().unique().tolist()
    if not districts:
        return ["Belum tersedia"] * n
    return random.sample(districts, min(n, len(districts)))

def build_zone_price_response(input_row, kota=None):
    zone = predict_zone_from_features(input_row)
    sample_n = 2
    locations = get_district_examples(zone, kota, n=sample_n)
    prices = estimate_price_in_zone(input_row, zone, sample_n=sample_n)
    # Pastikan jumlah locations cukup
    while len(locations) < len(prices):
        locations.append("Belum tersedia")
    response = (
    f"Rumah dengan spesifikasi yang Anda sebutkan yaitu:\n"
    f"- {int(input_row['bedrooms'][0])} kamar tidur\n"
    f"- {int(input_row['bathrooms'][0])} kamar mandi\n"
    f"- Luas tanah {int(input_row['land_area'][0])} m²\n"
    f"- Luas bangunan {int(input_row['building_area'][0])} m²\n"
    f"berlokasi di kota {kota.title()}.\n\n"
    f"Kota ini termasuk ke dalam zona {zone.upper()}.\n"
)

    for i, (real_price, est_price) in enumerate(prices):
        district = locations[i] if i < len(locations) and locations[i] else "Belum tersedia"
        response += (
            f"\n📍 Contoh lokasi: {district.title()}\n"
            f"💰 Estimasi harga: Rp {est_price:,.0f}"
        )
    return response.strip()

def get_spec_from_budget(budget_rp, kota=None, verbose=False):
    df = raw_df.copy()
    if kota:
        kota = kota.strip().lower()
        df["city"] = df["city"].astype(str).str.strip().str.lower()
        df = df[df["city"].str.contains(kota, case=False, na=False)]

    if verbose:
        print("\n[DEBUG] === get_spec_from_budget ===")
        print(f"Kota      : {kota}")
        print(f"Budget    : Rp {budget_rp:,.0f}")
        print(f"Data cocok: {len(df)} baris")

    df = df[df["price_in_rp"] <= budget_rp].sort_values("price_in_rp", ascending=False)

    if df.empty:
        if verbose:
            print("[DEBUG] Tidak ada rumah di bawah budget.")
        closest_df = raw_df.copy()
        if kota:
            closest_df["city"] = closest_df["city"].astype(str).str.strip().str.lower()
            closest_df = closest_df[closest_df["city"].str.contains(kota, case=False, na=False)]

        if closest_df.empty:
            return f"Maaf, kami belum menemukan rumah di {kota.title()}."

        closest_df["selisih"] = abs(closest_df["price_in_rp"] - budget_rp)
        closest_df = closest_df.sort_values("selisih")
        top = closest_df.iloc[0]

        return (
            f"Maaf, tidak ada rumah di {kota.title()} untuk budget tersebut.\n"
            f"Namun, Anda bisa mempertimbangkan rumah di {top['district'].title()} seharga Rp {top['price_in_rp']:,.0f} dengan spesifikasi:\n"
            f"- Luas tanah: {top['land_area']} m2\n"
            f"- Luas bangunan: {top['building_area']} m2\n"
            f"- {top['bedrooms']} kamar tidur\n"
            f"- {top['bathrooms']} kamar mandi"
        )

    top = df.iloc[0]
    return (
        f"Dari hasil analisis dan data yang kami miliki, rumah dengan harga sekitar Rp {budget_rp:,.0f} di kota {kota.title()} cocok dengan {len(df)} data.\n"
        f"Spesifikasi yang mungkin Anda dapatkan adalah:\n"
        f"- Luas tanah: {top['land_area']} m2\n"
        f"- Luas bangunan: {top['building_area']} m2\n"
        f"- {top['bedrooms']} kamar tidur\n"
        f"- {top['bathrooms']} kamar mandi\n"
        f"📍 Contoh lokasi: {top['district'].title()}"
    )
