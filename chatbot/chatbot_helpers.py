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

# Hitung batas bawah dan atas dari fitur numerik
PERCENTILE_MIN = 0.01
PERCENTILE_MAX = 0.99

feature_bounds = raw_df[[
    "bedrooms", "bathrooms", "land_area", "building_area",
    "carports", "garages", "floors"
]].quantile([PERCENTILE_MIN, PERCENTILE_MAX]).to_dict()

def normalize_input_keys(ctx):
    return {
        "bedrooms": ctx.get("jumlah_kamar"),
        "bathrooms": ctx.get("bathrooms"),
        "land_area": ctx.get("luas_tanah"),
        "building_area": ctx.get("building_area"),
        "carports": ctx.get("carports"),
        "garages": ctx.get("garasi"),
        "floors": ctx.get("floors")
    }

def check_unreasonable_input(info):
    warnings = []
    bounds_info = get_feature_bounds()

    def is_out_of_bounds(key, value):
        if key not in bounds_info or value is None:
            return False
        low, high = bounds_info[key]
        return value < low or value > high

    unreasonable_keys = []
    for key, label in [
        ("bedrooms", "jumlah kamar tidur"),
        ("bathrooms", "jumlah kamar mandi"),
        ("land_area", "luas tanah"),
        ("building_area", "luas bangunan"),
        ("floors", "jumlah lantai"),
        ("carports", "jumlah carport"),
        ("garages", "jumlah garasi"),
    ]:
        value = info.get(key)
        if value is not None and is_out_of_bounds(key, value):
            low, high = bounds_info[key]
            warnings.append(f"- {label} yang Anda masukkan ({value}) terlihat tidak umum.")
            unreasonable_keys.append((label, low, high))

    if not warnings:
        return None  # Input wajar

    # Format pesan peringatan lengkap
    warning_text = "⚠️ Perhatian:\n"
    warning_text += "Sistem mendeteksi bahwa beberapa input yang Anda masukkan tampaknya tidak wajar:\n"
    warning_text += "\n".join(warnings)
    warning_text += "\n\nBerdasarkan data yang kami miliki, umumnya:\n"
    for label, low, high in unreasonable_keys:
        warning_text += f"- {label} berada antara {low} hingga {high}\n"
    warning_text += "\nSilakan masukkan kembali data spesifikasi rumah dengan nilai yang lebih wajar."

    return warning_text.strip()

def is_unreasonable_budget(budget_rp):
    min_price = raw_df["price_in_rp"].quantile(0.01)
    max_price = raw_df["price_in_rp"].quantile(0.99)

    if budget_rp < min_price or budget_rp > max_price:
        return (
            f"⚠️ Perhatian:\n"
            f"Budget yang Anda masukkan ({budget_rp:,.0f}) terlihat tidak umum.\n\n"
            f"Berdasarkan data yang kami miliki, harga rumah biasanya berada antara "
            f"Rp {min_price:,.0f} hingga Rp {max_price:,.0f}.\n"
            f"Silakan masukkan budget yang lebih realistis agar kami dapat membantu Anda dengan akurat."
        )
    return None

def get_feature_bounds():
    return {
        key: (round(value[0.01], 2), round(value[0.99], 2))
        for key, value in feature_bounds.items()
    }

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

    # Pastikan jumlah lokasi cukup
    while len(locations) < len(prices):
        locations.append("Belum tersedia")

    # Mulai menyusun respons
    response = (
        f"Rumah dengan spesifikasi yang Anda sebutkan yaitu:\n"
        f"- {int(input_row['bedrooms'][0])} kamar tidur\n"
        f"- {int(input_row['bathrooms'][0])} kamar mandi\n"
        f"- Luas tanah {int(input_row['land_area'][0])} m²\n"
        f"- Luas bangunan {int(input_row['building_area'][0])} m²\n"
    )

    # Tambahkan fitur opsional jika tersedia dan > 0
    optional_features = {
        "garages": "garasi",
        "carports": "carport",
        "floors": "lantai"
    }

    for key, label in optional_features.items():
        if key in input_row.columns:
            val = input_row[key][0]
            if pd.notna(val) and val > 0:
                response += f"- {int(val)} {label}\n"

    response += f"berlokasi di kota {kota.title()}.\n\n"
    response += f"Mungkin anda bisa dapatkan dibeberapa lokasi, dengan estimasi harga : "

    for i, (real_price, est_price) in enumerate(prices):
        district = locations[i] if i < len(locations) and locations[i] else "Belum tersedia"
        response += (
            f"\n📍 Contoh lokasi: {district.title()}\n"
            f"💰 Estimasi harga: Rp {est_price:,.0f}\n"
        )
    
    response += f"\n\n menurut data yang kami miliki, lokasi ini termasuk ke dalam zona {zone.upper()}.\n"

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
