# scripts/predict_price.py

import pandas as pd
import numpy as np
import joblib
import json

# ==== 1. Load model & fitur ====
model = joblib.load("models/model_best_regression.pkl")
with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# ==== 2. Fungsi preprocessing input ====
def preprocess_input(user_input):
    df_input = pd.DataFrame([user_input])

    # Normalisasi teks kategori
    for col in ["city", "district", "property_type"]:
        if col in df_input:
            df_input[col] = df_input[col].astype(str).str.strip().str.lower()

    # Hitung fitur turunan
    df_input["building_ratio"] = df_input["building_area"] / (df_input["land_area"] + 1)
    df_input["room_ratio"] = df_input["bedrooms"] / (df_input["bathrooms"] + 1)
    df_input["luas_per_kamar"] = df_input["building_area"] / (df_input["bedrooms"] + 1)
    df_input["kamar_mandi_per_kamar"] = df_input["bathrooms"] / (df_input["bedrooms"] + 1)

    # One-hot encoding kategori
    df_input = pd.get_dummies(df_input)

    # Tambahkan kolom yang tidak ada sekaligus untuk efisiensi
    missing_cols = [col for col in feature_columns if col not in df_input.columns]
    df_missing = pd.DataFrame([[0]*len(missing_cols)], columns=missing_cols)
    df_input = pd.concat([df_input, df_missing], axis=1)

    # Urutkan kolom
    df_input = df_input[feature_columns]
    return df_input

# ==== 3. Fungsi prediksi ====
def predict_price(user_input):
    df_input = preprocess_input(user_input)
    log_price_pred = model.predict(df_input)[0]
    price_pred = np.expm1(log_price_pred)
    return int(price_pred)

# ==== 4. Contoh penggunaan ====
if __name__ == "__main__":
    sample_input = {
        "city": "depok",
        "district": "cilodong",
        "property_type": "rumah",
        "land_area": 140,
        "building_area": 90,
        "bedrooms": 5,
        "bathrooms": 4,
        "carports": 1,
        "floors": 1,
        "garages": 0,
        "lat": -6.4,
        "long": 106.8
    }

    predicted_price = predict_price(sample_input)
    print(f"Estimasi harga rumah: Rp {predicted_price:,.0f}")