# chatbot/chatbot_final.py

import re
import json
import random
import joblib
import numpy as np
import pandas as pd
from chatbot_helpers import build_zone_price_response
from chatbot_helpers import get_spec_from_budget

# === Load NLP intent classification model ===
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)

with open("models/nlp_model.pkl", "rb") as f:
    nlp_model, vectorizer, label_encoder = joblib.load(f)

with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# === Session context ===
session_context = {
    "kota": None,
    "luas_tanah": None,
    "jumlah_kamar": None,
    "garasi": None,
    "bathrooms": None,
    "floors": None,
    "carports": None,
    "building_area": None,
    "awaiting_luas_tanah": False,
    "awaiting_jumlah_kamar": False,
    "awaiting_jumlah_kamar_mandi": False,
    "awaiting_building_area": False  # ⬅️ Tambahan penting
}

# === Ekstraksi fitur dari teks ===
def extract_info(text):
    text = text.lower()
    result = {}

    if m := re.search(r'((jakarta|bogor|depok|tangerang|bekasi)( [a-z]+)?)', text):
        result["kota"] = m.group(1).strip()
    if m := re.search(r'(\d+)\s*(m\u00b2|m2|meter)', text):
        result["luas_tanah"] = int(m.group(1))
    if m := re.search(r'(\d+)\s*kamar mandi', text):
        result["bathrooms"] = int(m.group(1))
    if m := re.search(r'(\d+)\s*kamar\b(?! mandi)', text):
        result["jumlah_kamar"] = int(m.group(1))
    if m := re.search(r'(\d+)\s*lantai', text):
        result["floors"] = int(m.group(1))
    if m := re.search(r'(\d+)\s*garasi', text):
        result["garasi"] = int(m.group(1))
    if m := re.search(r'(\d+)\s*carport', text):
        result["carports"] = int(m.group(1))
    if m := re.search(r'(\d+)\s*(m2|m\u00b2|meter)?\s*(bangunan)?', text):
        result["building_area"] = int(m.group(1))

    return result

# === Bangun input untuk model ===
def build_input_row(info):
    row = {
        "bedrooms": int(info.get("jumlah_kamar", 0) or 0),
        "bathrooms": int(info.get("bathrooms", 0) or 0),
        "land_area": float(info.get("luas_tanah", 0) or 0),
        "building_area": float(info.get("building_area", 0) or 0),
        "carports": int(info.get("carports", 0) or 0),
        "floors": int(info.get("floors", 1) or 1),
        "garages": int(info.get("garasi", 0) or 0),
        "lat": -6.3,
        "long": 106.8
    }

    # Tambah city_ jika tersedia
    if session_context.get("kota"):
        city_col = f"city_{session_context['kota'].lower()}"
        row[city_col] = 1

    # Fitur tambahan
    row["building_ratio"] = row["building_area"] / (row["land_area"] + 1)
    row["room_ratio"] = row["bedrooms"] / (row["bathrooms"] + 1)
    row["luas_per_kamar"] = row["building_area"] / (row["bedrooms"] + 1)
    row["kamar_mandi_per_kamar"] = row["bathrooms"] / (row["bedrooms"] + 1)

    df = pd.DataFrame([row])

    # Tambahkan kolom kosong jika belum ada (dan isi dengan 0)
    for col in feature_columns:
        df = df.copy()
        if col not in df.columns:
            df[col] = 0

    # Hindari kolom object
    df = df.astype({col: 'float32' for col in df.columns if df[col].dtype == 'object'})

    return df[feature_columns]

# === Pilih respons dari intents ===
def get_response_by_tag(tag):
    for intent in intents_data["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "Maaf, saya tidak paham maksud Anda."

# === Fungsi utama ===
def chatbot_response(user_input):
    global session_context

    # Tangani kelanjutan input "lanjutkan" → jangan deteksi intent, cukup teruskan alur pertanyaan
    if user_input.lower() == "lanjutkan":
        if not session_context.get("jumlah_kamar"):
            session_context["awaiting_jumlah_kamar"] = True
            return "Berapa jumlah kamar tidur yang Anda butuhkan (misal : 2 kamar tidur)?"
        if not session_context.get("bathrooms"):
            session_context["awaiting_jumlah_kamar_mandi"] = True
            return "Berapa jumlah kamar mandi yang Anda butuhkan (misal : 2 kamar mandi)?"
        if not session_context.get("luas_tanah"):
            session_context["awaiting_luas_tanah"] = True
            return "Berapa luas tanah yang Anda inginkan (misal : 180 meter)?"
        if not session_context.get("building_area"):
            session_context["awaiting_building_area"] = True
            return "Berapa kira-kira luas bangunan yang Anda inginkan? (misal: 80 m2)"

        # Semua data lengkap → prediksi
        kota = session_context.get("kota")
        input_row = build_input_row(session_context)
        session_context = {k: None for k in session_context}
        return build_zone_price_response(input_row, kota)

    # Ekstrak info user dan update context
    info = extract_info(user_input)
    session_context.update({k: v for k, v in info.items() if v is not None})

    # Tangani input yang ditunggu sebelumnya
    if session_context["awaiting_jumlah_kamar"]:
        if "jumlah_kamar" in info:
            session_context["jumlah_kamar"] = info["jumlah_kamar"]
            session_context["awaiting_jumlah_kamar"] = False
            return chatbot_response("lanjutkan")
        else:
            return "Berapa jumlah kamar tidur yang Anda butuhkan (misal : 2 kamar)?"

    if session_context["awaiting_jumlah_kamar_mandi"]:
        if "bathrooms" in info:
            session_context["bathrooms"] = info["bathrooms"]
            session_context["awaiting_jumlah_kamar_mandi"] = False
            return chatbot_response("lanjutkan")
        else:
            return "Berapa jumlah kamar mandi yang Anda butuhkan? (misal : 2 kamar mandi)"

    if session_context["awaiting_building_area"]:
        if "building_area" in info:
            session_context["building_area"] = info["building_area"]
            session_context["awaiting_building_area"] = False
            return chatbot_response("lanjutkan")
        else:
            return "Berapa kira-kira luas bangunan yang Anda inginkan? (misal: 80 m2)"

    if session_context["awaiting_luas_tanah"]:
        if "luas_tanah" in info:
            session_context["luas_tanah"] = info["luas_tanah"]
            session_context["awaiting_luas_tanah"] = False
            return chatbot_response("lanjutkan")
        else:
            return "Berapa luas tanah yang Anda inginkan? (misal : 180 meter)"

    # === Deteksi intent hanya jika bukan input "lanjutkan"
    x_vec = vectorizer.transform([user_input]).toarray()
    intent = label_encoder.inverse_transform(nlp_model.predict(x_vec))[0]

    if intent == "tanya_dari_budget":
        match = re.search(r'(\d+[.,]?\d*)\s*(juta|miliar|m|jt)', user_input.lower())
        if match:
            raw_angka = match.group(1).replace(',', '.')
            try:
                angka = float(raw_angka)
            except ValueError:
                return "Nominal budget tidak dapat dipahami, coba ketik ulang misalnya: 1.2 miliar"
            
            satuan = match.group(2)

            if satuan in ['m', 'miliar']:
                budget_rp = int(angka * 1_000_000_000)
            elif satuan in ['jt', 'juta']:
                budget_rp = int(angka * 1_000_000)
            else:
                budget_rp = int(angka)


            kota = session_context.get("kota")
            if not kota:
                return "Di kota mana Anda ingin mencari rumah dengan budget tersebut?"
            return get_spec_from_budget(budget_rp, kota)
        else:
            return "Berapa budget yang Anda miliki untuk membeli rumah?"

    if intent in ["tanya_harga", "cari_rumah"]:
        if session_context.get("kota") == "jakarta":
            return "Jakarta mana yang Anda maksud? jakarta Utara, jakarta Selatan, jakarta Timur, jakarta Barat, atau jakarta Pusat?"
        if not session_context.get("kota"):
            return "Di kota mana Anda ingin mencari rumah?"
        if not session_context.get("jumlah_kamar"):
            session_context["awaiting_jumlah_kamar"] = True
            return "Berapa jumlah kamar tidur yang Anda butuhkan (misal : 2 kamar tidur)? "
        if not session_context.get("bathrooms"):
            session_context["awaiting_jumlah_kamar_mandi"] = True
            return "Berapa jumlah kamar mandi yang Anda butuhkan (misal : 2 kamar mandi)? "
        if not session_context.get("luas_tanah"):
            session_context["awaiting_luas_tanah"] = True
            return "Berapa luas tanah yang Anda inginkan (misal : 180 meter)?"
        if not session_context.get("building_area"):
            session_context["awaiting_building_area"] = True
            return "Berapa kira-kira luas bangunan yang Anda inginkan? (misal: 80 m2)"

        kota = session_context.get("kota")
        input_row = build_input_row(session_context)
        session_context = {k: None for k in session_context}
        return build_zone_price_response(input_row, kota)

    if intent == "unknown":
        # Deteksi fallback untuk pertanyaan tentang budget
        if re.search(r'(budget|dana|uang|punya|modal)\s+[\d,.]+\s*(miliar|jt|juta|m)?', user_input.lower()):
            intent = "tanya_dari_budget"
        else:
            return get_response_by_tag("unknown")

    return get_response_by_tag(intent)


# === CLI ===
if __name__ == "__main__":
    print("\n🏠 Chatbot Rumah Jabodetabek Siap Digunakan!")
    print("Ketik 'keluar' untuk berhenti.\n")
    while True:
        user_input = input("🧑 Anda: ")
        if user_input.lower() in ["keluar", "exit", "quit"]:
            print("🤖 Bot: Sampai jumpa! Semoga Anda segera menemukan rumah impian.")
            break
        response = chatbot_response(user_input)
        print(f"🤖 Bot: {response}\n")
