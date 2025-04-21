# chatbot/chatbot_final.py

import re
import json
import random
import joblib
import numpy as np
import pandas as pd
from chatbot_helpers import build_zone_price_response, get_spec_from_budget
from difflib import get_close_matches

# === Load NLP intent classification model ===
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents_data = json.load(f)
with open("models/nlp_model.pkl", "rb") as f:
    nlp_model, vectorizer, label_encoder = joblib.load(f)
with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

KOTA_LIST = [
    "jakarta pusat", "jakarta selatan", "jakarta timur", 
    "jakarta barat", "jakarta utara", "jakarta", 
    "bogor", "depok", "tangerang", "bekasi"
]

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
    "pending_budget_query": None,
    "awaiting_luas_tanah": False,
    "awaiting_jumlah_kamar": False,
    "awaiting_jumlah_kamar_mandi": False,
    "awaiting_building_area": False  # ⬅️ Tambahan penting
}
# === Ekstraksi fitur dari teks ===
def extract_info(text):
    text = text.lower()
    result = {}

    if m := re.search(r'\b(jakarta( [a-z]+)?|bogor|depok|tangerang|bekasi)\b', text):
        result["kota"] = m.group(1).strip()
    if m := re.search(r'(\d+)\s*(m2|m²|meter)?\s*(luas\s*tanah)', text):
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
    if m := re.search(r'(\d+)\s*(m2|m²|meter)?\s*(luas )?(bangunan)', text):
        result["building_area"] = int(m.group(1))
    
    if "kota" not in result:
        tokens = text.lower().split()
        for token in tokens:
            closest = get_close_matches(token, KOTA_LIST, n=1, cutoff=0.75)
            if closest:
                result["kota"] = closest[0]
                break

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
        city_key = session_context['kota'].lower().replace(" ", "_")
        city_col = f"city_{city_key}"
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
    
    if not user_input.strip():
        return "Silakan masukkan pertanyaan atau informasi yang Anda butuhkan. Saya bisa membantu memprediksi harga rumah dan spesifikasi berdasarkan budget Anda di wilayah Jabodetabek."

    # 1. Tangani jawaban terhadap pending budget (user menyebutkan kota saja)
    if session_context.get("pending_budget_query"):
        info = extract_info(user_input)
        if "kota" in info:
            kota = info["kota"]
            session_context["kota"] = kota
            if kota.lower() == "jakarta":
                return "Jakarta mana yang Anda maksud? Jakarta Utara, Selatan, Timur, Barat, atau Pusat?"
            budget_rp = session_context.pop("pending_budget_query")  # hapus setelah digunakan
            result = get_spec_from_budget(budget_rp, kota)
            session_context["kota"] = None
            
            return result

    # 2. Tangani lanjutan input "lanjutkan"
    if user_input.lower() == "lanjutkan":
        if not session_context.get("jumlah_kamar"):
            session_context["awaiting_jumlah_kamar"] = True
            return "Berapa jumlah kamar tidur yang Anda butuhkan (misal : 2 kamar tidur)?"
        if not session_context.get("bathrooms"):
            session_context["awaiting_jumlah_kamar_mandi"] = True
            return "Berapa jumlah kamar mandi yang Anda butuhkan (misal : 2 kamar mandi)?"
        if not session_context.get("luas_tanah"):
            session_context["awaiting_luas_tanah"] = True
            return "Berapa luas tanah yang Anda inginkan (misal : 180 meter luas tanah)?"
        if not session_context.get("building_area"):
            session_context["awaiting_building_area"] = True
            return "Berapa kira-kira luas bangunan yang Anda inginkan? (misal: 80 m2 luas bangunan)"

        kota = session_context.get("kota")
        input_row = build_input_row(session_context)
        response = build_zone_price_response(input_row, kota)
        session_context = {k: None for k in session_context}  # ← Reset di akhir
        return response


    # 3. Ekstrak info user & update session
    info = extract_info(user_input)
    session_context.update({k: v for k, v in info.items() if v is not None})

    # 4. Tangani input yang ditunggu sebelumnya
    if session_context["awaiting_jumlah_kamar"] and "jumlah_kamar" in info:
        session_context["awaiting_jumlah_kamar"] = False
        return chatbot_response("lanjutkan")

    if session_context["awaiting_jumlah_kamar_mandi"] and "bathrooms" in info:
        session_context["awaiting_jumlah_kamar_mandi"] = False
        return chatbot_response("lanjutkan")

    if session_context["awaiting_building_area"] and "building_area" in info:
        session_context["awaiting_building_area"] = False
        return chatbot_response("lanjutkan")

    if session_context["awaiting_luas_tanah"] and "luas_tanah" in info:
        session_context["awaiting_luas_tanah"] = False
        return chatbot_response("lanjutkan")

    # 5. Deteksi intent
    x_vec = vectorizer.transform([user_input]).toarray()
    intent = label_encoder.inverse_transform(nlp_model.predict(x_vec))[0]

    # 6. Tangani intent tanya_dari_budget
    if intent == "tanya_dari_budget":
        match = re.search(r'(\d+[.,]?\d*)\s*(juta|miliar|m|jt)', user_input.lower())
        if match:
            raw_angka = match.group(1).replace(',', '.')
            try:
                angka = float(raw_angka)
            except ValueError:
                return "Nominal budget tidak dapat dipahami, coba ketik ulang misalnya: 1.2 miliar"
            satuan = match.group(2)
            budget_rp = int(angka * 1_000_000_000 if satuan in ['m', 'miliar'] else angka * 1_000_000)
            
            # Simpan budget ke dalam session
            session_context["pending_budget_query"] = budget_rp
            
            kota = session_context.get("kota") or extract_info(user_input).get("kota")
            
            if kota:
                session_context["kota"] = kota  # simpan kota ke context
                
                if kota.lower() == "jakarta":
                    return "Jakarta mana yang Anda maksud? Jakarta Utara, Selatan, Timur, Barat, atau Pusat?"
                
                # Jika bukan "jakarta", lanjutkan prediksi
                session_context.pop("pending_budget_query", None)
                result = get_spec_from_budget(budget_rp, kota)
                session_context["kota"] = None
                
                return result
            else:
                return "Di kota mana Anda ingin mencari rumah dengan budget tersebut?"
            
        else:
            return "Berapa budget yang Anda miliki untuk membeli rumah?"

    # 7. Tangani intent cari_rumah dan tanya_harga
    if intent in ["tanya_harga", "cari_rumah"]:
        if session_context.get("kota") and session_context["kota"].lower() == "jakarta":
            return "Jakarta mana yang Anda maksud? Jakarta Utara, Jakarta Selatan, Jakarta Timur, Jakarta Barat, atau Jakarta Pusat?"
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
            return "Berapa luas tanah yang Anda inginkan (misal : 180 meter luas tanah)?"
        if not session_context.get("building_area"):
            session_context["awaiting_building_area"] = True
            return "Berapa kira-kira luas bangunan yang Anda inginkan? (misal: 80 m2 luas bangunan)"

        kota = session_context.get("kota")
        input_row = build_input_row(session_context)
        response = build_zone_price_response(input_row, kota)
        session_context = {k: None for k in session_context}  # ← Reset di akhir
        return response


    # 8. Fallback untuk unknown intent
    if intent == "unknown":
        if re.search(r'(budget|dana|uang|punya|modal)\s+[\d,.]+\s*(miliar|jt|juta|m)?', user_input.lower()):
            return chatbot_response(user_input)
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
