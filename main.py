# main.py

from scripts.nlp_parser import parse_user_input
from scripts.predict_price import predict_price
from scripts.filter_house_by_spek import filter_houses_by_spec
from scripts.filter_house_by_budget import filter_houses_by_budget
from scripts.fallback_estimator import get_city_avg_price

# ==== Memory sementara untuk multi-turn (sederhana) ====
session_memory = {}

def run_chatbot():
    print("\n🏠 Chatbot Rumah Jabodetabek Siap Digunakan!")
    print("Ketik 'keluar' untuk berhenti.\n")

    while True:
        user_input = input("🧑 Anda: ").strip()
        if user_input.lower() in ["keluar", "exit", "bye"]:
            print("🤖 Bot: Terima kasih! Sampai jumpa.")
            break

        parsed = parse_user_input(user_input)
        intent = parsed["intent"]
        entities = parsed["entities"]

        # === Lanjutkan percakapan sebelumnya (multi-turn cari rumah) ===
        if "pending_cari_rumah" in session_memory and "city" in entities:
            previous = session_memory.pop("pending_cari_rumah")
            land = previous.get("land_area")
            bed = previous.get("bedrooms")
            bath = previous.get("bathrooms")
            city = entities["city"]
            hasil = filter_houses_by_spec(city=city, land_area=land or 100, bedrooms=bed or 2, bathrooms=bath)
            if hasil:
                print("🤖 Bot: Berikut beberapa rumah yang cocok dengan permintaan Anda:")
                for h in hasil:
                    print(f"- Zona {h['zona'].capitalize()} → {h['lokasi']} | Rp {h['harga']:,} | Tanah {h['land_area']}m | {h['bedrooms']} KT, {h['bathrooms']} KM")
            else:
                fallback = get_city_avg_price(city)
                if fallback:
                    print(f"🤖 Bot: Tidak ditemukan rumah dengan spek tersebut. Namun rata-rata harga rumah di {fallback['city'].title()} adalah Rp {fallback['harga_rata_rata']:,} dari {fallback['jumlah_data']} data.")
            continue

        # ===== Handle tiap intent =====
        if intent == "cari_rumah":
            city = entities.get("city")
            land = entities.get("land_area")
            bed = entities.get("bedrooms")
            bath = entities.get("bathrooms")

            if not city and any([land, bed, bath]):
                session_memory["pending_cari_rumah"] = {
                    "land_area": land,
                    "bedrooms": bed,
                    "bathrooms": bath
                }
                print("🤖 Bot: Anda belum menyebutkan kota. Di kota mana Anda ingin mencari rumah?")
                continue

            if not city:
                print("🤖 Bot: Anda belum menyebutkan kota. Di kota mana Anda ingin mencari rumah?")
                continue

            if not any([land, bed, bath]):
                print("🤖 Bot: Mohon sebutkan minimal satu spesifikasi rumah seperti luas tanah, jumlah kamar tidur, atau kamar mandi.")
                continue

            hasil = filter_houses_by_spec(city=city, land_area=land or 100, bedrooms=bed or 2, bathrooms=bath)
            if hasil:
                print("🤖 Bot: Saya menemukan beberapa opsi rumah untuk Anda:")
                for h in hasil:
                    print(f"- Zona {h['zona'].capitalize()} → {h['lokasi']} | Rp {h['harga']:,} | Tanah {h['land_area']}m | {h['bedrooms']} KT, {h['bathrooms']} KM")
            else:
                fallback = get_city_avg_price(city)
                if fallback:
                    print(f"🤖 Bot: Maaf, tidak ada rumah dengan spek tersebut. Tapi rata-rata harga rumah di {fallback['city'].title()} adalah Rp {fallback['harga_rata_rata']:,} dari {fallback['jumlah_data']} data.")
                else:
                    print("🤖 Bot: Maaf, data untuk kota tersebut belum tersedia.")

        elif intent == "estimasi_harga":
            if "city" in entities and "land_area" in entities and "bedrooms" in entities:
                pred_input = {
                    "city": entities["city"],
                    "district": "unknown",
                    "property_type": "rumah",
                    "land_area": entities["land_area"],
                    "building_area": entities["land_area"] * 0.8,
                    "bedrooms": entities["bedrooms"],
                    "bathrooms": entities.get("bathrooms", 1),
                    "carports": 1,
                    "floors": 1,
                    "garages": 0,
                    "lat": -6.3,
                    "long": 106.8
                }
                price = predict_price(pred_input)
                print(f"🤖 Bot: Estimasi harga rumah adalah sekitar Rp {price:,}.")
            else:
                print("🤖 Bot: Untuk estimasi harga, mohon sebutkan kota, luas tanah, dan jumlah kamar tidur.")

        elif intent == "rata2_harga":
            if "city" in entities:
                fallback = get_city_avg_price(entities["city"])
                if fallback:
                    print(f"🤖 Bot: Rata-rata harga rumah di {fallback['city'].title()} adalah Rp {fallback['harga_rata_rata']:,} dari {fallback['jumlah_data']} data.")
                else:
                    print("🤖 Bot: Data rata-rata harga untuk kota tersebut tidak tersedia.")
            else:
                print("🤖 Bot: Mohon sebutkan kota yang ingin Anda ketahui rata-rata harganya.")

        elif intent == "cari_spesifikasi_dari_budget":
            if "city" in entities and "budget" in entities:
                hasil = filter_houses_by_budget(entities["city"], entities["budget"])
                if hasil:
                    print(f"🤖 Bot: Berikut beberapa rumah yang sesuai dengan budget Anda:")
                    for h in hasil:
                        print(f"- {h['lokasi']} | Rp {h['harga']:,} | Tanah {h['land_area']}m | {h['bedrooms']} KT, {h['bathrooms']} KM")
                else:
                    fallback = get_city_avg_price(entities["city"])
                    if fallback:
                        print(f"🤖 Bot: Tidak ada rumah dalam budget tersebut. Namun rata-rata harga rumah di {fallback['city'].title()} adalah Rp {fallback['harga_rata_rata']:,} dari {fallback['jumlah_data']} data.")
                    else:
                        print("🤖 Bot: Maaf, data untuk kota tersebut belum tersedia.")
            elif "budget" in entities and "city" not in entities:
                session_memory["pending_budget"] = entities["budget"]
                print("🤖 Bot: Anda belum menyebutkan kota. Di kota mana Anda ingin mencari rumah?")
            elif "city" in entities and "pending_budget" in session_memory:
                hasil = filter_houses_by_budget(entities["city"], session_memory["pending_budget"])
                del session_memory["pending_budget"]
                if hasil:
                    print(f"🤖 Bot: Berikut rumah yang cocok dengan dana Anda:")
                    for h in hasil:
                        print(f"- {h['lokasi']} | Rp {h['harga']:,} | Tanah {h['land_area']}m | {h['bedrooms']} KT, {h['bathrooms']} KM")
                else:
                    print("🤖 Bot: Tidak ditemukan rumah dalam rentang harga tersebut.")
            else:
                print("🤖 Bot: Mohon sebutkan dana dan kota tujuan Anda.")

        elif intent == "keluar":
            print("🤖 Bot: Sampai jumpa, semoga hari Anda menyenangkan!")
            break

        else:
            print("🤖 Bot: Maaf, saya belum mengerti maksud Anda. Bisa dijelaskan lagi?")

if __name__ == "__main__":
    run_chatbot()