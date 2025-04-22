# scripts/fallback_estimator.py

import pandas as pd

# ==== 1. Load data zona ====
df = pd.read_csv("data/processed/zone_label.csv")

# ==== 2. Fungsi fallback rata-rata harga per kota ====
def get_city_avg_price(city):
    city = city.strip().lower()
    filtered = df[df["city"].str.lower() == city]

    if filtered.empty:
        return None

    avg_price = int(filtered["price_in_rp"].mean())
    count = filtered.shape[0]
    return {
        "city": city,
        "jumlah_data": count,
        "harga_rata_rata": avg_price
    }

# ==== 3. Contoh penggunaan ====
if __name__ == "__main__":
    hasil = get_city_avg_price("Bogor")
    if hasil:
        print(f"Rata-rata harga rumah di {hasil['city'].title()} adalah Rp {hasil['harga_rata_rata']:,} dari {hasil['jumlah_data']} data.")
    else:
        print("Kota tidak ditemukan dalam data.")
