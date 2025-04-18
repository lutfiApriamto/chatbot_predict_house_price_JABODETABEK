import pandas as pd

# ==== 1. Load dataset zone ====
df = pd.read_csv("data/processed/zone_label.csv")

# ==== 2. Fungsi pencarian rumah berdasarkan budget ====
def filter_houses_by_budget(city, budget, tolerance=0.1):
    city = city.strip().lower()
    min_price = budget * (1 - tolerance)
    max_price = budget * (1 + tolerance)

    filtered = df[df["city"].str.lower() == city]
    filtered = filtered[
        (filtered["price_in_rp"] >= min_price) &
        (filtered["price_in_rp"] <= max_price)
    ]

    results = []
    if not filtered.empty:
        for _, row in filtered.sample(min(3, len(filtered))).iterrows():
            results.append({
                "lokasi": row["district"].title(),
                "harga": int(row["price_in_rp"]),
                "land_area": row.get("land_area", "-"),
                "bedrooms": row.get("bedrooms", "-"),
                "bathrooms": row.get("bathrooms", "-")
            })
    return results

# ==== 3. Contoh penggunaan ====
if __name__ == "__main__":
    city = "bekasi"
    budget = 300_000_000

    hasil = filter_houses_by_budget(city, budget)

    if hasil:
        print(f"Rekomendasi rumah di {city.title()} untuk budget Rp {budget:,}:")
        for rumah in hasil:
            print(f"- {rumah['lokasi']} | Rp {rumah['harga']:,} | Tanah {rumah['land_area']}m | {rumah['bedrooms']} KT, {rumah['bathrooms']} KM")
    else:
        print(f"Tidak ditemukan rumah di {city.title()} dengan budget Rp {budget:,}.")
