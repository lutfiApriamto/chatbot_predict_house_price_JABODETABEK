# scripts/filter_house_by_spek.py

import pandas as pd

# ==== 1. Load dataset zona harga ====
df = pd.read_csv("data/processed/zone_label.csv")

# ==== 2. Fungsi filtering rumah ====
def filter_houses_by_spec(city, land_area, bedrooms, bathrooms=None, tolerance=0.1):
    # Normalisasi input
    city = city.strip().lower()

    # Toleransi rentang
    land_min = land_area * (1 - tolerance)
    land_max = land_area * (1 + tolerance)
    bed_min = bedrooms * (1 - tolerance)
    bed_max = bedrooms * (1 + tolerance)

    if bathrooms is not None:
        bath_min = bathrooms * (1 - tolerance)
        bath_max = bathrooms * (1 + tolerance)

    # Filter berdasarkan kota dan spek
    filtered = df[df["city"].str.lower() == city]
    filtered = filtered[
        (filtered["land_area"] >= land_min) & (filtered["land_area"] <= land_max) &
        (filtered["bedrooms"] >= bed_min) & (filtered["bedrooms"] <= bed_max)
    ]
    if bathrooms is not None:
        filtered = filtered[(filtered["bathrooms"] >= bath_min) & (filtered["bathrooms"] <= bath_max)]

    # Ambil satu rumah dari tiap zona jika ada
    results = []
    for zona in ["murah", "sedang", "mahal"]:
        subset = filtered[filtered["price_category"] == zona]
        if not subset.empty:
            rumah = subset.sample(1).iloc[0]
            results.append({
                "zona": zona,
                "lokasi": rumah["district"],
                "harga": int(rumah["price_in_rp"]),
                "land_area": rumah["land_area"],
                "bedrooms": rumah["bedrooms"],
                "bathrooms": rumah["bathrooms"]
            })

    return results

# ==== 3. Contoh penggunaan ====
if __name__ == "__main__":
    hasil = filter_houses_by_spec(city="bogor", land_area=120, bedrooms=3)
    for h in hasil:
        print(f"Zona {h['zona'].capitalize()} → {h['lokasi'].title()} | Rp {h['harga']:,} | Tanah {h['land_area']}m | {h['bedrooms']} KT, {h['bathrooms']} KM")
