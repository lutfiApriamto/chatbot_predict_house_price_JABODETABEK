# scripts/preprocessing/visualize_preprocessing.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load preprocessed data
df = pd.read_csv("data/processed/preprocessed_final.csv")

# Set style
sns.set(style="whitegrid")

# ==== 1. Distribusi Harga Asli vs Harga Log ====
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df["price_in_rp"], kde=True, bins=50, color='skyblue')
plt.title("Distribusi Harga Rumah Asli")
plt.xlabel("Harga (Rp)")
plt.ylabel("Jumlah Rumah")

plt.subplot(1, 2, 2)
sns.histplot(df["log_price"], kde=True, bins=50, color='salmon')
plt.title("Distribusi Harga Rumah Setelah Transformasi Log")
plt.xlabel("Log Harga")
plt.ylabel("Jumlah Rumah")

plt.tight_layout()
plt.show()

# ==== 2. Histogram Harga per Kategori Harga (Sebelum Log Transform) ====
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="price_in_rp", hue="price_category", element="step", common_norm=False, palette="Set2", bins=50)
plt.title("Distribusi Harga Rumah (Asli) per Zona Harga")
plt.xlabel("Harga Rumah (Rp)")
plt.ylabel("Jumlah Rumah")
plt.xscale("log")  # Supaya lebih proporsional, harga dalam log-scale sumbu X
plt.show()


# ==== 3. Histogram Harga per Kategori Harga (Setelah Log Transform) ====
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x="log_price", hue="price_category", element="step", common_norm=False, palette="Set2", bins=50)
plt.title("Distribusi Harga Rumah (Log Transform) per Zona Harga")
plt.xlabel("Log Harga")
plt.ylabel("Jumlah Rumah")
plt.show()

# ==== 4. Heatmap Korelasi Antar Fitur Numerik ====
num_cols = [
    "bedrooms", "bathrooms", "land_area", "building_area",
    "carports", "floors", "garages", "lat", "long",
    "building_ratio", "room_ratio", "luas_per_kamar", "kamar_mandi_per_kamar"
]

plt.figure(figsize=(14, 10))
corr = df[num_cols + ["log_price"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Heatmap Korelasi Fitur Numerik")
plt.show()

# ==== 5. Scatter Plot Luas Tanah vs Harga ====
plt.figure(figsize=(10, 6))
sns.scatterplot(x="land_area", y="price_in_rp", data=df, alpha=0.5)
plt.yscale("log")
plt.title("Scatterplot: Luas Tanah vs Harga Rumah")
plt.xlabel("Luas Tanah (m²)")
plt.ylabel("Harga Rumah (Rp, Log Scale)")
plt.show()

# ==== 6. Scatter Plot Tambahan: Building Area vs Harga & Bathrooms vs Harga ====
plt.figure(figsize=(14, 6))

# Scatterplot 1: Building Area vs Price
plt.subplot(1, 2, 1)
sns.scatterplot(x="building_area", y="price_in_rp", data=df, alpha=0.5, color="seagreen")
plt.yscale("log")
plt.title("Building Area vs Harga Rumah")
plt.xlabel("Luas Bangunan (m²)")
plt.ylabel("Harga Rumah (Rp, Log Scale)")

# Scatterplot 2: Bathrooms vs Price
plt.subplot(1, 2, 2)
sns.scatterplot(x="bathrooms", y="price_in_rp", data=df, alpha=0.5, color="orange")
plt.yscale("log")
plt.title("Jumlah Kamar Mandi vs Harga Rumah")
plt.xlabel("Jumlah Kamar Mandi")
plt.ylabel("Harga Rumah (Rp, Log Scale)")

plt.tight_layout()
plt.show()

print("Visualisasi selesai ditampilkan ✅")
