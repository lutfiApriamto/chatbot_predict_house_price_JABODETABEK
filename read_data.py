import pandas as pd

# df = pd.read_csv("data/processed/preprocessed_final.csv")
df = pd.read_csv("data/processed/zone_label.csv")


# 5 baris pertama
print(df.head())

# Jumlah baris dan kolom
print("Jumlah baris:", df.shape[0])
print("Jumlah kolom:", df.shape[1])
