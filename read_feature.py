import json

# Baca file JSON
with open("models/feature_columns.json", "r") as f:
    feature_columns = json.load(f)

# 1. Tampilkan 5 fitur pertama
print("🔹 5 fitur pertama:")
print(feature_columns[:5])

# 2. Tampilkan jumlah total fitur
print("\n🔹 Jumlah total fitur:")
print(len(feature_columns))

# 3. Tampilkan semua nama fitur (opsional)
print("\n🔹 Daftar semua nama fitur:")
for i, column in enumerate(feature_columns, start=1):
    print(f"{i}. {column}")
