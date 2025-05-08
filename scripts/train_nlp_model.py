# scripts/train_nlp_model.py

import json
import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# === Load file intents ===
with open("data/intents.json", "r", encoding="utf-8") as f:
    intents = json.load(f)

# === Persiapkan dataset ===
X = []
y = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern.lower())  # Menormalkan teks
        y.append(intent["tag"])

# === Ekstraksi fitur dengan CountVectorizer ===
vectorizer = CountVectorizer()
X_vec = vectorizer.fit_transform(X).toarray()

# === Label encoding untuk target ===
label_encoder = LabelEncoder()
y_enc = label_encoder.fit_transform(y)

# Tampilkan label hasil encoding
label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
print("\n🏷️ Mapping label:")
for tag, enc in label_map.items():
    print(f"{tag} ➝ {enc}")

# === Split data (training & testing) ===
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)

print(f"\n📁 Jumlah data training: {len(X_train)}")
print(f"📁 Jumlah data testing : {len(X_test)}")

# === Training model ===
model = MultinomialNB()
model.fit(X_train, y_train)

# === Evaluasi model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n📊 Akurasi model NLP: {acc * 100:.2f}%")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

from sklearn.metrics import classification_report
report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Hapus 'accuracy', 'macro avg', dan 'weighted avg' dari visualisasi utama
report_df_main = report_df.iloc[:-3, :]

# Visualisasi Classification Report
plt.figure(figsize=(10, 6))
report_df_main[["precision", "recall", "f1-score"]].plot(kind="bar", figsize=(12, 6), colormap="Set2")
plt.title("📈 Classification Report per Label")
plt.xlabel("Label")
plt.ylabel("Score")
plt.ylim(0, 1.1)
plt.xticks(rotation=45)
plt.legend(loc="lower right")
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("📊 Confusion Matrix")
plt.tight_layout()
plt.show()

# === Simpan model, vectorizer, dan label encoder ===
os.makedirs("models", exist_ok=True)
joblib.dump((model, vectorizer, label_encoder), "models/nlp_model.pkl")

print("\n✅ Model NLP berhasil disimpan ke models/nlp_model.pkl")

# === Persiapkan dataset ===
# Hitung jumlah data per intent
label_counts = pd.Series(y).value_counts().sort_index()
label_names = label_encoder.classes_

print("\n📦 Jumlah data per intent:")
for i, count in enumerate(label_counts.values):
    print(f"- {label_names[i]} (label {i}) berjumlah : {count} data")


# Visualisasi jumlah data per intent
plt.figure(figsize=(10, 6))
sns.barplot(x=label_names, y=label_counts.values, palette="viridis")
plt.title("📊 Jumlah Data per Intent")
plt.xlabel("Intent")
plt.ylabel("Jumlah Data")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
