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

# === Split data (training & testing) ===
X_train, X_test, y_train, y_test = train_test_split(X_vec, y_enc, test_size=0.2, random_state=42)

# === Training model ===
model = MultinomialNB()
model.fit(X_train, y_train)

# === Evaluasi model ===
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print(f"\n📊 Akurasi model NLP: {acc * 100:.2f}%")
print("\n📋 Classification Report:\n")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# === Simpan model, vectorizer, dan label encoder ===
os.makedirs("models", exist_ok=True)
joblib.dump((model, vectorizer, label_encoder), "models/nlp_model.pkl")

print("\n✅ Model NLP berhasil disimpan ke models/nlp_model.pkl")
