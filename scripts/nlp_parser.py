# scripts/nlp_parser.py

import json
import re
import os
from difflib import SequenceMatcher

# ==== 1. Load intents ====
INTENT_PATH = os.path.join("data/intents_full.json")
with open(INTENT_PATH, "r") as f:
    intents = json.load(f)["intents"]

# ==== 2. Intent detection ====
def detect_intent(text):
    text = text.lower()
    best_match = None
    best_score = 0

    for intent in intents:
        for pattern in intent["patterns"]:
            score = SequenceMatcher(None, pattern.lower(), text).ratio()
            if score > best_score:
                best_score = score
                best_match = intent["tag"]

    return best_match if best_score >= 0.5 else "fallback"

# ==== 3. Entity extraction ====
def extract_entities(text):
    entities = {}
    text = text.lower()

    # Kota & wilayah Jakarta
    kota_list = ["jakarta", "bogor", "depok", "tangerang", "bekasi"]
    jakarta_area = ["jakarta pusat", "jakarta timur", "jakarta barat", "jakarta selatan", "jakarta utara"]
    for area in jakarta_area:
        if area in text:
            entities["city"] = area
    for kota in kota_list:
        if kota in text and "city" not in entities:
            entities["city"] = kota

    # Angka-angka
    number = lambda label, kw: re.search(rf"{kw} *(\d+[\.,]?\d*)", text)

    luas = number("land_area", r"(?:luas|tanah)")
    if luas:
        entities["land_area"] = int(float(luas.group(1).replace(",", ".")))

    kamar = number("bedrooms", r"(?:kamar( tidur)?)")
    if kamar:
        entities["bedrooms"] = int(float(kamar.group(1)))

    mandi = number("bathrooms", r"(?:kamar mandi|wc|toilet)")
    if mandi:
        entities["bathrooms"] = int(float(mandi.group(1)))

    budget = number("budget", r"(?:budget|dana|uang|harga|maximal|max)")
    if budget:
        val = float(budget.group(1).replace(",", "."))
        if val < 1000:
            val *= 1_000_000  # asumsikan juta
        entities["budget"] = int(val)

    return entities

# ==== 4. Combine parser ====
def parse_user_input(text):
    intent = detect_intent(text)
    entities = extract_entities(text)
    return {
        "intent": intent,
        "entities": entities
    }

# ==== 5. Contoh penggunaan ====
if __name__ == "__main__":
    sample_text = "Saya punya dana 300 juta, bisa dapat rumah seperti apa di Bekasi?"
    result = parse_user_input(sample_text)
    print("Intent:", result["intent"])
    print("Entities:", result["entities"])
