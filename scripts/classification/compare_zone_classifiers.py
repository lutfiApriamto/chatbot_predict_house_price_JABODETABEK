# scripts/classification/compare_zone_classifiers.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

# ==== Load data ====
print("📦 Memuat data...")
df = pd.read_csv("data/processed/preprocessed_final.csv")
df = df.dropna(subset=["price_category"])
X = df.drop(columns=["log_price", "price_category", "price_in_rp"])
y = df["price_category"]

# ==== Split data ====
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# ==== Define models ====
models = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "GradientBoosting": GradientBoostingClassifier(),
    "SVC": SVC()
}

results = []

# ==== Train & Evaluate ====
print("\n📊 Evaluasi Model Klasifikasi:")
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1
    })

    print(f"- {name}: Acc={acc:.3f} | Prec={prec:.3f} | Rec={rec:.3f} | F1={f1:.3f}")

# ==== Visualisasi ====
df_results = pd.DataFrame(results)

metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
colors = ["Blues_d", "Greens_d", "Oranges_d", "Purples_d"]

for metric, color in zip(metrics, colors):
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Model", y=metric, data=df_results, palette=color)
    plt.title(f"Perbandingan {metric} antar Model")
    plt.ylim(0, 1)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ==== Confusion Matrix untuk Model Terbaik ====
best_model_name = df_results.sort_values("F1-Score", ascending=False).iloc[0]["Model"]
print(f"\n🏆 Model terbaik berdasarkan F1-Score: {best_model_name}")

best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test)

conf_matrix = confusion_matrix(y_test, y_pred_best)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title(f"Confusion Matrix - {best_model_name}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()
