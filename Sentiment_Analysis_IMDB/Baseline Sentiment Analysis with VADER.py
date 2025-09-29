# baseline_vader.py
"""
Rule-based baseline using VADER (standalone package, no NLTK downloads).
Loads the IMDb CSV, predicts sentiment, prints metrics, and saves a confusion matrix.

Outputs:
  results_baseline/vader_confusion_matrix.png
  results_baseline/vader_classification_report.txt
"""

import os
import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------
# CONFIG
# -------------------
INPUT_CSV = r"C:\Users\Dell\Desktop\Leonard\IMDB Dataset.csv"
TEXT_COL  = "review"
LABEL_COL = "sentiment"
OUTPUT_DIR = "results_baseline"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------
# LOAD DATA
# -------------------
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)
print(f"Dataset shape: {df.shape}")
df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
df[LABEL_COL] = df[LABEL_COL].astype(str).str.lower().str.strip()

# -------------------
# VADER PREDICTION
# -------------------
sia = SentimentIntensityAnalyzer()

def vader_label(text: str) -> str:
    c = sia.polarity_scores(str(text))["compound"]
    # Standard VADER thresholds are: pos >= 0.05, neg <= -0.05, neutral otherwise.
    # The IMDb task is binary; we can map neutrals to nearest side or use 0 threshold.
    # Here we use 0 threshold to force binary labels.
    return "positive" if c >= 0 else "negative"

print("Scoring with VADER (this may take a moment)...")
df["predicted"] = df[TEXT_COL].apply(vader_label)

# -------------------
# EVALUATION
# -------------------
y_true = df[LABEL_COL].values
y_pred = df["predicted"].values

acc = accuracy_score(y_true, y_pred)
print(f"\nAccuracy: {acc:.4f}\n")

report_str = classification_report(y_true, y_pred, digits=3, target_names=["negative","positive"])
print("Classification Report:")
print(report_str)

# Save report
with open(os.path.join(OUTPUT_DIR, "vader_classification_report.txt"), "w", encoding="utf-8") as f:
    f.write(f"Accuracy: {acc:.4f}\n\n")
    f.write(report_str)

# Confusion matrix
labels = ["positive", "negative"]  # keep order consistent with dataset labels
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("VADER Baseline Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
out_path = os.path.join(OUTPUT_DIR, "vader_confusion_matrix.png")
plt.savefig(out_path, dpi=200)
plt.close()

print(f"Saved confusion matrix to: {out_path}")
print(f"Saved classification report to: {os.path.join(OUTPUT_DIR, 'vader_classification_report.txt')}")
