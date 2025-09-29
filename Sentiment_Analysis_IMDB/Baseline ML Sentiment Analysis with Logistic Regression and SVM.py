import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# === Load TF-IDF features and labels ===
print("Loading features...")
X_train = pickle.load(open("features/X_train_tfidf.pkl", "rb"))
X_test  = pickle.load(open("features/X_test_tfidf.pkl", "rb"))
y_train = pickle.load(open("features/y_train.pkl", "rb"))
y_test  = pickle.load(open("features/y_test.pkl", "rb"))

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# make sure results directory exists
os.makedirs("results", exist_ok=True)
os.makedirs("figures", exist_ok=True)

results_text = []

# === Logistic Regression ===
print("\nTraining Logistic Regression...")
log_reg = LogisticRegression(max_iter=1000, n_jobs=-1, solver='liblinear')
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
report_lr = classification_report(y_test, y_pred_lr)

print(f"Logistic Regression Accuracy: {acc_lr:.4f}")
print(report_lr)

results_text.append("=== Logistic Regression ===")
results_text.append(f"Accuracy: {acc_lr:.4f}")
results_text.append(report_lr)

# Confusion Matrix for Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix - Logistic Regression")
plt.savefig("figures/confusion_matrix_logreg.png")
plt.close()

# === Linear SVM ===
print("\nTraining Linear SVM...")
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

acc_svm = accuracy_score(y_test, y_pred_svm)
report_svm = classification_report(y_test, y_pred_svm)

print(f"Linear SVM Accuracy: {acc_svm:.4f}")
print(report_svm)

results_text.append("\n=== Linear SVM ===")
results_text.append(f"Accuracy: {acc_svm:.4f}")
results_text.append(report_svm)

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt="d", cmap="Greens",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.title("Confusion Matrix - Linear SVM")
plt.savefig("figures/confusion_matrix_svm.png")
plt.close()

print("\nSaved confusion matrices to 'figures/'")

# === Save text results ===
with open("results/baseline_ml_results.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(results_text))

print("Saved classification reports to 'results/baseline_ml_results.txt'")

# --- Add this block BEFORE you build the distribution DataFrame/plot ---

# 1) Recreate the SAME test split on raw text to align with y_test
import pandas as pd
from sklearn.model_selection import train_test_split

df_clean = pd.read_csv("data_clean/cleaned_sentiment.csv")
X_text_all = df_clean["clean_text"]
y_all      = df_clean["label"]

# Use the SAME split parameters to get the same test fold as your features
_, X_test_text, _, y_test_text = train_test_split(
    X_text_all, y_all, test_size=0.2, random_state=42, shuffle=True
)

# Sanity check: y_test_text should match y_test order/length
assert len(y_test_text) == len(y_test), "Test label lengths differ; check your split settings."

# 2) Generate VADER predictions on the test texts
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

def compound_to_label(c):
    if c >= 0.05:
        return "positive"
    elif c <= -0.05:
        return "negative"
    else:
        return "neutral"   # include if your dataset has neutral; otherwise you can map to nearest class

y_pred_vader = [
    compound_to_label(sia.polarity_scores(txt)["compound"]) for txt in X_test_text
]

# If your ground truth has only positive/negative (IMDB), optionally fold neutral into the nearest class:
# y_pred_vader = ["positive" if lab == "neutral" else lab for lab in y_pred_vader]

# 3) Build sentiment distribution comparison
import matplotlib.pyplot as plt
import seaborn as sns

true_counts  = pd.Series(y_test).value_counts(normalize=True).sort_index()
vader_counts = pd.Series(y_pred_vader).value_counts(normalize=True).sort_index()
lr_counts    = pd.Series(y_pred_lr).value_counts(normalize=True).sort_index()
svm_counts   = pd.Series(y_pred_svm).value_counts(normalize=True).sort_index()

dist_df = pd.DataFrame({
    "True": true_counts,
    "VADER": vader_counts,
    "Logistic Regression": lr_counts,
    "SVM": svm_counts
}).T.fillna(0)

plt.figure(figsize=(8,5))
sns.barplot(data=dist_df)
plt.title("Sentiment Distribution Across Models vs True Labels")
plt.ylabel("Proportion of Reviews")
plt.xlabel("Sentiment Class")
plt.legend(title="Model", labels=dist_df.index)
plt.tight_layout()
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/sentiment_distribution.png", dpi=200)
plt.close()
print("Saved: figures/sentiment_distribution.png")
