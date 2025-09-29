"""
Section 2.2 Data Preprocessing (offline safe)
Loads a labelled dataset, cleans text, runs sanity checks, and saves outputs.

Outputs:
  data_clean/cleaned_sentiment.csv
  figures/class_distribution.png
  figures/token_length_hist.png
  data_clean/preprocessing_log.json
"""

# =======================
# Imports
# =======================
import os
import re
import json
import time
import random

import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer


# =======================
# Configurations
# =======================
INPUT_CSV = r"C:\Users\Dell\Desktop\Leonard\IMDB Dataset.csv"
TEXT_COL = "review"
LABEL_COL = "sentiment"

# Make output folders
os.makedirs("figures", exist_ok=True)
os.makedirs("data_clean", exist_ok=True)

random.seed(42)

CFG = {
    "lowercase": True,
    "strip_urls": True,
    "keep_letters_only": True,
    "min_token_length": 3,
    "remove_stopwords": True,
    "stem": True,                # use Porter stemming
    "negation_handling": True,   # "not good" -> "not_good"
    "preview_rows": 5
}

STOPWORDS = set(ENGLISH_STOP_WORDS) if CFG["remove_stopwords"] else set()
STEMMER = PorterStemmer() if CFG["stem"] else None

print("Configuration:", json.dumps(CFG, indent=2))
print(f"Loaded {len(STOPWORDS)} stopwords from scikit-learn.")


# =======================
# Helper Functions
# =======================
def attach_negations(tokens):
    """
    Simple negation handling.
    Turns 'not good' into 'not_good' so the polarity cue is preserved.
    """
    out = []
    skip = False
    for i, tok in enumerate(tokens):
        if skip:
            skip = False
            continue
        if tok == "not" and i + 1 < len(tokens):
            out.append(f"not_{tokens[i+1]}")
            skip = True
        else:
            out.append(tok)
    return out


def clean_text(text: str, cfg: dict) -> str:
    """
    Apply a reproducible set of transformations to one document.
    Returns a single cleaned string.
    """
    s = str(text)

    if cfg["lowercase"]:
        s = s.lower()
    if cfg["strip_urls"]:
        s = re.sub(r"http\S+|www\.\S+", " ", s)
    if cfg["keep_letters_only"]:
        s = re.sub(r"[^a-z\s]", " ", s)

    tokens = s.split()

    if cfg["negation_handling"]:
        tokens = attach_negations(tokens)

    minlen = cfg["min_token_length"]
    if cfg["remove_stopwords"]:
        tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= minlen]
    else:
        tokens = [t for t in tokens if len(t) >= minlen]

    if cfg["stem"] and STEMMER is not None:
        tokens = [STEMMER.stem(t) for t in tokens]

    return " ".join(tokens)


# =======================
# Main Execution
# =======================
if __name__ == "__main__":
    # Load dataset
    df = pd.read_csv(INPUT_CSV, encoding="utf-8")
    if TEXT_COL not in df.columns or LABEL_COL not in df.columns:
        raise ValueError(
            f"Columns '{TEXT_COL}' and '{LABEL_COL}' must exist in {INPUT_CSV}. "
            f"Found: {list(df.columns)}"
        )

    df = df[[TEXT_COL, LABEL_COL]].dropna().copy()
    df.columns = ["text", "label"]  # standard names

    print("Raw shape:", df.shape)
    print(df.head(3))

    # Normalise labels
    df["label"] = df["label"].astype(str).str.lower().str.strip()
    label_map = {
        "pos": "positive",
        "neg": "negative",
        "positive": "positive",
        "negative": "negative",
        "neutral": "neutral"
    }
    df["label"] = df["label"].map(lambda x: label_map.get(x, x))

    print("Label counts before cleaning:")
    print(df["label"].value_counts(dropna=False))

    # Preview before/after cleaning
    preview = df.sample(min(CFG["preview_rows"], len(df)), random_state=42)[["text", "label"]].copy()
    preview["clean_text"] = preview["text"].apply(lambda x: clean_text(x, CFG))
    print("\nPreview of before and after cleaning:")
    print(preview)

    # Apply cleaning to full dataset
    tqdm.pandas(desc="Cleaning texts")
    t0 = time.time()
    df["clean_text"] = df["text"].progress_apply(lambda x: clean_text(x, CFG))
    t1 = time.time()

    # Remove empty rows and duplicates
    before_len = len(df)
    df = df[df["clean_text"].str.strip().str.len() > 0].copy()
    empties_removed = before_len - len(df)

    dup_count = df.duplicated(subset=["clean_text", "label"]).sum()
    df = df.drop_duplicates(subset=["clean_text", "label"])

    print(f"\nCleaning time seconds: {t1 - t0:.2f}")
    print(f"Rows removed because empty after cleaning: {empties_removed}")
    print(f"Duplicate rows removed on clean_text and label: {dup_count}")
    print("Shape after cleaning:", df.shape)

    # Class distribution figure
    label_counts = df["label"].value_counts()
    plt.figure(figsize=(5, 4))
    plt.bar(label_counts.index, label_counts.values)
    plt.title("Class distribution after cleaning")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("figures/class_distribution.png", dpi=200)
    plt.close()

    # Token length histogram
    df["token_count"] = df["clean_text"].str.split().apply(len)
    desc = df["token_count"].describe()
    print("\nToken count summary:")
    print(desc)

    plt.figure(figsize=(6, 4))
    plt.hist(df["token_count"], bins=40)
    plt.title("Token count per document after cleaning")
    plt.xlabel("Tokens per document")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("figures/token_length_hist.png", dpi=200)
    plt.close()

    # Save outputs
    clean_path = "data_clean/cleaned_sentiment.csv"
    df[["clean_text", "label"]].to_csv(clean_path, index=False)

    log = {
        "rows_after_cleaning": int(len(df)),
        "class_distribution": df["label"].value_counts().to_dict(),
        "token_count_summary": {k: float(v) for k, v in desc.to_dict().items()},
        "config": CFG,
        "outputs": {
            "clean_csv": clean_path,
            "class_distribution_png": "figures/class_distribution.png",
            "token_length_hist_png": "figures/token_length_hist.png"
        }
    }
    with open("data_clean/preprocessing_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print("\nSaved cleaned CSV to:", clean_path)
    print("Saved preprocessing log to: data_clean/preprocessing_log.json")
    print("Saved figures to: figures/class_distribution.png and figures/token_length_hist.png")
