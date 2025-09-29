import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy import sparse
import joblib

# Optional wordclouds
try:
    from wordcloud import WordCloud
    WORDCLOUD_AVAILABLE = True
except Exception:
    WORDCLOUD_AVAILABLE = False

os.makedirs("figures", exist_ok=True)
os.makedirs("features", exist_ok=True)

# ---- Load cleaned data ----
df = pd.read_csv("data_clean/cleaned_sentiment.csv")
df = df.dropna(subset=["clean_text","label"]).copy()
print("Cleaned shape:", df.shape)

# ---- 3.1 Dataset description & visualisation ----
label_counts = df["label"].value_counts()
plt.figure(figsize=(5,4))
plt.bar(label_counts.index, label_counts.values)
plt.title("Class distribution (after cleaning)")
plt.xlabel("Label"); plt.ylabel("Count"); plt.tight_layout()
plt.savefig("figures/class_distribution_after_clean.png", dpi=200); plt.close()

def top_ngrams(texts, n=20, ngram=2):
    c = Counter()
    for s in texts:
        toks = s.split()
        grams = ["_".join(toks[i:i+ngram]) for i in range(len(toks)-ngram+1)]
        c.update(grams)
    return c.most_common(n)

def plot_top(items, title, outpath):
    labels = [k for k,_ in items][::-1]
    vals   = [v for _,v in items][::-1]
    plt.figure(figsize=(7,6))
    plt.barh(labels, vals); plt.title(title); plt.tight_layout()
    plt.savefig(outpath, dpi=200); plt.close()

pos_txt = df.loc[df["label"]=="positive","clean_text"]
neg_txt = df.loc[df["label"]=="negative","clean_text"]
plot_top(top_ngrams(pos_txt, n=20, ngram=2), "Top bigrams – positive", "figures/top_ngrams_pos.png")
plot_top(top_ngrams(neg_txt, n=20, ngram=2), "Top bigrams – negative", "figures/top_ngrams_neg.png")

if WORDCLOUD_AVAILABLE:
    for subset, name in [(pos_txt,"pos"), (neg_txt,"neg")]:
        wc = WordCloud(width=1000, height=700, background_color="white").generate(" ".join(subset.tolist()))
        plt.figure(figsize=(8,6)); plt.imshow(wc); plt.axis("off"); plt.tight_layout()
        plt.savefig(f"figures/wordcloud_{name}.png", dpi=200); plt.close()

# ---- 3.2 Feature extraction: TF–IDF ----
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
)

tfidf = TfidfVectorizer(max_features=50_000, ngram_range=(1,2), min_df=5)
X_train_csr = tfidf.fit_transform(X_train)
X_test_csr  = tfidf.transform(X_test)

print("TF–IDF shapes:", X_train_csr.shape, X_test_csr.shape)
print("Vocabulary size:", len(tfidf.get_feature_names_out()))

# Persist features for modelling
os.makedirs("features", exist_ok=True)
sparse.save_npz("features/X_train_csr.npz", X_train_csr)
sparse.save_npz("features/X_test_csr.npz", X_test_csr)
np.save("features/y_train.npy", y_train.to_numpy())
np.save("features/y_test.npy", y_test.to_numpy())
joblib.dump(tfidf, "features/tfidf_vectorizer.joblib")
print("Saved features to 'features/'")
