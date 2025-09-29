import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

print("Loading cleaned dataset...")
df = pd.read_csv("data_clean/cleaned_sentiment.csv")

# Use the actual column names from your CSV
X = df["clean_text"]
y = df["label"]

print("Extracting TF-IDF features...")
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1,2))
X_tfidf = tfidf.fit_transform(X)

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X_tfidf, y, test_size=0.2, random_state=42
)

print("Saving features...")
# Save features for reuse
pickle.dump(X_train, open("features/X_train_tfidf.pkl", "wb"))
pickle.dump(X_test, open("features/X_test_tfidf.pkl", "wb"))
pickle.dump(y_train, open("features/y_train.pkl", "wb"))
pickle.dump(y_test, open("features/y_test.pkl", "wb"))
pickle.dump(tfidf, open("features/tfidf_vectorizer.pkl", "wb"))

print("Feature extraction completed and saved!")
