# Sentiment Analysis on IMDb Reviews

**Goal** Build a baseline and classical ML pipeline for binary sentiment classification on movie reviews.

## Pipeline
1. **Preprocessing** lowercasing, URL stripping, letters-only, stopword removal, Porter stemming, and simple negation handling (`not_good` pattern).
2. **EDA** class distribution, token length histogram, top n-grams, optional wordclouds.
3. **Features** TF–IDF with unigrams and bigrams.
4. **Models**
   - Rule-based baseline with VADER
   - Logistic Regression
   - Linear SVM
5. **Evaluation** accuracy, precision, recall, F1, confusion matrices, and sentiment distribution comparison across models.

## Folders and outputs
- `data_clean/cleaned_sentiment.csv` cleaned dataset
- `features/` saved sparse matrices and vectorizer for reuse
- `figures/` plots for class distribution, n-grams, wordclouds, confusion matrices

## How to run
1. Run preprocessing to create `data_clean/cleaned_sentiment.csv`.
2. Run feature extraction to produce TF–IDF features.
3. Train ML baselines and the VADER baseline.
4. Review metrics and figures in `results/` and `figures/`.
