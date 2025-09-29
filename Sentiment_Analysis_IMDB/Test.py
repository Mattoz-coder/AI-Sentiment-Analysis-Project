import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# True distribution
true_counts = pd.Series(y_test).value_counts(normalize=True).sort_index()

# VADER distribution (assuming you saved y_pred_vader earlier)
vader_counts = pd.Series(y_pred_vader).value_counts(normalize=True).sort_index()

# Logistic Regression distribution
lr_counts = pd.Series(y_pred_lr).value_counts(normalize=True).sort_index()

# SVM distribution
svm_counts = pd.Series(y_pred_svm).value_counts(normalize=True).sort_index()

# Combine into a dataframe for plotting
dist_df = pd.DataFrame({
    "True": true_counts,
    "VADER": vader_counts,
    "Logistic Regression": lr_counts,
    "SVM": svm_counts
}).T.fillna(0)  # fill any missing classes

# Plot
plt.figure(figsize=(8,5))
sns.barplot(data=dist_df)
plt.title("Sentiment Distribution Across Models vs True Labels")
plt.ylabel("Proportion of Reviews")
plt.xlabel("Sentiment Class")
plt.legend(title="Model", labels=dist_df.index)
plt.tight_layout()
plt.savefig("figures/sentiment_distribution.png", dpi=200)
plt.close()
