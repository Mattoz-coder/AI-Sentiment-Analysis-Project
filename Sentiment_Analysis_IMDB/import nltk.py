import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

LEXICON = r"C:\Users\Dell\AppData\Roaming\nltk_data\sentiment\vader_lexicon.txt"

print("Exists?", os.path.exists(LEXICON), LEXICON)

# Make sure NLTK searches your Roaming dir (harmless if already present)
nltk.data.path.append(r"C:\Users\Dell\AppData\Roaming\nltk_data")

# Force the analyzer to use your file directly
sia = SentimentIntensityAnalyzer(lexicon_file=LEXICON)

print("OK ->", sia.polarity_scores("This movie was great!"))
