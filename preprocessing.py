import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure stopwords are available (user should run nltk.download before first run)
try:
    STOP = set(stopwords.words('english'))
except:
    STOP = set()

def clean_text(text: str) -> str:
    if text is None:
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    tokens = text.split()
    if STOP:
        tokens = [t for t in tokens if t not in STOP and len(t) > 2]
    else:
        tokens = [t for t in tokens if len(t) > 2]
    return " ".join(tokens)

def build_vectorizer(corpus, max_features=5000):
    vect = TfidfVectorizer(max_features=max_features, ngram_range=(1,2))
    vect.fit(corpus)
    return vect
