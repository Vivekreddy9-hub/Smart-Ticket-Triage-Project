import pandas as pd
import joblib
import sys
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
from src.preprocessing import clean_text, build_vectorizer

def train_pipeline(data_path, model_out='models/model.pkl', vectorizer_out='models/vectorizer.pkl'):
    df = pd.read_csv(data_path)
    df = df.dropna(subset=['ticket_text','category'])
    df['cleaned'] = df['ticket_text'].map(clean_text)
    X = df['cleaned']
    y = df['category']
    vect = build_vectorizer(X)
    X_vec = vect.transform(X)
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_vec, y)
    joblib.dump(clf, model_out)
    joblib.dump(vect, vectorizer_out)
    # Simple eval (train)
    y_pred = clf.predict(X_vec)
    print('Train Accuracy:', accuracy_score(y, y_pred))
    print('Classification report:')
    print(classification_report(y, y_pred))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python src/train.py data/train.csv')
    else:
        train_pipeline(sys.argv[1])
