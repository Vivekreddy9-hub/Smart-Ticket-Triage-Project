import joblib
from src.preprocessing import clean_text
import os

MODEL_PATH = os.path.join('models', 'model.pkl')
VECT_PATH = os.path.join('models', 'vectorizer.pkl')

def load_models():
    clf = None; vect = None
    try:
        clf = joblib.load(MODEL_PATH)
        vect = joblib.load(VECT_PATH)
    except Exception as e:
        raise RuntimeError('Model artifacts not found. Train the model first.') from e
    return clf, vect

def predict_ticket(text: str):
    clf, vect = load_models()
    cleaned = clean_text(text)
    vec = vect.transform([cleaned])
    probs = clf.predict_proba(vec)[0]
    pred = clf.classes_[probs.argmax()]
    confidence = float(probs.max())
    return pred, confidence

if __name__ == '__main__':
    import sys
    text = ' '.join(sys.argv[1:]) or 'My invoice shows wrong amount'
    pred, conf = predict_ticket(text)
    print(pred, conf)
