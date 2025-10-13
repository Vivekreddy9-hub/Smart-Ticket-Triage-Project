# Smart Ticket Triage (AI-powered Support Routing)

**Author:** Your Name  
**Tech Stack:** Python, scikit-learn, Flask, Streamlit

## Project Overview
Smart Ticket Triage is an MVP to classify customer support tickets into categories (Billing, Technical, Account, Feature Request). The system returns both a predicted category and a confidence score. Low-confidence tickets are flagged for human review. Users can correct misclassifications, which are logged for later retraining.

## Quick Start (local)
1. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Download NLTK stopwords (only once):
```python
python -c "import nltk; nltk.download('stopwords')"
```

3. Train a baseline model:
```bash
python src/train.py data/train.csv
```

4. Start the API:
```bash
python src/api.py
```

5. Start the UI (in another terminal):
```bash
streamlit run ui/app.py
```

Then open `http://localhost:8501` in your browser.

## Folder Structure
```
smart-ticket-triage/
├── data/
│   ├── train.csv
│   └── feedback.csv
├── models/
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   ├── api.py
│   └── feedback_loop.py
├── ui/
│   └── app.py
├── requirements.txt
├── README.md
└── LICENSE
```

## Notes
- This is an educational MVP designed to run locally without any paid APIs.
- The dataset included is synthetic and intended for demonstration and evaluation.
- Improve model by swapping MultinomialNB with logistic regression or transformer embeddings for production readiness.
