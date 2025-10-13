import pandas as pd
from pathlib import Path

FEEDBACK_LOG = Path('data/feedback.csv')

def log_feedback(text: str, correct_label: str):
    row = {'ticket_text': text, 'category': correct_label}
    header = not FEEDBACK_LOG.exists()
    df = pd.DataFrame([row])
    df.to_csv(FEEDBACK_LOG, mode='a', header=header, index=False)
    return True

def merge_and_retrain(original_csv='data/train.csv'):
    # Simple helper: append feedback to original and retrain using src.train
    import shutil
    merged = Path('data/merged_train.csv')
    if FEEDBACK_LOG.exists():
        import pandas as pd
        orig = pd.read_csv(original_csv)
        fb = pd.read_csv(FEEDBACK_LOG)
        merged_df = pd.concat([orig, fb], ignore_index=True)
        merged_df.to_csv(merged, index=False)
        print('Merged data saved to', merged)
        print('Now run: python src/train.py data/merged_train.csv')
    else:
        print('No feedback logged yet.')
