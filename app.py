import streamlit as st
import requests
import os
from src.feedback_loop import log_feedback

API_URL = os.environ.get('TRIAGE_API', 'http://localhost:5000/predict')

st.title("Smart Ticket Triage")

st.write("Enter a support ticket and get an automated category + confidence score.")

text = st.text_area("Ticket text", height=150)

if st.button("Classify"):
    if not text.strip():
        st.warning("Please enter some ticket text.")
    else:
        try:
            resp = requests.post(API_URL, json={'ticket_text': text}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                st.success(f"Predicted: **{data['prediction']}** (confidence: {data['confidence']:.2f})")
            else:
                st.error(f'API error: {resp.text}')
        except Exception as e:
            st.error(f'Could not reach API: {e}')

st.markdown('---')
st.write('If the prediction is incorrect, submit a correction below (this logs feedback).')

correct = st.selectbox('Correct category (leave blank if correct)', ['', 'Billing', 'Technical', 'Account', 'Feature Request'])

if st.button('Submit Correction'):
    if not text.strip():
        st.warning('Please provide the ticket text above first.')
    elif not correct:
        st.warning('Select a correct category or leave blank.')
    else:
        log_feedback(text, correct)
        st.info('Thanks — your correction has been logged.')
