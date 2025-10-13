from flask import Flask, request, jsonify
from src.predict import predict_ticket

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data.get('ticket_text', '')
    if not text:
        return jsonify({'error': 'no ticket_text provided'}), 400
    label, confidence = predict_ticket(text)
    return jsonify({'prediction': label, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
