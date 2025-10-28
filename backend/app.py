from flask import Flask, request, jsonify, send_from_directory
from backend.model_serving import ModelServer
from backend.kail import KAIL
from backend.rules import RULE_SET
import os

app = Flask(__name__, static_folder='../frontend')

MODEL_PATH = os.environ.get('MODEL_PATH', 'model.pth')
server = ModelServer(MODEL_PATH)

# init KAIL with default weights
kail = KAIL(rule_weights={k: v[1] for k,v in RULE_SET.items()})

@app.route('/')
def home():
    return send_from_directory('../frontend', 'index.html')

@app.route('/api/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error':'no image uploaded'}), 400
    f = request.files['image']
    data = f.read()
    pred = server.predict_from_bytes(data)
    facts = kail.concept_to_fact(pred['concepts'], pred['concept_names'])
    activations = kail.weighted_reasoning(facts, RULE_SET)
    decision_score = kail.aggregate_decision(activations)
    explanation = {
'concepts': facts,
        'rule_activations': activations,
        'decision_score': decision_score,
        'class_probs': pred['probs']
    }
    return jsonify(explanation)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
