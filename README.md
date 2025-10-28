# Explainable-Medical-Diagnosis-Assistant
An AI system that analyzes medical images (e.g., X-rays, MRIs) using deep learning to detect abnormalities, while symbolic reasoning provides step-by-step explanations and rule-based validation, allowing doctors to trust and understand the AI’s decisions.
# Explainable Medical Diagnosis Assistant (Prototype)

Prototype combining a CNN-based perception module with a simple symbolic reasoning layer and a Knowledge-Adaptive Integration Layer (KAIL) for explainable decisions.

**Warning:** This is a research prototype only — not for clinical diagnosis.

## Quick start

1. Create virtual environment and install:

bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
`

2. Train model (example with small dataset):

bash
python pipeline/train.py --data ./data

3. Run backend server:

bash
python backend/app.py


4. Open `frontend/index.html` in your browser or visit `http://localhost:5000`.

## What is inside

* `pipeline/` — training scripts and model definition (PyTorch).
* `backend/` — Flask API, KAIL implementation, rule engine and serving code.
* `frontend/` — minimal UI to upload images and view explainable outputs.
