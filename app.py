from flask import Flask, request, jsonify, render_template
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from flask_cors import CORS

app = Flask(_name_)
CORS(app)

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

def generate_study_plan(goal, strengths, weaknesses, preferences):
    input_text = f"Goal: {goal}, Strengths: {strengths}, Weaknesses: {weaknesses}, Preferences: {preferences}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        outputs = bert_model(**inputs)
    
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    study_plan = [
        {"day": "Monday", "task": f"Revise {strengths}. Strengthen core concepts."},
        {"day": "Tuesday", "task": f"Focus on {weaknesses}. Use {preferences} for better understanding."},
        {"day": "Wednesday", "task": f"Practice problems related to {goal}."},
        {"day": "Thursday", "task": f"Take a mock test on {goal}."},
        {"day": "Friday", "task": "Review mistakes and improve weak areas."},
        {"day": "Saturday", "task": "Revise everything & relax."},
        {"day": "Sunday", "task": "Self-evaluation & set goals for next week."}
    ]
    
    return study_plan

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/generate-plan', methods=['POST'])
def generate_plan():
    data = request.json
    goal = data.get("goal", "")
    strengths = data.get("strengths", "")
    weaknesses = data.get("weaknesses", "")
    preferences = data.get("preferences", "")

    plan = generate_study_plan(goal, strengths, weaknesses, preferences)
    return jsonify(plan)

if _name_ == '_main_':
    app.run(debug=True)