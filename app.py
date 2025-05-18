from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np
import ast  # For safely evaluating strings as Python literals
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load datasets
try:
    precautions_df = pd.read_csv("datasets/precautions_df.csv")
    description_df = pd.read_csv("datasets/description.csv")
    workout_df = pd.read_csv("datasets/workout_df.csv")
    medications_df = pd.read_csv("datasets/medications.csv")
    diets_df = pd.read_csv("datasets/diets.csv")
    training_data = pd.read_csv('datasets/Training.csv')
except Exception as e:
    print(f"Error loading datasets: {e}")
    precautions_df = pd.DataFrame()
    description_df = pd.DataFrame()
    workout_df = pd.DataFrame()
    medications_df = pd.DataFrame()
    diets_df = pd.DataFrame()
    training_data = pd.DataFrame()

# Load model and encoder
try:
    with open('knn_model.pkl', 'rb') as f:
        knn = pickle.load(f)
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
except Exception as e:
    print(f"Error loading model files: {e}")
    knn = None
    le = None

symptom_columns = training_data.columns.drop('prognosis').tolist() if 'prognosis' in training_data.columns else []

def parse_string_list(list_str):
    """Convert string representation of list to actual list"""
    try:
        if isinstance(list_str, str) and list_str.startswith('[') and list_str.endswith(']'):
            return ast.literal_eval(list_str)
        return []
    except (ValueError, SyntaxError):
        return []

def get_recommendations(disease):
    result = {
        'description': "No description available",
        'precautions': [],
        'medications': [],
        'workouts': [],
        'diets': []
    }
    
    try:
        # Get description
        if not description_df.empty:
            desc = description_df[description_df['Disease'] == disease]['Description']
            result['description'] = desc.values[0] if not desc.empty else result['description']
        
        # Get precautions
        if not precautions_df.empty:
            items = precautions_df[precautions_df['Disease'] == disease].iloc[:, 1:].values
            result['precautions'] = [p for p in items[0] if pd.notna(p)] if len(items) > 0 else []
        
        # Get medications
        if not medications_df.empty:
            meds = medications_df[medications_df['Disease'] == disease]['Medication']
            if not meds.empty:
                meds_str = meds.values[0]
                result['medications'] = parse_string_list(meds_str)
        
        # Get workouts
        if not workout_df.empty:
            workouts = workout_df[workout_df['disease'] == disease]['workout']
            result['workouts'] = workouts.tolist() if not workouts.empty else []
        
        # Get diets - using the same parsing function as medications
        if not diets_df.empty:
            diets = diets_df[diets_df['Disease'] == disease]['Diet']
            if not diets.empty:
                diets_str = diets.values[0]
                result['diets'] = parse_string_list(diets_str)
            
    except Exception as e:
        print(f"Error getting recommendations: {e}")
    
    return result

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/symptoms')
def get_symptoms():
    return jsonify({
        'symptoms': symptom_columns,
        'count': len(symptom_columns)
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not knn or not le:
            return jsonify({"error": "Diagnosis system is not properly initialized"}), 500
            
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms or len(symptoms) < 1:
            return jsonify({"error": "Please select at least 1 symptom"}), 400
        
        # Validate symptoms
        invalid_symptoms = [s for s in symptoms if s not in symptom_columns]
        if invalid_symptoms:
            return jsonify({
                "error": f"Invalid symptoms detected: {', '.join(invalid_symptoms)}"
            }), 400
        
        # Create input vector
        input_vector = np.zeros(len(symptom_columns))
        for symptom in symptoms:
            idx = symptom_columns.index(symptom)
            input_vector[idx] = 1
        
        # Predict
        prediction = knn.predict([input_vector])[0]
        disease = le.inverse_transform([prediction])[0]
        
        return jsonify({
            "status": "success",
            "prediction": disease,
            **get_recommendations(disease)
        })
        
    except Exception as e:
        return jsonify({
            "status": "error", 
            "message": str(e),
            "error": "An error occurred during diagnosis"
        }), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)