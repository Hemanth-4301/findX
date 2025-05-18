from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

# Load datasets
precautions_df = pd.read_csv("datasets/precautions_df.csv")
description_df = pd.read_csv("datasets/description.csv")
workout_df = pd.read_csv("datasets/workout_df.csv")
medications_df = pd.read_csv("datasets/medications.csv")
diets_df = pd.read_csv("datasets/diets.csv")
training_data = pd.read_csv('datasets/Training.csv')

# Load model and encoder
with open('knn_model.pkl', 'rb') as f:
    knn = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

# Get all symptoms
symptom_columns = training_data.columns.drop('prognosis').tolist()

def get_recommendations(disease):
    result = {}
    # Get description
    desc = description_df[description_df['Disease'] == disease]['Description']
    result['description'] = desc.values[0] if not desc.empty else "No description available"
    
    # Get other recommendations
    for df, name in [(precautions_df, 'precautions'), 
                     (medications_df, 'medications'),
                     (workout_df, 'workouts'),
                     (diets_df, 'diets')]:
        if name == 'precautions':
            items = df[df['Disease'] == disease].iloc[:, 1:].values
            result[name] = [p for p in items[0] if pd.notna(p)] if len(items) > 0 else []
        else:
            col = 'workout' if name == 'workouts' else name[:-1].capitalize()
            items = df[df['Disease' if name != 'workouts' else 'disease'] == disease][col]
            result[name] = items.tolist() if not items.empty else []
    
    return result

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptom_columns)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if len(symptoms) < 1:
            return jsonify({"error": "Please select at least 1 symptom"}), 400
        
        # Create input vector
        input_vector = np.zeros(len(symptom_columns))
        for symptom in symptoms:
            if symptom in symptom_columns:
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
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)