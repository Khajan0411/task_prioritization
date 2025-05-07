# app.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocess import preprocess_data

app = Flask(__name__)

# Load the trained model
model = joblib.load('task_priority_model.pkl')

@app.route('/')
def home():
    return "Task Prioritization Model API"

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from POST request
    data = request.get_json()
    
    # Convert data into DataFrame (for preprocessing)
    df = pd.DataFrame([data])

    # Preprocess data (to match the format used in training)
    X, _, le_priority = preprocess_data(df)
    
    # Make prediction
    prediction = model.predict(X)
    
    # Return the prediction
    priority = le_priority.inverse_transform(prediction)[0]
    return jsonify({'priority': priority})

if __name__ == '__main__':
    app.run(debug=True)
