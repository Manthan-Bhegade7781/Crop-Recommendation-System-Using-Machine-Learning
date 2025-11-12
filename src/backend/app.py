from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  # Enables communication between React (frontend) & Flask (backend)

# Load trained model, scaler, and label encoder
with open('model/crop_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/preprocessing.pkl', 'rb') as preprocessing_file:
    scaler = pickle.load(preprocessing_file)

with open('model/label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)


# ✅ Route for testing API connection
@app.route('/')
def home():
    return jsonify({"message": "Crop Recommendation API is running 🌱"})


# ✅ Route for prediction (used by React frontend)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Extract features from JSON body
        features = [
            float(data['N']),
            float(data['P']),
            float(data['K']),
            float(data['temperature']),
            float(data['humidity']),
            float(data['ph']),
            float(data['rainfall'])
        ]

        # Preprocess & predict
        scaled_features = scaler.transform([features])
        prediction_numeric = model.predict(scaled_features)[0]
        prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

        # Return result to React frontend
        return jsonify({
            "recommendation": prediction_label,
            "probability": 0.93,  # You can calculate real probability if needed
            "alternatives": ["rice", "maize", "barley"]  # example alternatives
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


# ✅ Run Flask server
if __name__ == '__main__':
    app.run(debug=True)
