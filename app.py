from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model, scaler, and label encoder
with open('model/crop_recommendation_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('model/preprocessing.pkl', 'rb') as preprocessing_file:
    scaler = pickle.load(preprocessing_file)

with open('model/label_encoder.pkl', 'rb') as encoder_file:
    label_encoder = pickle.load(encoder_file)

# Homepage route
@app.route('/')
def index():
    return render_template('index2.html')  # Homepage

# Route for Crop Recommendation Form
@app.route('/crop.html')
def crop_form():
    return render_template('crop.html')  # Form page

# Route to handle form submission and show result
@app.route('/result.html', methods=['POST'])
def recommend():
    try:
        # Retrieve form data
        features = [
            float(request.form['N']),
            float(request.form['P']),
            float(request.form['K']),
            float(request.form['temperature']),
            float(request.form['humidity']),
            float(request.form['ph']),
            float(request.form['rainfall'])
        ]

        # Preprocess and predict
        scaled_features = scaler.transform([features])
        prediction_numeric = model.predict(scaled_features)[0]
        prediction_label = label_encoder.inverse_transform([prediction_numeric])[0]

        return render_template('result.html', recommendation=prediction_label)

    except Exception as e:
        return f"Error occurred: {str(e)}"

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
