from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

# Load the trained model and encoders
model = joblib.load("crop_prediction_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

app = Flask(__name__)

@app.route("/")
def home():
    return "Crop Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract features from request
        feature_names = ["Temperature", "Humidity", "Soil_pH", "N", "P", "K"]
        features = np.array([[data["Temperature"], data["Humidity"], data["Soil_pH"], data["N"], data["P"], data["K"]]])
        
        # Convert input to DataFrame with correct feature names
        features_df = pd.DataFrame(features, columns=feature_names)
        
        # Scale the input features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        crop = label_encoder.inverse_transform([prediction])[0]

        return jsonify({"Predicted Crop": crop})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
