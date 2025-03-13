from flask import Flask, request, jsonify
import numpy as np
import joblib
import pandas as pd

# Load trained models & encoders
try:
    crop_model = joblib.load("crop_prediction_model.pkl")
    scaler = joblib.load("scaler.pkl")
    label_encoder = joblib.load("label_encoder.pkl")

    # Load yield prediction models & scalers
    tomato_yield_model = joblib.load("tomato_yield_model.pkl")
    capsicum_yield_model = joblib.load("capsicum_yield_model.pkl")
    tomato_scaler = joblib.load("tomato_scaler.pkl")
    capsicum_scaler = joblib.load("capsicum_scaler.pkl")

except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit(1)  # Stop the API if model loading fails

app = Flask(__name__)

@app.route("/")
def home():
    return "🌱 Crop & Yield Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data
        data = request.get_json()

        # Validate required input fields
        required_fields = ["Temperature", "Humidity", "Soil_pH", "N", "P", "K", "Area"]
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400

        # Extract features for crop prediction
        crop_features = np.array([[data["Temperature"], data["Humidity"], data["Soil_pH"], 
                                   data["N"], data["P"], data["K"]]])

        # Convert to DataFrame & Scale
        crop_features_df = pd.DataFrame(crop_features, columns=["Temperature", "Humidity", "Soil_pH", "N", "P", "K"])
        crop_features_scaled = scaler.transform(crop_features_df)

        # Predict Crop
        predicted_crop_idx = crop_model.predict(crop_features_scaled)[0]
        predicted_crop = label_encoder.inverse_transform([predicted_crop_idx])[0]

        # Extract & Scale Area for Yield Prediction
        area = np.array([[data["Area"]]])
        area_df = pd.DataFrame(area, columns=["Area"])  # Ensure correct format

        if predicted_crop == "Tomato":
            area_scaled = tomato_scaler.transform(area_df)
            predicted_yield = tomato_yield_model.predict(area_scaled)[0]
        else:
            area_scaled = capsicum_scaler.transform(area_df)
            predicted_yield = capsicum_yield_model.predict(area_scaled)[0]

        return jsonify({
            "Predicted Crop": predicted_crop,
            "Predicted Yield (metric tons)": round(predicted_yield, 2)
        })

    except Exception as e:
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
