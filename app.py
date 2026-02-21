from flask import Flask, request # pyright: ignore[reportMissingImports]
import joblib # pyright: ignore[reportMissingImports]
import numpy as np
import pandas as pd

app = Flask(__name__)

# -------- Load Stage 1 Model (NPK Prediction) --------
model_npk = joblib.load("npk_model.pkl")
scaler_npk = joblib.load("npk_scaler.pkl")

# -------- Load Stage 2 Model (Crop Prediction) --------
model_crop = joblib.load("crop_model.pkl")
scaler_crop = joblib.load("crop_scaler.pkl")


@app.route('/')
def home():
    return "Smart Crop Recommendation API Running (2-Stage ML)"


@app.route('/predict')
def predict():
    try:
        # -------- Get Sensor Values --------
        temp = float(request.args.get('temp'))
        humidity = float(request.args.get('humidity'))
        soil = float(request.args.get('soil'))

        # -------- Stage 1: Predict NPK --------
        npk_input = pd.DataFrame(
            [[temp, humidity, soil]],
            columns=['temperature', 'humidity', 'soil_moisture']
        )

        npk_scaled = scaler_npk.transform(npk_input)
        predicted_npk = model_npk.predict(npk_scaled)

        N, P, K = predicted_npk[0]

        # -------- Stage 2: Predict Crop --------
        crop_input = pd.DataFrame(
            [[N, P, K, temp, humidity, soil]],
            columns=['N', 'P', 'K', 'temperature', 'humidity', 'soil_moisture']
        )

        crop_scaled = scaler_crop.transform(crop_input)
        crop_prediction = model_crop.predict(crop_scaled)

        result = {
            "Temperature": temp,
            "Humidity": humidity,
            "Soil_Moisture": soil,
            "Predicted_N": round(N, 2),
            "Predicted_P": round(P, 2),
            "Predicted_K": round(K, 2),
            "Recommended_Crop": crop_prediction[0]
        }

        return result

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    app.run()