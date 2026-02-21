from flask import Flask, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models
npk_model = joblib.load("smart-crop-ai/npk_model.pkl")
npk_scaler = joblib.load("smart-crop-ai/npk_scaler.pkl")

crop_model = joblib.load("smart-crop-ai/crop_model.pkl")
crop_scaler = joblib.load("smart-crop-ai/crop_scaler.pkl")


@app.route('/')
def home():
    return "Smart Crop API Running Successfully"


@app.route('/predict')
def predict():

    # Get sensor values
    temp = float(request.args.get('temp'))
    humidity = float(request.args.get('humidity'))
    soil = float(request.args.get('soil'))

    # ----- Step 1: Predict NPK -----
    sensor_df = pd.DataFrame([[temp, humidity, soil]],
                             columns=['temperature','humidity','soil'])

    sensor_scaled = npk_scaler.transform(sensor_df)
    predicted_npk = npk_model.predict(sensor_scaled)

    N = predicted_npk[0][0]
    P = predicted_npk[0][1]
    K = predicted_npk[0][2]

    # ----- Step 2: Predict Crop -----
    crop_df = pd.DataFrame([[N, P, K, temp, humidity]],
                           columns=['N','P','K','temperature','humidity'])

    crop_scaled = crop_scaler.transform(crop_df)
    crop_prediction = crop_model.predict(crop_scaled)

    return {
        "Temperature": temp,
        "Humidity": humidity,
        "Soil Moisture": soil,
        "Predicted N": float(N),
        "Predicted P": float(P),
        "Predicted K": float(K),
        "Recommended Crop": crop_prediction[0]
    }


if __name__ == "__main__":
    app.run()
