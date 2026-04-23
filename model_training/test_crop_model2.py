# === predict_crop.py ===
import joblib
import numpy as np

# === Step 1: Load the Trained Model ===
model = joblib.load("models/crop_model2.pkl")

print("🌾 Crop Recommendation System")
print("----------------------------------")

# === Step 2: Take User Inputs ===
try:
    N = float(input("Enter Nitrogen content (N): "))
    P = float(input("Enter Phosphorus content (P): "))
    K = float(input("Enter Potassium content (K): "))
    pH = float(input("Enter soil pH value: "))
    rainfall = float(input("Enter rainfall (mm): "))
    temperature = float(input("Enter temperature (°C): "))

    # === Step 3: Prepare Data for Prediction ===
    features = np.array([[N, P, K, pH, rainfall, temperature]])

    # === Step 4: Make Prediction ===
    prediction = model.predict(features)

    # === Step 5: Display Result ===
    print("----------------------------------")
    print(f"✅ Recommended Crop: {prediction[0]}")
    print("----------------------------------")

except Exception as e:
    print("❌ Error:", e)
