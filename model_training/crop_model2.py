# === train_crop_model.py ===
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# === Step 1: Load Dataset ===
data = pd.read_excel(r"Train_Dataset.xlsx")

# === Step 2: Define Features and Target ===
# Your CSV columns: Crop, N, P, K, pH, rainfall, temperature
X = data[['N', 'P', 'K', 'pH', 'rainfall', 'temperature']]
y = data['Crop']

# === Step 3: Split Data ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# === Step 4: Train Model ===
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Step 5: Evaluate Model ===
accuracy = model.score(X_test, y_test)
print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

# === Step 6: Save Model ===
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/crop_model2.pkl")

print("🌾 Crop Recommendation Model Saved Successfully at 'models/crop_model2.pkl'")
