# train_crop_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import os

# Create reports folder if not exists
os.makedirs("reports", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Features and target
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
y = data['label']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Calculate accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Plot accuracy
plt.figure(figsize=(6,4))
plt.bar(['Training Accuracy', 'Testing Accuracy'], 
        [train_accuracy, test_accuracy])
plt.ylabel("Accuracy")
plt.title("Training vs Testing Accuracy")

# Save plot
plt.savefig("reports/accuracy2.png")
plt.close()

print("📊 Accuracy plot saved at: reports/accuracy2.png")

# Save model
joblib.dump(model, "models/crop_model.pkl")
print("✅ Crop Recommendation Model Saved Successfully at 'models/crop_model.pkl'")
