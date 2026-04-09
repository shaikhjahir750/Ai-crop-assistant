# train_excel_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# Create reports folder if not exists
os.makedirs("report 3", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load dataset
try:
    data = pd.read_csv("Train_Dataset.csv")
except Exception as e:
    print(f"Error loading CSV file: {e}")
    exit(1)

# Features and target
X = data[['N', 'P', 'K', 'pH', 'rainfall', 'temperature']]
y = data['Crop']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Ensemble Model to prevent overfitting while keeping RF Primary
print("Training Ensemble Learning Model (VotingClassifier) on Excel Dataset...")

# Primary Classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=8, min_samples_leaf=10, min_samples_split=15, random_state=42)

# Secondary Supporting Classifiers
lr = Pipeline([
    ('scaler', StandardScaler()),
    ('logistic', LogisticRegression(max_iter=2000, random_state=42))
])
dt = DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42)

# Soft Voting Ensemble emphasizing the Random Forest
model = VotingClassifier(
    estimators=[('RandomForest', rf), ('LogisticRegression', lr), ('DecisionTree', dt)],
    voting='soft',
    weights=[3, 1, 1], # Weight Random Forest heavily so it acts as primary
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate accuracy
train_accuracy = model.score(X_train, y_train)
test_accuracy = model.score(X_test, y_test)

print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"Testing Accuracy: {test_accuracy * 100:.2f}%")

# Save classification report
report = classification_report(y_test, y_pred)
with open("report 3/classification_report.txt", "w") as f:
    f.write("Classification Report:\n\n")
    f.write(report)
print("Classification report saved at: report 3/classification_report.txt")

# Plot and save confusion matrix (Advanced Visualization)
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='flare', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix - Crop Recommendation (Excel)", fontsize=18, pad=20)
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("report 3/confusion_matrix.png", dpi=300)
plt.close()
print("Confusion matrix plot saved at: report 3/confusion_matrix.png")

# Plot accuracy (Advanced Visualization)
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")
ax = sns.barplot(
    x=['Training Accuracy', 'Testing Accuracy'], 
    y=[train_accuracy, test_accuracy], 
    palette='viridis',
    hue=['Training Accuracy', 'Testing Accuracy'],
    legend=False
)
plt.ylim(0.0, 1.1)
plt.ylabel("Accuracy Score", fontsize=14)
plt.title("Model Accuracy: Training vs Testing", fontsize=16, pad=15)

# Add text on top of bars
for i, v in enumerate([train_accuracy, test_accuracy]):
    ax.text(i, v + 0.02, f"{v*100:.2f}%", ha='center', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig("report 3/accuracy.png", dpi=300)
plt.close()

print("Accuracy plot saved at: report 3/accuracy.png")

# Save model
joblib.dump(model, "models/excel_crop_model.pkl")
print("Excel Crop Recommendation Model Saved Successfully at 'models/excel_crop_model.pkl'")
