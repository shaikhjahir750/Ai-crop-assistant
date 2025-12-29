import joblib
from tensorflow.keras.models import load_model
import warnings

warnings.filterwarnings("ignore")

try:
    disease_model = load_model("models/disease_model_gpu.h5")
    print("✅ Disease model loaded successfully!")
except Exception as e:
    print("❌ Error loading disease model:", e)

try:
    crop_model = joblib.load("models/crop_model.pkl")
    print("✅ Crop model loaded successfully!")
except Exception as e:
    print("❌ Error loading crop model:", e)
