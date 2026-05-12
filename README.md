# AI Crop Assistant 

AI Crop Assistant is a precision agriculture web application designed to help farmers optimize their yield and maintain plant health. Powered by Machine Learning and Computer Vision, the platform offers data-driven insights tailored to local soil metrics and environmental conditions.

## Features

1. **Disease Detection:** 
   - A Computer Vision model (CNN) analyzes leaf images to detect diseases across various plants (e.g., Apple Scab, Corn Northern Leaf Blight, Potato Early Blight, Tomato Target Spot).
   - Returns confidence scores and provides organic treatment suggestions based on the identified disease.
2. **Crop Recommendation:** 
   - A Machine Learning classifier (Random Forest) suggests the best crops based on soil metrics (Nitrogen, Phosphorous, Potassium, pH) and environmental factors (Temperature, Humidity, Rainfall).
   - Offers location-based climate autofill using the Open-Meteo API.
3. **Fertilizer Strategy:** 
   - Evaluates user soil metrics against ideal NPK ratios and recommends customized fertilizer action plans to balance soil nutrients.
4. **Authentication & History:** 
   - Secure user signup/signin powered by Firebase (with an SQLite fallback).
   - Tracks past crop recommendations and disease predictions in a unified user dashboard.

## Tech Stack

- **Frontend / Fullstack:** Streamlit (`app.py`) with Custom CSS (Glassmorphism, animations)
- **Machine Learning:** 
  - TensorFlow / Keras (CNN for image classification)
  - Scikit-Learn (Random Forest for crop prediction)
- **Database:** Firebase / SQLite
- **APIs:** Open-Meteo (Geocoding and Climate data)

## Directory Structure

```text
Ai-crop-assistant/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── PROJECT_CONTEXT.md          # Internal developer context map
├── database/                   # Firebase configuration and SQLite database files
├── data/                       # CSVs, Excel data, and JSON dictionary files (crop_tips.json, etc.)
├── models/                     # Production-ready compiled model weights (.pkl, .h5)
├── model_training/             # Training scripts used to compile and test ML models
├── scripts/                    # Utility scripts
│   ├── data_prep/              # Scripts for dataset arrangement and leakage analysis
│   └── testing/                # Scripts for dataloader testing
├── plots/                      # Evaluation plots (confusion matrices, PR curves, etc.)
├── reports/                    # Classification reports and model summaries
├── logs/                       # Application logs (error.log)
└── static/                     # Static assets and temporary cache for uploads
```

## Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shaikhjahir750/Ai-crop-assistant.git
   cd Ai-crop-assistant
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Setup:**
   Create a `.env` file in the root directory and add your Firebase configuration:
   ```env
   FIREBASE_API_KEY=your_api_key
   FIREBASE_AUTH_DOMAIN=your_auth_domain
   FIREBASE_DATABASE_URL=your_database_url
   FIREBASE_PROJECT_ID=your_project_id
   FIREBASE_STORAGE_BUCKET=your_storage_bucket
   FIREBASE_MESSAGING_SENDER_ID=your_messaging_sender_id
   FIREBASE_APP_ID=your_app_id
   ```

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## UI/UX Note
The application uses a highly customized Streamlit UI with a sleek dark mode. It injects custom CSS for gradient buttons, hover animations, and glassmorphic panels. Ensure you do not overwrite `.streamlit/config.toml` unless you intend to alter the base theme.
