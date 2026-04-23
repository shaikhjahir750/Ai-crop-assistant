# AI Crop Assistant - Project Context

This file serves as the system prompt and context map for any AI model working on this repository. It provides a high-level overview of the architecture, stack, state management, and file structure to ensure seamless contributions.

## 1. Project Overview
"AI Crop Assistant" is a precision agriculture web application built to help farmers optimize their yield. It provides two primary ML-powered services:
1. **Disease Detection:** A Computer Vision model that analyzes leaf images to detect diseases (Healthy, Powdery Mildew, Rust) and returns organic treatment suggestions.
2. **Crop Recommendation:** A Machine Learning classifier that suggests the best crops based on soil metrics (N, P, K, pH) and environmental factors (Temperature, Humidity, Rainfall).

## 2. Tech Stack
- **Frontend / Fullstack:** Streamlit (`app.py`)
- **Machine Learning:** 
  - TensorFlow / Keras (CNN for image classification: `disease_model_gpu.h5`)
  - Scikit-Learn (Random Forest for crop prediction: `crop_model.pkl`)
- **Database:** SQLite (`database/crop_app.db`) for local tracking, with optional Supabase Auth hooks.
- **APIs:** Open-Meteo (for free Geocoding and Climate data).
- **Styling:** Custom CSS injected via Streamlit markdown (Glassmorphism, animations) + `.streamlit/config.toml` (Sleek Dark Mode).

## 3. Directory Structure
- `app.py`: The core application file containing UI logic, session control, model loading, API calls, and custom CSS styling (`apply_custom_theme()`).
- `model_training/`: Contains all scripts used to originally compile, test, and graph the machine learning models (`train_crop_model.py`, `graphs.py`, etc.).
- `models/`: Production-ready compiled model weights (`.pkl`, `.h5`).
- `data/`: Raw testing CSVs, Excel data, and JSON dictionary files (`crop_tips.json`, `disease_tips.json`) which the app reads for dynamic recommendation text.
- `database/`: Local `.db` files and SQL schema references.
- `static/uploads/`: Temporary cache for leaf images uploaded by users.

## 4. Key Architectural Patterns & AI Instructions
When modifying this codebase, future AI models should strictly adhere to the following patterns:

### A. Streamlit Session State & Widgets
- **API UI Updates:** Streamlit intrinsically caches user widget inputs (like `st.number_input`). When programmatically updating UI values (e.g., auto-filling weather from an API), you *must* bind the widget to a unique `key` (e.g., `key="temp_val"`) and update `st.session_state.temp_val` rather than passing the `value=` argument directly.
- **Auth Flow:** Basic auth uses `st.session_state.user`. Do not implement complex token management without checking the existing SQLite boilerplate in `app.py`.

### B. Machine Learning Integration
- **Loading:** Models are loaded into memory globally via `@st.cache_resource` in `app.py`. Do not re-load models inside functional prediction loops.
- **Retraining:** If model features change, the scripts in `model_training/` must be updated, and `.h5` or `.pkl` binaries must be replaced in `models/`. 
- **Fertilizer Logic:** The fertilizer strategy is handled heuristically by comparing user input to an `IDEAL_NPK` dictionary located inside `app.py` before the `crop_recommendation_page()` block.

### C. UI / Aesthetics
- Avoid default Streamlit UI elements when possible. The app uses a highly customized CSS block inside `apply_custom_theme()` located near the top of `app.py`.
- If adding new buttons, rely on the global CSS wrapper (`.stButton > button`) which already provides gradients and hover animations.
- Emojis are used structurally to define headers and UX feedback (e.g., 🌿, 🌱, ⚠️, ✅). Maintain this visual consistency.
