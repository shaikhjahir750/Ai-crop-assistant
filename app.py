import os
from dotenv import load_dotenv
load_dotenv(override=True)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import streamlit as st
from translations import get_text
import numpy as np
import json
import sqlite3
import requests
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

try:
    from database.firebase_client import insert_prediction, fetch_recommendations, get_client as get_firebase_client, sign_up, sign_in
except Exception:
    insert_prediction = None
    fetch_recommendations = None
    get_firebase_client = None
    sign_up = None
    sign_in = None

import google.generativeai as genai
from datetime import datetime, timedelta

@st.cache_data(ttl=3600, show_spinner=False)
def get_gemini_response(prompt):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return None

# ==============================
# APP CONFIG
# ==============================
st.set_page_config(page_title="🌾 AI Crop Assistant", layout="wide")

def apply_custom_theme():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif !important;
        }
        
        /* Modern gradients for buttons */
        .stButton > button {
            background: linear-gradient(135deg, #00e676, #1de9b6);
            color: #121212 !important;
            border-radius: 8px;
            border: none;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            font-weight: 600;
        }
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 12px rgba(0, 230, 118, 0.4);
            color: #000 !important;
            border: none;
        }
        
        /* Hide default header and footer for clean app feel */
        header {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Glassmorphism for sidebar */
        [data-testid="stSidebar"] {
            background-color: rgba(28, 31, 38, 0.7) !important;
            backdrop-filter: blur(10px);
        }
        
        /* Input styling */
        .stTextInput > div > div > input, .stNumberInput > div > div > input {
            border-radius: 8px;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_theme()

# ==============================
# LANGUAGE CONFIGURATION
# ==============================
if "language" not in st.session_state:
    st.session_state.language = "en"

lang_map = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "मराठी (Marathi)": "mr"
}
default_idx = list(lang_map.values()).index(st.session_state.language) if st.session_state.language in lang_map.values() else 0
selected_lang = st.sidebar.selectbox("🌐 Language / भाषा", list(lang_map.keys()), index=default_idx)
st.session_state.language = lang_map[selected_lang]
lang = st.session_state.language

MODEL_DISEASE_PATH = "models/plant_village_model_20260511_204122.pth"
MODEL_CROP_PATH = "models/crop_model.pkl"
INDICES_PATH = "models/class_indices_20260511_204122.json"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_crop_model():
    try:
        try:
            import joblib
        except Exception:
            st.sidebar.error("❌ Python package 'joblib' is not installed. Install it with `pip install joblib` and redeploy the app.")
            return None
        if not os.path.exists(MODEL_CROP_PATH):
            st.sidebar.error(f"❌ Crop model file not found at {MODEL_CROP_PATH}")
            return None
        model = joblib.load(MODEL_CROP_PATH)
        st.sidebar.success("✅ Crop model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading crop model: {e}")
        return None

@st.cache_resource
def load_disease_model():
    try:
        if not os.path.exists(MODEL_DISEASE_PATH):
            st.sidebar.error(f"❌ Disease model file not found at {MODEL_DISEASE_PATH}")
            return None, None
        if not os.path.exists(INDICES_PATH):
            st.sidebar.error(f"❌ Disease class indices file not found at {INDICES_PATH}")
            return None, None
            
        with open(INDICES_PATH, "r") as f:
            class_to_idx = json.load(f)
        idx_to_class = {v: k for k, v in class_to_idx.items()}
        num_classes = len(idx_to_class)

        weights = models.EfficientNet_B3_Weights.DEFAULT
        model = models.efficientnet_b3(weights=weights)
        in_features = model.classifier[1].in_features
        
        # Must match the architecture from train_plant_village_model.py
        model.classifier = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(p=0.5),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.4),
            nn.Linear(512, num_classes)
        )
        
        # We need weights_only=False because of the BatchNorm layers or depending on how it was saved, but True is usually fine for state_dicts.
        # Actually torch.load with state_dict is fine with weights_only=True usually, but let's just do it securely.
        model.load_state_dict(torch.load(MODEL_DISEASE_PATH, map_location=device))
        model = model.to(device)
        model.eval()
        
        st.sidebar.success("✅ Disease model loaded successfully!")
        return model, idx_to_class
    except Exception as e:
        st.sidebar.error(f"❌ Error loading disease model: {e}")
        return None, None

# Load models at startup
disease_model, idx_to_class = load_disease_model()
crop_model = load_crop_model()

# ==============================
# DATABASE (SQLite)
# ==============================
def init_db():
    os.makedirs("database", exist_ok=True)
    conn = sqlite3.connect("database/crop_app.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT UNIQUE,
                    password TEXT
                )''')
    conn.commit()
    conn.close()

def get_user(email, password):
    conn = sqlite3.connect("database/crop_app.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    user = c.fetchone()
    conn.close()
    return user

def register_user(name, email, password):
    try:
        conn = sqlite3.connect("database/crop_app.db")
        c = conn.cursor()
        c.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)", (name, email, password))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        return False

init_db()

# ==============================
# AUTHENTICATION STATE
# ==============================
if "user" not in st.session_state:
    st.session_state.user = None

if "page" not in st.session_state:
    st.session_state.page = "login"

def logout():
    st.session_state.user = None
    st.session_state.page = "login"
    st.success(get_text("logout_success", lang))

# ==============================
# LOGIN PAGE
# ==============================
def login_page():
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title(get_text("title", lang))
        
        # Simple login form in a container
        with st.container():
            st.subheader(get_text("login", lang))
            email = st.text_input(get_text("email_address", lang), placeholder=get_text("enter_email", lang))
            password = st.text_input(get_text("password", lang), type="password", placeholder=get_text("enter_password", lang))
            
            if st.button(get_text("sign_in", lang), use_container_width=True):
                if not email or not password:
                    st.warning(get_text("warn_enter_credentials", lang))
                else:
                    # If Firebase client is configured, try Firebase Auth first
                    if get_firebase_client is not None and sign_in is not None and get_firebase_client():
                        try:
                            resp = sign_in(email, password)
                            success = False
                            display_name = email
                            # Support multiple response shapes
                            if resp is None:
                                success = False
                            elif isinstance(resp, tuple) and len(resp) == 2:
                                data, err = resp
                                if not err:
                                    success = True
                            elif isinstance(resp, dict):
                                # Look for user in common keys
                                user_obj = None
                                if resp.get('user'):
                                    user_obj = resp.get('user')
                                elif resp.get('data') and isinstance(resp.get('data'), dict) and resp['data'].get('user'):
                                    user_obj = resp['data']['user']
                                elif resp.get('session') and isinstance(resp.get('session'), dict) and resp['session'].get('user'):
                                    user_obj = resp['session']['user']
                                if user_obj:
                                    success = True
                                    display_name = user_obj.get('name') or user_obj.get('email') or display_name
                            if success:
                                st.session_state.user = {
                                    "id": user_obj.get('localId') if user_obj else None,
                                    "name": display_name,
                                    "email": email,
                                    "idToken": user_obj.get('idToken') if user_obj else None
                                }
                                st.session_state.page = "dashboard"
                                st.success(get_text("success_welcome_back", lang, name=display_name))
                            else:
                                st.error(get_text("error_invalid_credentials", lang))
                        except Exception as e:
                            st.error(f"{get_text('error_invalid_credentials', lang)}: {e}")
                    else:
                        user = get_user(email, password)
                        if user:
                            st.session_state.user = {"id": user[0], "name": user[1], "email": user[2]}
                            st.session_state.page = "dashboard"
                            st.success(get_text("success_welcome_back", lang, name=user[1]))
                        else:
                            st.error(get_text("error_invalid_credentials", lang))
            
            st.markdown("---")
            st.write(get_text("dont_have_account", lang))
            if st.button(get_text("create_new_account", lang), use_container_width=True):
                st.session_state.page = "register"

# ==============================
# REGISTER PAGE
# ==============================
def register_page():
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title(get_text("title", lang))
        
        # Simple registration form in a container
        with st.container():
            st.subheader(get_text("create_account", lang))
            
            # Form fields with placeholders and help
            name = st.text_input(get_text("full_name", lang), placeholder=get_text("enter_full_name", lang))
            email = st.text_input(get_text("email_address", lang), placeholder=get_text("enter_email", lang))
            password = st.text_input(get_text("password", lang), type="password", 
                                   placeholder=get_text("choose_password", lang),
                                   help=get_text("password_help", lang))
            
            if st.button(get_text("create_account", lang), use_container_width=True):
                if not name or not email or not password:
                    st.warning(get_text("fill_all_fields", lang))
                elif len(password) < 6:
                    st.warning(get_text("password_length_warn", lang))
                else:
                    # If Firebase is available, create user there
                    if get_firebase_client is not None and sign_up is not None and get_firebase_client():
                        try:
                            resp = sign_up(email, password, user_metadata={"name": name})
                            created = False
                            if resp is None:
                                created = False
                            elif isinstance(resp, tuple) and len(resp) == 2:
                                data, err = resp
                                if not err:
                                    created = True
                            elif isinstance(resp, dict) and (resp.get('user') or resp.get('data')):
                                created = True
                            if created:
                                st.success(get_text("acc_created_success", lang))
                                st.info(get_text("login_info", lang))
                                st.session_state.page = "login"
                            else:
                                st.error(get_text("could_not_create", lang))
                        except Exception as e:
                            st.error(f"{get_text('could_not_create', lang)}: {e}")
                    else:
                        if register_user(name, email, password):
                            st.success(get_text("acc_created_success", lang))
                            st.info(get_text("login_info", lang))
                            st.session_state.page = "login"
                        else:
                            st.error(get_text("email_exists", lang))
                            st.info(get_text("login_instead_info", lang))
            
            st.markdown("---")
            st.write(get_text("already_have_account", lang))
            if st.button(get_text("back_to_login", lang), use_container_width=True):
                st.session_state.page = "login"

# ==============================
# SIDEBAR NAVIGATION
# ==============================
def show_navigation():
    st.sidebar.title(get_text("sidebar_welcome", lang, name=st.session_state.user['name']))
    st.sidebar.markdown("---")
    
    if st.sidebar.button(get_text("sidebar_dashboard", lang)):
        st.session_state.page = "dashboard"
        st.rerun()
        
    if st.sidebar.button(get_text("sidebar_disease", lang)):
        st.session_state.page = "disease_detection"
        st.rerun()
        
    if st.sidebar.button(get_text("sidebar_crop", lang)):
        st.session_state.page = "crop_recommendation"
        st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button(get_text("sidebar_logout", lang)):
        logout()
        st.rerun()

# ==============================
# DISEASE DETECTION PAGE
# ==============================
def disease_detection_page():
    show_navigation()
    
    st.title(get_text("disease_detection_title", lang))
    
    if disease_model is None:
        st.error("❌ Disease detection model is not loaded. Please check if the model file exists in the models folder.")
        return

    # Simple instructions
    with st.expander(get_text("how_to_use", lang)):
        st.write(get_text("disease_step1", lang))
        st.write(get_text("disease_step2", lang))
        st.write(get_text("disease_step3", lang))
        st.write(get_text("disease_step4", lang))

    # Simple file upload
    uploaded_file = st.file_uploader(get_text("upload_leaf_img", lang), type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        temp_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display image
        st.image(temp_path, caption=get_text("uploaded_image_caption", lang), use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            analyze_button = st.button(get_text("analyze", lang), use_container_width=True)
        
        if analyze_button:
            try:
                with st.spinner(get_text("analyzing_image", lang)):
                    # PyTorch Image Preprocessing
                    img = Image.open(temp_path).convert('RGB')
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                    ])
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    with torch.no_grad():
                        outputs = disease_model(img_tensor)
                        probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

                    top_idx = int(np.argmax(probs))
                    label = idx_to_class[top_idx]
                    confidence = float(probs[top_idx])

                    # Display results clearly
                    st.subheader(get_text("results", lang))
                    st.info(get_text("detected_condition", lang, label=label))
                    st.write(get_text("confidence_level", lang, confidence=confidence * 100))

                    st.subheader(get_text("recommendations", lang))
                    if label == "Healthy":
                        st.success(get_text("healthy_msg", lang))
                        st.write(get_text("healthy_desc", lang))
                    elif confidence >= 0.15:
                        if confidence >= 0.25:
                            st.warning(get_text("warning_high_conf", lang, label=label, confidence=confidence*100))
                        else:
                            st.warning(get_text("warning_possible", lang, label=label, confidence=confidence*100))
                            
                        with st.spinner(get_text("generating_ai_tips", lang)):
                            lang_name = "English" if lang == "en" else ("Hindi" if lang == "hi" else "Marathi")
                            prompt = f"A plant is affected by the disease '{label}'. Provide a brief 3-point list of actionable recommendations, treatments, or specific organic/chemical remedy names (e.g., names of specific fungicides/pesticides/bio-fertilizers) to manage this plant disease. Keep the response short, concise, and in markdown format. Write the response in {lang_name}."
                            gemini_tips = get_gemini_response(prompt)
                            
                        if gemini_tips:
                            st.markdown(get_text("ai_treatment_plan", lang))
                            st.markdown(gemini_tips)
                        else:
                            # Fallback
                            st.write(get_text("fallback_tip1", lang))
                            st.write(get_text("fallback_tip2", lang))
                            
                        if confidence >= 0.6:
                            st.info(get_text("extension_info", lang))
                        else:
                            st.info(get_text("retake_info", lang))
                    else:
                        st.success(get_text("low_confidence_msg", lang))
                        st.write(get_text("low_confidence_desc", lang))

                # Additional short messages
                if label == "Healthy":
                    st.success(get_text("healthy_success", lang))
                elif label == "Powdery":
                    st.warning(get_text("powdery_warning", lang))
                elif label == "Rust":
                    st.warning(get_text("rust_warning", lang))

                # Attempt to save prediction to Firebase
                try:
                    if insert_prediction is not None and get_firebase_client is not None and get_firebase_client():
                        with st.spinner(get_text("syncing_firebase", lang)):
                            payload = {
                                "user_email": st.session_state.user['email'] if st.session_state.user else "Anonymous",
                                "type": "disease",
                                "label": label,
                                "confidence": float(confidence),
                                "image_path": temp_path
                            }
                            token = st.session_state.user.get('idToken') if st.session_state.user else None
                            insert_prediction(payload, token=token)
                        st.success(get_text("sync_success", lang))
                    else:
                        st.info(get_text("firebase_not_detected", lang))
                except Exception as e:
                    st.error(get_text("firebase_error", lang, error=str(e)))

            except Exception as e:
                st.error(get_text("prediction_failed", lang, error=str(e)))

IDEAL_NPK = {
    'rice': {'N': 90, 'P': 40, 'K': 40},
    'maize': {'N': 100, 'P': 40, 'K': 30},
    'jute': {'N': 60, 'P': 30, 'K': 30},
    'cotton': {'N': 100, 'P': 40, 'K': 40},
    'coconut': {'N': 50, 'P': 20, 'K': 50},
    'papaya': {'N': 50, 'P': 50, 'K': 50},
    'orange': {'N': 50, 'P': 20, 'K': 20},
    'apple': {'N': 50, 'P': 20, 'K': 50},
    'muskmelon': {'N': 80, 'P': 40, 'K': 40},
    'watermelon': {'N': 80, 'P': 40, 'K': 40},
    'grapes': {'N': 40, 'P': 40, 'K': 80},
    'mango': {'N': 50, 'P': 20, 'K': 40},
    'banana': {'N': 100, 'P': 40, 'K': 100},
    'pomegranate': {'N': 50, 'P': 20, 'K': 40},
    'lentil': {'N': 20, 'P': 40, 'K': 20},
    'blackgram': {'N': 20, 'P': 40, 'K': 20},
    'mungbean': {'N': 20, 'P': 40, 'K': 20},
    'mothbeans': {'N': 20, 'P': 40, 'K': 20},
    'pigeonpeas': {'N': 20, 'P': 40, 'K': 20},
    'kidneybeans': {'N': 20, 'P': 40, 'K': 20},
    'chickpea': {'N': 20, 'P': 40, 'K': 20},
    'coffee': {'N': 80, 'P': 30, 'K': 60}
}

def get_fertilizer_recommendation(crop_name, user_n, user_p, user_k):
    crop = crop_name.lower()
    ideal = IDEAL_NPK.get(crop, {'N': 50, 'P': 50, 'K': 50})
    recs = []
    
    # Get current language from session state
    lang = st.session_state.get('language', 'en')
    
    if user_n < ideal['N'] - 10:
        if lang == "hi":
            recs.append("🌱 **नाइट्रोजन** कम है। नाइट्रोजन युक्त उर्वरक डालें जैसे **यूरिया (46% N)**, **अमोनियम सल्फेट**, या **अमोनियम नाइट्रेट** (रासायनिक), या **ब्लड मील / कम्पोस्ट** (जैविक)।")
        elif lang == "mr":
            recs.append("🌱 **नायट्रोजन** कमी आहे. नायट्रोजनयुक्त खत घाला जसे की **युरिया (46% N)**, **अमोनियम सल्फेट**, किंवा **अमोनियम नायट्रेट** (रासायनिक), किंवा **ब्लड मील / कंपोस्ट** (सेंद्रिय).")
        else:
            recs.append("🌱 **Nitrogen** is low. Add a Nitrogen-rich fertilizer such as **Urea (46% N)**, **Ammonium Sulfate**, or **Ammonium Nitrate** (chemical), or **Blood Meal / Compost** (organic).")
    elif user_n > ideal['N'] + 15:
        if lang == "hi":
            recs.append("⚠️ **नाइट्रोजन** बहुत अधिक है! अधिक एन-उर्वरक (जैसे यूरिया) डालने से बचें क्योंकि यह फलने को रोक सकता है।")
        elif lang == "mr":
            recs.append("⚠️ **नायट्रोजन** जास्त आहे! जास्त एन-खते (जसे की युरिया) घालणे टाळा कारण यामुळे फळ यायला अडथळा येऊ शकतो.")
        else:
            recs.append("⚠️ **Nitrogen** is too high! Avoid adding more N-fertilizers (like Urea) as it may stunt fruiting.")
        
    if user_p < ideal['P'] - 10:
        if lang == "hi":
            recs.append("🌱 **फास्फोरस** कम है। फास्फोरस युक्त उर्वरक डालें जैसे **सिंगल सुपरफॉस्फेट (SSP)**, **डाई-अमोनियम फॉस्फेट (DAP)** (रासायनिक), या **बोन मील** (जैविक)।")
        elif lang == "mr":
            recs.append("🌱 **फॉस्फरस** कमी आहे. फॉस्फरसयुक्त खत घाला जसे की **सिंगल सुपरफॉस्फेट (SSP)**, **डाय-अमोनियम फॉस्फेट (DAP)** (रासायनिक), किंवा **बोन मील** (सेंद्रिय).")
        else:
            recs.append("🌱 **Phosphorus** is low. Add Phosphorus-rich fertilizer such as **Single Superphosphate (SSP)**, **Di-ammonium Phosphate (DAP)** (chemical), or **Bone Meal** (organic).")
    elif user_p > ideal['P'] + 15:
        if lang == "hi":
            recs.append("⚠️ **फास्फोरस** अधिक है। जस्ता/लोहे की कमी को रोकने के लिए पी-उर्वरक (जैसे डीएपी या एसएसपी) डालने से बचें।")
        elif lang == "mr":
            recs.append("⚠️ **फॉस्फरस** जास्त आहे. जस्त/लोहाची कमतरता रोखण्यासाठी पी-खते (जसे की डीएपी किंवा एसएसपी) घालणे टाळा.")
        else:
            recs.append("⚠️ **Phosphorus** is high. Avoid P-fertilizers (like DAP or SSP) to prevent zinc/iron deficiency.")
        
    if user_k < ideal['K'] - 10:
        if lang == "hi":
            recs.append("🌱 **पोटेशियम** कम है। पोटेशियम युक्त उर्वरक डालें जैसे **म्यूरिएट ऑफ पोटाश (MOP)**, **पोटेशियम सल्फेट (SOP)** (रासायनिक), या **केल्प मील / लकड़ी की राख** (जैविक)।")
        elif lang == "mr":
            recs.append("🌱 **पोटॅशियम** कमी आहे. पोटॅशियमयुक्त खत घाला जसे की **म्युरिएट ऑफ पोटॅश (MOP)**, **पोटॅशियम सल्फेट (SOP)** (रासायनिक), किंवा **केल्प मील / लाकडाची राख** (सेंद्रिय).")
        else:
            recs.append("🌱 **Potassium** is low. Add Potassium-rich fertilizer such as **Muriate of Potash (MOP)**, **Potassium Sulfate (SOP)** (chemical), or **Kelp Meal / Wood Ash** (organic).")
    elif user_k > ideal['K'] + 15:
        if lang == "hi":
            recs.append("⚠️ **पोटेशियम** अधिक है। पोषक तत्वों के जमाव को रोकने के लिए के-उर्वरक (जैसे एमओपी) डालने से बचें।")
        elif lang == "mr":
            recs.append("⚠️ **पोटॅशियम** जास्त आहे. पोषक घटकांचे ब्लॉकेज रोखण्यासाठी के-खते (जसे की एमओपी) घालणे टाळा.")
        else:
            recs.append("⚠️ **Potassium** is high. Avoid K-fertilizers (like MOP) to prevent nutrient lock-out.")
        
    if not recs:
        if lang == "hi":
            recs.append("✅ इस फसल के लिए मिट्टी का एनपीके स्तर इष्टतम सीमा में है! नियमित खाद डालना जारी रखें।")
        elif lang == "mr":
            recs.append("✅ या पिकासाठी मातीचे एनपीके पातळी इष्टतम मर्यादेत आहे! नियमित कंपोस्ट खत घालणे सुरू ठेवा.")
        else:
            recs.append("✅ Soil NPK levels are in the optimal range for this crop! Maintain regular composting.")
        
    return recs

def fetch_weather(city_name):
    try:
        # 1. Geocode the city using Open-Meteo
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={city_name}&count=1&format=json"
        geo_res = requests.get(geo_url).json()
        if 'results' not in geo_res or len(geo_res['results']) == 0:
            return None, None, None
        
        lat = geo_res['results'][0]['latitude']
        lon = geo_res['results'][0]['longitude']
        
        # 2. Get past 40 days date range
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=40)
        
        start_str = start_date.strftime('%Y-%m-%d')
        end_str = end_date.strftime('%Y-%m-%d')
        
        # 3. Fetch historical weather
        hist_url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_str}&end_date={end_str}&daily=temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean&timezone=auto"
        res = requests.get(hist_url).json()
        
        if 'daily' in res:
            daily = res['daily']
            temps = [t for t in daily.get('temperature_2m_mean', []) if t is not None]
            rains = [r for r in daily.get('precipitation_sum', []) if r is not None]
            hums = [h for h in daily.get('relative_humidity_2m_mean', []) if h is not None]
            
            avg_temp = sum(temps) / len(temps) if temps else 0
            avg_hum = sum(hums) / len(hums) if hums else 0
            total_rain = sum(rains) if rains else 0
            
            return avg_temp, avg_hum, total_rain
            
        return None, None, None
    except Exception as e:
        return None, None, None

# ==============================
# CROP RECOMMENDATION PAGE
# ==============================
def crop_recommendation_page():
    show_navigation()
    lang = st.session_state.get('language', 'en')

    st.title(get_text("crop_rec_title", lang))

    if crop_model is None:
        st.error("❌ Crop recommendation model is not loaded. Please check if the model file exists in the models folder.")
        return

    # Simple instructions in expander
    with st.expander(get_text("how_to_use", lang)):
        st.write(get_text("crop_rec_step1", lang))
        st.write(get_text("crop_rec_step2", lang))
        st.write(get_text("crop_rec_step3", lang))
        st.write(get_text("crop_rec_step4", lang))

    # Two main sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(get_text("soil_parameters", lang))
        N = st.number_input(get_text("nitrogen", lang), 0, 200, 50, help=get_text("mg_kg", lang))
        P = st.number_input(get_text("phosphorus", lang), 0, 200, 50, help=get_text("mg_kg", lang))
        K = st.number_input(get_text("potassium", lang), 0, 200, 50, help=get_text("mg_kg", lang))
        ph = st.number_input(get_text("soil_ph", lang), 0.0, 14.0, 6.5)
    
    with col2:
        st.subheader(get_text("environmental_conditions", lang))
        city = st.text_input(get_text("auto_fetch_weather", lang), placeholder=get_text("weather_placeholder", lang))
        
        # Initialize default session states for weather
        if 'temp_val' not in st.session_state:
            st.session_state.temp_val = 25.0
        if 'hum_val' not in st.session_state:
            st.session_state.hum_val = 60.0
        if 'rain_val' not in st.session_state:
            st.session_state.rain_val = 100.0
            
        if st.button(get_text("fetch_climate_data", lang), use_container_width=True):
            if city:
                with st.spinner(get_text("getting_weather_data", lang, city=city)):
                    t, h, r = fetch_weather(city)
                    if t is not None:
                        st.session_state.temp_val = float(t)
                        st.session_state.hum_val = float(h)
                        st.session_state.rain_val = float(r)
                        st.success(get_text("weather_loaded_success", lang, city=city))
                    else:
                        st.error(get_text("weather_load_failed", lang))
        
        temperature = st.number_input(get_text("temperature", lang), 0.0, 50.0, key="temp_val", help="°C")
        humidity = st.number_input(get_text("humidity", lang), 0.0, 100.0, key="hum_val", help="%")
        rainfall = st.number_input(get_text("rainfall", lang), 0.0, 500.0, key="rain_val", help="mm")

    # Show only the top recommendation
    k = 1

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button(get_text("get_recommendation", lang), use_container_width=True)

    current_input = (float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall))

    if analyze:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        try:
            with st.spinner(get_text("analyzing_conditions", lang)):
                recommendations = []
                if hasattr(crop_model, 'predict_proba'):
                    probs = crop_model.predict_proba(input_data)[0]
                    classes = list(crop_model.classes_)
                    sorted_idx = np.argsort(probs)[::-1]
                    for idx in sorted_idx[:k]:
                        recommendations.append((classes[idx], float(probs[idx])))
                else:
                    pred = crop_model.predict(input_data)[0]
                    recommendations.append((pred, 1.0))

                # store results in session_state so slider changes won't clear them
                st.session_state['last_recommendations'] = recommendations
                st.session_state['last_input'] = current_input
                st.session_state['last_k'] = k

        except Exception as e:
            st.error(get_text("model_pred_failed", lang, error=str(e)))
            st.error(get_text("model_pred_failed_desc", lang))

    # Display stored recommendations (if any)
    if st.session_state.get('last_recommendations'):
        last_recs = st.session_state['last_recommendations']
        last_input = st.session_state.get('last_input')
        last_k = st.session_state.get('last_k', k)

        # warn if inputs changed since recommendation
        if last_input != current_input:
            st.warning(get_text("inputs_changed_warning", lang))

        st.subheader(get_text("recommended_crop_title", lang))
        for crop_name, prob in last_recs[:k]:
            st.success(get_text("recommended_crop_result", lang, crop_name=crop_name, prob=prob*100))

        st.subheader(get_text("cultivation_tips_title", lang))
        
        for crop_name, prob in last_recs[:k]:
            st.markdown(f"### 🌾 **{crop_name.capitalize()}**")
            
            with st.spinner(get_text("generating_ai_guide", lang, crop_name=crop_name)):
                lang_name = "English" if lang == "en" else ("Hindi" if lang == "hi" else "Marathi")
                prompt = f"The user wants to grow '{crop_name}'. Their soil conditions are: Nitrogen: {current_input[0]} mg/kg, Phosphorus: {current_input[1]} mg/kg, Potassium: {current_input[2]} mg/kg, pH: {current_input[5]}. The temperature is {current_input[3]}°C with {current_input[4]}% humidity. Provide a short 3-point cultivation tip and a brief tailored fertilizer recommendation based on these specific soil metrics, naming specific fertilizers (e.g., Urea, Single Superphosphate (SSP), Muriate of Potash (MOP), DAP, or organic composts) that should be applied. Keep it concise in markdown. Write the response in {lang_name}."
                gemini_tips = get_gemini_response(prompt)
                
            if gemini_tips:
                st.markdown(get_text("ai_cultivation_guide", lang))
                st.markdown(gemini_tips)
            else:
                st.write(get_text("fallback_cult_tip1", lang))
                st.write(get_text("fallback_cult_tip2", lang))
                st.write(get_text("fallback_cult_tip3", lang))
                
            # Embed Fallback Fertilizer logic dynamically based on current user inputs
            fert_recs = get_fertilizer_recommendation(crop_name, current_input[0], current_input[1], current_input[2])
            with st.expander(get_text("fertilizer_plan_title", lang, crop_name=crop_name.capitalize())):
                for rec in fert_recs:
                    st.write(rec)

       
# ==============================
# DASHBOARD PAGE
# ==============================
def dashboard_page():
    show_navigation()
    
    st.title(get_text("dashboard_title", lang))
    st.write(get_text("dashboard_desc", lang))
    
    # Main services in two columns
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.info(f"🧪 **{get_text('disease_detection', lang)}**\n\n{get_text('disease_detection_desc', lang)}")
        if st.button(get_text("disease_detection", lang), use_container_width=True):
            st.session_state.page = "disease_detection"
            st.rerun()
            
    with col2:
        st.info(f"🌱 **{get_text('crop_recommendation', lang)}**\n\n{get_text('crop_recommendation_desc', lang)}")
        if st.button(get_text("crop_recommendation", lang), use_container_width=True):
            st.session_state.page = "crop_recommendation"
            st.rerun()
    
    # Quick guide
    st.markdown("---")
    st.subheader(get_text("quick_guide", lang))
    st.write(get_text("quick_guide_step1", lang))
    st.write(get_text("quick_guide_step2", lang))
    
    # Simple image at the bottom
    st.image(
        "https://img.freepik.com/free-photo/farmer-hand-holding-young-plant_1150-11014.jpg",
        use_container_width=True
    )

# ==============================
# PAGE ROUTING
# ==============================
if st.session_state.user:
    if st.session_state.page == "dashboard":
        dashboard_page()
    elif st.session_state.page == "disease_detection":
        disease_detection_page()
    elif st.session_state.page == "crop_recommendation":
        crop_recommendation_page()
    else:
        dashboard_page()
else:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "register":
        register_page()
