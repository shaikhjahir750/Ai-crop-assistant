import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
import streamlit as st
import numpy as np
import json
import sqlite3
import joblib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# ==============================
# APP CONFIG
# ==============================
st.set_page_config(page_title="🌾 AI Crop Assistant", layout="wide")

MODEL_DISEASE_PATH = "models/disease_model_gpu.h5"
MODEL_CROP_PATH = "models/crop_model.pkl"
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD MODELS
# ==============================
@st.cache_resource
def load_crop_model():
    try:
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
            return None
        model = load_model(MODEL_DISEASE_PATH)
        st.sidebar.success("✅ Disease model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Error loading disease model: {e}")
        return None

# Load models at startup
disease_model = load_disease_model()
crop_model = load_crop_model()
class_labels = ['Healthy', 'Powdery', 'Rust']
class_labels = ['Healthy', 'Powdery', 'Rust']

# ==============================
# DATABASE (SQLite)
# ==============================
def init_db():
    conn = sqlite3.connect("crop_app.db")
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
    conn = sqlite3.connect("crop_app.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email, password))
    user = c.fetchone()
    conn.close()
    return user

def register_user(name, email, password):
    try:
        conn = sqlite3.connect("crop_app.db")
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
    st.success("You have been logged out!")

# ==============================
# LOGIN PAGE
# ==============================
def login_page():
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🌿 AI Crop Assistant")
        
        # Simple login form in a container
        with st.container():
            st.subheader("Login")
            email = st.text_input("Email address", placeholder="Enter your email")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            
            if st.button("Sign In", use_container_width=True):
                if not email or not password:
                    st.warning("Please enter both email and password.")
                else:
                    user = get_user(email, password)
                    if user:
                        st.session_state.user = {"id": user[0], "name": user[1], "email": user[2]}
                        st.session_state.page = "dashboard"
                        st.success(f"Welcome back, {user[1]}!")
                    else:
                        st.error("Invalid email or password.")
            
            st.markdown("---")
            st.write("Don't have an account?")
            if st.button("Create New Account", use_container_width=True):
                st.session_state.page = "register"

# ==============================
# REGISTER PAGE
# ==============================
def register_page():
    # Center the form
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("� AI Crop Assistant")
        
        # Simple registration form in a container
        with st.container():
            st.subheader("Create Account")
            
            # Form fields with placeholders and help
            name = st.text_input("Full Name", placeholder="Enter your full name")
            email = st.text_input("Email address", placeholder="Enter your email")
            password = st.text_input("Password", type="password", 
                                   placeholder="Choose a password",
                                   help="Choose a strong password")
            
            if st.button("Create Account", use_container_width=True):
                if not name or not email or not password:
                    st.warning("Please fill in all fields.")
                elif len(password) < 6:
                    st.warning("Password should be at least 6 characters long.")
                else:
                    if register_user(name, email, password):
                        st.success("Account created successfully!")
                        st.info("Please log in with your new account.")
                        st.session_state.page = "login"
                    else:
                        st.error("Email already exists.")
                        st.info("Please try logging in instead.")
            
            st.markdown("---")
            st.write("Already have an account?")
            if st.button("Back to Login", use_container_width=True):
                st.session_state.page = "login"

# ==============================
# SIDEBAR NAVIGATION
# ==============================
def show_navigation():
    st.sidebar.title(f"👋 Welcome {st.session_state.user['name']}")
    st.sidebar.markdown("---")
    
    if st.sidebar.button("🏠 Dashboard"):
        st.session_state.page = "dashboard"
        st.rerun()
        
    if st.sidebar.button("🧪 Disease Detection"):
        st.session_state.page = "disease_detection"
        st.rerun()
        
    if st.sidebar.button("🌱 Crop Recommendation"):
        st.session_state.page = "crop_recommendation"
        st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout"):
        logout()
        st.rerun()

# ==============================
# DISEASE DETECTION PAGE
# ==============================
def disease_detection_page():
    show_navigation()
    
    st.title("🧪 Disease Detection")
    
    if disease_model is None:
        st.error("❌ Disease detection model is not loaded. Please check if the model file exists in the models folder.")
        return

    # Simple instructions
    with st.expander("📋 How to use"):
        st.write("1. Take a clear photo of the leaf")
        st.write("2. Make sure the image is well-lit")
        st.write("3. Upload the image below")
        st.write("4. Click 'Analyze' to get results")

    # Simple file upload
    uploaded_file = st.file_uploader("Upload a leaf image:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        temp_path = os.path.join(UPLOAD_FOLDER, uploaded_file.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Display image
        st.image(temp_path, caption="Uploaded Image", use_container_width=True)
        
        col1, col2 = st.columns([2, 1])
        with col2:
            analyze_button = st.button("Analyze", use_container_width=True)
        
        if analyze_button:
            try:
                with st.spinner('Analyzing image...'):
                    img = image.load_img(temp_path, target_size=(128, 128))
                    img_array = image.img_to_array(img) / 255.0
                    img_array = np.expand_dims(img_array, axis=0)

                    predictions = disease_model.predict(img_array)
                    probs = predictions[0]
                    top_idx = int(np.argmax(probs))
                    label = class_labels[top_idx]
                    confidence = float(probs[top_idx])

                    # Display results clearly
                    st.subheader("Results:")
                    st.info(f"**Detected Condition**: {label}")
                    st.write(f"Confidence: {confidence * 100:.1f}%")

                    # Show per-class percentages in columns
                    st.subheader("Model Output (percentages)")
                    cols = st.columns(len(class_labels))
                    for i, cls in enumerate(class_labels):
                        with cols[i]:
                            pct = float(probs[i]) * 100
                            st.metric(label=cls, value=f"{pct:.1f}%")

                    # Recommendations mapping — load from JSON file for easier editing
                    st.subheader("Recommendations")
                    tips_file = os.path.join("data", "disease_tips.json")
                    try:
                        if os.path.exists(tips_file):
                            with open(tips_file, 'r', encoding='utf-8') as dtf:
                                recommendations = json.load(dtf)
                        else:
                            recommendations = {
                                'Healthy': [
                                    "No treatment required.",
                                    "Continue regular monitoring and good cultural practices."
                                ],
                                'Powdery': [
                                    "Apply sulfur-based fungicides or potassium bicarbonate sprays.",
                                    "Use neem oil or horticultural oils as organic options.",
                                    "Improve air circulation and reduce leaf wetness; avoid overhead irrigation.",
                                    "Remove severely affected leaves and dispose of them safely."
                                ],
                                'Rust': [
                                    "Use copper-based fungicides or other recommended protectants.",
                                    "Consider systemic fungicides for severe infections per local guidance.",
                                    "Remove and destroy infected material and improve airflow."
                                ]
                            }
                    except Exception as e:
                        st.error(f"Failed to load disease tips: {e}")
                        recommendations = {
                            'Healthy': ["No treatment required.", "Monitor regularly."],
                            'Powdery': ["Consider cultural controls and consult extension services."],
                            'Rust': ["Consult extension services for management options."]
                        }

                    # Provide tailored recommendation based on confidence
                    if confidence >= 0.6 and label in recommendations:
                        st.warning(f"⚠️ {label} detected with high confidence ({confidence*100:.1f}%).")
                        for rec in recommendations[label]:
                            st.write(f"• {rec}")
                        st.info("Consult local agricultural extension services for pesticides, dosages and timings.")
                    elif confidence >= 0.3 and label in recommendations:
                        st.warning(f"⚠️ Possible {label} detected ({confidence*100:.1f}%). Consider retesting or manual inspection.")
                        for rec in recommendations[label][:2]:
                            st.write(f"• {rec}")
                        st.info("If unsure, collect additional images from different leaves/angles.")
                    else:
                        st.success("Detection confidence is low; no treatment recommended automatically.")
                        st.write("Consider taking clearer images or consulting an expert if symptoms are visible.")

                # Additional short messages
                if label == "Healthy":
                    st.success("✅ Your crop looks healthy!")
                elif label == "Powdery":
                    st.warning("⚠️ Powdery mildew detected. See recommendations above.")
                elif label == "Rust":
                    st.warning("⚠️ Rust detected. See recommendations above.")

            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ==============================
# CROP RECOMMENDATION PAGE
# ==============================
def crop_recommendation_page():
    show_navigation()

    st.title("🌱 Crop Recommendation")

    if crop_model is None:
        st.error("❌ Crop recommendation model is not loaded. Please check if the model file exists in the models folder.")
        return

    # Simple instructions in expander
    with st.expander("📋 How to use"):
        st.write("1. Enter soil test results (NPK values)")
        st.write("2. Add environmental conditions")
        st.write("3. Click 'Get Recommendation'")
        st.write("4. View suggested crop for your conditions")

    # Two main sections
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Soil Parameters")
        N = st.number_input("Nitrogen (N)", 0, 200, 50, help="mg/kg")
        P = st.number_input("Phosphorus (P)", 0, 200, 50, help="mg/kg")
        K = st.number_input("Potassium (K)", 0, 200, 50, help="mg/kg")
        ph = st.number_input("Soil pH", 0.0, 14.0, 6.5)
    
    with col2:
        st.subheader("Environmental Conditions")
        temperature = st.number_input("Temperature", 0.0, 50.0, 25.0, help="°C")
        humidity = st.number_input("Humidity", 0.0, 100.0, 60.0, help="%")
        rainfall = st.number_input("Rainfall", 0.0, 500.0, 100.0, help="mm")

    # Slider and analysis button (slider outside button to avoid losing results on change)
    k = st.slider('Number of recommendations', min_value=1, max_value=5, value=3)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze = st.button("Get Recommendation", use_container_width=True)

    current_input = (float(N), float(P), float(K), float(temperature), float(humidity), float(ph), float(rainfall))

    if analyze:
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        try:
            with st.spinner('Analyzing conditions...'):
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
            st.error(f"❌ Model prediction failed: {str(e)}")
            st.error("Please ensure the model file is properly trained and saved.")

    # Display stored recommendations (if any)
    if st.session_state.get('last_recommendations'):
        last_recs = st.session_state['last_recommendations']
        last_input = st.session_state.get('last_input')
        last_k = st.session_state.get('last_k', k)

        # warn if inputs changed since recommendation
        if last_input != current_input:
            st.warning('Inputs have changed since the last recommendation — click Get Recommendation to update results.')

        st.subheader("Recommended Crops")
        for crop_name, prob in last_recs[:k]:
            st.success(f"{crop_name} — Confidence: {prob*100:.1f}%")

        st.subheader("Cultivation Tips")
        tips_file = os.path.join("data", "crop_tips.json")
        try:
            if os.path.exists(tips_file):
                with open(tips_file, 'r', encoding='utf-8') as tf:
                    crop_tips = json.load(tf)
            else:
                crop_tips = {
                    'default': [
                        'Follow local variety and sowing-time recommendations.',
                        'Base fertilizer applications on a soil test and maintain soil organic matter.',
                        'Monitor regularly for pests and diseases and adopt IPM.'
                    ]
                }
        except Exception as e:
            st.error(f"Failed to load crop tips: {e}")
            crop_tips = {
                'default': [
                    'Follow local variety and sowing-time recommendations.',
                    'Base fertilizer applications on a soil test and maintain soil organic matter.'
                ]
            }

        for crop_name, prob in last_recs[:k]:
            tips = crop_tips.get(crop_name.lower(), crop_tips.get('default', []))
            st.markdown(f"**{crop_name}**")
            for t in tips:
                st.write(f"• {t}")

       
# ==============================
# DASHBOARD PAGE
# ==============================
def dashboard_page():
    show_navigation()
    
    st.title("🌾 AI Crop Assistant")
    st.write("Welcome! Choose a service to get started:")
    
    # Main services in two columns
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.info("🧪 **Disease Detection**\n\nUpload leaf images to check for diseases")
        if st.button("Disease Detection", use_container_width=True):
            st.session_state.page = "disease_detection"
            st.rerun()
            
    with col2:
        st.info("🌱 **Crop Recommendation**\n\nGet crop suggestions based on soil data")
        if st.button("Crop Recommendation", use_container_width=True):
            st.session_state.page = "crop_recommendation"
            st.rerun()
    
    # Quick guide
    st.markdown("---")
    st.subheader("📋 Quick Guide")
    st.write("1. **Disease Detection**: Take a clear photo of a leaf and upload it")
    st.write("2. **Crop Recommendation**: Enter your soil test results and weather data")
    
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
