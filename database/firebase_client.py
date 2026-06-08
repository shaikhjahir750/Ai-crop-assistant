import os
from dotenv import load_dotenv

try:
    import pyrebase
except Exception:
    pyrebase = None

# Load .env using absolute path relative to this script
current_dir = os.path.dirname(os.path.abspath(__file__))
dotenv_path = os.path.join(current_dir, "..", ".env")
load_dotenv(dotenv_path=dotenv_path, override=True)

def get_client():
    if pyrebase is None:
        return None
        
    api_key = os.getenv("FIREBASE_API_KEY")
    if not api_key:
        return None
        
    config = {
        "apiKey": api_key,
        "authDomain": os.getenv("FIREBASE_AUTH_DOMAIN"),
        "databaseURL": os.getenv("FIREBASE_DATABASE_URL"),
        "projectId": os.getenv("FIREBASE_PROJECT_ID"),
        "storageBucket": os.getenv("FIREBASE_STORAGE_BUCKET"),
        "messagingSenderId": os.getenv("FIREBASE_MESSAGING_SENDER_ID"),
        "appId": os.getenv("FIREBASE_APP_ID")
    }
    return pyrebase.initialize_app(config)

def insert_prediction(payload: dict, token: str = None):
    client = get_client()
    if client is None:
        raise RuntimeError("Firebase client not configured. Set FIREBASE keys in .env.")
    db = client.database()
    return db.child("predictions").push(payload, token=token)

def fetch_recommendations(limit: int = 10):
    client = get_client()
    if client is None:
        return []
    db = client.database()
    # Pyrebase4 fetch limit isn't direct strictly natively unless ordered, returning all or slicing manually for UI
    all_recs = db.child("recommendations").get()
    if all_recs.val() is None: return []
    return list(all_recs.val().values())[:limit]

def sign_up(email: str, password: str, user_metadata: dict = None):
    client = get_client()
    if client is None:
        raise RuntimeError("Firebase client not configured.")
    auth = client.auth()
    db = client.database()
    
    user = auth.create_user_with_email_and_password(email, password)
    
    if user_metadata and 'name' in user_metadata:
        db.child("users").child(user['localId']).set({"name": user_metadata['name'], "email": email}, token=user['idToken'])
    return {"user": {"email": email, "name": user_metadata.get('name', ''), "idToken": user['idToken'], "localId": user['localId']}}

def sign_in(email: str, password: str):
    client = get_client()
    if client is None:
        raise RuntimeError("Firebase client not configured.")
    auth = client.auth()
    try:
        user = auth.sign_in_with_email_and_password(email, password)
        # Fetch name if available
        db = client.database()
        user_info = db.child("users").child(user['localId']).get(token=user['idToken'])
        name = user_info.val().get('name') if (user_info and user_info.val()) else email
        return {"user": {"email": email, "name": name, "idToken": user['idToken'], "localId": user['localId']}}
    except Exception as e:
        raise e
