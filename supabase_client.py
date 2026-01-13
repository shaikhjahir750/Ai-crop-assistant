import os
from dotenv import load_dotenv

try:
    from supabase import create_client
except Exception:
    create_client = None

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def get_client():
    if not SUPABASE_URL or not SUPABASE_KEY or create_client is None:
        return None
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_prediction(payload: dict):
    client = get_client()
    if client is None:
        raise RuntimeError("Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY.")
    return client.table("predictions").insert(payload).execute()

def fetch_recommendations(limit: int = 10):
    client = get_client()
    if client is None:
        return []
    res = client.table("recommendations").select("*").limit(limit).execute()
    try:
        return res.data
    except Exception:
        return res

def sign_up(email: str, password: str, user_metadata: dict = None):
    client = get_client()
    if client is None:
        raise RuntimeError("Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY.")
    # Try multiple possible client auth method signatures for compatibility
    try:
        return client.auth.sign_up({"email": email, "password": password, "options": {"data": user_metadata}})
    except Exception:
        try:
            return client.auth.sign_up(email=email, password=password, user_metadata=user_metadata)
        except Exception:
            try:
                return client.auth.api.sign_up(email=email, password=password, data=user_metadata)
            except Exception as e:
                raise e

def sign_in(email: str, password: str):
    client = get_client()
    if client is None:
        raise RuntimeError("Supabase client not configured. Set SUPABASE_URL and SUPABASE_KEY.")
    try:
        return client.auth.sign_in_with_password({"email": email, "password": password})
    except Exception:
        try:
            return client.auth.sign_in(email=email, password=password)
        except Exception:
            try:
                return client.auth.api.sign_in(email=email, password=password)
            except Exception as e:
                raise e
