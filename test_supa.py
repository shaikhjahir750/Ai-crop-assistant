# test_supabase.py
from supabase_client import get_client

c = get_client()
if not c:
    print("NO CLIENT: check SUPABASE_URL / SUPABASE_KEY")
else:
    try:
        res = c.table("predictions").select("*").limit(1).execute()
        print("CONNECTED. Query result:", getattr(res, 'data', res))
    except Exception as e:
        print("CONNECTED but query failed:", e)