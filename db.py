# db.py  – Supabase connector
import os, datetime
from supabase import create_client

SUPA_URL  = os.getenv("SUPABASE_URL")
SUPA_KEY  = os.getenv("SUPABASE_SERVICE_KEY")  # service role key (server‑side only)

supabase = create_client(SUPA_URL, SUPA_KEY)

def log_chat(user_query: str, bot_answer: str):
    supabase.table("chat_log").insert({
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "query":     user_query,
        "answer":    bot_answer
    }).execute()
