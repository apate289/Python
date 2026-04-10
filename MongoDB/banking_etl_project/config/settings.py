"""
Configuration — loaded from environment variables with sensible defaults.
Copy .env.example → .env and fill in your values.
"""
import os
from dotenv import load_dotenv

load_dotenv()

# ── MongoDB ───────────────────────────────────
MONGO_URI  = os.getenv("MONGO_URI",  "mongodb://localhost:27017/")
DB_NAME    = os.getenv("DB_NAME",    "banking_etl")

# ── ETL ───────────────────────────────────────
ETL_SOURCE = os.getenv("ETL_SOURCE", "data/clients.json")

# ── App ───────────────────────────────────────
APP_TITLE  = os.getenv("APP_TITLE",  "Banking Client Intelligence")
PAGE_SIZE  = int(os.getenv("PAGE_SIZE", "25"))
