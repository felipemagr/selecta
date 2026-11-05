import os
from pathlib import Path


class DISCOGS:
    TOKEN = os.environ.get("DISCOGS_TOKEN")
    API_URL = "https://api.discogs.com"
    USER_AGENT = "selecta/0.1"
    OUT_DIR = Path("data/discogs")
    PAGE_SIZE = 100
    TIMEOUT_SECONDS = 30
    RATE_LIMIT_WAIT_SECONDS = 60
    RETRY_WAIT_SECONDS = 60
    MAX_RETRIES = 5


class DATABASE:
    URL = os.environ.get("DATABASE_URL", "postgresql://selecta:selecta@localhost:5432/selecta")


class HOUSEHOLD:
    ID = os.environ.get("SELECTA_HOUSEHOLD_ID", "default")
