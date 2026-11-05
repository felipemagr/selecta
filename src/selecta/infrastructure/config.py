import os
from pathlib import Path


class APP:
    DATA_DIR = Path(os.environ.get("SELECTA_DATA_DIR", "data"))


class DISCOGS:
    TOKEN = os.environ.get("DISCOGS_TOKEN")
    API_URL = "https://api.discogs.com"
    USER_AGENT = "selecta/0.1"
    OUT_DIR = APP.DATA_DIR / "discogs"
    PAGE_SIZE = 100
    TIMEOUT_SECONDS = 30
    RATE_LIMIT_WAIT_SECONDS = 60
    RETRY_WAIT_SECONDS = 60
    MAX_RETRIES = 5


class DATABASE:
    URL = os.environ.get("DATABASE_URL")
