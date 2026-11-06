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


class TAVILY:
    API_KEY = os.environ.get("TAVILY_API_KEY")
    API_URL = "https://api.tavily.com"
    TIMEOUT_SECONDS = 5
    MAX_RESULTS = 3
    MAX_CONTENT_CHARS = 600


class COHERE:
    API_KEY = os.environ.get("COHERE_API_KEY")
    API_URL = "https://api.cohere.com"
    RERANK_MODEL = "rerank-v3.5"
    TIMEOUT_SECONDS = 3


class MISTRAL:
    API_KEY = os.environ.get("MISTRAL_API_KEY")
    API_URL = "https://api.mistral.ai/v1"
    CHAT_MODEL = "ministral-8b-2512"
    AGENT_MODEL = "mistral-small-2603"
    AGENT_TEMPERATURE = 0.3
    EMBED_MODEL = "mistral-embed-2312"
    EMBED_BATCH_SIZE = 16
    MAX_OUTPUT_TOKENS = 400
    TIMEOUT_SECONDS = 60
    RETRY_WAIT_SECONDS = 2
    MAX_RETRIES = 5
