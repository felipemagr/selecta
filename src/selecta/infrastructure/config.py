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


class NAVIDROME:
    URL = os.environ.get("NAVIDROME_URL")
    USER = os.environ.get("NAVIDROME_USER")
    PASSWORD = os.environ.get("NAVIDROME_PASSWORD")
    TIMEOUT_SECONDS = 30
    PAGE_SIZE = 500


class COHERE:
    API_KEY = os.environ.get("COHERE_API_KEY")
    API_URL = "https://api.cohere.com"
    RERANK_MODEL = "rerank-v3.5"
    TIMEOUT_SECONDS = 3


class MISTRAL:
    API_KEY = os.environ.get("MISTRAL_API_KEY")
    API_URL = "https://api.mistral.ai/v1"
    CHAT_MODEL = "ministral-8b-2512"
    AGENT_MODEL = os.environ.get("SELECTA_AGENT_MODEL", "mistral-small-2603")
    AGENT_TEMPERATURE = 0.3
    EMBED_MODEL = "mistral-embed-2312"
    EMBED_BATCH_SIZE = 16
    MAX_OUTPUT_TOKENS = 400
    TIMEOUT_SECONDS = 60
    RETRY_WAIT_SECONDS = 2
    MAX_RETRIES = 5


class VOICE:
    STT_MODEL = "voxtral-mini-transcribe-3"
    STT_SAMPLE_RATE = 16000
    TTS_MODEL = "voxtral-mini-tts-3"
    TIMEOUT_SECONDS = 15
    TTS_ATTEMPTS = 2
    MAX_SILENCE_SECONDS = 10
    VOICES = {
        "en": "82c99ee6-f932-423f-a4a3-d403c8914b8d",  # Jane, neutral
        "es": "01a0f318-4afd-77a5-aa5e-f4d741c30187",  # Elena
        "fr": "5a271406-039d-46fe-835b-fbbb00eaf08d",  # Marie, neutral
    }


class SERVER:
    HOST = os.environ.get("SELECTA_HOST", "127.0.0.1")
    PORT = int(os.environ.get("SELECTA_PORT", "8000"))
    SESSION_IDLE_MINUTES = 60
    COVER_TIMEOUT_SECONDS = 10
