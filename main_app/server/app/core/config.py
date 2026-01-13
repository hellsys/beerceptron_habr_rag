import os

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
EMBED_BASE_URL = os.getenv("EMBED_BASE_URL", BASE_URL)
EMBED_API_KEY = os.getenv("EMBED_API_KEY", API_KEY)

# Модель для генерации
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3-32b")

# Модель / метод эмбеддингов
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# Параметры генерации
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.0"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "512"))

# Управление логами
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Параметры Qdrant
QDRANT_API_URL = os.getenv("QDRANT_API_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
