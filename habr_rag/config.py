import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASETS_DIR = DATA_DIR / "datasets"
RESULTS_DIR = DATA_DIR / "results"

CHUNKS_FILE = PROCESSED_DATA_DIR / "chunked_docs.pkl"
SINGLE_HOP_DATASET = DATASETS_DIR / "dataset_single_hop.json"
MULTI_HOP_DATASET = DATASETS_DIR / "dataset_multi_hop.json"
META_COLS_FILE = RAW_DATA_DIR / "meta_cols.csv"

EVAL_RESULTS_FILE = RESULTS_DIR / "evaluation_results_retrieval.json"
EVAL_METRICS_FILE = RESULTS_DIR / "evaluation_metrics_retrieval.csv"

# Qdrant Configuration
QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "https://590d8b86-25a0-42c7-bf89-43a3fe0387ac.eu-central-1-0.aws.cloud.qdrant.io:6333",
)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_PREFIX = "habr_rag_bge_m3"

# Embeddings Configuration
EMBEDDING_MODEL = "bge-m3"
EMBEDDING_BASE_URL = "https://gpt.mwsapis.ru/projects/mws-ai-automation/openai/v1"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Upload Configuration
BATCH_SIZE = 500
MAX_CONCURRENT_UPLOADS = 10
MAX_CHUNKS_TO_UPLOAD = 5000000

# Generation Configuration
MODEL_NAME = "gpt-5.1-2025-11-13"

