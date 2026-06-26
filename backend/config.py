import os
import json

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_CONFIG_PATH = os.path.join(BASE_DIR, "..", "config.json")

# Default values
DEFAULT_CONFIG = {
    "model_name": "all-MiniLM-L6-v2",
    "top_k": 3,
    "score_threshold": 0.35,
    "max_chunk_size": 500,
    "overlap": 50,
    "synonyms": {}
}

# Load root config if exists
root_config = {}
if os.path.exists(ROOT_CONFIG_PATH):
    try:
        with open(ROOT_CONFIG_PATH, "r", encoding="utf-8") as f:
            root_config = json.load(f)
    except Exception as e:
        print(f"Warning: Failed to load root config.json ({e}). Using base defaults.")

def get_setting(key, default):
    # Try Env var, then Root Config, then Default Config
    env_key = key.upper()
    if os.getenv(env_key) is not None:
        return os.getenv(env_key)
    return root_config.get(key, DEFAULT_CONFIG.get(key, default))

# Config parameters
MODEL_NAME = get_setting("model_name", "all-MiniLM-L6-v2")
TOP_K = int(get_setting("top_k", 3))
SCORE_THRESHOLD = float(get_setting("score_threshold", 0.35))
MAX_CHUNK_SIZE = int(get_setting("max_chunk_size", 500))
OVERLAP = int(get_setting("overlap", 50))
SYNONYMS = get_setting("synonyms", {})

# Database settings
MONGODB_URL = get_setting("mongodb_url", "mongodb://localhost:27017")
MONGODB_DB_NAME = get_setting("mongodb_db_name", "semantic_search")
CHROMA_PERSIST_DIRECTORY = get_setting("chroma_persist_directory", os.path.join(BASE_DIR, "chroma_db"))
CHROMA_COLLECTION_NAME = get_setting("chroma_collection_name", "documents")
