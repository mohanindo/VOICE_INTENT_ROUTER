import os
from dotenv import load_dotenv

load_dotenv()

APP_NAME = "SourceBytes Agent Engine"
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is missing in environment variables")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "1536"))
##text-embedding-3-small
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
RULE_CACHE_SIZE = int(os.getenv("RULE_CACHE_SIZE", "500"))

FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "rule_faiss.index")
FAISS_METADATA_FILE = os.getenv("FAISS_METADATA_FILE", "rule_faiss_metadata.pkl")