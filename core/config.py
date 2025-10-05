from dotenv import load_dotenv
import os

load_dotenv()
DB_URL = os.getenv("DATABASE_URL", "sqlite:///reviews.db")
APP_META = {
    "title": "Decathlon Review Analytics",
    "version": "0.2.0",
    "docs_url": "/docs",
    "openapi_url": "/openapi.json",
}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5-nano")
EMBEDDING_CACHE_PATH = os.getenv("EMBEDDING_CACHE_PATH", "./data/embeddings.pkl")
TOP_K = int(os.getenv("TOP_K", "5"))

