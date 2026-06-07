"""Central RAG configuration.

All hyperparameters / model names come from environment variables so the SAME
values are used at ingestion time (scripts/ingest.py) and serving time (the API),
and so GET /api/stats always reflects the live config.
"""
import os

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))          # tokens, max 1024
OVERLAP_RATIO = float(os.getenv("OVERLAP_RATIO", "0.15"))  # 0..0.3
TOP_K = int(os.getenv("TOP_K", "8"))                       # 1..30 chunks to the model

# After over-fetching, keep at most this many chunks per article so that
# "list 3 distinct articles" questions get article diversity, while fact /
# summary questions still get the strongest passages.
MAX_CHUNKS_PER_ARTICLE = int(os.getenv("MAX_CHUNKS_PER_ARTICLE", "2"))

# Models exactly as provided by the assignment gateway.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small")
CHAT_MODEL = os.getenv("CHAT_MODEL", "4UHRUIN-gpt-5-mini")

# text-embedding-3-small default dimensions = 1536 (must match the Pinecone index).
EMBEDDING_DIM = 1536

# OpenAI-compatible gateway.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None  # leave unset for real OpenAI

# Pinecone.
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX", "medium-rag")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")


def overlap_tokens() -> int:
    return int(CHUNK_SIZE * OVERLAP_RATIO)


def validate() -> None:
    if CHUNK_SIZE > 1024:
        raise ValueError("CHUNK_SIZE must be <= 1024 tokens")
    if not (0 <= OVERLAP_RATIO <= 0.3):
        raise ValueError("OVERLAP_RATIO must be between 0 and 0.3")
    if not (1 <= TOP_K <= 30):
        raise ValueError("TOP_K must be between 1 and 30")


validate()
