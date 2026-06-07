"""Lazy LangChain + Pinecone client factories.

Everything is built on first use so importing this module (e.g. during a Vercel
build) never requires credentials and never opens a network connection.
"""
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

from . import config


@lru_cache(maxsize=1)
def get_embeddings() -> OpenAIEmbeddings:
    # check_embedding_ctx_length=False: our chunks are already <= CHUNK_SIZE
    # tokens, so we skip LangChain's tiktoken re-batching (which would also
    # choke on the non-standard "4UHRUIN-" model name). We send the strings as-is.
    return OpenAIEmbeddings(
        model=config.EMBEDDING_MODEL,
        dimensions=config.EMBEDDING_DIM,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
        check_embedding_ctx_length=False,
        chunk_size=100,  # embed up to 100 texts per request
    )


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    # temperature=1 is the only value gpt-5-mini style models accept; setting it
    # explicitly avoids LangChain sending a default the gateway would reject.
    return ChatOpenAI(
        model=config.CHAT_MODEL,
        temperature=1,
        api_key=config.OPENAI_API_KEY,
        base_url=config.OPENAI_BASE_URL,
    )


@lru_cache(maxsize=1)
def get_pinecone():
    from pinecone import Pinecone

    return Pinecone(api_key=config.PINECONE_API_KEY)


@lru_cache(maxsize=1)
def get_index():
    return get_pinecone().Index(config.PINECONE_INDEX)


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    return PineconeVectorStore(index=get_index(), embedding=get_embeddings())


def ensure_index() -> None:
    """Create the serverless index (dim=1536, cosine) if it does not exist.
    Used by the ingestion script."""
    from pinecone import ServerlessSpec

    pc = get_pinecone()
    existing = {i["name"] for i in pc.list_indexes()}
    if config.PINECONE_INDEX in existing:
        print(f'Index "{config.PINECONE_INDEX}" already exists.')
        return
    print(f'Creating index "{config.PINECONE_INDEX}" (dim={config.EMBEDDING_DIM})...')
    pc.create_index(
        name=config.PINECONE_INDEX,
        dimension=config.EMBEDDING_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION),
    )
    # Wait until ready.
    import time

    while not pc.describe_index(config.PINECONE_INDEX).status["ready"]:
        time.sleep(1)
    print("Index ready.")
