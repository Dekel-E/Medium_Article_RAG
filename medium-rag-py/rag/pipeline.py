"""The RAG pipeline: retrieve relevant chunks (with per-article diversity) and
generate a grounded answer with the chat model via LangChain.
"""
from typing import List, Dict

from langchain_core.messages import HumanMessage, SystemMessage

from . import config
from .clients import get_vectorstore, get_llm
from .prompts import SYSTEM_PROMPT, build_user_prompt

_FALLBACK = "I don't know based on the provided Medium articles data."


def retrieve(question: str) -> List[Dict]:
    """Embed the question, similarity-search Pinecone, then greedily keep chunks
    while capping per-article chunks. Returns exactly up to TOP_K chunks ordered
    by similarity."""
    vs = get_vectorstore()
    fetch_k = min(config.TOP_K * 4, 100)
    results = vs.similarity_search_with_score(question, k=fetch_k)  # [(Document, score)]

    per_article: Dict[str, int] = {}
    selected: List[Dict] = []

    for doc, score in results:
        md = doc.metadata or {}
        article_id = str(md.get("article_id", ""))
        if per_article.get(article_id, 0) >= config.MAX_CHUNKS_PER_ARTICLE:
            continue
        per_article[article_id] = per_article.get(article_id, 0) + 1

        selected.append(
            {
                "article_id": article_id,
                "title": md.get("title", ""),
                "chunk": doc.page_content,
                "score": float(score),
                "url": md.get("url", ""),
                "authors": md.get("authors", ""),
                "tags": md.get("tags", ""),
            }
        )
        if len(selected) >= config.TOP_K:
            break

    return selected


def answer(question: str) -> Dict:
    """Full RAG turn -> the exact response shape the assignment requires."""
    items = retrieve(question)
    user_prompt = build_user_prompt(question, items)

    llm = get_llm()
    resp = llm.invoke(
        [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=user_prompt)]
    )
    text = (resp.content or "").strip() or _FALLBACK

    return {
        "response": text,
        "context": [
            {
                "article_id": c["article_id"],
                "title": c["title"],
                "authors": c["authors"],
                "url": c["url"],
                "tags": c["tags"],
                "chunk": c["chunk"],
                "score": round(c["score"], 4),
            }
            for c in items
        ],
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
    }
