"""Prompt construction. The constrained system-prompt text is REQUIRED by the
assignment and kept verbatim; the response-style lines are permitted additions.
"""
from typing import List, Dict

SYSTEM_PROMPT = (
    "You are a Medium-article assistant that answers questions strictly and only "
    "based on the Medium articles dataset context provided to you (metadata and "
    "article passages). You must not use any external knowledge, the open "
    "internet, or information that is not explicitly contained in the retrieved "
    'context. If the answer cannot be determined from the provided context, '
    'respond: "I don\'t know based on the provided Medium articles data." Always '
    "explain your answer using the given context, quoting or paraphrasing the "
    "relevant article passage or metadata when helpful.\n\n"
    "Response style:\n"
    "- Be concise and directly answer what was asked.\n"
    "- When the user asks for a title and/or author, state them explicitly.\n"
    '- When the user asks for "exactly N" articles, return exactly N DISTINCT '
    "articles (different titles), never multiple passages of the same article.\n"
    "- Do not invent titles, authors, or facts that are not present in the context."
)


def build_user_prompt(question: str, items: List[Dict]) -> str:
    blocks = []
    for i, c in enumerate(items, start=1):
        meta_parts = [f"Title: {c.get('title', '')}"]
        if c.get("authors"):
            meta_parts.append(f"Author(s): {c['authors']}")
        if c.get("tags"):
            meta_parts.append(f"Tags: {c['tags']}")
        if c.get("url"):
            meta_parts.append(f"URL: {c['url']}")
        meta_parts.append(f"article_id: {c.get('article_id')}")
        meta = " | ".join(meta_parts)
        blocks.append(f"[#{i}] {meta}\nPassage: {c.get('chunk', '')}")

    context = "\n\n".join(blocks)
    return (
        "Use ONLY the following retrieved Medium article context to answer the "
        "question. Each item shows article metadata and a passage from that "
        "article.\n\n"
        "--- RETRIEVED CONTEXT START ---\n"
        f"{context}\n"
        "--- RETRIEVED CONTEXT END ---\n\n"
        f"Question: {question}"
    )
