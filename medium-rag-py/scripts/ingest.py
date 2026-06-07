"""Ingest the Medium CSV into Pinecone using LangChain.

Run locally (NOT on Vercel). Embeds once; query-time params need no re-ingest.

Usage:
    python scripts/ingest.py <path-to-csv> [--limit N] [--start N]

Cost control:
    python scripts/ingest.py data/medium.csv --limit 200   # validate cheaply
    python scripts/ingest.py data/medium.csv               # full corpus

text-embedding-3-small is ~$0.02 / 1M tokens, so a full ingest is a few cents.
"""
import argparse
import csv
import sys
import time

from dotenv import load_dotenv

load_dotenv(".env.local")
load_dotenv()  # fall back to .env

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag import config
from rag.clients import ensure_index, get_vectorstore

csv.field_size_limit(min(sys.maxsize, 2_000_000_000))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--start", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.time()
    print(f"Config: chunk_size={config.CHUNK_SIZE}, overlap_ratio={config.OVERLAP_RATIO}")

    # Token-based splitter: chunk_size/overlap are real cl100k tokens (the
    # encoding text-embedding-3-* uses), so /api/stats reports true tokens.
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.overlap_tokens(),
    )

    print(f"Reading CSV: {args.csv_path}")
    with open(args.csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Parsed {len(rows)} rows.")

    end = len(rows) if args.limit is None else args.start + args.limit
    rows_slice = rows[args.start : end]

    documents, ids = [], []
    for local_idx, row in enumerate(rows_slice):
        article_id = str(args.start + local_idx)
        text = (row.get("text") or "").strip()
        if not text:
            continue
        for ci, piece in enumerate(splitter.split_text(text)):
            documents.append(
                Document(
                    page_content=piece,
                    metadata={
                        "article_id": article_id,
                        "title": row.get("title") or "",
                        "url": row.get("url") or "",
                        "authors": row.get("authors") or "",
                        "timestamp": row.get("timestamp") or "",
                        "tags": row.get("tags") or "",
                        "chunk_index": ci,
                    },
                )
            )
            ids.append(f"{article_id}-{ci}")

    print(f"Prepared {len(documents)} chunks from {len(rows_slice)} articles.")

    ensure_index()
    vs = get_vectorstore()

    BATCH = 100
    done = 0
    for i in range(0, len(documents), BATCH):
        vs.add_documents(documents[i : i + BATCH], ids=ids[i : i + BATCH])
        done += len(documents[i : i + BATCH])
        print(f"\rEmbedded + upserted {done}/{len(documents)}", end="", flush=True)

    print(f"\nDone. {done} chunks indexed in {time.time() - t0:.1f}s.")


if __name__ == "__main__":
    main()
