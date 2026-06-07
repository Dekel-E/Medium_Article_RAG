# Medium Article RAG Assistant (Python + LangChain)

A Retrieval-Augmented Generation system that answers questions **only** from a
dataset of ~7,600 English Medium articles. Built with **FastAPI + LangChain +
Pinecone**, deployable to **Vercel** as a Python serverless function.

Supports the four required question categories: precise fact retrieval,
multi-result topic listing (up to 3 distinct articles), key-idea summary
extraction, and recommendation-with-justification.

---

## How it works

```
question ─▶ OpenAIEmbeddings (text-embedding-3-small, 1536-d via LangChain)
        ─▶ PineconeVectorStore similarity search (over-fetch + per-article cap)
        ─▶ build augmented prompt (required system prompt + retrieved passages)
        ─▶ ChatOpenAI (gpt-5-mini) answers strictly from context
        ─▶ { response, context, Augmented_prompt }
```

Ingestion is a **separate, one-time, local script**, so you never re-embed the
corpus on each deploy or parameter tweak.

```
api/index.py     FastAPI app (Vercel entrypoint): /api/prompt, /api/stats, /
rag/config.py    hyperparameters + model names (from env)
rag/clients.py   lazy LangChain embeddings / chat / Pinecone vector store
rag/prompts.py   required system prompt (verbatim) + user-prompt builder
rag/pipeline.py  diversity retrieval + grounded answer generation
scripts/ingest.py  CSV -> token chunks -> embed -> Pinecone
scripts/eval.py    runs the 4 example questions against a running URL
```

---

## 1. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt      # includes uvicorn for local dev
cp .env.example .env.local               # fill in the values
```

Fill `.env.local`:

| Variable | What to put |
|---|---|
| `OPENAI_API_KEY` | the key the course gave you |
| `OPENAI_BASE_URL` | gateway base URL (e.g. `https://.../v1`); blank only for real OpenAI |
| `EMBEDDING_MODEL` | `4UHRUIN-text-embedding-3-small` (default) |
| `CHAT_MODEL` | `4UHRUIN-gpt-5-mini` (default) |
| `PINECONE_API_KEY` | from the Pinecone console |
| `PINECONE_INDEX` | `medium-rag` (default; created automatically) |
| `CHUNK_SIZE` / `OVERLAP_RATIO` / `TOP_K` | `512` / `0.15` / `8` |
| `MAX_CHUNKS_PER_ARTICLE` | `2` |

The index is created with dimension **1536** to match `text-embedding-3-small`.

## 2. Ingest the data (run locally, once)

```bash
# Start SMALL to validate cheaply:
python scripts/ingest.py data/medium.csv --limit 200

# Then ingest the full corpus:
python scripts/ingest.py data/medium.csv
```

`--start N` lets you resume / ingest a slice. Chunks are embedded 100 at a time
and upserted to Pinecone in batches.

## 3. Run locally

```bash
uvicorn api.index:app --reload --port 8000
# open http://localhost:8000   (status page; docs at /docs)

BASE_URL=http://localhost:8000 python scripts/eval.py   # smoke-test the 4 Qs
```

## 4. Deploy to Vercel

1. Push to GitHub.
2. Import in Vercel → **New Project** (it auto-detects Python via `requirements.txt`).
3. Add the same env vars under **Settings → Environment Variables**.
4. Deploy. Endpoints:
   - `POST https://<your-app>.vercel.app/api/prompt`
   - `GET  https://<your-app>.vercel.app/api/stats`

`vercel.json` rewrites every path to the single FastAPI function, which routes
on the original path. Keep the Pinecone index active until grading is done.

---

## API

### `POST /api/prompt`
```json
{ "question": "List exactly 3 articles about education. Return only the titles." }
```
Response:
```json
{
  "response": "Final natural language answer from gpt-5-mini.",
  "context": [
    { "article_id": "1234", "title": "Sample title", "chunk": "retrieved chunk", "score": 0.8123 }
  ],
  "Augmented_prompt": {
    "System": "the system prompt used to query the chat model",
    "User": "the user prompt used to query the chat model"
  }
}
```

### `GET /api/stats`
```json
{ "chunk_size": 512, "overlap_ratio": 0.15, "top_k": 8 }
```

---

## Hyperparameter choices (and why)

| Param | Value | Rationale |
|---|---|---|
| `chunk_size` | **512 tokens** | Keeps an idea intact while staying mostly on-topic; under the 1024 cap. |
| `overlap_ratio` | **0.15** | Avoids splitting sentences across boundaries without paying to embed the same text twice; under the 0.3 cap. |
| `top_k` | **8** | Enough breadth for "list 3 distinct articles" and for grounding summaries, without bloating context. |
| `max_chunks_per_article` | **2** | Retrieval over-fetches `top_k × 4`, then caps chunks per article so multi-result questions get **distinct** articles while fact/summary questions still get the strongest passages. |

**Compare settings cheaply (the assignment asks for this):** ingest a small
subset once (`--limit 200`), then change only `TOP_K` / `MAX_CHUNKS_PER_ARTICLE`
in `.env.local` and re-run `eval.py` — these are query-time params needing **no**
re-embedding. Only `CHUNK_SIZE` / `OVERLAP_RATIO` require re-ingesting, so test
those last on the small subset before a full ingest.

## Budget notes

`text-embedding-3-small` is ~$0.02 / 1M tokens; a full ingest of the corpus costs
only a few cents, and gpt-5-mini queries are tiny. Starting with `--limit` keeps
early experiments effectively free — well within the $5 budget.

## Notes / gotchas

- **tiktoken** is used only at **ingest** time (for token-accurate chunking) and
  downloads its vocab on first run, so run ingest where there's internet. The
  serving path does **not** tokenize (`check_embedding_ctx_length=False`), so the
  Vercel function never needs that download.
- `ChatOpenAI` is created with `temperature=1` because gpt-5-mini style models
  only accept the default temperature. If your gateway rejects the parameter
  entirely, remove it in `rag/clients.py`.
- The required system prompt is sent **verbatim**; the model is told to answer
  only from context and to reply *"I don't know based on the provided Medium
  articles data."* when context is insufficient.
