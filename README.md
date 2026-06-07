# Medium Article RAG Assistant

A RAG system built on ~7,600 English Medium articles. You ask a question, it finds the most relevant chunks from the corpus, and the chat model answers strictly from that retrieved context — no internet, no hallucinated facts.

Built with FastAPI, LangChain, Pinecone, and deployed to Vercel.

---

## How it works

```
question
  → embed with text-embedding-3-small (1536-d)
  → similarity search in Pinecone (over-fetch × 4, then cap per article)
  → build prompt: required system prompt + retrieved passages
  → gpt-5-mini answers only from context
  → { response, context, Augmented_prompt }
```

The retrieval over-fetches (`top_k × 4`) and then greedily caps how many chunks can come from the same article. This keeps the context diverse for "list 3 distinct articles" questions while still returning multiple strong passages for fact lookups and summaries.

---

## Project layout

```
api/
  index.py          FastAPI app — Vercel entrypoint
  ui.html           browser UI
  static/           CSS + JS for the UI
rag/
  config.py         all hyperparameters, read from env
  clients.py        lazy LangChain + Pinecone factories
  prompts.py        system prompt (verbatim required text) + user-prompt builder
  pipeline.py       retrieval + answer generation
scripts/
  ingest.py         one-time CSV → chunk → embed → Pinecone
  eval.py           smoke-tests all 4 question categories against a running URL
```

---

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
```

Create `.env.local` with these values:

| Variable | Value |
|---|---|
| `OPENAI_API_KEY` | your course API key |
| `OPENAI_BASE_URL` | gateway base URL (e.g. `https://.../v1`) |
| `EMBEDDING_MODEL` | `4UHRUIN-text-embedding-3-small` |
| `CHAT_MODEL` | `4UHRUIN-gpt-5-mini` |
| `PINECONE_API_KEY` | from your Pinecone console |
| `PINECONE_INDEX` | `medium-rag` (created automatically on first ingest) |
| `CHUNK_SIZE` | `512` |
| `OVERLAP_RATIO` | `0.15` |
| `TOP_K` | `12` |
| `MAX_CHUNKS_PER_ARTICLE` | `3` |

---

## Ingest

Run this **locally**, once. It embeds the corpus and upserts everything into Pinecone. You never need to re-run it unless you change `CHUNK_SIZE` or `OVERLAP_RATIO`.

```bash
# Start with a small slice to validate cheaply
python scripts/ingest.py data/medium-english-50mb.csv --limit 200

# Full corpus once you're happy
python scripts/ingest.py data/medium-english-50mb.csv
```

`--start N` lets you resume from a row offset if something interrupted mid-ingest. The chunker uses tiktoken (`cl100k_base`) so chunk sizes are real tokens, not characters.

`TOP_K` and `MAX_CHUNKS_PER_ARTICLE` are query-time parameters — you can tune them in `.env.local` and re-run `eval.py` without touching Pinecone.

---

## Run locally

```bash
uvicorn api.index:app --reload --port 8000
```

Open `http://localhost:8000` for the browser UI, or hit the endpoints directly:

```bash
# Smoke-test all 4 question types
BASE_URL=http://localhost:8000 python scripts/eval.py
```

---

## Deploy to Vercel

1. Push the repo to GitHub.
2. Import the project in Vercel (it picks up `requirements.txt` automatically).
3. Add all the env vars under **Settings → Environment Variables**.
4. Deploy.

`vercel.json` rewrites every incoming path to `api/index.py`, which FastAPI then routes internally. The Pinecone index needs to stay active until grading is done.

Live endpoints:
- `POST https://<your-app>.vercel.app/api/prompt`
- `GET  https://<your-app>.vercel.app/api/stats`

---

## API

### `POST /api/prompt`

**Request:**
```json
{ "question": "List exactly 3 articles about education. Return only the titles." }
```

**Response:**
```json
{
  "response": "Here are 3 articles about education: ...",
  "context": [
    {
      "article_id": "1234",
      "title": "Learning at Scale",
      "authors": "Jane Doe",
      "url": "https://medium.com/...",
      "tags": "education,learning",
      "chunk": "...retrieved passage...",
      "score": 0.8741
    }
  ],
  "Augmented_prompt": {
    "System": "...the system prompt sent to the model...",
    "User": "...the full user prompt with injected context..."
  }
}
```

Returns `400` if `question` is missing or empty, `500` if the pipeline fails.

### `GET /api/stats`

Returns the active RAG hyperparameters:
```json
{ "chunk_size": 512, "overlap_ratio": 0.15, "top_k": 12 }
```

---

## Hyperparameter choices

**Chunk size: 512 tokens.** Keeps a single idea or argument intact. Going larger risks mixing unrelated content in one chunk; going smaller risks cutting sentences mid-thought. 512 is the sweet spot for editorial-length Medium articles.

**Overlap: 0.15 (15%).** Enough to avoid hard breaks at chunk boundaries without paying to embed the same text twice. 30% would double the index size for marginal retrieval gain.

**Top-k: 12.** With a cap of 3 chunks per article, 12 guarantees at least 4 distinct articles reach the model — enough headroom for "list exactly 3" questions. Well under the 30 cap, so the prompt doesn't balloon.

**Max chunks per article: 3.** The retriever over-fetches `top_k × 4 = 48` candidates, then greedily selects up to 3 from each article. This gives fact and summary questions several strong passages from the same source while forcing multi-listing questions to pull from different articles.

---

## Budget notes

`text-embedding-3-small` runs at ~$0.02 / 1M tokens. A full ingest of the ~7,600-article corpus costs a few cents. Testing with `--limit 200` first is effectively free. `gpt-5-mini` query costs are negligible. Well within the $5 budget if you avoid re-ingesting the whole corpus every time you tune a parameter.
