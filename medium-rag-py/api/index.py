"""Vercel serverless entrypoint. All requests are rewritten to this file by
vercel.json, and FastAPI routes on the original path (/api/prompt, /api/stats).
"""
import os
import sys

# Make the project root importable so `rag` resolves on Vercel.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(_ROOT, ".env.local"))
load_dotenv(os.path.join(_ROOT, ".env"))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from rag import config
from rag.pipeline import answer

app = FastAPI(title="Medium Article RAG Assistant")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

_UI_HTML = open(os.path.join(os.path.dirname(__file__), "ui.html"), encoding="utf-8").read()


class Query(BaseModel):
    question: str


@app.post("/api/prompt")
async def prompt(body: Query):
    question = (body.question or "").strip()
    if not question:
        return JSONResponse(
            {"error": "Body must include a non-empty 'question' string."},
            status_code=400,
        )
    try:
        return JSONResponse(answer(question))
    except Exception as e:  # noqa: BLE001
        return JSONResponse(
            {"error": "Failed to process question", "detail": str(e)},
            status_code=500,
        )


@app.get("/api/stats")
async def stats():
    # Field names must match the assignment spec exactly.
    return {
        "chunk_size": config.CHUNK_SIZE,
        "overlap_ratio": config.OVERLAP_RATIO,
        "top_k": config.TOP_K,
    }


@app.get("/")
async def root():
    return HTMLResponse(_UI_HTML)
