"""Smoke-test the 4 assignment question categories against a running API.

Usage:
    BASE_URL=http://localhost:8000 python scripts/eval.py
    BASE_URL=https://your-app.vercel.app python scripts/eval.py
"""
import json
import os
import time
import urllib.request

BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")

QUESTIONS = [
    # 1. Precise fact retrieval
    "Find an article that reframes marketing as a conversation with readers, "
    "aimed at writers who find self-promotion uncomfortable. Provide the title "
    "and author.",
    # 2. Multi-result topic listing
    "List exactly 3 articles about education. Return only the titles.",
    # 3. Key idea summary extraction
    "Find an article that argues past pandemics (such as the bubonic plague) "
    "can spur innovation and recovery, and summarise its central argument.",
    # 4. Recommendation with justification
    "I want practical, beginner-friendly advice on building habits that actually "
    "stick. Which article would you recommend, and why?",
]


def get_json(url):
    with urllib.request.urlopen(url) as r:
        return json.load(r)


def post_json(url, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.load(r)


def main():
    print(f"Testing {BASE_URL}\n")
    print("GET /api/stats ->", get_json(f"{BASE_URL}/api/stats"), "\n")

    for q in QUESTIONS:
        print("Q:", q)
        t0 = time.perf_counter()
        out = post_json(f"{BASE_URL}/api/prompt", {"question": q})
        elapsed = time.perf_counter() - t0
        print(f"A ({elapsed:.2f}s):", out.get("response"))
        ctx = out.get("context", [])
        print("Retrieved:", " | ".join(f"{c['title']} ({c['score']})" for c in ctx))
        print("-" * 80)


if __name__ == "__main__":
    main()
