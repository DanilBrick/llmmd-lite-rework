"""Lite stdio MCP server: search and RAG over the running llmmd RAG API."""

from __future__ import annotations

import os

import httpx
from mcp.server.fastmcp import FastMCP

RAG_BASE = os.environ.get("LLMMD_RAG_BASE_URL", "http://127.0.0.1:8765").rstrip("/")

mcp = FastMCP("llmmd-rag")


def _post(path: str, payload: dict) -> dict:
    with httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0)) as client:
        response = client.post(f"{RAG_BASE}{path}", json=payload)
        response.raise_for_status()
        data = response.json()
        return data if isinstance(data, dict) else {"result": data}


@mcp.tool()
def rag_search(query: str, limit: int = 6) -> str:
    """Semantic/hybrid search in the indexed Qdrant corpus."""
    data = _post("/v1/search", {"query": query, "limit": max(1, min(limit, 32))})
    hits = data.get("hits") or []
    if not hits:
        return "No hits."
    lines = []
    for i, hit in enumerate(hits, start=1):
        text = (hit.get("text") or hit.get("payload", {}).get("text") or "")[:1200]
        score = hit.get("score")
        source = hit.get("source_path") or hit.get("metadata", {}).get("source_file") or "?"
        suffix = f" (score={score:.3f})" if isinstance(score, (int, float)) else ""
        lines.append(f"[{i}] {source}{suffix}\n{text}")
    return "\n\n".join(lines)


@mcp.tool()
def rag_ask(query: str, limit: int = 6) -> str:
    """Ask a question; RAG API searches the corpus and generates an answer."""
    data = _post("/v1/rag", {"query": query, "limit": max(1, min(limit, 32))})
    answer = data.get("answer") or ""
    sources = data.get("context") or data.get("sources") or []
    if not sources:
        return answer
    refs = []
    for i, src in enumerate(sources, start=1):
        path = src.get("source_path") or src.get("metadata", {}).get("source_file") or "?"
        refs.append(f"[{i}] {path}")
    return f"{answer}\n\nSources:\n" + "\n".join(refs)


if __name__ == "__main__":
    mcp.run(transport="stdio")
