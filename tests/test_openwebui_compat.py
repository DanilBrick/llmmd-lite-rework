from __future__ import annotations

import asyncio

from rag_service import main as rag_main


def test_openai_content_to_text_supports_multimodal_text_blocks():
    content = [
        {"type": "text", "text": "first"},
        {"type": "input_text", "text": "second"},
        {"type": "image_url", "image_url": {"url": "ignored"}},
    ]

    assert rag_main._openai_content_to_text(content) == "first\nsecond"


def test_last_openai_user_text_prefers_last_user_message():
    messages = [
        rag_main.OpenAIChatMessage(role="system", content="system"),
        rag_main.OpenAIChatMessage(role="user", content="old"),
        rag_main.OpenAIChatMessage(role="assistant", content="answer"),
        rag_main.OpenAIChatMessage(role="user", content=[{"type": "text", "text": "new"}]),
    ]

    assert rag_main._last_openai_user_text(messages) == "new"


def test_openai_chat_completion_wraps_rag(monkeypatch):
    async def fake_rag(body: rag_main.RagRequest):
        assert body.query == "Что в корпусе?"
        assert body.topic == "Что в корпусе?"
        assert body.model is None
        return {
            "answer": "Ответ по корпусу [1].",
            "sources": [{"source_path": "book.md", "heading": "Глава"}],
        }

    monkeypatch.setattr(rag_main, "rag", fake_rag)

    body = rag_main.OpenAIChatCompletionRequest(
        model="llmmd-rag",
        messages=[rag_main.OpenAIChatMessage(role="user", content="Что в корпусе?")],
    )

    payload = asyncio.run(rag_main.openai_chat_completions(body))

    assert payload["object"] == "chat.completion"
    assert payload["model"] == "llmmd-rag"
    content = payload["choices"][0]["message"]["content"]
    assert "Ответ по корпусу [1]." in content
    assert "[1] book.md - Глава" in content
    assert payload["llmmd"]["sources"][0]["source_path"] == "book.md"
