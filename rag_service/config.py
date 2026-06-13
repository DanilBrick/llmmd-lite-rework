from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="RAG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = Field(default="http://localhost:6333", description="URL Qdrant")
    qdrant_api_key: str | None = Field(default=None)
    collection_name: str = Field(default="llmmd_corpus")

    corpus_root: Path | None = Field(
        default=None,
        description="Корень корпуса по умолчанию (например путь к outputs MarkItDown)",
    )

    embedding_model: str = Field(default="intfloat/multilingual-e5-large")
    embedding_device: str | None = Field(
        default=None,
        description="cuda / cpu; None — авто",
    )
    enable_hybrid: bool = Field(
        default=True,
        description="Sparse BM25 (fastembed) + dense E5; при ошибке — только dense",
    )

    heading_level: int = Field(
        default=2,
        ge=1,
        le=6,
        description="Уровень заголовка для границ секций: 1=H1, 2=H2 (как «## (H2)» в GUI), …",
    )
    chunk_max_chars: int = Field(
        default=0,
        ge=0,
        description="0 — только по заголовкам; >0 — донарезка длинных секций по символам (перекрытие chunk_overlap_chars)",
    )
    chunk_overlap_chars: int = Field(default=400, ge=0)

    rag_context_max_chars: int = Field(
        default=12000,
        ge=1000,
        le=200000,
        description="Максимальный суммарный размер контекста, передаваемого в LLM для /v1/rag",
    )
    rag_source_max_chars: int = Field(
        default=4000,
        ge=500,
        le=100000,
        description="Максимальный размер одного найденного источника внутри RAG-контекста",
    )
    rag_dedupe_sources: bool = Field(
        default=True,
        description="Убирать точные дубли источников перед сборкой RAG-контекста",
    )

    chunking_mode: str = Field(
        default="heading",
        description="heading | semantic | heading_semantic",
    )

    lm_studio_base_url: str = Field(
        default="http://127.0.0.1:1234/v1",
        description="OpenAI-совместимый API LM Studio (вкладка Developer → Local Server)",
    )
    lm_studio_api_key: str | None = Field(
        default=None,
        description="Обычно не нужен; если в LM Studio включили ключ",
    )
    semantic_chunk_model: str = Field(
        default="",
        description="Идентификатор загруженной модели для чанкинга (как в запросе chat/completions)",
    )
    semantic_llm_timeout_s: float = Field(default=180.0, ge=10.0, le=3600.0)
    semantic_chunk_max_input_chars: int = Field(
        default=20000,
        ge=4000,
        le=120000,
        description="Максимум символов на один запрос к LM (окно при длинном тексте)",
    )
    semantic_subchunk_min_chars: int = Field(
        default=3500,
        ge=800,
        le=200000,
        description="В режиме heading_semantic: секции длиннее — донарезка через LM",
    )
    semantic_chunk_temperature: float = Field(default=0.05, ge=0.0, le=1.0)

    api_host: str = Field(default="127.0.0.1")
    api_port: int = Field(default=8765, ge=1, le=65535)

    openai_base_url: str | None = Field(default=None)
    openai_api_key: str | None = Field(default=None)
    default_llm_model: str = Field(default="gpt-4o-mini")

    # Генерация ответа RAG: Claude (отдельно от OpenAI-совместимого облака)
    anthropic_api_key: str | None = Field(default=None)
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Идентификатор модели в API Anthropic",
    )
    anthropic_max_tokens: int = Field(default=4096, ge=256, le=32000)

    # Имя модели в LM Studio для POST /v1/rag (если не передано в теле запроса)
    lm_studio_rag_model: str = Field(
        default="",
        description="Пусто — брать RAG_SEMANTIC_CHUNK_MODEL или поле model в запросе",
    )

    default_rag_llm_provider: str = Field(
        default="auto",
        description="auto | lm_studio | openai | anthropic — кто отвечает на POST /v1/rag, если в теле не указан llm_provider",
    )

    ui_settings_file: Path | None = Field(
        default=None,
        description="Путь к JSON с настройками из UI; None — rag_service/ui_settings.json",
    )
