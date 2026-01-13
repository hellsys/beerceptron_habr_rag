from langchain_core.tools import tool

import logging
from typing import Any, Dict

from app.services.llm import provider
from app.services.retriever import context_builder, qdrant_client
from app.core.prompts import RAG_PROMPT

log = logging.getLogger("app")

@tool
def run_rag(question: str) -> str:
    """Инструмент поиска для RAG, чтобы найти релевантные фрагменты документации Хабра"""
    collection_name = "habr_rag_bge_m3_test_2025_12_05"
    top_k = 7

    chunks = qdrant_client.get_chunks(
        text=question,
        collection_name=collection_name,
        top_k=top_k
    )
    context = context_builder.build_context(chunks)
    # prompt = RAG_PROMPT.replace("{context}", context).replace("{question}", question)
    # answer = provider.generate(prompt, temperature=0.7)
    return context

