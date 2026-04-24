# src/generation/llm_client.py
# Author: [Your Name] | Index: [Your Index Number]
# Handles LLM calls via the OpenAI-compatible client (OpenAI, Groq, etc.).
# Supports both RAG mode (with context) and pure LLM mode (no retrieval).

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError(
                "Set OPENAI_API_KEY or GROQ_API_KEY in .env (see README for Groq setup)."
            )
        base_url = (os.getenv("OPENAI_BASE_URL") or "").strip() or None
        kw = {"api_key": api_key}
        if base_url:
            kw["base_url"] = base_url.rstrip("/")
        _client = OpenAI(**kw)
    return _client


def generate_response(prompt: str, model: str | None = None, max_tokens: int = 512) -> str:
    """
    Send a prompt to the chat completion endpoint and return the response text.
    This is used in RAG mode — the prompt already contains retrieved context.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    client = _get_client()

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.2,   # Low temperature for factual, consistent answers
    )
    return response.choices[0].message.content.strip()


def generate_pure_llm(query: str, model: str | None = None, max_tokens: int = 512) -> str:
    """
    Generate a response WITHOUT any retrieved context.
    Used for RAG vs pure LLM evaluation comparison.
    """
    model = model or os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
    client = _get_client()

    system_msg = (
        "You are a knowledgeable assistant. Answer the user's question "
        "as accurately as possible using your own knowledge."
    )
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": query}
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()
