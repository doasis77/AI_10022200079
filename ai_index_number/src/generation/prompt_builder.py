# src/generation/prompt_builder.py
# Author: [Your Name] | Index: [Your Index Number]
#
# Three prompt versions are implemented:
#
# V1 - Basic: Simple context injection, no guardrails
# V2 - Hallucination-controlled: Explicit grounding instruction, fallback phrase
# V3 - Structured final: Chunk IDs, strict grounding, structured output request
#
# Prompt version comparison allows evaluation of how template design
# affects accuracy, hallucination rate, and response quality.

from typing import List, Dict


# Maximum words of context to include (context window management)
MAX_CONTEXT_WORDS = 1800


def _build_context_block(chunks_with_scores: List[tuple], max_words: int = MAX_CONTEXT_WORDS) -> str:
    """
    Build a context string from retrieved chunks.
    Truncates total context to max_words to stay within LLM context limits.
    Deduplicates identical chunk texts.
    """
    seen_texts = set()
    context_parts = []
    total_words = 0

    for chunk, scores in chunks_with_scores:
        text = chunk["text"].strip()
        if text in seen_texts:
            continue
        seen_texts.add(text)
        words = text.split()
        if total_words + len(words) > max_words:
            # Truncate this chunk to fit
            remaining = max_words - total_words
            if remaining > 50:
                text = " ".join(words[:remaining]) + "..."
                context_parts.append(
                    f"[{chunk['chunk_id']} | source: {chunk['source']}]\n{text}"
                )
            break
        context_parts.append(
            f"[{chunk['chunk_id']} | source: {chunk['source']}]\n{text}"
        )
        total_words += len(words)

    return "\n\n".join(context_parts)


def build_prompt_v1(query: str, chunks_with_scores: List[tuple]) -> str:
    """
    V1 - Basic prompt. Minimal instruction, raw context injection.
    Risk: model may hallucinate if context is weak.
    """
    context = _build_context_block(chunks_with_scores)
    return f"""You are a helpful assistant. Use the context below to answer the question.

Context:
{context}

Question: {query}

Answer:"""


def build_prompt_v2(query: str, chunks_with_scores: List[tuple]) -> str:
    """
    V2 - Hallucination-controlled. Explicit instruction to stay grounded.
    Model is told to say "I do not have enough information" if context is insufficient.
    """
    context = _build_context_block(chunks_with_scores)
    return f"""You are a knowledgeable assistant specializing in Ghana's elections and government budgets.
Answer ONLY using the provided context. Do not use outside knowledge.
If the answer is not clearly supported by the context, respond with:
"I do not have enough information from the provided documents to answer this question."

Context:
{context}

Question: {query}

Answer:"""


def build_prompt_v3(query: str, chunks_with_scores: List[tuple]) -> str:
    """
    V3 - Structured final prompt. Chunk IDs are cited, strict grounding enforced,
    structured output requested. This is the production prompt.
    """
    context = _build_context_block(chunks_with_scores)

    # List chunk IDs for citation reference
    chunk_ids = [c["chunk_id"] for c, _ in chunks_with_scores]
    ids_str = ", ".join(chunk_ids)

    return f"""You are an AI assistant for Academic City. Your role is to answer questions about Ghana's \
election results and the 2025 Ghana Budget Statement strictly using the provided context chunks.

Retrieved Context Chunks (IDs: {ids_str}):
{context}

Instructions:
- Base your answer ONLY on the context above. Do not use prior knowledge.
- Be concise and factual. Prefer specific numbers, names, and dates where present.
- If multiple chunks are relevant, synthesize them clearly.
- If the context does not contain enough information to answer, say exactly:
  "I do not have enough information from the provided documents to answer this question."
- Do NOT speculate or make assumptions.

Question: {query}

Answer:"""


PROMPT_BUILDERS = {
    "v1": build_prompt_v1,
    "v2": build_prompt_v2,
    "v3": build_prompt_v3,
}


def build_prompt(query: str, chunks_with_scores: List[tuple], version: str = "v3") -> str:
    """Build a prompt using the specified version (v1, v2, v3)."""
    builder = PROMPT_BUILDERS.get(version, build_prompt_v3)
    return builder(query, chunks_with_scores)
