"""
Chat Handler
-------------
Handles:
  1. Running the RAG pipeline for a user query
  2. Extracting only the final answer + citations
  3. Returning clean response to Streamlit UI
  4. All internal logs suppressed — UI only sees final answer
"""

import sys
import os
import time
import re


def run_rag_pipeline(question: str, rag_app) -> dict:
    """
    Runs the full LangGraph RAG pipeline silently.
    Suppresses all internal node logs from appearing on UI.

    Args:
        question : Raw user question
        rag_app  : Compiled LangGraph app (built once, reused)

    Returns:
        dict with keys: answer, domain, citations, validation
    """
    from rag.state import RAGState

    # ── Initialize State ─────────────────────────────────────────────
    initial_state: RAGState = {
        "question": question,
        "cleaned_question": "",
        "domain": "",
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "answer": "",
        "validation_result": "",
        "retry_count": 0,
        "guardrail_triggered": False,
        "output_flagged": False,
    }

    # ── Suppress all stdout during pipeline run ──────────────────────
    old_stdout = sys.stdout
    # sys.stdout = open(os.devnull, "w")
    sys.stdout = open(os.devnull, "w", encoding="utf-8")

    try:
        final_state = rag_app.invoke(initial_state)
        sys.stdout = old_stdout
    except Exception as e:
        sys.stdout = old_stdout
        return {
            "answer": "An error occurred while processing your question. Please try again.",
            "domain": "N/A",
            "citations": [],
            "error": str(e),
        }

    # ── Extract answer ───────────────────────────────────────────────
    answer = final_state.get("answer", "No answer generated.")
    domain = final_state.get("domain", "N/A")

    # ── Extract citations from answer ────────────────────────────────
    citations = _extract_citations(answer)

    return {
        "answer": answer,
        "domain": domain,
        "citations": citations,
        "validation": final_state.get("validation_result", "N/A"),
    }


def _extract_citations(answer: str) -> list:
    """
    Extracts source citations from the answer text.
    Looks for patterns like: (Source: filename.pdf) or (Document: filename.pdf)

    Args:
        answer : Raw answer string from summarizer

    Returns:
        List of unique citation strings
    """
    patterns = [
        r'\(Source:\s*([^)]+)\)',
        r'\(Document:\s*([^)]+)\)',
        r'Source:\s*([^\n]+)',
    ]

    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, answer, re.IGNORECASE)
        citations.extend([m.strip() for m in matches])

    # Deduplicate while preserving order
    seen = set()
    unique_citations = []
    for c in citations:
        if c not in seen:
            seen.add(c)
            unique_citations.append(c)

    return unique_citations


def stream_answer(answer: str):
    """
    Generator that yields answer word by word for streaming effect.

    Args:
        answer : Full answer string

    Yields:
        One word at a time with small delay
    """
    words = answer.split(" ")
    for i, word in enumerate(words):
        yield word + (" " if i < len(words) - 1 else "")
        time.sleep(0.025)
