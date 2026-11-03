"""Compatibility exports for the RAG-side embedding client."""

from app.rag.embedding import embed_text, embed_texts

__all__ = ["embed_text", "embed_texts"]