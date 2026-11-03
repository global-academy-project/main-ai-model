"""Compatibility exports for the RAG search backend."""

from app.rag.tool_backend import BACKEND_ERRORS, build_search_backend

__all__ = ["BACKEND_ERRORS", "build_search_backend"]