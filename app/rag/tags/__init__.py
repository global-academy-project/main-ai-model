"""Deterministic tag extraction for RAG and structured safety data."""

from .regulatory import classify_regulations

__all__ = ["classify_regulations"]