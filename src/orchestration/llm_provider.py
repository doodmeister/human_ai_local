"""Orchestration-layer access to model providers.

Phase 6: interfaces must depend on orchestration only.
"""

from __future__ import annotations

from src.model.llm_provider import LLMProviderFactory

__all__ = ["LLMProviderFactory"]
