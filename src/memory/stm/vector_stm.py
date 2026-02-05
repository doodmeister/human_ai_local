"""Compatibility shim for the refactored Vector STM implementation.

Historically, callers imported symbols from `src.memory.stm.vector_stm`.
The STM implementation was refactored into `vector_stm_refactored.py`.

This module preserves the old import path by re-exporting the public API.
"""

from .vector_stm_refactored import (
    MemoryItem,
    STMConfiguration,
    VectorMemoryResult,
    VectorShortTermMemory,
    VectorSTMConfigError,
    VectorSTMError,
    VectorSTMRetrievalError,
    VectorSTMStorageError,
)

__all__ = [
    "VectorShortTermMemory",
    "VectorMemoryResult",
    "MemoryItem",
    "STMConfiguration",
    "VectorSTMError",
    "VectorSTMConfigError",
    "VectorSTMStorageError",
    "VectorSTMRetrievalError",
]
