"""Attention-specific exceptions."""


class AttentionError(Exception):
    """Base exception for attention mechanism errors."""


class CapacityExceededError(AttentionError):
    """Raised when attention capacity is exceeded."""


class InvalidStimulus(AttentionError):
    """Raised when stimulus parameters are invalid."""