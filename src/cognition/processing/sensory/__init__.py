"""
Sensory Processing Module

This module provides biologically-inspired sensory input processing for the
Human-AI Cognition Framework. It implements:

- Multimodal input preprocessing and filtering
- Embedding generation and vector conversion
- Information entropy scoring and salience detection
- Adaptive filtering and attention-based preprocessing

Key Components:
- SensoryProcessor: Core processing engine
- SensoryInput: Input data structure
- ProcessedSensoryData: Output data structure
- SensoryInterface: High-level interface utilities
"""

from .sensory_processor import (
    SensoryProcessor,
    SensoryInput,
    ProcessedSensoryData
)

from .sensory_interface import (
    SensoryInterface,
    SensoryInputBuilder,
    quick_text_input,
    quick_process_text
)

__all__ = [
    "SensoryProcessor",
    "SensoryInput", 
    "ProcessedSensoryData",
    "SensoryInterface",
    "SensoryInputBuilder",
    "quick_text_input",
    "quick_process_text"
]
