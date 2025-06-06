"""
Optimization Module for Human-AI Cognition Framework

This module provides comprehensive performance optimization capabilities including:
- Batch processing optimization
- Memory efficiency improvements  
- Neural training speed optimization
- Parallel processing capabilities
- Tensor operation optimization
"""

from .performance_optimizer import (
    PerformanceOptimizer,
    PerformanceConfig,
    BatchProcessor,
    MemoryMonitor,
    EmbeddingCache,
    NeuralTrainingOptimizer,
    PerformanceTracker,
    ParallelProcessor,
    create_performance_optimizer,
    optimize_tensor_operations
)

__all__ = [
    'PerformanceOptimizer',
    'PerformanceConfig',
    'BatchProcessor',
    'MemoryMonitor',
    'EmbeddingCache',
    'NeuralTrainingOptimizer',
    'PerformanceTracker',
    'ParallelProcessor',
    'create_performance_optimizer',
    'optimize_tensor_operations'
]
