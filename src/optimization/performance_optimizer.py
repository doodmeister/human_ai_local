"""
Performance Optimization Framework for Human-AI Cognition System

This module implements comprehensive performance optimizations including:
- Batch processing optimization for neural networks
- Memory efficiency improvements
- Neural training speed optimization
- Throughput improvements for sensory-cognitive integration
"""

import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import gc
import psutil
import asyncio

logger = logging.getLogger(__name__)

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    # Batch processing
    enable_batch_optimization: bool = True
    dynamic_batch_sizing: bool = True
    max_batch_size: int = 64
    min_batch_size: int = 4
    target_memory_usage: float = 0.8  # Target GPU memory usage
    
    # Memory efficiency
    enable_memory_pooling: bool = True
    enable_gradient_checkpointing: bool = True
    memory_cleanup_interval: int = 100  # Steps between cleanup
    tensor_cache_size: int = 1000
    
    # Neural training optimization
    enable_mixed_precision: bool = True
    enable_adaptive_learning_rate: bool = True
    enable_early_stopping: bool = True
    performance_monitoring_window: int = 50
    
    # Throughput optimization
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    enable_embedding_cache: bool = True
    cache_ttl: int = 300  # Cache time-to-live in seconds
    
    # GPU optimization
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.9
    enable_tensor_cores: bool = True


class BatchProcessor:
    """Optimized batch processing for neural networks"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.current_batch_size = config.min_batch_size
        self.performance_history = []
        self.memory_monitor = MemoryMonitor()
        
    def optimize_batch_size(self, processing_time: float, memory_usage: float) -> int:
        """Dynamically optimize batch size based on performance metrics"""
        if not self.config.dynamic_batch_sizing:
            return self.config.max_batch_size
        
        # Track performance
        self.performance_history.append({
            'batch_size': self.current_batch_size,
            'processing_time': processing_time,
            'memory_usage': memory_usage,
            'throughput': self.current_batch_size / processing_time
        })
        
        # Keep only recent history
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-10:]
        
        # Optimize batch size
        if memory_usage < self.config.target_memory_usage and self.current_batch_size < self.config.max_batch_size:
            # Increase batch size if memory allows
            self.current_batch_size = min(
                self.current_batch_size * 2,
                self.config.max_batch_size
            )
        elif memory_usage > self.config.target_memory_usage * 1.1:
            # Decrease batch size if memory pressure
            self.current_batch_size = max(
                self.current_batch_size // 2,
                self.config.min_batch_size
            )
        
        return self.current_batch_size
    
    def create_optimized_batches(
        self, 
        embeddings: List[torch.Tensor], 
        importance_scores: Optional[List[float]] = None
    ) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Create optimized batches from embeddings"""
        if not embeddings:
            return []
        
        batch_size = self.current_batch_size
        batches = []
        
        for i in range(0, len(embeddings), batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_importance = None
            
            if importance_scores:
                batch_importance = torch.tensor(
                    importance_scores[i:i + batch_size], 
                    dtype=torch.float32
                )
            
            # Stack embeddings into batch tensor
            try:
                batch_tensor = torch.stack(batch_embeddings)
                batches.append((batch_tensor, batch_importance))
            except Exception as e:
                logger.warning(f"Failed to create batch {i}: {e}")
                # Fallback to individual processing
                for j, emb in enumerate(batch_embeddings):
                    imp = batch_importance[j:j+1] if batch_importance is not None else None
                    batches.append((emb.unsqueeze(0), imp))
        
        return batches
    

class MemoryMonitor:
    """Monitor and optimize memory usage"""
    
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        stats = {'cpu_percent': psutil.virtual_memory().percent / 100.0}
        
        if self.gpu_available:
            try:
                gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                stats['gpu_percent'] = gpu_memory
            except:
                stats['gpu_percent'] = 0.0
        else:
            stats['gpu_percent'] = 0.0
            
        return stats
    
    def cleanup_memory(self):
        """Perform memory cleanup"""
        gc.collect()
        if self.gpu_available:
            torch.cuda.empty_cache()


class EmbeddingCache:
    """Cache for frequently accessed embeddings"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self.cache = {}
        self.access_times = {}
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[torch.Tensor]:
        """Get cached embedding"""
        with self.lock:
            if key in self.cache:
                # Check TTL
                if time.time() - self.access_times[key] < self.ttl:
                    self.access_times[key] = time.time()
                    return self.cache[key].clone()
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
        return None
    
    def put(self, key: str, embedding: torch.Tensor):
        """Cache embedding"""
        with self.lock:
            # Clean expired entries
            self._cleanup_expired()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_size:
                self._evict_oldest()
            
            self.cache[key] = embedding.clone()
            self.access_times[key] = time.time()
    
    def _cleanup_expired(self):
        """Remove expired cache entries"""
        current_time = time.time()
        expired_keys = [
            k for k, t in self.access_times.items() 
            if current_time - t >= self.ttl
        ]
        for key in expired_keys:
            del self.cache[key]
            del self.access_times[key]
    
    def _evict_oldest(self):
        """Evict oldest cache entry"""
        if self.access_times:
            oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
            del self.cache[oldest_key]
            del self.access_times[oldest_key]


class NeuralTrainingOptimizer:
    """Optimize neural network training performance"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.mixed_precision_enabled = config.enable_mixed_precision and torch.cuda.is_available()
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision_enabled else None
        self.performance_tracker = PerformanceTracker()
        
    def optimize_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimized training step with mixed precision and monitoring"""
        start_time = time.time()
        
        model.train()
        optimizer.zero_grad()
        
        # Mixed precision forward pass
        if self.mixed_precision_enabled:
            with torch.cuda.amp.autocast():
                outputs = model(inputs, **kwargs)
                loss = loss_fn(outputs, targets)
            
            # Mixed precision backward pass
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
        else:
            # Standard precision
            outputs = model(inputs, **kwargs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
        
        # Track performance
        step_time = time.time() - start_time
        self.performance_tracker.record_step(
            loss=loss.item(),
            step_time=step_time,
            batch_size=inputs.size(0)
        )
        
        return {
            'loss': loss.item(),
            'step_time': step_time,
            'outputs': outputs
        }
    
    def should_early_stop(self, patience: int = 10) -> bool:
        """Check if training should stop early"""
        if not self.config.enable_early_stopping:
            return False
        
        return self.performance_tracker.check_early_stopping(patience)


class PerformanceTracker:
    """Track and analyze training performance"""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.history = []
        self.best_loss = float('inf')
        self.patience_counter = 0
        
    def record_step(self, loss: float, step_time: float, batch_size: int):
        """Record a training step"""
        self.history.append({
            'loss': loss,
            'step_time': step_time,
            'batch_size': batch_size,
            'throughput': batch_size / step_time,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.history) > self.window_size:
            self.history = self.history[-self.window_size // 2:]
        
        # Update best loss and patience
        if loss < self.best_loss:
            self.best_loss = loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.history:
            return {}
        
        recent_losses = [h['loss'] for h in self.history[-10:]]
        recent_times = [h['step_time'] for h in self.history[-10:]]
        recent_throughput = [h['throughput'] for h in self.history[-10:]]
        
        return {
            'avg_loss': np.mean(recent_losses),
            'loss_std': np.std(recent_losses),
            'avg_step_time': np.mean(recent_times),
            'avg_throughput': np.mean(recent_throughput),
            'best_loss': self.best_loss,
            'loss_trend': self._calculate_trend(recent_losses),
            'total_steps': len(self.history)
        }
    
    def check_early_stopping(self, patience: int) -> bool:
        """Check if early stopping criteria is met"""
        return self.patience_counter >= patience
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 5:
            return "insufficient_data"
        
        recent = np.mean(values[-3:])
        older = np.mean(values[-6:-3])
        
        if recent < older * 0.95:
            return "improving"
        elif recent > older * 1.05:
            return "degrading"
        else:
            return "stable"


class ParallelProcessor:
    """Parallel processing for cognitive operations"""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.executor = ThreadPoolExecutor(max_workers=config.max_worker_threads)
        
    async def process_inputs_parallel(
        self,
        inputs: List[str],
        process_func: callable,
        **kwargs
    ) -> List[Any]:
        """Process multiple inputs in parallel"""
        if not self.config.enable_parallel_processing or len(inputs) == 1:
            # Fall back to sequential processing
            results = []
            for inp in inputs:
                result = await process_func(inp, **kwargs)
                results.append(result)
            return results
        
        # Parallel processing
        loop = asyncio.get_event_loop()
        futures = []
        
        for inp in inputs:
            future = loop.run_in_executor(
                self.executor,
                lambda x: asyncio.run(process_func(x, **kwargs)),
                inp
            )
            futures.append(future)
        
        results = await asyncio.gather(*futures, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing input {i}: {result}")
                processed_results.append(None)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def shutdown(self):
        """Shutdown parallel processor"""
        self.executor.shutdown(wait=True)


class PerformanceOptimizer:
    """Main performance optimization coordinator"""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Initialize components
        self.batch_processor = BatchProcessor(self.config)
        self.memory_monitor = MemoryMonitor()
        self.embedding_cache = EmbeddingCache(
            max_size=self.config.tensor_cache_size,
            ttl=self.config.cache_ttl
        )
        self.training_optimizer = NeuralTrainingOptimizer(self.config)
        self.parallel_processor = ParallelProcessor(self.config)
        
        # Performance tracking
        self.optimization_stats = {
            'total_optimizations': 0,
            'memory_cleanups': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'batch_optimizations': 0
        }
        
        self.cleanup_counter = 0
        
        logger.info("Performance Optimizer initialized")
    
    def optimize_neural_forward_pass(
        self,
        model: nn.Module,
        embeddings: List[torch.Tensor],
        importance_scores: Optional[List[float]] = None,
        **forward_kwargs
    ) -> Dict[str, Any]:
        """Optimize neural network forward pass with batching"""
        start_time = time.time()
          # Create optimized batches
        batches = self.batch_processor.create_optimized_batches(
            embeddings, importance_scores
        )
        
        results = []
        total_memory_usage = 0
        
        for batch_embeddings, batch_importance in batches:
            # Monitor memory before processing
            memory_before = self.memory_monitor.get_memory_usage()
            
            # Forward pass - adapt to model type
            with torch.no_grad():
                # Check if model is LSHN (has consolidate_memories method)
                if hasattr(model, 'consolidate_memories'):
                    # LSHN network - convert list to tensor if needed
                    if isinstance(batch_embeddings, list):
                        batch_tensor = torch.stack(batch_embeddings)
                    else:
                        batch_tensor = batch_embeddings
                        
                    if isinstance(batch_importance, list):
                        importance_tensor = torch.tensor(batch_importance, dtype=torch.float32)
                    else:
                        importance_tensor = batch_importance
                    
                    batch_result = model(
                        batch_tensor,
                        importance_scores=importance_tensor,
                        **forward_kwargs
                    )
                elif hasattr(model, 'behavior_path'):
                    # DPAD network - convert list to tensor if needed
                    if isinstance(batch_embeddings, list):
                        batch_tensor = torch.stack(batch_embeddings)
                    else:
                        batch_tensor = batch_embeddings
                    
                    batch_result = model(batch_tensor, **forward_kwargs)
                else:
                    # Generic model - try basic forward pass
                    if isinstance(batch_embeddings, list):
                        batch_tensor = torch.stack(batch_embeddings)
                    else:
                        batch_tensor = batch_embeddings
                    
                    batch_result = model(batch_tensor, **forward_kwargs)
            
            results.append(batch_result)
            
            # Monitor memory after processing
            memory_after = self.memory_monitor.get_memory_usage()
            total_memory_usage += memory_after['gpu_percent']
        
        # Optimize batch size for next time
        processing_time = time.time() - start_time
        avg_memory_usage = total_memory_usage / len(batches) if batches else 0
        
        optimized_batch_size = self.batch_processor.optimize_batch_size(
            processing_time, avg_memory_usage
        )
        
        # Periodic memory cleanup
        self.cleanup_counter += 1
        if self.cleanup_counter >= self.config.memory_cleanup_interval:
            self.memory_monitor.cleanup_memory()
            self.optimization_stats['memory_cleanups'] += 1
            self.cleanup_counter = 0
        
        self.optimization_stats['batch_optimizations'] += 1
        
        return {
            'results': results,
            'processing_time': processing_time,
            'optimized_batch_size': optimized_batch_size,
            'memory_usage': avg_memory_usage,
            'num_batches': len(batches)
        }
    
    def get_cached_embedding(self, cache_key: str) -> Optional[torch.Tensor]:
        """Get cached embedding if available"""
        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            self.optimization_stats['cache_hits'] += 1
            return cached
        else:
            self.optimization_stats['cache_misses'] += 1
            return None
    
    def cache_embedding(self, cache_key: str, embedding: torch.Tensor):
        """Cache embedding for future use"""
        self.embedding_cache.put(cache_key, embedding)
    
    def optimize_training_step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: callable,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        **kwargs
    ) -> Dict[str, Any]:
        """Optimize a training step"""
        return self.training_optimizer.optimize_training_step(
            model, optimizer, loss_fn, inputs, targets, **kwargs
        )
    
    async def process_parallel(
        self,
        inputs: List[str],
        process_func: callable,
        **kwargs
    ) -> List[Any]:
        """Process inputs in parallel"""
        return await self.parallel_processor.process_inputs_parallel(
            inputs, process_func, **kwargs
        )
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        memory_stats = self.memory_monitor.get_memory_usage()
        training_stats = self.training_optimizer.performance_tracker.get_performance_stats()
        
        return {
            'optimization_stats': self.optimization_stats,
            'memory_stats': memory_stats,
            'training_stats': training_stats,
            'current_batch_size': self.batch_processor.current_batch_size,
            'cache_hit_rate': (
                self.optimization_stats['cache_hits'] / 
                max(1, self.optimization_stats['cache_hits'] + self.optimization_stats['cache_misses'])
            )
        }
    
    def should_early_stop_training(self, patience: int = 10) -> bool:
        """Check if training should stop early"""
        return self.training_optimizer.should_early_stop(patience)
    
    def shutdown(self):
        """Shutdown performance optimizer"""
        self.parallel_processor.shutdown()
        logger.info("Performance Optimizer shutdown completed")


# Utility functions for integration
def create_performance_optimizer(config: Optional[PerformanceConfig] = None) -> PerformanceOptimizer:
    """Factory function to create performance optimizer"""
    return PerformanceOptimizer(config)


def optimize_tensor_operations(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Optimize tensor operations for better performance"""
    optimized = []
    
    for tensor in tensors:
        # Ensure contiguous memory layout
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        
        # Move to GPU if available and beneficial
        if torch.cuda.is_available() and tensor.device.type == 'cpu' and tensor.numel() > 1000:
            tensor = tensor.cuda()
        
        optimized.append(tensor)
    
    return optimized
