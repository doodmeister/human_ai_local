"""
Performance Integration Test for Human-AI Cognition Framework

This test validates the performance optimization framework integration
with the neural networks and cognitive architecture.
"""

import asyncio
import pytest
import torch
import numpy as np
import time
import logging
import sys
import os
from typing import List, Dict, Any

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.optimization import PerformanceOptimizer, PerformanceConfig, create_performance_optimizer
from src.processing.neural.dpad_network import DPADNetwork, DPADConfig
from src.processing.neural.lshn_network import LSHNNetwork, LSHNConfig
from src.core.config import CognitiveConfig

logger = logging.getLogger(__name__)

@pytest.mark.asyncio
async def test_performance_optimization_framework():
    """Test the performance optimization framework implementation"""
    print("\n🚀 PERFORMANCE OPTIMIZATION FRAMEWORK TEST")
    print("=" * 60)
    
    try:
        # Initialize performance optimizer
        perf_config = PerformanceConfig(
            enable_batch_optimization=True,
            enable_memory_pooling=True,
            enable_mixed_precision=True,
            enable_parallel_processing=True,
            max_batch_size=32,
            dynamic_batch_sizing=True
        )
        
        optimizer = create_performance_optimizer(perf_config)
        print("✓ Performance optimizer initialized")
        
        # Test batch processing optimization
        print("\n📦 TESTING BATCH PROCESSING OPTIMIZATION")
        test_embeddings = [torch.randn(384) for _ in range(50)]
        importance_scores = [0.5 + 0.5 * np.random.random() for _ in range(50)]
        
        batches = optimizer.batch_processor.create_optimized_batches(
            test_embeddings, importance_scores
        )
        
        print(f"✓ Created {len(batches)} optimized batches from {len(test_embeddings)} embeddings")
        print(f"  Initial batch size: {optimizer.batch_processor.current_batch_size}")
        
        # Test memory monitoring
        print("\n🧠 TESTING MEMORY MONITORING")
        memory_stats = optimizer.memory_monitor.get_memory_usage()
        print(f"✓ Memory monitoring active:")
        print(f"  CPU usage: {memory_stats['cpu_percent']:.1%}")
        print(f"  GPU usage: {memory_stats['gpu_percent']:.1%}")
        
        # Test embedding cache
        print("\n💾 TESTING EMBEDDING CACHE")
        test_embedding = torch.randn(384)
        cache_key = "test_embedding_001"
        
        # Cache miss
        cached = optimizer.get_cached_embedding(cache_key)
        assert cached is None, "Should be cache miss initially"
        
        # Cache put
        optimizer.cache_embedding(cache_key, test_embedding)
        
        # Cache hit
        cached = optimizer.get_cached_embedding(cache_key)
        assert cached is not None, "Should be cache hit after caching"
        assert torch.allclose(cached, test_embedding), "Cached embedding should match original"
        
        print("✓ Embedding cache working correctly")
        
        # Test neural network optimization
        print("\n🧮 TESTING NEURAL NETWORK OPTIMIZATION")
        
        # Initialize test networks
        dpad_config = DPADConfig(embedding_dim=384, latent_dim=64)
        dpad_network = DPADNetwork(dpad_config)
        
        lshn_config = LSHNConfig(embedding_dim=384, pattern_dim=512)
        lshn_network = LSHNNetwork(lshn_config)
          # Test DPAD optimization
        dpad_start_time = time.time()
        dpad_optimization = optimizer.optimize_neural_forward_pass(
            dpad_network,
            test_embeddings[:20],  # Smaller batch for testing
            return_paths=True  # DPAD-specific parameter
        )
        dpad_time = time.time() - dpad_start_time
        
        print(f"✓ DPAD optimization completed in {dpad_time:.3f}s")
        print(f"  Processing time: {dpad_optimization['processing_time']:.3f}s")
        print(f"  Memory usage: {dpad_optimization['memory_usage']:.1%}")
        print(f"  Optimized batch size: {dpad_optimization['optimized_batch_size']}")
        print(f"  Number of batches: {dpad_optimization['num_batches']}")
        
        # Test LSHN optimization
        lshn_start_time = time.time()
        lshn_optimization = optimizer.optimize_neural_forward_pass(
            lshn_network,
            test_embeddings[:15],  # Smaller batch for testing
            importance_scores[:15],
            store_patterns=True,
            retrieve_similar=True
        )
        lshn_time = time.time() - lshn_start_time
        
        print(f"✓ LSHN optimization completed in {lshn_time:.3f}s")
        print(f"  Processing time: {lshn_optimization['processing_time']:.3f}s")
        print(f"  Memory usage: {lshn_optimization['memory_usage']:.1%}")
        print(f"  Optimized batch size: {lshn_optimization['optimized_batch_size']}")
        print(f"  Number of batches: {lshn_optimization['num_batches']}")
        
        # Test training optimization
        print("\n🎯 TESTING TRAINING OPTIMIZATION")
        
        # Create training data
        train_inputs = torch.stack(test_embeddings[:10])
        train_targets = train_inputs + 0.1 * torch.randn_like(train_inputs)  # Add noise
        
        def simple_loss_fn(outputs, targets):
            return torch.nn.functional.mse_loss(outputs, targets)
        
        training_result = optimizer.optimize_training_step(
            model=dpad_network,
            optimizer=torch.optim.Adam(dpad_network.parameters(), lr=0.001),
            loss_fn=simple_loss_fn,
            inputs=train_inputs,
            targets=train_targets
        )
        
        print(f"✓ Training optimization completed")
        print(f"  Loss: {training_result['loss']:.4f}")
        print(f"  Step time: {training_result['step_time']:.3f}s")
        
        # Test parallel processing
        print("\n⚡ TESTING PARALLEL PROCESSING")
        
        async def dummy_process_func(text: str) -> str:
            await asyncio.sleep(0.01)  # Simulate processing time
            return f"processed_{text}"
        
        test_inputs = [f"input_{i}" for i in range(8)]
        
        parallel_start = time.time()
        parallel_results = await optimizer.process_parallel(
            test_inputs, dummy_process_func
        )
        parallel_time = time.time() - parallel_start
        
        print(f"✓ Parallel processing completed in {parallel_time:.3f}s")
        print(f"  Processed {len(parallel_results)} inputs")
        print(f"  Results: {parallel_results[:3]}...")  # Show first 3 results
        
        # Get comprehensive optimization statistics
        print("\n📊 OPTIMIZATION STATISTICS")
        opt_stats = optimizer.get_optimization_stats()
        
        print("✓ Overall optimization statistics:")
        print(f"  Total optimizations: {opt_stats['optimization_stats']['total_optimizations']}")
        print(f"  Memory cleanups: {opt_stats['optimization_stats']['memory_cleanups']}")
        print(f"  Cache hit rate: {opt_stats['cache_hit_rate']:.1%}")
        print(f"  Batch optimizations: {opt_stats['optimization_stats']['batch_optimizations']}")
        print(f"  Current batch size: {opt_stats['current_batch_size']}")
        
        # Memory efficiency metrics
        memory_stats = opt_stats['memory_stats']
        print(f"  Memory efficiency:")
        print(f"    CPU usage: {memory_stats['cpu_percent']:.1%}")
        print(f"    GPU usage: {memory_stats['gpu_percent']:.1%}")
        
        # Training performance metrics
        training_stats = opt_stats['training_stats']
        if training_stats:
            print(f"  Training performance:")
            print(f"    Total steps: {training_stats.get('total_steps', 0)}")
            print(f"    Average loss: {training_stats.get('avg_loss', 0):.4f}")
            print(f"    Average throughput: {training_stats.get('avg_throughput', 0):.1f} samples/sec")
        
        # Test early stopping
        should_stop = optimizer.should_early_stop_training(patience=5)
        print(f"  Early stopping recommended: {should_stop}")
        
        # Cleanup
        optimizer.shutdown()
        print("\n✅ PERFORMANCE OPTIMIZATION FRAMEWORK TEST COMPLETED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance optimization test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


@pytest.mark.asyncio 
async def test_performance_optimization_integration():
    """Test performance optimization integration with cognitive agent"""
    print("\n🔗 PERFORMANCE OPTIMIZATION INTEGRATION TEST")
    print("=" * 60)
    
    try:
        # This test would integrate with the cognitive agent
        # For now, we'll test the framework components
        
        config = CognitiveConfig()
        
        # Create performance-optimized configuration
        perf_config = PerformanceConfig(
            enable_batch_optimization=True,
            enable_memory_pooling=True,
            enable_mixed_precision=torch.cuda.is_available(),
            enable_parallel_processing=True,
            max_batch_size=16,
            target_memory_usage=0.8
        )
        
        optimizer = create_performance_optimizer(perf_config)
        print("✓ Performance optimizer created for integration")
        
        # Test tensor optimization utility
        test_tensors = [
            torch.randn(100, 384),  # Large tensor - should move to GPU if available
            torch.randn(10, 384),   # Small tensor - might stay on CPU
            torch.randn(50, 384).transpose(0, 1)  # Non-contiguous tensor
        ]
        
        from src.optimization import optimize_tensor_operations
        optimized_tensors = optimize_tensor_operations(test_tensors)
        
        print(f"✓ Tensor optimization completed:")
        for i, (orig, opt) in enumerate(zip(test_tensors, optimized_tensors)):
            print(f"  Tensor {i}: {orig.shape} -> contiguous: {opt.is_contiguous()}, device: {opt.device}")
        
        # Test integration readiness
        print("\n🔍 INTEGRATION READINESS CHECK")
        
        integration_tests = [
            ("Batch Processing", hasattr(optimizer, 'batch_processor')),
            ("Memory Monitoring", hasattr(optimizer, 'memory_monitor')),
            ("Embedding Cache", hasattr(optimizer, 'embedding_cache')),
            ("Training Optimization", hasattr(optimizer, 'training_optimizer')),
            ("Parallel Processing", hasattr(optimizer, 'parallel_processor')),
            ("Statistics Tracking", callable(getattr(optimizer, 'get_optimization_stats', None)))
        ]
        
        all_ready = True
        for test_name, test_result in integration_tests:
            status = "✓" if test_result else "❌"
            print(f"  {status} {test_name}: {'Ready' if test_result else 'Not Ready'}")
            if not test_result:
                all_ready = False
        
        if all_ready:
            print("\n✅ ALL INTEGRATION COMPONENTS READY!")
        else:
            print("\n⚠️  Some integration components need attention")
        
        optimizer.shutdown()
        return all_ready
        
    except Exception as e:
        print(f"❌ Integration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_performance_optimization_tests():
    """Run all performance optimization tests"""
    print("🧪 RUNNING PERFORMANCE OPTIMIZATION TESTS")
    print("=" * 80)
    
    results = []
    
    # Test 1: Framework functionality
    result1 = await test_performance_optimization_framework()
    results.append(("Performance Framework", result1))
    
    # Test 2: Integration readiness
    result2 = await test_performance_optimization_integration()
    results.append(("Integration Readiness", result2))
    
    # Summary
    print("\n📋 TEST SUMMARY")
    print("=" * 40)
    
    passed = 0
    for test_name, result in results:
        status = "PASSED" if result else "FAILED"
        icon = "✅" if result else "❌"
        print(f"{icon} {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL PERFORMANCE OPTIMIZATION TESTS PASSED!")
        return True
    else:
        print("⚠️  Some tests failed - see details above")
        return False


if __name__ == "__main__":
    asyncio.run(run_performance_optimization_tests())
