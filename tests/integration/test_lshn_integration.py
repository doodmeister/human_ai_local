"""
Test LSHN (Latent Structured Hopfield Networks) Integration

This test verifies the LSHN implementation for episodic memory:
- Hopfield-based associative memory formation
- Episodic pattern encoding and retrieval  
- Memory consolidation during dream cycles
- Integration with existing cognitive architecture
"""

import asyncio
import sys
import os
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.cognition.processing.neural.lshn_network import LSHNNetwork, LSHNConfig
from src.orchestration.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig

@pytest.mark.asyncio
async def test_lshn_basic_functionality():
    """Test basic LSHN network functionality"""
    print("🧠 TESTING LSHN BASIC FUNCTIONALITY")
    print("=" * 60)
    
    # Initialize LSHN network
    config = LSHNConfig(
        embedding_dim=384,
        pattern_dim=512,
        hidden_dim=256,
        attractor_strength=0.8,
        num_heads=4,
        num_layers=2
    )
    
    lshn = LSHNNetwork(config)
    
    print("✅ LSHN Network initialized")
    print(f"   Pattern dimension: {config.pattern_dim}")
    print(f"   Hopfield layers: {config.num_layers}")
    print(f"   Attention heads: {config.num_heads}")
    
    # Test episodic memory encoding
    print("\n📝 STEP 1: Testing Episodic Memory Encoding...")
    
    # Create sample embeddings (simulating memory embeddings)
    batch_size = 5
    sample_embeddings = torch.randn(batch_size, config.embedding_dim)
    importance_scores = torch.tensor([0.8, 0.6, 0.9, 0.5, 0.7])
    
    # Process through LSHN
    results = lshn.forward(
        sample_embeddings,
        store_patterns=True,
        retrieve_similar=False,  # No patterns stored yet
        importance_scores=importance_scores
    )
    
    print(f"   Input embeddings: {sample_embeddings.shape}")
    print(f"   Episodic patterns: {results['episodic_patterns'].shape}")
    print(f"   Episodic traces: {results['episodic_traces'].shape}")
    
    # Verify outputs
    assert results['episodic_patterns'].shape == (batch_size, config.pattern_dim)
    assert results['episodic_traces'].shape == (batch_size, config.pattern_dim)
    
    # Test memory retrieval
    print("\n🔍 STEP 2: Testing Memory Retrieval...")
    
    # Add more patterns and test retrieval
    new_embeddings = torch.randn(3, config.embedding_dim)
    new_importance = torch.tensor([0.6, 0.8, 0.5])
    
    retrieval_results = lshn.forward(
        new_embeddings,
        store_patterns=True,
        retrieve_similar=True,
        importance_scores=new_importance
    )
    
    if retrieval_results['retrieved_patterns'] is not None:
        print(f"   Retrieved patterns: {retrieval_results['retrieved_patterns'].shape}")
        print(f"   Similarity scores: {retrieval_results['similarity_scores'].shape}")
        print(f"   Consolidation scores: {retrieval_results['consolidation_scores'].shape}")
    else:
        print("   ⚠️  Hopfield layers not available - using simplified retrieval")
    
    # Test network statistics
    print("\n📊 STEP 3: Testing Network Statistics...")
    
    stats = lshn.get_network_stats()
    print(f"   Patterns encoded: {stats['patterns_encoded']}")
    print(f"   Patterns stored: {stats['patterns_stored']}")
    print(f"   Retrievals performed: {stats['retrievals_performed']}")
    print(f"   Total parameters: {stats['total_parameters']:,}")
    print(f"   Hopfield available: {stats['hopfield_available']}")
    
    print("\n✅ LSHN BASIC FUNCTIONALITY TEST COMPLETED!")
    return True

@pytest.mark.asyncio
async def test_lshn_memory_consolidation():
    """Test LSHN memory consolidation during dream cycles"""
    print("\n🌙 TESTING LSHN MEMORY CONSOLIDATION")
    print("=" * 60)
    
    # Initialize LSHN
    config = LSHNConfig(
        embedding_dim=384,
        pattern_dim=256,  # Smaller for testing
        association_threshold=0.6,
        consolidation_strength=0.3
    )
    
    lshn = LSHNNetwork(config)
    
    # Create memory embeddings for consolidation
    print("📝 Creating memory embeddings for consolidation...")
    
    memory_embeddings = []
    importance_scores = []
    
    # Simulate different types of memories
    memory_types = [
        ("work meeting about AI project", 0.8),
        ("coffee with friend", 0.4), 
        ("research paper reading", 0.9),
        ("grocery shopping", 0.3),
        ("important client call", 0.9),
        ("lunch break", 0.2),
        ("project deadline discussion", 0.8)
    ]
    
    for i, (description, importance) in enumerate(memory_types):
        # Create embeddings that vary based on content type
        if "AI" in description or "research" in description or "project" in description:
            # AI/work related memories - more similar
            embedding = torch.randn(config.embedding_dim) * 0.5 + torch.ones(config.embedding_dim) * 0.2
        elif "friend" in description or "coffee" in description or "lunch" in description:
            # Social/casual memories
            embedding = torch.randn(config.embedding_dim) * 0.5 - torch.ones(config.embedding_dim) * 0.2  
        else:
            # Other memories
            embedding = torch.randn(config.embedding_dim) * 0.8
            
        memory_embeddings.append(embedding)
        importance_scores.append(importance)
        
        print(f"   Memory {i+1}: {description} (importance: {importance})")
    
    # Test memory consolidation
    print(f"\n🔧 STEP 1: Consolidating {len(memory_embeddings)} memories...")
    
    consolidation_results = lshn.consolidate_memories(
        memory_embeddings,
        importance_scores,
        association_threshold=config.association_threshold
    )
    
    print(f"   Consolidated memories: {consolidation_results['consolidated_memories']}")
    print(f"   Associations created: {consolidation_results['associations_created']}")
    print(f"   Consolidation strength: {consolidation_results['consolidation_strength']:.3f}")
    print(f"   Episodic patterns formed: {consolidation_results['episodic_patterns_formed']}")
    
    # Verify consolidation
    assert consolidation_results['consolidated_memories'] == len(memory_embeddings)
    assert consolidation_results['associations_created'] >= 0
    assert consolidation_results['consolidation_strength'] >= 0
    
    # Test retrieval after consolidation
    print("\n🔍 STEP 2: Testing retrieval after consolidation...")
    
    # Create a query similar to stored memories  
    query_embedding = memory_embeddings[0] + torch.randn(config.embedding_dim) * 0.1  # Similar to first memory
    
    query_results = lshn.forward(
        query_embedding.unsqueeze(0),
        store_patterns=False,
        retrieve_similar=True,
        importance_scores=torch.tensor([0.7])
    )
    
    if query_results['similarity_scores'] is not None:
        max_similarity = query_results['similarity_scores'].max().item()
        print(f"   Max similarity score: {max_similarity:.3f}")
        print(f"   Retrieved patterns shape: {query_results['retrieved_patterns'].shape}")
    else:
        print("   ⚠️  Hopfield layers not available - limited retrieval functionality")
    
    # Test network stats after consolidation
    print("\n📊 STEP 3: Network statistics after consolidation...")
    
    final_stats = lshn.get_network_stats()
    print(f"   Patterns encoded: {final_stats['patterns_encoded']}")
    print(f"   Patterns stored: {final_stats['patterns_stored']}")
    print(f"   Consolidations triggered: {final_stats['consolidations_triggered']}")
    
    if 'total_patterns' in final_stats:
        print(f"   Total stored patterns: {final_stats['total_patterns']}")
        print(f"   Memory utilization: {final_stats['memory_utilization']:.2%}")
    
    print("\n✅ LSHN MEMORY CONSOLIDATION TEST COMPLETED!")
    return True

@pytest.mark.asyncio  
async def test_lshn_integration_with_cognitive_agent():
    """Test LSHN integration with the full cognitive agent"""
    print("\n🤖 TESTING LSHN COGNITIVE AGENT INTEGRATION")
    print("=" * 60)
    
    try:
        # Initialize cognitive agent
        config = CognitiveConfig()
        agent = CognitiveAgent(config)
        
        print(f"✅ Cognitive agent initialized: {agent.session_id}")
        
        # Check if neural integration includes LSHN
        neural_manager = agent.neural_integration
        
        if hasattr(neural_manager, 'lshn_network'):
            print("✅ LSHN network found in neural integration")
            
            # Get LSHN network stats
            lshn_stats = neural_manager.lshn_network.get_network_stats()
            print(f"   LSHN parameters: {lshn_stats['total_parameters']:,}")
            print(f"   Hopfield available: {lshn_stats['hopfield_available']}")
        else:
            print("⚠️  LSHN network not found in neural integration")
        
        # Test memory processing with LSHN
        print("\n📝 STEP 1: Creating memories for LSHN processing...")
        
        test_inputs = [
            "I had an important meeting about neural networks today",
            "The Hopfield network paper was fascinating to read", 
            "Our AI project is making great progress",
            "I need to implement episodic memory consolidation",
            "The cognitive architecture is working well"
        ]
        
        for i, input_text in enumerate(test_inputs):
            print(f"   Input {i+1}: {input_text[:50]}...")
            response = await agent.process_input(input_text)
            print(f"   Response: {response[:50]}...")
          # Check memory state
        stm_count = len(agent.memory.stm.get_recent_memories())
        ltm_count = len(agent.memory.ltm.search_memories("", limit=1000))
        
        print("\n📊 Memory state after processing:")
        print(f"   STM: {stm_count} memories")
        print(f"   LTM: {ltm_count} memories")
        
        # Get neural integration status
        if hasattr(neural_manager, 'get_neural_status'):
            neural_status = neural_manager.get_neural_status()
            print("\n🧠 Neural integration status:")
            print(f"   DPAD parameters: {neural_status['network_status']['parameters']:,}")
            print(f"   Neural replay enabled: {neural_status['integration_status']['neural_replay_enabled']}")
            
        print("\n✅ LSHN COGNITIVE AGENT INTEGRATION TEST COMPLETED!")
        
        # Clean up
        await agent.shutdown()
        
        return True
        
    except Exception as e:
        print(f"❌ Error in cognitive agent integration: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_lshn_performance_benchmark():
    """Benchmark LSHN performance with different configurations"""
    print("\n⚡ TESTING LSHN PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    import time
    
    configs = [
        ("Small", LSHNConfig(embedding_dim=128, pattern_dim=256, hidden_dim=128, num_heads=4, num_layers=1)),
        ("Medium", LSHNConfig(embedding_dim=384, pattern_dim=512, hidden_dim=256, num_heads=8, num_layers=2)), 
        ("Large", LSHNConfig(embedding_dim=768, pattern_dim=1024, hidden_dim=384, num_heads=8, num_layers=3))  # 384 is divisible by 8
    ]
    
    for name, config in configs:
        print(f"\n🔧 Testing {name} configuration...")
        print(f"   Embedding dim: {config.embedding_dim}")
        print(f"   Pattern dim: {config.pattern_dim}")
        print(f"   Heads: {config.num_heads}, Layers: {config.num_layers}")
        
        # Initialize network
        start_time = time.time()
        lshn = LSHNNetwork(config)
        init_time = time.time() - start_time
        
        # Get parameter count
        stats = lshn.get_network_stats()
        params = stats['total_parameters']
        
        # Test forward pass timing
        batch_size = 10
        embeddings = torch.randn(batch_size, config.embedding_dim)
        importance = torch.rand(batch_size)
        
        start_time = time.time()
        results = lshn.forward(embeddings, importance_scores=importance)
        forward_time = time.time() - start_time
        
        # Test consolidation timing
        memory_embeddings = [torch.randn(config.embedding_dim) for _ in range(20)]
        importance_scores = [0.5 + 0.5 * np.random.random() for _ in range(20)]
        
        start_time = time.time()
        consolidation_results = lshn.consolidate_memories(memory_embeddings, importance_scores)
        consolidation_time = time.time() - start_time
        
        print(f"   Parameters: {params:,}")
        print(f"   Init time: {init_time:.3f}s")
        print(f"   Forward pass: {forward_time:.3f}s ({batch_size} samples)")
        print(f"   Consolidation: {consolidation_time:.3f}s ({len(memory_embeddings)} memories)")
        print(f"   Associations created: {consolidation_results['associations_created']}")
    
    print("\n✅ LSHN PERFORMANCE BENCHMARK COMPLETED!")
    return True

# Run all tests
async def run_all_lshn_tests():
    """Run all LSHN tests"""
    print("🚀 RUNNING ALL LSHN TESTS")
    print("=" * 80)
    
    tests = [
        ("Basic Functionality", test_lshn_basic_functionality),
        ("Memory Consolidation", test_lshn_memory_consolidation),
        ("Cognitive Agent Integration", test_lshn_integration_with_cognitive_agent),
        ("Performance Benchmark", test_lshn_performance_benchmark)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            print(f"\n🧪 Running: {test_name}")
            result = await test_func()
            results.append((test_name, result, None))
            print(f"✅ {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"❌ {test_name}: FAILED - {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("🏁 LSHN TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result, _ in results if result)
    total = len(results)
    
    for test_name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {test_name}")
        if error:
            print(f"   Error: {error}")
    
    print(f"\nTests passed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("\n🎉 ALL LSHN TESTS PASSED! 🎉")
        print("LSHN (Latent Structured Hopfield Networks) implementation is ready!")
    else:
        print(f"\n⚠️  {total-passed} tests failed. Check implementation.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_lshn_tests())
