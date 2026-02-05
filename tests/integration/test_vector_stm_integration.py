#!/usr/bin/env python3
"""
Test script for Vector STM integration with MemorySystem
"""
import sys
import os
import logging
import shutil
import tempfile

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.memory.memory_system import MemorySystem, MemorySystemConfig
from src.memory.stm import VectorShortTermMemory

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_vector_stm_integration():
    """Test Vector STM integration with MemorySystem"""
    assert _vector_stm_integration_impl() is True


def _vector_stm_integration_impl() -> bool:
    print("=== Vector STM Integration Test ===")

    test_dir = tempfile.mkdtemp(prefix="vector_stm_integration_")
    memory_system = None
    try:
        # Initialize memory system with vector STM enabled
        print("\n1. Initializing MemorySystem with Vector STM...")
        config = MemorySystemConfig(
            stm_capacity=50,
            use_vector_stm=True,
            use_vector_ltm=True,
            chroma_persist_dir=os.path.join(test_dir, "chroma"),
        )
        memory_system = MemorySystem(config)
        
        # Check if Vector STM is properly initialized
        print(f"   STM Type: {type(memory_system.stm).__name__}")
        status = memory_system.get_status()
        print(f"   Vector STM Enabled: {status.get('use_vector_stm', 'N/A')}")
        print(f"   LTM Type: {type(memory_system.ltm).__name__}")
        
        # Test basic storage
        print("\n2. Testing memory storage...")
        test_memories = [
            ("python_concept", "Python is a high-level programming language", 0.8),
            ("ai_learning", "Machine learning involves training models on data", 0.9),
            ("database_design", "Vector databases store high-dimensional embeddings", 0.7),
            ("neural_networks", "Neural networks are inspired by biological neurons", 0.6),
            ("algorithm_complexity", "Big O notation describes algorithmic complexity", 0.5)
        ]
        
        for memory_id, content, importance in test_memories:
            success = memory_system.store_memory(
                memory_id=memory_id,
                content=content,
                importance=importance,
                emotional_valence=0.1,
                memory_type="semantic"
            )
            print(f"   Stored '{memory_id}': {success}")
        
        # Test status reporting
        print("\n3. Testing status reporting...")
        status = memory_system.get_status()
        print(f"   STM Status: {status['stm']}")
        print(f"   Vector STM: {status.get('use_vector_stm', 'N/A')}")
        print(f"   Vector LTM: {status.get('use_vector_ltm', 'N/A')}")
        
        # Test semantic search in STM
        print("\n4. Testing semantic search...")
        if isinstance(memory_system.stm, VectorShortTermMemory):
            search_query = "programming concepts and algorithms"
            results = memory_system.search_stm_semantic(
                query=search_query,
                max_results=3,
                min_similarity=0.3
            )
            print(f"   Semantic search for '{search_query}':")
            for i, (item, score) in enumerate(results):
                print(f"   {i+1}. Score: {score:.3f} - {item.content[:50]}...")
        
        # Test context retrieval
        print("\n5. Testing context retrieval...")
        context_query = "machine learning and neural networks"
        context = memory_system.get_context_for_query(
            query=context_query,
            max_stm_context=3,
            max_ltm_context=2,
            min_relevance=0.2
        )
        print(f"   Context for '{context_query}':")
        print(f"   STM contexts: {len(context['stm'])}")
        print(f"   LTM contexts: {len(context['ltm'])}")
        
        # Test regular search for comparison
        print("\n6. Testing regular memory search...")
        search_results = memory_system.search_memories(
            query="programming",
            max_results=3
        )
        print(f"   Regular search results: {len(search_results)}")
        for memory, score, source in search_results:
            content = getattr(memory, 'content', str(memory))
            print(f"   {source}: {score:.3f} - {content[:50]}...")
        
        print("\n=== Vector STM Integration Test Completed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if memory_system is not None:
            try:
                memory_system.shutdown()
            except Exception:
                pass
        shutil.rmtree(test_dir, ignore_errors=True)


def test_non_vector_fallback():
    """Test fallback to regular STM when vector support is disabled"""
    assert _non_vector_fallback_impl() is True


def _non_vector_fallback_impl() -> bool:
    print("\n=== Non-Vector Fallback Test ===")

    test_dir = tempfile.mkdtemp(prefix="non_vector_fallback_")
    memory_system = None
    try:
        # Initialize memory system with vector STM disabled
        print("\n1. Initializing MemorySystem without Vector STM...")
        config = MemorySystemConfig(
            stm_capacity=20,
            use_vector_stm=False,
            use_vector_ltm=True,
            chroma_persist_dir=os.path.join(test_dir, "chroma"),
        )
        memory_system = MemorySystem(config)
        
        print(f"   STM Type: {type(memory_system.stm).__name__}")
        status = memory_system.get_status()
        print(f"   Vector STM Enabled: {status.get('use_vector_stm', 'N/A')}")
        
        # Test basic storage
        print("\n2. Testing memory storage with regular STM...")
        success = memory_system.store_memory(
            memory_id="test_memory",
            content="This is a test memory for regular STM",
            importance=0.6
        )
        print(f"   Storage success: {success}")
        
        # Test search methods (should use fallback)
        print("\n3. Testing search methods...")
        results = memory_system.search_stm_semantic(
            query="test memory",
            max_results=3
        )
        print(f"   Semantic search fallback results: {len(results)}")
        
        context = memory_system.get_context_for_query(
            query="test",
            max_stm_context=2
        )
        print(f"   Context retrieval fallback: STM={len(context['stm'])}, LTM={len(context['ltm'])}")
        
        print("\n=== Non-Vector Fallback Test Completed Successfully! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Fallback test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        if memory_system is not None:
            try:
                memory_system.shutdown()
            except Exception:
                pass
        shutil.rmtree(test_dir, ignore_errors=True)

if __name__ == "__main__":
    print("Starting Vector STM Integration Tests...")
    
    # Run tests
    test1_success = _vector_stm_integration_impl()
    test2_success = _non_vector_fallback_impl()
    
    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY:")
    print(f"Vector STM Integration: {'✅ PASS' if test1_success else '❌ FAIL'}")
    print(f"Non-Vector Fallback: {'✅ PASS' if test2_success else '❌ FAIL'}")
    
    if test1_success and test2_success:
        print("\n🎉 All tests passed! Vector STM integration is complete.")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)
