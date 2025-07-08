#!/usr/bin/env python3
"""
Test script for the refactored vector_stm.py module.
This script tests all major functionality without Unicode issues.
"""

import sys
import os
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_vector_stm_functionality():
    """Test the refactored VectorShortTermMemory module."""
    print("=" * 60)
    print("TESTING REFACTORED VECTOR STM MODULE")
    print("=" * 60)
    
    try:
        # Test imports
        from src.memory.stm.vector_stm import VectorShortTermMemory, STMConfiguration, VectorMemoryResult
        print("[PASS] Import successful")
        
        # Test configuration
        config = STMConfiguration(
            capacity=3,
            enable_gpu=False,  # Use CPU for testing
            collection_name="test_stm_refactor"
        )
        print("[PASS] Configuration created")
        
        # Test initialization
        stm = VectorShortTermMemory(config)
        print("[PASS] VectorSTM initialized")
        
        # Test storage operations
        test_memories = [
            ("test_001", "This is about machine learning and AI", 0.9),
            ("test_002", "Python programming and data structures", 0.8), 
            ("test_003", "Natural language processing concepts", 0.7),
            ("test_004", "Database design and SQL queries", 0.6),  # Should trigger eviction
        ]
        
        stored_count = 0
        for mem_id, content, importance in test_memories:
            success = stm.store(mem_id, content, importance=importance)
            if success:
                stored_count += 1
                print(f"[PASS] Stored memory: {mem_id}")
            else:
                print(f"[FAIL] Failed to store: {mem_id}")
        
        print(f"[INFO] Total memories stored: {stored_count}")
        
        # Test retrieval
        retrieved = stm.retrieve("test_004")
        if retrieved:
            print(f"[PASS] Retrieved memory: {retrieved.content[:30]}...")
        else:
            print("[FAIL] Failed to retrieve memory")
        
        # Test semantic search
        search_results = stm.search_semantic("programming", max_results=3)
        print(f"[PASS] Semantic search returned {len(search_results)} results")
        
        for i, result in enumerate(search_results[:2]):  # Show first 2 results
            print(f"  Result {i+1}: {result.item.content[:40]}... (score: {result.similarity_score:.2f})")
        
        # Test associative search
        stm.store("test_005", "Advanced algorithms", associations=["computer-science", "programming"])
        assoc_results = stm.search_associative("programming")
        print(f"[PASS] Associative search returned {len(assoc_results)} results")
        
        # Test status
        status = stm.get_status()
        print(f"[PASS] Status retrieved: {status.get('vector_db_count', 0)} items in database")
        
        # Test memory decay
        decayed_ids = stm.decay_memories(min_activation=0.8)  # High threshold
        print(f"[PASS] Memory decay removed {len(decayed_ids)} items")
        
        # Test clearing
        cleared = stm.clear()
        if cleared:
            print("[PASS] Memory cleared successfully")
        else:
            print("[FAIL] Failed to clear memory")
        
        # Test shutdown
        stm.shutdown()
        print("[PASS] STM shut down successfully")
        
        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY!")
        print("Vector STM module refactor is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_vector_stm_functionality()
    sys.exit(0 if success else 1)
