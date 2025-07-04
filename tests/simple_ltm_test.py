#!/usr/bin/env python3
"""
Simple test to debug LTM vector search issues
"""
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.ltm.vector_ltm import VectorLongTermMemory

def main():
    # Create temporary directory
    temp_dir = tempfile.mkdtemp()
    print(f"Using temp dir: {temp_dir}")
    
    try:
        # Initialize LTM
        print("Initializing LTM...")
        ltm = VectorLongTermMemory(
            chroma_persist_dir=f"{temp_dir}/chroma_ltm",
            collection_name="simple_test"
        )
        
        # Store a simple memory
        print("Storing memory...")
        success = ltm.store(
            memory_id="test_1",
            content="This is a test about artificial intelligence",
            memory_type="test",
            importance=0.8
        )
        print(f"Store success: {success}")
        
        # Retrieve the memory
        print("Retrieving memory...")
        retrieved = ltm.retrieve("test_1")
        print(f"Retrieved: {retrieved}")
        
        # Simple search with very low threshold
        print("Searching with low threshold...")
        results = ltm.search_semantic(
            query="artificial",
            max_results=10,
            min_similarity=0.0
        )
        print(f"Search results: {len(results)}")
        for i, result in enumerate(results):
            print(f"  {i}: {result.get('content', '')} (score: {result.get('similarity_score', 0)})")
        
        print("✓ Simple LTM test completed successfully")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    main()
