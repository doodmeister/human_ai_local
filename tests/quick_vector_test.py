#!/usr/bin/env python3
"""
Quick test for vector-only memory systems
"""
import sys
sys.path.append('.')

from src.memory.stm.vector_stm import VectorShortTermMemory
from src.memory.stm.short_term_memory import MemoryItem
from datetime import datetime
import tempfile

def main():
    print("Testing STM basic operations...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Test STM
            stm = VectorShortTermMemory(
                chroma_persist_dir=temp_dir + "/stm_test",
                collection_name="quick_test"
            )
            
            print("✓ STM initialized")
            
            # Test store
            success = stm.store(
                memory_id="test_1",
                content="Test memory content",
                importance=0.7
            )
            print(f"Store result: {success}")
            
            if success:
                # Test retrieve
                retrieved = stm.retrieve("test_1")
                print(f"Retrieved: {retrieved}")
                
                if retrieved:
                    print(f"Content: {retrieved.content}")
                    print("✓ All tests passed!")
                else:
                    print("❌ Retrieve failed")
            else:
                print("❌ Store failed")
                
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()
