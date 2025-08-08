#!/usr/bin/env python3
"""
Quick test for vector-only memory systems
"""
import sys
sys.path.append('.')

from src.memory.stm.vector_stm import VectorShortTermMemory, STMConfiguration
import tempfile

def main():
    print("Testing STM basic operations...")
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        stm = None  # Ensure stm is always defined
        try:
            # Test STM
            config = STMConfiguration(
                chroma_persist_dir=temp_dir + "/stm_test",
                collection_name="quick_test"
            )
            stm = VectorShortTermMemory(config)
            
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
        finally:
            # Properly shutdown STM to release file locks
            if stm is not None:
                try:
                    stm.shutdown()
                except Exception:
                    pass  # Ignore shutdown errors

if __name__ == "__main__":
    main()
