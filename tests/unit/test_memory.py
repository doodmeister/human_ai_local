#!/usr/bin/env python3
"""
Simple test script to verify memory components work independently using Vector STM
"""
import sys
import os
import tempfile

# Add the project root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def test_memory_components():
    """Test memory components independently"""
    print("Testing Vector STM import...")
    from src.memory.stm.vector_stm import VectorShortTermMemory, STMConfiguration
    import uuid
    import atexit
    import shutil
    print("✓ Successfully imported VectorShortTermMemory")
    
    # Use a persistent temp dir to avoid Windows file locking issues
    temp_dir = tempfile.mkdtemp()
    
    def cleanup():
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
    atexit.register(cleanup)
    
    config = STMConfiguration(
        chroma_persist_dir=os.path.join(temp_dir, "test_stm"),
        collection_name="test_memory",
        capacity=10
    )
    stm = VectorShortTermMemory(config)
    print("✓ Successfully created VectorShortTermMemory")
    
    try:
        memory_id = str(uuid.uuid4())
        content = "test item"
        assert stm.store(memory_id, content) is True
        print(f"✓ Stored item: {content}")
        
        item = stm.retrieve(memory_id)
        assert item is not None
        assert item.content == content
        print(f"✓ Retrieved memory: {item.content if item else 'None'}")
        print("\n🎉 Vector STM tests passed!")
    finally:
        # Properly shutdown STM to release file locks
        stm.shutdown()
        
    # Test LTM
    print("\nTesting LTM import...")
    from src.memory.ltm.vector_ltm import VectorLongTermMemory
    print("✓ Successfully imported VectorLongTermMemory")
    ltm = VectorLongTermMemory()
    print("✓ Successfully created VectorLongTermMemory")
    print("\n🎉 All memory component tests passed!")

if __name__ == "__main__":
    test_memory_components()
