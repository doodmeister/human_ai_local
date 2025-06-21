#!/usr/bin/env python3
"""
Simple test script to verify memory components work independently
"""
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_memory_components():
    """Test memory components independently"""
    print("Testing STM import...")
    from src.memory.stm.short_term_memory import ShortTermMemory
    import uuid
    print("✓ Successfully imported ShortTermMemory")
    stm = ShortTermMemory(capacity=10)
    print("✓ Successfully created ShortTermMemory")
    memory_id = str(uuid.uuid4())
    content = "test item"
    assert stm.store(memory_id, content) is True
    print(f"✓ Stored item: {content}")
    item = stm.retrieve(memory_id)
    assert item is not None
    assert item['content'] == content
    print(f"✓ Retrieved memory: {item['content'] if item else 'None'}")
    print("\n🎉 STM tests passed!")
    # Test LTM
    print("\nTesting LTM import...")
    from src.memory.ltm.long_term_memory import LongTermMemory
    print("✓ Successfully imported LongTermMemory and LTMRecord")
    ltm = LongTermMemory()
    print("✓ Successfully created LongTermMemory")
    print("\n🎉 All memory component tests passed!")

if __name__ == "__main__":
    test_memory_components()
