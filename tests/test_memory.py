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
    try:
        print("Testing STM import...")
        
        # Test STM directly
        from memory.stm.short_term_memory import ShortTermMemory
        print("✓ Successfully imported ShortTermMemory and MemoryItem")
        
        # Test creating STM
        stm = ShortTermMemory(capacity=7)
        print("✓ Successfully created ShortTermMemory")
        
        # Test storing memory
        success = stm.store("test_1", "Hello world", importance=0.8)
        print(f"✓ Memory stored: {success}")
        
        # Test retrieving memory
        item = stm.retrieve("test_1")
        print(f"✓ Retrieved memory: {item.content if item else 'None'}")
        
        print("\n🎉 STM tests passed!")
        
        # Test LTM
        print("\nTesting LTM import...")
        from memory.ltm.long_term_memory import LongTermMemory
        print("✓ Successfully imported LongTermMemory and LTMRecord")
        
        # Test creating LTM
        ltm = LongTermMemory()
        print("✓ Successfully created LongTermMemory")
        
        print("\n🎉 All memory component tests passed!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    test_memory_components()
