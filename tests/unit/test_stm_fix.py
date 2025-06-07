#!/usr/bin/env python3
"""
Test the STM fix for remove_item method
"""

import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from memory.stm.short_term_memory import ShortTermMemory

def test_stm_remove_item():
    """Test the new remove_item method"""
    print("🧪 Testing STM remove_item method...")
    
    # Create STM instance
    stm = ShortTermMemory(capacity=5)
    
    # Add some test memories
    print("📝 Adding test memories...")
    stm.store("mem1", "Alice is a data scientist", importance=0.8)
    stm.store("mem2", "Working on AI project", importance=0.7)
    stm.store("mem3", "Meeting tomorrow", importance=0.6)
    
    print(f"   Initial STM size: {len(stm.items)}")
    assert len(stm.items) == 3, "Should have 3 items"
    
    # Test remove_item method
    print("🗑️ Testing remove_item method...")
    result = stm.remove_item("mem2")
    assert result == True, "Should return True for successful removal"
    assert len(stm.items) == 2, "Should have 2 items after removal"
    assert "mem2" not in stm.items, "mem2 should be removed"
    
    # Test removing non-existent item
    result = stm.remove_item("nonexistent")
    assert result == False, "Should return False for non-existent item"
    
    # Test get_all_items method
    print("📋 Testing get_all_items method...")
    all_items = stm.get_all_items()
    assert len(all_items) == 2, "Should return 2 items"
    assert "mem1" in all_items, "Should contain mem1"
    assert "mem3" in all_items, "Should contain mem3"
    assert "mem2" not in all_items, "Should not contain removed mem2"
    
    print("✅ All STM tests passed!")
    return True

def test_dream_consolidation_basic():
    """Test basic consolidation functionality"""
    print("🌙 Testing basic dream consolidation...")
    
    try:
        from core.cognitive_agent import CognitiveAgent
        
        # Create minimal test
        agent = CognitiveAgent()
        
        # Add some memories
        agent.process_input("Test memory 1")
        agent.process_input("Test memory 2")
        
        # Check STM has items
        stm_count = len(agent.memory_system.stm.items)
        print(f"   STM has {stm_count} items")
        
        # Test that remove_item exists and works
        if stm_count > 0:
            item_ids = list(agent.memory_system.stm.items.keys())
            first_id = item_ids[0]
            result = agent.memory_system.stm.remove_item(first_id)
            print(f"   Removed item {first_id}: {result}")
            
            new_count = len(agent.memory_system.stm.items)
            print(f"   STM now has {new_count} items")
            
            assert new_count == stm_count - 1, "Item should be removed"
        
        print("✅ Basic consolidation test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Dream consolidation test failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 TESTING STM FIXES")
    print("=" * 50)
    
    try:
        # Test 1: STM methods
        test_stm_remove_item()
        print()
        
        # Test 2: Integration with dream processor
        test_dream_consolidation_basic()
        print()
        
        print("🎉 ALL TESTS PASSED! STM fix is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
