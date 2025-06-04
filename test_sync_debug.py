"""
Non-async test to debug the hanging issue
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.core.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig

def test_sync_operations():
    """Test non-async operations only"""
    print("🔍 Testing synchronous operations only...")
    
    try:
        # Initialize agent
        config = CognitiveConfig()
        print("✅ Config created")
        
        agent = CognitiveAgent(config)
        print(f"✅ Agent initialized: {agent.session_id}")
        
        # Test sync methods only
        print("\n--- Testing get_cognitive_status ---")
        status = agent.get_cognitive_status()
        print(f"✅ Status retrieved: {status}")
        
        print("\n--- Testing _calculate_attention_allocation ---")
        dummy_input = {"raw_input": "test", "type": "text"}
        dummy_memory = []
        attention = agent._calculate_attention_allocation(dummy_input, dummy_memory)
        print(f"✅ Attention calculated: {attention}")
        
        print("\n--- Testing _update_cognitive_state ---")
        agent._update_cognitive_state(attention)
        print(f"✅ Cognitive state updated")
        
        print("\n🎉 All sync operations work!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_sync_operations()
