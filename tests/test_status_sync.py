#!/usr/bin/env python3
"""
Synchronous test to check attention mechanism status
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_attention_status_sync():
    """Test attention status synchronously"""
    print("🔍 Testing Attention Status (Sync)...")
    
    try:
        from src.core.cognitive_agent import CognitiveAgent
        
        print("Creating agent...")
        agent = CognitiveAgent()
        print(f"Agent created: {agent.session_id}")
        
        print("\nGetting initial status...")
        status = agent.get_cognitive_status()
        
        print("\n📊 ATTENTION STATUS FIELDS:")
        attention_status = status.get("attention_status", {})
        for key, value in attention_status.items():
            print(f"  {key}: {value}")
        
        print("\n📊 SENSORY PROCESSING FIELDS:")
        sensory_status = status.get("sensory_processing", {})
        for key, value in sensory_status.items():
            print(f"  {key}: {value}")
        
        print("\n📊 INTEGRATION FIELDS:")
        integration = status.get("cognitive_integration", {})
        for key, value in integration.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Sync test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_attention_status_sync()
