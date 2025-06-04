#!/usr/bin/env python3
"""
Test script to verify that the CognitiveAgent can be imported and instantiated
"""
import sys
import os
import asyncio

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_cognitive_agent():
    """Test basic CognitiveAgent functionality"""
    try:
        print("Testing imports...")
        
        # Test importing the core module
        from src.core import CognitiveAgent, CognitiveConfig
        print("✓ Successfully imported CognitiveAgent and CognitiveConfig")
        
        # Test creating a configuration
        config = CognitiveConfig.from_env()
        print("✓ Successfully created CognitiveConfig")
        
        # Test creating a cognitive agent
        agent = CognitiveAgent(config)
        print("✓ Successfully created CognitiveAgent")
        
        # Test getting cognitive status
        status = agent.get_cognitive_status()
        print(f"✓ Cognitive status: {status}")
        
        # Test processing a simple input
        response = await agent.process_input("Hello, how are you?")
        print(f"✓ Agent response: {response}")
        
        print("\n🎉 All tests passed! The cognitive agent is working correctly.")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    asyncio.run(test_cognitive_agent())
