"""
Test basic integration of cognitive agent with direct configuration
"""
import asyncio
from src.core.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig

async def test_basic_integration():
    """Test basic cognitive agent functionality"""
    print("🧠 Testing basic cognitive agent integration...")
    
    # Create direct config to avoid from_env() issues
    config = CognitiveConfig()
    
    try:
        # Initialize cognitive agent
        agent = CognitiveAgent(config)
        print(f"✅ Agent initialized: {agent.session_id}")
        
        # Test cognitive status
        status = agent.get_cognitive_status()
        print(f"✅ Cognitive status: {status}")
        
        # Test basic input processing
        response = await agent.process_input("Hello, how are you?")
        print(f"✅ Response: {response}")
        
        # Test cognitive status after processing
        status_after = agent.get_cognitive_status()
        print(f"✅ Status after processing: {status_after}")
        
        # Test dream state
        await agent.enter_dream_state()
        print("✅ Dream state processing completed")
        
        # Test shutdown
        await agent.shutdown()
        print("✅ Agent shutdown completed")
        
        print("\n🎉 Basic integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_basic_integration())
    if success:
        print("\n✅ All tests passed! Basic cognitive agent is working.")
    else:
        print("\n❌ Tests failed. Check the output above.")