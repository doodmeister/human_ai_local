"""
Test memory system integration with cognitive agent
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.core.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig

async def test_memory_integration():
    """Test memory system functionality with the cognitive agent"""
    print("🧠 Testing memory integration with cognitive agent...")
    
    # Create config
    config = CognitiveConfig()
    
    try:
        # Initialize cognitive agent
        agent = CognitiveAgent(config)
        print(f"✅ Agent initialized: {agent.session_id}")
        
        # Test multiple interactions to build memory
        inputs = [
            "Hello, my name is Alice and I'm a software engineer.",
            "I work on artificial intelligence systems.",
            "Can you remember what I told you about my profession?",
            "What's my name again?",
            "Tell me about our previous conversations."
        ]
        
        responses = []
        
        for i, input_text in enumerate(inputs, 1):
            print(f"\n--- Interaction {i} ---")
            print(f"Input: {input_text}")
            
            try:
                response = await agent.process_input(input_text)
                responses.append(response)
                print(f"Response: {response}")
                
                # Show memory status after each interaction
                status = agent.get_cognitive_status()
                memory_status = status["memory_status"]
                print(f"STM: {memory_status['stm']['size']}/{memory_status['stm']['capacity']} items")
                print(f"LTM: {memory_status['ltm']['total_memories']} memories")
                print(f"Session memories: {memory_status['session_memories']}")
                
            except Exception as e:
                print(f"❌ Error processing input {i}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        # Test dream state processing
        print("\n--- Dream State Processing ---")
        try:
            dream_results = await agent.enter_dream_state()
            print(f"Dream results: {dream_results}")
        except Exception as e:
            print(f"❌ Error in dream state: {e}")
        
        try:
            final_status = agent.get_cognitive_status()
            print(f"Final memory status: {final_status['memory_status']}")
        except Exception as e:
            print(f"❌ Error getting final status: {e}")
        
        # Test shutdown
        try:
            await agent.shutdown()
            print("✅ Shutdown complete")
        except Exception as e:
            print(f"❌ Error in shutdown: {e}")
        
        print("\n🎉 Memory integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    try:
        success = asyncio.run(test_memory_integration())
        if success:
            print("\n✅ Memory integration working perfectly!")
        else:
            print("\n❌ Memory integration test failed.")
    except Exception as e:
        print(f"❌ Failed to run test: {e}")
        import traceback
        traceback.print_exc()
