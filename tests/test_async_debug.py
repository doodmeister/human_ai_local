"""
Simple async test to debug the hanging issue
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.core.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig

async def test_process_input_step_by_step():
    """Test process_input method step by step"""
    print("🔍 Testing process_input step by step...")
    
    try:
        # Initialize agent
        config = CognitiveConfig()
        agent = CognitiveAgent(config)
        print(f"✅ Agent initialized: {agent.session_id}")
        
        # Test each step of process_input individually
        input_text = "Hello, test input!"
        
        print("\n--- Testing _process_sensory_input ---")
        processed = await agent._process_sensory_input(input_text, "text")
        print(f"✅ Sensory processing complete: {processed}")
        
        print("\n--- Testing _retrieve_memory_context ---")
        memory_context = await agent._retrieve_memory_context(processed)
        print(f"✅ Memory retrieval complete: {memory_context}")
        
        print("\n--- Testing _calculate_attention_allocation ---")
        attention_scores = agent._calculate_attention_allocation(processed, memory_context)
        print(f"✅ Attention calculation complete: {attention_scores}")
        
        print("\n--- Testing _generate_response ---")
        response = await agent._generate_response(processed, memory_context, attention_scores)
        print(f"✅ Response generation complete: {response}")
        
        print("\n--- Testing _consolidate_memory ---")
        await agent._consolidate_memory(input_text, response, attention_scores)
        print("✅ Memory consolidation complete")
        
        print("\n--- Testing _update_cognitive_state ---")
        agent._update_cognitive_state(attention_scores)
        print("✅ Cognitive state update complete")
        
        print("\n🎉 All individual steps work!")
        
    except Exception as e:
        print(f"❌ Error at step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_process_input_step_by_step())
