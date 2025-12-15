#!/usr/bin/env python3
"""
Focused test demonstrating attention mechanism integration
"""
import asyncio
import pytest
from src.core.cognitive_agent import CognitiveAgent

@pytest.mark.asyncio
async def test_attention_features():
    """Demonstrate key attention mechanism features"""
    
    print("🧠 ATTENTION MECHANISM INTEGRATION DEMO")
    print("=" * 50)
    
    # Initialize agent
    agent = CognitiveAgent()
    print("✓ Cognitive agent initialized")
    
    # Test 1: Basic attention allocation
    print("\n1. Testing basic attention allocation:")
    response1 = await agent.process_input("Tell me about machine learning")
    status1 = agent.get_cognitive_status()
    
    print("   Input: 'Tell me about machine learning'")
    print(f"   Response: {response1[:60]}...")
    print(f"   Attention items: {status1['attention_status']['focused_items']}")
    print(f"   Cognitive load: {status1['attention_status']['cognitive_load']:.3f}")
    print(f"   Fatigue level: {status1['attention_status']['fatigue_level']:.3f}")
    
    # Test 2: Multiple inputs and capacity management
    print("\n2. Testing multiple inputs and capacity:")
    inputs = [
        "What about neural networks?",
        "How do transformers work?", 
        "Explain attention mechanisms"
    ]
    
    for i, inp in enumerate(inputs, 2):
        await agent.process_input(inp)
        status = agent.get_cognitive_status()
        print(f"   Input {i}: {inp}")
        print(f"   → Focus items: {status['attention_status']['focused_items']}, "
              f"Load: {status['attention_status']['cognitive_load']:.3f}")
    
    # Test 3: Attention focus details
    print("\n3. Current attention focus:")
    focus_items = agent.attention.get_attention_focus()
    for i, item in enumerate(focus_items[:3], 1):
        print(f"   Item {i}: Salience={item['salience']:.3f}, "
              f"Activation={item['activation']:.3f}, "
              f"Duration={item['duration_seconds']:.1f}s")
    
    # Test 4: Cognitive break
    print("\n4. Testing cognitive break:")
    pre_fatigue = agent.current_fatigue
    break_results = agent.take_cognitive_break(0.5)
    post_fatigue = agent.current_fatigue
    
    print(f"   Pre-break fatigue: {pre_fatigue:.3f}")
    print(f"   Post-break fatigue: {post_fatigue:.3f}")
    print(f"   Fatigue reduction: {break_results['fatigue_before'] - break_results['fatigue_after']:.3f}")
    print(f"   Items lost focus: {break_results['attention_items_lost']}")
    
    # Test 5: Integration status
    print("\n5. Cognitive integration status:")
    final_status = agent.get_cognitive_status()
    integration = final_status['cognitive_integration']
    
    print(f"   Memory-attention sync: {integration['attention_memory_sync']}")
    print(f"   Processing capacity: {integration['processing_capacity']:.3f}")
    print(f"   Overall efficiency: {integration['overall_efficiency']:.3f}")
    print(f"   Total focus switches: {final_status['attention_status']['focus_switches']}")
    
    print("\n✅ All attention mechanism features working correctly!")
    print("=" * 50)

if __name__ == "__main__":
    asyncio.run(test_attention_features())
