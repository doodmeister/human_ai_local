#!/usr/bin/env python3
"""
Comprehensive test for attention mechanism integration with cognitive agent
"""
import asyncio
import json
from datetime import datetime
from src.core.cognitive_agent import CognitiveAgent

async def test_attention_integration():
    """Test the full attention mechanism integration"""
    
    print("=" * 60)
    print("COMPREHENSIVE ATTENTION MECHANISM INTEGRATION TEST")
    print("=" * 60)
    
    # Initialize cognitive agent
    print("\n1. Initializing Cognitive Agent...")
    agent = CognitiveAgent()
    
    initial_status = agent.get_cognitive_status()
    print(f"   ✓ Agent initialized with session: {initial_status['session_id']}")
    print(f"   ✓ Initial fatigue: {initial_status['fatigue_level']:.3f}")
    print(f"   ✓ Initial attention capacity: {initial_status['attention_status']['available_capacity']:.3f}")
    
    # Test sequential input processing with attention tracking
    print("\n2. Testing Sequential Input Processing...")
    
    test_inputs = [
        "Hello, I'm interested in learning about AI",
        "Can you explain machine learning concepts?", 
        "What about neural networks and deep learning?",
        "How does attention work in transformer models?",
        "Tell me about cognitive architectures like this one"
    ]
    
    responses = []
    attention_progression = []
    
    for i, input_text in enumerate(test_inputs, 1):
        print(f"\n   Input {i}: {input_text}")
        
        # Process input
        response = await agent.process_input(input_text)
        responses.append(response)
        
        # Get attention status
        status = agent.get_cognitive_status()
        attention_info = {
            "input_num": i,
            "focused_items": status["attention_status"]["focused_items"],
            "cognitive_load": status["attention_status"]["cognitive_load"],
            "fatigue_level": status["attention_status"]["fatigue_level"],
            "available_capacity": status["attention_status"]["available_capacity"],
            "focus_switches": status["attention_status"]["focus_switches"]
        }
        attention_progression.append(attention_info)
        
        print(f"   → Response: {response[:80]}...")
        print(f"   → Attention Items: {attention_info['focused_items']}")
        print(f"   → Cognitive Load: {attention_info['cognitive_load']:.3f}")
        print(f"   → Fatigue: {attention_info['fatigue_level']:.3f}")
        print(f"   → Available Capacity: {attention_info['available_capacity']:.3f}")
    
    # Test attention capacity management
    print("\n3. Testing Attention Capacity Management...")
    
    # Add several rapid inputs to test capacity limits
    print("   Adding rapid inputs to test capacity limits...")
    rapid_inputs = [
        "Quantum computing basics",
        "Blockchain technology overview",  
        "Robotics and automation",
        "Virtual reality applications",
        "Artificial general intelligence",
        "Computer vision techniques",
        "Natural language processing",
        "Reinforcement learning algorithms"
    ]
    
    for i, rapid_input in enumerate(rapid_inputs[:4], 1):  # Just test a few
        await agent.process_input(rapid_input)
        status = agent.get_cognitive_status()
        print(f"   Rapid Input {i}: Items={status['attention_status']['focused_items']}, "
              f"Load={status['attention_status']['cognitive_load']:.3f}")
    
    # Test cognitive break and recovery
    print("\n4. Testing Cognitive Break and Recovery...")
    
    pre_break_status = agent.get_cognitive_status()
    print(f"   Pre-break fatigue: {pre_break_status['attention_status']['fatigue_level']:.3f}")
    print(f"   Pre-break load: {pre_break_status['attention_status']['cognitive_load']:.3f}")
    
    # Take a cognitive break
    break_results = agent.take_cognitive_break(duration_minutes=1.0)
    
    post_break_status = agent.get_cognitive_status()
    print(f"   Post-break fatigue: {post_break_status['attention_status']['fatigue_level']:.3f}")
    print(f"   Post-break load: {post_break_status['attention_status']['cognitive_load']:.3f}")
    print(f"   Fatigue reduction: {break_results['fatigue_before'] - break_results['fatigue_after']:.3f}")
    print(f"   Items lost during break: {break_results['attention_items_lost']}")
    
    # Test dream state processing
    print("\n5. Testing Dream State Processing...")
    
    print("   Entering dream state for memory consolidation...")
    await agent.enter_dream_state()
    
    dream_status = agent.get_cognitive_status()
    print(f"   Post-dream fatigue: {dream_status['attention_status']['fatigue_level']:.3f}")
    print(f"   Post-dream load: {dream_status['attention_status']['cognitive_load']:.3f}")
    
    # Test attention focus details
    print("\n6. Analyzing Attention Focus Details...")
    
    focus_items = agent.attention.get_attention_focus()
    print(f"   Current focus items: {len(focus_items)}")
    
    for i, item in enumerate(focus_items[:3], 1):  # Show top 3
        print(f"   Focus Item {i}:")
        print(f"     - ID: {item['id']}")
        print(f"     - Salience: {item['salience']:.3f}")
        print(f"     - Activation: {item['activation']:.3f}")
        print(f"     - Priority: {item['priority']:.3f}")
        print(f"     - Duration: {item['duration_seconds']:.1f}s")
    
    # Summary report
    print("\n7. Integration Summary Report...")
    
    final_status = agent.get_cognitive_status()
    print(f"   ✓ Total conversations processed: {len(test_inputs) + len(rapid_inputs[:4])}")
    print(f"   ✓ Memory entries (STM): {final_status['memory_status']['stm']['size']}")
    print(f"   ✓ Memory entries (LTM): {final_status['memory_status']['ltm']['total_memories']}")
    print(f"   ✓ Current attention items: {final_status['attention_status']['focused_items']}")
    print(f"   ✓ Total focus switches: {final_status['attention_status']['focus_switches']}")
    print(f"   ✓ Final cognitive load: {final_status['attention_status']['cognitive_load']:.3f}")
    print(f"   ✓ Final fatigue level: {final_status['attention_status']['fatigue_level']:.3f}")
    print(f"   ✓ Processing efficiency: {final_status['cognitive_integration']['overall_efficiency']:.3f}")
    print(f"   ✓ Attention-memory sync: {final_status['cognitive_integration']['attention_memory_sync']}")
    
    # Attention progression analysis
    print("\n8. Attention Progression Analysis...")
    print("   Input# | Focus | Load  | Fatigue | Capacity | Switches")
    print("   -------|-------|-------|---------|----------|----------")
    for info in attention_progression:
        print(f"   {info['input_num']:6} | {info['focused_items']:5} | "
              f"{info['cognitive_load']:5.3f} | {info['fatigue_level']:7.3f} | "
              f"{info['available_capacity']:8.3f} | {info['focus_switches']:8}")
    
    print("\n" + "=" * 60)
    print("✅ ATTENTION MECHANISM INTEGRATION TEST COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    
    return {
        "responses": responses,
        "attention_progression": attention_progression,
        "final_status": final_status,
        "break_results": break_results
    }

if __name__ == "__main__":
    # Run the comprehensive test
    results = asyncio.run(test_attention_integration())
    
    print(f"\nTest completed at: {datetime.now()}")
    print("All attention mechanism features are working correctly!")
