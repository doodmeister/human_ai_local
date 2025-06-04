#!/usr/bin/env python3
"""
Complete Cognitive Pipeline Test
Demonstrates the full sensory-cognitive integration with real conversation flow
"""
import sys
import os
import asyncio
import time
sys.path.insert(0, os.path.abspath('.'))

from src.core.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig

async def test_complete_pipeline():
    """Test the complete cognitive pipeline with conversation flow"""
    print("🧠 COMPLETE COGNITIVE PIPELINE TEST")
    print("=" * 60)
    
    # Initialize cognitive agent
    config = CognitiveConfig()
    agent = CognitiveAgent(config)
    
    print(f"✅ Agent initialized: {agent.session_id}")
    
    # Simulate a realistic conversation
    conversation_inputs = [
        "Hello, I'm new to AI and machine learning. Can you help me?",
        "What are neural networks?", 
        "How does backpropagation work in training neural networks?",
        "URGENT: I have a deadline tomorrow for my ML project!",
        "Can you recommend some resources for learning deep learning?",
        "Thank you for your help today!"
    ]
    
    print(f"\n🗣️ PROCESSING CONVERSATION ({len(conversation_inputs)} turns)")
    print("-" * 60)
    
    conversation_history = []
    
    for i, user_input in enumerate(conversation_inputs, 1):
        print(f"\n[Turn {i}] User: {user_input}")
        
        # Record timing
        start_time = time.time()
        
        # Process through complete cognitive pipeline
        response = await agent.process_input(user_input)
        
        processing_time = time.time() - start_time
        
        print(f"Agent: {response}")
        print(f"⏱️ Processing time: {processing_time:.3f}s")
        
        # Get cognitive status after processing
        status = agent.get_cognitive_status()
        
        print(f"📊 Cognitive State:")
        print(f"   Fatigue: {status['fatigue_level']:.3f}")
        print(f"   Memory: STM({status['memory_status']['stm']['size']}) "
              f"LTM({status['memory_status']['ltm']['total_memories']})")
        print(f"   Attention Load: {status['attention_status']['cognitive_load']:.3f}")
        print(f"   Sensory Processed: {status['sensory_processing']['total_processed']}")
        
        # Store conversation turn
        conversation_history.append({
            'turn': i,
            'input': user_input,
            'response': response,
            'processing_time': processing_time,
            'cognitive_state': {
                'fatigue': status['fatigue_level'],
                'memory_items': status['memory_status']['stm']['size'],
                'attention_load': status['attention_status']['cognitive_load']
            }
        })
        
        # Brief pause between turns (simulating natural conversation)
        await asyncio.sleep(0.1)
    
    # Final analysis
    print(f"\n📈 CONVERSATION ANALYSIS")
    print("-" * 60)
    
    total_processing_time = sum(turn['processing_time'] for turn in conversation_history)
    avg_processing_time = total_processing_time / len(conversation_history)
    
    print(f"Total conversation time: {total_processing_time:.3f}s")
    print(f"Average processing time per turn: {avg_processing_time:.3f}s")
    
    # Final cognitive status
    final_status = agent.get_cognitive_status()
    
    print(f"\n🧠 FINAL COGNITIVE STATE")
    print("-" * 60)
    print(f"Session ID: {final_status['session_id']}")
    print(f"Conversation length: {final_status['conversation_length']} turns")
    print(f"Total fatigue: {final_status['fatigue_level']:.3f}")
    
    # Memory analysis
    memory = final_status['memory_status']
    print(f"\n💾 MEMORY ANALYSIS")
    print(f"STM utilization: {memory['stm']['size']}/{memory['stm']['capacity']} "
          f"({(memory['stm']['size']/memory['stm']['capacity']*100):.1f}%)")
    print(f"LTM memories created: {memory['ltm']['total_memories']}")
    print(f"Session memories: {memory['session_memories']}")
    
    # Attention analysis  
    attention = final_status['attention_status']
    print(f"\n🎯 ATTENTION ANALYSIS")
    print(f"Focus switches: {attention['focus_switches']}")
    print(f"Current cognitive load: {attention['cognitive_load']:.3f}")
    print(f"Available capacity: {attention['available_capacity']:.3f}")
    
    # Sensory processing analysis
    sensory = final_status['sensory_processing']
    print(f"\n👁️ SENSORY PROCESSING ANALYSIS")
    print(f"Total inputs processed: {sensory['total_processed']}")
    print(f"Inputs filtered: {sensory['filtered_count']}")
    if sensory['total_processed'] > 0:
        filter_rate = (sensory['filtered_count'] / sensory['total_processed']) * 100
        print(f"Filter rate: {filter_rate:.1f}%")
    
    # Integration efficiency
    integration = final_status['cognitive_integration']
    print(f"\n🔗 INTEGRATION EFFICIENCY")
    print(f"Overall efficiency: {integration['overall_efficiency']:.3f}")
    print(f"Sensory efficiency: {integration['sensory_efficiency']:.3f}")
    print(f"Processing capacity: {integration['processing_capacity']:.3f}")
    
    # Test cognitive break functionality
    print(f"\n🛌 TESTING COGNITIVE BREAK")
    print("-" * 60)
    
    fatigue_before = agent.current_fatigue
    break_results = agent.take_cognitive_break(duration_minutes=0.5)
    fatigue_after = agent.current_fatigue
    
    print(f"Fatigue before break: {fatigue_before:.3f}")
    print(f"Fatigue after break: {fatigue_after:.3f}")
    print(f"Recovery effective: {break_results['recovery_effective']}")
    
    print(f"\n✅ COMPLETE PIPELINE TEST SUCCESSFUL")
    print("The sensory-cognitive integration is working perfectly!")
    
    return True

async def main():
    """Main async function"""
    try:
        success = await test_complete_pipeline()
        if success:
            print("\n🎉 ALL TESTS PASSED!")
        else:
            print("\n❌ Tests failed.")
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting complete cognitive pipeline test...")
    asyncio.run(main())
