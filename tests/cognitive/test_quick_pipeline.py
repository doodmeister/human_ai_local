#!/usr/bin/env python3
"""
Quick Pipeline Test - Simple version to verify integration
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath('.'))

from src.core.cognitive_agent import CognitiveAgent

async def quick_test():
    """Quick test of the cognitive pipeline"""
    print("🧠 QUICK PIPELINE TEST")
    print("=" * 40)
    
    # Initialize
    agent = CognitiveAgent()
    print(f"✅ Agent: {agent.session_id}")
    
    # Test single input
    test_input = "Hello, I'm interested in AI!"
    print(f"\nProcessing: {test_input}")
    
    response = await agent.process_input(test_input)
    print(f"Response: {response}")
    
    # Check status
    status = agent.get_cognitive_status()
    print("\nStatus:")
    print(f"  Fatigue: {status['fatigue_level']:.3f}")
    print(f"  Memories: {status['memory_status']['ltm']['total_memories']}")
    print(f"  Sensory: {status['sensory_processing']['total_processed']}")
    
    print("\n✅ PIPELINE WORKING!")
    return True

if __name__ == "__main__":
    asyncio.run(quick_test())
