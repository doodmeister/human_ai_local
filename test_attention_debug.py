#!/usr/bin/env python3
"""
Quick debug test for attention score integration
"""
import sys
import os
import asyncio
sys.path.insert(0, os.path.abspath('.'))

from src.core.cognitive_agent import CognitiveAgent

async def debug_attention_scores():
    """Debug attention score integration"""
    print("🔍 Debugging Attention Score Integration...")
    
    agent = CognitiveAgent()
    
    # Process a single input
    print("Processing test input...")
    response = await agent.process_input("Hello, this is a test message!")
    print(f"Response: {response}")
    
    # Get detailed status
    status = agent.get_cognitive_status()
    
    # Display all available status information
    print("\n📊 FULL STATUS BREAKDOWN:")
    
    for key, value in status.items():
        print(f"\n{key.upper()}:")
        if isinstance(value, dict):
            for subkey, subvalue in value.items():
                print(f"  {subkey}: {subvalue}")
        else:
            print(f"  {value}")
    
    await agent.shutdown()
    print("Debug completed!")

if __name__ == "__main__":
    asyncio.run(debug_attention_scores())
