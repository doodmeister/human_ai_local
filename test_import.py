"""
Simple import test
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Starting import test...")

try:
    print("1. Importing CognitiveConfig...")
    from src.core.config import CognitiveConfig
    print("✅ CognitiveConfig imported")
    
    print("2. Creating config...")
    config = CognitiveConfig()
    print("✅ Config created")
    
    print("3. Importing CognitiveAgent...")
    from src.core.cognitive_agent import CognitiveAgent
    print("✅ CognitiveAgent imported")
    
    print("4. Creating agent...")
    agent = CognitiveAgent(config)
    print("✅ Agent created")
    
    print("\n🎉 All imports and initialization successful!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
