"""
Debug test for cognitive agent - step by step
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Starting step-by-step debug test...")

try:
    print("1. Testing basic Python functionality...")
    x = 1 + 1
    print(f"✅ Basic math works: 1 + 1 = {x}")
    
    print("2. Importing CognitiveConfig...")
    from src.core.config import CognitiveConfig
    print("✅ CognitiveConfig imported successfully")
    
    print("3. Creating config...")
    config = CognitiveConfig()
    print("✅ Config created successfully")
    
    print("4. Importing CognitiveAgent...")
    from src.core.cognitive_agent import CognitiveAgent
    print("✅ CognitiveAgent imported successfully")
    
    print("5. Creating cognitive agent...")
    agent = CognitiveAgent(config)
    print(f"✅ Cognitive agent created with session: {agent.session_id}")
    
    print("\n🎉 All imports and initialization successful!")
    
except Exception as e:
    print(f"❌ Error at step: {e}")
    import traceback
    traceback.print_exc()
