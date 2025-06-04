import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Starting...", flush=True)

from src.core.config import CognitiveConfig
print("Config imported", flush=True)

from src.core.cognitive_agent import CognitiveAgent
print("Agent imported", flush=True)

config = CognitiveConfig()
print("Config created", flush=True)

agent = CognitiveAgent(config)
print("Agent created", flush=True)

status = agent.get_cognitive_status()
print(f"Status: {status}", flush=True)

print("Done!", flush=True)
