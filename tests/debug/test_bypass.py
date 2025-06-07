import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Testing with bypass...", flush=True)

# Import the components we need
from src.memory.stm.short_term_memory import ShortTermMemory
from src.core.config import CognitiveConfig

print("Imports done", flush=True)

# Create just STM
stm = ShortTermMemory(capacity=20, decay_threshold=0.3)
print("STM created", flush=True)

# Create config
config = CognitiveConfig()
print("Config created", flush=True)

print("All tests passed!", flush=True)
