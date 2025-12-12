import sys
import os
import tempfile
import shutil
import atexit
sys.path.insert(0, os.path.abspath('.'))

print("Testing with bypass...", flush=True)

# Import the components we need
from src.memory.stm.vector_stm import VectorShortTermMemory, STMConfiguration
from src.core.config import CognitiveConfig

print("Imports done", flush=True)

# Create Vector STM with temporary directory that persists until process exit
temp_dir = tempfile.mkdtemp()

def cleanup():
    try:
        shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

atexit.register(cleanup)

config = STMConfiguration(
    chroma_persist_dir=os.path.join(temp_dir, "bypass_stm"),
    collection_name="bypass_test",
    capacity=20
)
stm = VectorShortTermMemory(config)
print("Vector STM created", flush=True)

try:
    # Create config
    cognitive_config = CognitiveConfig()
    print("Config created", flush=True)
    
    print("All tests passed!", flush=True)
finally:
    # Properly shutdown to release file locks
    stm.shutdown()
