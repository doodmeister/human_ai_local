"""
Test memory system in isolation
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Testing memory system imports...")

try:
    print("1. Importing MemorySystem...")
    from src.memory.memory_system import MemorySystem
    print("✅ MemorySystem imported")
    
    print("2. Creating MemorySystem...")
    memory = MemorySystem(
        stm_capacity=20,
        stm_decay_threshold=0.3,
        ltm_storage_path="data/memory_stores/ltm"
    )
    print("✅ MemorySystem created")
    
    print("3. Testing memory operations...")
    status = memory.get_memory_status()
    print(f"✅ Memory status: {status}")
    
    print("\n🎉 Memory system works independently!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
