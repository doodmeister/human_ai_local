"""
Test individual memory components
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

print("Testing individual memory component imports...")

try:
    print("1. Testing STM import...")
    from src.memory.stm.short_term_memory import ShortTermMemory
    print("✅ STM components imported")
    
    print("2. Testing LTM import...")
    from src.memory.ltm.long_term_memory import LongTermMemory
    print("✅ LTM components imported")
    
    print("3. Creating STM...")
    stm = ShortTermMemory(capacity=20, decay_threshold=0.3)
    print("✅ STM created")
    
    print("4. Creating LTM...")
    ltm = LongTermMemory(storage_path="data/memory_stores/ltm")
    print("✅ LTM created")
    
    print("\n🎉 Individual memory components work!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
