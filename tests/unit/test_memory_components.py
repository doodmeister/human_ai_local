"""
Test individual memory components using the new Vector STM
"""
import sys
import os
import tempfile
sys.path.insert(0, os.path.abspath('.'))

print("Testing individual memory component imports...")

try:
    print("1. Testing Vector STM import...")
    from src.memory.stm.vector_stm import VectorShortTermMemory, STMConfiguration
    print("✅ Vector STM components imported")
    
    print("2. Testing LTM import...")
    from src.memory.ltm.vector_ltm import VectorLongTermMemory
    print("✅ LTM components imported")
    
    print("3. Creating Vector STM...")
    with tempfile.TemporaryDirectory() as temp_dir:
        config = STMConfiguration(
            chroma_persist_dir=os.path.join(temp_dir, "stm_test"),
            collection_name="test_components",
            capacity=20
        )
        stm = VectorShortTermMemory(config)
        print("✅ Vector STM created")
        
        print("4. Creating LTM...")
        ltm = VectorLongTermMemory()
        print("✅ LTM created")
        
        # Properly shutdown to release locks
        stm.shutdown()
        
        print("\n🎉 Individual memory components work!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
