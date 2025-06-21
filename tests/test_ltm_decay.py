import time
from datetime import datetime, timedelta
from src.memory.ltm.long_term_memory import LongTermMemory
import uuid

def test_ltm_decay():
    ltm = LongTermMemory(storage_path="data/memory_stores/ltm_test_decay")
    ltm.memories.clear()  # Ensure clean state
    # Create a memory with last_access 40 days ago
    memory_id = str(uuid.uuid4())
    content = "decay test memory"
    record_time = datetime.now() - timedelta(days=40)
    ltm.store(
        memory_id=memory_id,
        content=content,
        memory_type="episodic",
        importance=0.8,
        emotional_valence=0.0,
        source="test",
        tags=["decay"],
        associations=[]
    )
    # Manually set last_access and encoding_time to 40 days ago
    ltm.memories[memory_id].last_access = record_time
    ltm.memories[memory_id].encoding_time = record_time
    ltm.memories[memory_id].importance = 0.8
    ltm.memories[memory_id].confidence = 1.0
    # Run decay
    decayed = ltm.decay_memories(decay_rate=0.1, half_life_days=30.0)
    assert decayed >= 1
    decayed_record = ltm.memories[memory_id]
    assert decayed_record.importance < 0.8, f"Importance did not decay: {decayed_record.importance}"
    assert decayed_record.confidence < 1.0, f"Confidence did not decay: {decayed_record.confidence}"
    print(f"Decay test passed: importance={decayed_record.importance}, confidence={decayed_record.confidence}")

if __name__ == "__main__":
    test_ltm_decay()
