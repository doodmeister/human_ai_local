import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.memory.ltm.vector_ltm import VectorLongTermMemory
from datetime import datetime, timedelta

def test_ltm_decay():
    ltm = VectorLongTermMemory()
    # Store a memory with old last_access
    ltm.store(memory_id="decay1", content="old memory", importance=0.8, emotional_valence=0.1)
    # Simulate old last_access by updating metadata if collection is available
    rec = ltm.retrieve("decay1")
    if rec and getattr(ltm, 'collection', None):
        old_time = (datetime.now() - timedelta(days=40)).isoformat()
        rec["last_access"] = old_time
        # Sanitize metadata: convert lists to comma-separated strings, ensure allowed types
        meta = rec.copy()
        for key in list(meta.keys()):
            v = meta[key]
            if isinstance(v, list):
                meta[key] = ",".join(str(x) for x in v) if v else ""
            elif not isinstance(v, (str, int, float, bool)) and v is not None:
                meta[key] = str(v)
        if ltm.collection is not None:
            ltm.collection.update(ids=["decay1"], metadatas=[meta])
    # Decay
    decayed = ltm.decay_memories()
    assert decayed >= 1
    rec2 = ltm.retrieve("decay1")
    assert rec2 is not None
    assert rec2["importance"] < 0.8

if __name__ == "__main__":
    test_ltm_decay()
