import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime, timedelta
from src.memory.ltm.vector_ltm import VectorLongTermMemory

def test_ltm_consolidation_tracking():
    """Test consolidation tracking functionality in LTM"""
    ltm = VectorLongTermMemory()
    
    # Store and consolidate STM items using ltm.consolidate_from_stm()
    class STMItem:
        def __init__(self, id, content, importance=0.5, emotional_valence=0.0, access_count=0, associations=None):
            self.id = id
            self.content = content
            self.importance = importance
            self.emotional_valence = emotional_valence
            self.access_count = access_count
            self.associations = associations or []
    
    stm_item1 = STMItem("stm1", "episodic event 1", importance=0.7, access_count=3)
    stm_item2 = STMItem("stm2", "episodic event 2", importance=0.4, access_count=1)
    consolidated = ltm.consolidate_from_stm([stm_item1, stm_item2])
    
    assert consolidated == 1
    rec1 = ltm.retrieve("stm1")
    rec2 = ltm.retrieve("stm2")
    assert rec1 is not None, "STM item 1 should be in LTM after consolidation"
    assert rec2 is None, "STM item 2 should not be in LTM (below thresholds)"
    
    print("✅ Consolidation tracking test passed!")
    print(f"   - STM item 1 consolidated: {rec1 is not None}")
    print(f"   - STM item 2 consolidated: {rec2 is not None}")

if __name__ == "__main__":
    test_ltm_consolidation_tracking()
