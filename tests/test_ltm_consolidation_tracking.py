import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from datetime import datetime, timedelta
from src.memory.ltm.long_term_memory import LongTermMemory
from src.memory.stm.short_term_memory import MemoryItem
import uuid

def test_consolidation_tracking():
    """Test consolidation tracking functionality in LTM"""
    ltm = LongTermMemory(storage_path="data/memory_stores/ltm_test_consolidation")
    ltm.memories.clear()  # Ensure clean state
    
    # Test 1: Direct storage (non-consolidated)
    memory_id1 = str(uuid.uuid4())
    ltm.store(
        memory_id=memory_id1,
        content="direct memory",
        memory_type="semantic",
        importance=0.7,
        source="manual"
    )
    
    record1 = ltm.memories[memory_id1]
    assert record1.consolidation_source == "direct", f"Expected 'direct', got {record1.consolidation_source}"
    assert record1.consolidated_at is None, "Direct storage should not have consolidated_at timestamp"
    
    # Test 2: STM consolidation
    # Create mock STM items
    stm_item1 = MemoryItem(
        id=str(uuid.uuid4()),
        content="stm memory 1",
        encoding_time=datetime.now() - timedelta(minutes=30),
        last_access=datetime.now() - timedelta(minutes=5),
        importance=0.7,  # Above threshold for consolidation
        access_count=3,  # Above threshold for consolidation
        emotional_valence=0.2
    )
    
    stm_item2 = MemoryItem(
        id=str(uuid.uuid4()),
        content="stm memory 2",
        encoding_time=datetime.now() - timedelta(hours=25),  # Old enough to not get recency bonus
        last_access=datetime.now() - timedelta(hours=24),
        importance=0.3,  # Well below threshold
        access_count=1,  # Below threshold
        emotional_valence=0.0
    )
    
    # Consolidate from STM
    consolidation_start = datetime.now()
    consolidated_count = ltm.consolidate_from_stm([stm_item1, stm_item2])
    consolidation_end = datetime.now()
    
    # Should only consolidate stm_item1 (meets thresholds)
    assert consolidated_count == 1, f"Expected 1 consolidated item, got {consolidated_count}"
    
    # Check consolidation tracking for consolidated item
    assert stm_item1.id in ltm.memories, "STM item 1 should be in LTM after consolidation"
    consolidated_record = ltm.memories[stm_item1.id]
    assert consolidated_record.consolidation_source == "stm", f"Expected 'stm', got {consolidated_record.consolidation_source}"
    assert consolidated_record.consolidated_at is not None, "Consolidated memory should have consolidated_at timestamp"
    assert consolidation_start <= consolidated_record.consolidated_at <= consolidation_end, "Consolidation timestamp should be within expected range"
    
    # Check that non-consolidated item is not in LTM
    assert stm_item2.id not in ltm.memories, "STM item 2 should not be in LTM (below thresholds)"
    
    # Test 3: Query recently consolidated memories
    recent_consolidated = ltm.get_recently_consolidated(hours=1, consolidation_source="stm")
    assert len(recent_consolidated) == 1, f"Expected 1 recently consolidated memory, got {len(recent_consolidated)}"
    assert recent_consolidated[0].id == stm_item1.id, "Recently consolidated memory should match STM item 1"
    
    # Test 4: Get consolidation stats
    stats = ltm.get_consolidation_stats()
    assert stats["total_consolidated"] == 1, f"Expected 1 total consolidated, got {stats['total_consolidated']}"
    assert stats["consolidation_sources"]["stm"] == 1, f"Expected 1 STM consolidation, got {stats['consolidation_sources'].get('stm', 0)}"
    assert stats["recent_24h"] == 1, f"Expected 1 recent consolidation, got {stats['recent_24h']}"
    assert stats["avg_consolidation_count"] >= 0, "Average consolidation count should be non-negative"
    
    print("✅ Consolidation tracking test passed!")
    print(f"   - Direct storage: {record1.consolidation_source}")
    print(f"   - STM consolidation: {consolidated_record.consolidation_source}")
    print(f"   - Consolidated at: {consolidated_record.consolidated_at}")
    print(f"   - Stats: {stats}")

if __name__ == "__main__":
    test_consolidation_tracking()
