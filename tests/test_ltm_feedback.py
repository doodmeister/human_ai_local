import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from src.memory.ltm.long_term_memory import LongTermMemory
import uuid

def test_ltm_feedback():
    ltm = LongTermMemory(storage_path="data/memory_stores/ltm_test_feedback")
    ltm.memories.clear()
    # Store a memory
    memory_id = str(uuid.uuid4())
    ltm.store(memory_id=memory_id, content="feedback test memory", memory_type="semantic", importance=0.5)
    # Add feedback
    ltm.add_feedback(memory_id, feedback_type="relevance", value=4, comment="Useful", user_id="user1")
    ltm.add_feedback(memory_id, feedback_type="importance", value=5, comment="Critical info", user_id="user2")
    ltm.add_feedback(memory_id, feedback_type="emotion", value=0.8, comment="Positive feeling", user_id="user3")
    # Retrieve feedback
    feedback = ltm.get_feedback(memory_id)
    assert len(feedback) == 3, f"Expected 3 feedback events, got {len(feedback)}"
    # Check summary
    summary = ltm.get_feedback_summary(memory_id)
    assert summary["relevance"] == 4.0, f"Expected relevance 4.0, got {summary['relevance']}"
    assert summary["importance"] == 5.0, f"Expected importance 5.0, got {summary['importance']}"
    assert abs(summary["emotion"] - 0.8) < 1e-6, f"Expected emotion 0.8, got {summary['emotion']}"
    assert summary["count"] == 3, f"Expected count 3, got {summary['count']}"
    # Check memory fields updated
    record = ltm.memories[memory_id]
    assert abs(record.confidence - 0.8) < 1e-6, f"Expected confidence 0.8, got {record.confidence}"
    assert abs(record.importance - 1.0) < 1e-6, f"Expected importance 1.0, got {record.importance}"
    assert abs(record.emotional_valence - 0.8) < 1e-6, f"Expected emotional_valence 0.8, got {record.emotional_valence}"
    print("LTM feedback test passed!")

if __name__ == "__main__":
    test_ltm_feedback()
