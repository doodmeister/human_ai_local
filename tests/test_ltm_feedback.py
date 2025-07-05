import sys
import os
sys.path.insert(0, os.path.abspath('.'))
from src.memory.ltm.vector_ltm import VectorLongTermMemory
import uuid

def test_ltm_feedback():
    ltm = VectorLongTermMemory()
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
    rec = ltm.retrieve(memory_id)
    assert rec is not None, "Memory should exist after feedback"
    assert abs(rec.get("confidence", 0.0) - 0.8) < 1e-6, f"Expected confidence 0.8, got {rec.get('confidence')}"
    assert abs(rec.get("importance", 0.0) - 1.0) < 1e-6, f"Expected importance 1.0, got {rec.get('importance')}"
    assert abs(rec.get("emotional_valence", 0.0) - 0.8) < 1e-6, f"Expected emotional_valence 0.8, got {rec.get('emotional_valence')}"
    print("LTM feedback test passed!")

if __name__ == "__main__":
    test_ltm_feedback()
