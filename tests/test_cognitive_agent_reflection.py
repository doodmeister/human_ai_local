"""
Test for CognitiveAgent metacognitive reflection scheduler and reporting
"""
import time
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.core.cognitive_agent import CognitiveAgent

def test_manual_reflection():
    agent = CognitiveAgent()
    report = agent.manual_reflect()
    assert isinstance(report, dict)
    assert "timestamp" in report
    assert "ltm_status" in report
    assert "stm_status" in report
    # Reflection report should be stored
    assert agent.reflection_reports[-1] == report

def test_reflection_scheduler():
    agent = CognitiveAgent()
    # Simulate periodic reflection directly for test speed
    for _ in range(3):
        agent.reflect()
        time.sleep(0.01)
    # Should have at least 2 reflection reports
    assert len(agent.reflection_reports) >= 2
    # Reports should have timestamps
    for r in agent.reflection_reports:
        assert "timestamp" in r
        assert "ltm_status" in r
        assert "stm_status" in r
