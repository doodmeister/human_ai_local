"""
Phase 1 Chat Interface Implementation Plan (Scaffold)

1. Components:
   - ConversationSession / SessionManager
   - ContextBuilder (stages: recent, stm, ltm, episodic, attention, executive)
   - Provenance & metrics data models
2. Thresholds (initial):
   - stm_activation_min = 0.15
   - ltm_similarity_threshold = 0.62
3. Metrics:
   turn_latency_ms, retrieval_time_ms, stm_hits, ltm_hits, episodic_hits,
   attention_boost, fatigue_delta, consolidation_time_ms, fallback_used
4. Fallback:
   Word overlap (token Jaccard) if vector retrieval timeout (>400ms) or exception.
5. Testing (see tests/test_chat_interface_pipeline.py):
   Determinism, fallback path, consolidation trigger, streaming protocol, performance.
6. Delivery Order:
   Session + context scaffold -> provenance scoring -> real retrieval integration ->
   attention/executive enrichment -> fallback -> streaming -> metrics/perf/resilience.
"""
