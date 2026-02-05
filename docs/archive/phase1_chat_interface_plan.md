# Phase 1 Chat Interface Implementation Plan

## 1. Architecture Additions
- ConversationSession: holds turns, metadata, rolling context window (configurable max turns, size budget).
- ContextBuilder: staged pipeline
  1. RecentTurnsSelector
  2. STMSemanticSelector
  3. LTMSemanticSelector
  4. EpisodicAnchorSelector
  5. AttentionFocusInjector
  6. ExecutiveModeAnnotator
  (Each stage returns items + stage metrics)

## 2. Data Schemas (proposed)
TurnRecord {
  id, role(user|assistant|system), content, timestamp,
  salience, emotional_valence, importance, embedding_ref, consolidation_status
}

ContextItem {
  source_id, source_system(stm|ltm|episodic|attention|turn),
  rank, scores { similarity, activation, recency, salience, decay_factor },
  reason, included(bool)
}

Trace {
  pipeline_stages: [
    { name, candidates_in, candidates_out, latency_ms, rationale }
  ],
  scoring_version
}

Metrics {
  turn_latency_ms, retrieval_time_ms, stm_hits, ltm_hits,
  attention_boost, fatigue_delta, consolidation_time_ms
}

## 3. Inclusion Logic (initial thresholds)
- STM: top K (<=4) by activation where activation >= 0.15
- LTM: similarity >= 0.62; cap 3; penalize age via exp(-age_hours * λ)
- Episodic: max 2 recent related by temporal_window (default 6h) or tag overlap
- Attention focus: always include currently focused items (mark forced=True)
- Deduplicate by normalized content hash

## 4. Consolidation Hook
On assistant response:
- Combine user + assistant turn into interaction summary string
- Score importance = max(user.salience, average attention weight * 0.5)
- If importance >= 0.55 or emotional |valence| >= 0.6 => enqueue for STM storage
- Periodic background task promotes to LTM using existing consolidation pipeline

## 5. Fallback Retrieval
If vector search raises or times out (>300ms):
- Tokenize query; compute word overlap score with recent LTM/STM textual fields
- Select top N by overlap; mark provenance: strategy="fallback_word_overlap"

## 6. Streaming Strategy
- First chunk: metadata { session_id, trace_seed_id }
- Subsequent chunks: tokens
- Final chunk: full trace + metrics
- Timeout guard: if generation > configured soft limit emit partial trace + graceful stop

## 7. Observability
Expose metrics (Prometheus style):
- chat_turn_latency_ms (histogram)
- chat_context_items
- chat_retrieval_failures_total
- chat_vector_fallback_total
- chat_consolidations_total
- chat_attention_boost_applied_total

## 8. Error Handling
GracefulError(Exception) wrapper at endpoint; returns structured error payload with degraded_mode flag.

## 9. Testing Matrix
| Test | Focus |
|------|-------|
| test_context_builder_determinism | deterministic ordering |
| test_session_eviction_policy | proper removal when max sessions reached |
| test_vector_fallback_path | fallback triggers & labels provenance |
| test_attention_integration_turn | attention load & focus updated |
| test_consolidation_trigger | correct thresholds drive storage |
| test_streaming_response_chunks | ordering & final trace emission |
| test_performance_p95_latency | performance budget compliance |

## 10. Incremental Delivery Order
1. ConversationSession + ContextBuilder (non-stream)
2. Provenance + metrics
3. Streaming responses
4. Fallback retrieval
5. Emotional/salience tagging & consolidation integration
6. Streamlit panels (Chat + Trace)
7. Remaining panels (Attention, STM snapshot, Executive)
8. Performance tuning & resilience tests

## 11. Configuration (add to config.py)
chat:
  max_recent_turns: 8
  max_context_items: 16
  ltm_similarity_threshold: 0.62
  stm_activation_min: 0.15
  fallback_timeout_ms: 300
  streaming: enabled
  performance_targets:
    turn_latency_ms_p95: 1000

## 12. Risks & Mitigations
- Latency creep → stage timing logging + abort thresholds
- Memory bloat (many sessions) → LRU session eviction
- Overfitting retrieval → scoring versioning & A/B toggle flags
- Trace verbosity → client flag include_trace

## 13. Success Metrics
- p95 turn latency < 1s (baseline hardware)
- ≥70% of included items show user-validated relevance (future user feedback loop)
- Fallback rate <3% steady-state
- Zero unhandled exceptions in 500-turn soak test

(End of Plan)
