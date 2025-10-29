from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from .executive_core import MemoryService, AttentionService, LLMService, ActuatorService

"""Adapter classes mapping existing subsystems to ExecutiveController Protocols.

These are intentionally thin and defensive. They should be safe to use even
when certain optional subsystems (semantic memory, actuator tools) are absent.

Each adapter confines translation logic (method names, return shapes) so the
executive layer remains stable even if underlying implementations evolve.
"""


class MemoryAdapter(MemoryService):
    """Wraps the existing MemorySystem (and/or Consolidator) to satisfy MemoryService."""

    def __init__(self, memory_system: Any):
        self.ms = memory_system

    async def stm_add(self, text: str, tags: Optional[List[str]] = None) -> None:  # type: ignore[override]
        try:
            stm = getattr(self.ms, "stm", None)
            if stm is not None and hasattr(stm, "store"):
                stm.store(content=text, metadata={"tags": tags or []})  # legacy vector STM interface
            elif hasattr(self.ms, "store_memory"):
                # Fallback generic storage; produce a synthetic ID
                self.ms.store_memory(memory_id=f"stm::{len(text)}", content=text, importance=0.4)  # type: ignore
        except Exception:
            pass  # Non-fatal

    async def stm_recent(self, k: int = 10) -> List[str]:  # type: ignore[override]
        try:
            stm = getattr(self.ms, "stm", None)
            if stm is not None:
                def getter():  # type: ignore
                    return []
                if hasattr(stm, "get_all_memories"):
                    getter = stm.get_all_memories  # type: ignore
                elif hasattr(stm, "memories"):
                    def getter():  # type: ignore
                        return getattr(stm, "memories")
                raw = getter()
                items: List[str] = []
                for it in raw[-k:]:  # assume list-like ordering by recency
                    if isinstance(it, dict):
                        items.append(str(it.get("content", ""))[:400])
                    else:
                        content = getattr(it, "content", None)
                        if content:
                            items.append(str(content)[:400])
                return items
            return []
        except Exception:
            return []

    async def ltm_similar(self, query: str, k: int = 8) -> List[Tuple[str, float]]:  # type: ignore[override]
        try:
            ltm = getattr(self.ms, "ltm", None)
            if ltm is not None and hasattr(ltm, "search"):
                results = ltm.search(query)  # type: ignore
                out: List[Tuple[str, float]] = []
                for r in results[:k]:
                    if isinstance(r, dict):
                        content = r.get("content") or r.get("text") or r.get("item")
                        score = r.get("similarity") or r.get("similarity_score") or r.get("confidence") or 0.0
                        if content:
                            out.append((str(content)[:500], float(score)))
                    elif isinstance(r, tuple) and r:
                        content = r[0]
                        score = float(r[1]) if len(r) > 1 else 0.0
                        out.append((str(content)[:500], score))
                return out
        except Exception:
            pass
        return []

    async def ltm_add(self, text: str, tags: Optional[List[str]] = None) -> str:  # type: ignore[override]
        try:
            ltm = getattr(self.ms, "ltm", None)
            if ltm is not None and hasattr(ltm, "store"):
                res = ltm.store(content=text, tags=tags or [])  # type: ignore
                return str(res)
            elif hasattr(self.ms, "store_memory"):
                self.ms.store_memory(memory_id=f"ltm::{len(text)}", content=text, importance=0.7)  # type: ignore
                return f"ltm::{len(text)}"
        except Exception:
            pass
        return "ltm::failed"

    async def consolidate(self, items: List[str]) -> str:  # type: ignore[override]
        try:
            if hasattr(self.ms, "consolidate_memories"):
                stats = self.ms.consolidate_memories(force=True)  # type: ignore
                return f"consolidated:{stats.get('consolidated_count', 0)}"
        except Exception:
            pass
        return "consolidated:0"


class AttentionAdapter(AttentionService):
    """Simple top-k passthrough; optionally wrap AttentionMechanism later."""

    def __init__(self, attention_mechanism: Any | None = None):
        self.att = attention_mechanism

    async def allocate(self, *, query: str, candidates: List[str], capacity: int = 5) -> List[str]:  # type: ignore[override]
        if not candidates:
            return []
        # If attention mechanism exists, you could score; for now, just slice.
        return candidates[:capacity]


class LLMAdapter(LLMService):
    """Wraps an LLM client; placeholder deterministic response if none provided."""

    def __init__(self, client: Any | None = None):
        self.client = client

    async def complete(self, prompt: str, context: List[str], max_tokens: int = 512) -> Tuple[str, Dict[str, Any]]:  # type: ignore[override]
        if self.client is None:
            # Deterministic fallback
            text = f"(deterministic) {prompt.splitlines()[0][:120]} | ctx={len(context)}"
            return text, {"tokens_used": min(max_tokens, 64), "fatigue": 0.25}
        # Expect client has a 'generate' or 'complete' method; attempt both.
        try:
            if hasattr(self.client, "generate"):
                resp = self.client.generate(prompt=prompt, context=context, max_tokens=max_tokens)  # type: ignore
            elif hasattr(self.client, "complete"):
                resp = self.client.complete(prompt=prompt, context=context, max_tokens=max_tokens)  # type: ignore
            else:
                text = f"(no-llm-method) {prompt[:60]}"
                return text, {"tokens_used": 0, "fatigue": 0.3}
            if isinstance(resp, tuple) and len(resp) == 2:
                return resp  # (text, meta)
            if isinstance(resp, dict):
                return str(resp.get("text", "")), resp
            return str(resp), {"tokens_used": 0}
        except Exception:
            text = f"(llm-error-fallback) {prompt[:80]}"
            return text, {"error": True, "tokens_used": 0, "fatigue": 0.4}


class ActuatorAdapter(ActuatorService):
    """Tool/action stub; extend with real tools later."""

    async def act(self, action_text: str) -> Dict[str, Any]:  # type: ignore[override]
        return {"executed": False, "action": action_text, "reason": "no-op"}
