from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple


logger = logging.getLogger(__name__)


class MemoryRecallService:
    def __init__(
        self,
        *,
        is_initialized: Callable[[], bool],
        get_stm: Callable[[], Any],
        get_ltm: Callable[[], Any],
        get_episodic: Callable[[], Any],
        get_semantic: Callable[[], Any],
        use_vector_ltm: Callable[[], bool],
    ) -> None:
        self._is_initialized = is_initialized
        self._get_stm = get_stm
        self._get_ltm = get_ltm
        self._get_episodic = get_episodic
        self._get_semantic = get_semantic
        self._use_vector_ltm = use_vector_ltm

    def hierarchical_search(
        self,
        query: str,
        max_results: int = 10,
        max_per_system: int = 5,
    ) -> List[Tuple[Any, float, str]]:
        all_results = []

        try:
            stm_results = self._get_stm().search(query, max_results=max_per_system)
            for memory, score in stm_results:
                all_results.append((memory, score, "STM"))
        except Exception as exc:
            logger.error("Error searching STM for query '%s': %s", query, exc)

        try:
            ltm = self._get_ltm()
            if self._use_vector_ltm():
                ltm_results = getattr(ltm, "search_semantic", lambda **kwargs: [])(query=query, max_results=max_per_system)
                for memory in ltm_results:
                    score = memory.get("similarity_score", 0.0)
                    all_results.append((memory, score, "LTM"))
            else:
                ltm_results = getattr(ltm, "search_by_content", lambda **kwargs: [])(query=query, max_results=max_per_system)
                for memory, score in ltm_results:
                    all_results.append((memory, score, "LTM"))
        except Exception as exc:
            logger.error("Error searching LTM for query '%s': %s", query, exc)

        try:
            episodic_results = self._get_episodic().search_memories(query=query, limit=max_per_system)
            for result in episodic_results:
                all_results.append((result.memory, result.relevance, "Episodic"))
        except Exception as exc:
            logger.error("Error searching Episodic Memory for query '%s': %s", query, exc)

        try:
            semantic = self._get_semantic()
            subject_matches = semantic.find_facts(subject=query)
            object_matches = semantic.find_facts(object_val=query)
            for fact in subject_matches:
                all_results.append((fact, 0.9, "Semantic"))
            for fact in object_matches:
                all_results.append((fact, 0.8, "Semantic"))
        except Exception as exc:
            logger.error("Error searching Semantic Memory for query '%s': %s", query, exc)

        seen_content = set()
        unique_results = []
        for memory, score, source in sorted(all_results, key=lambda item: item[1], reverse=True):
            content_repr = ""
            if source == "Episodic":
                content_repr = repr(memory.detailed_content)
            elif source == "Semantic":
                content_repr = repr(memory)
            elif isinstance(memory, dict):
                content_repr = repr(memory.get("content"))
            elif hasattr(memory, "content"):
                content_repr = repr(memory.content)

            if content_repr and content_repr not in seen_content:
                unique_results.append((memory, score, source))
                seen_content.add(content_repr)

        return unique_results[:max_results]

    def proactive_recall(
        self,
        query: str,
        max_results: int = 5,
        min_relevance: float = 0.7,
        context_window: int = 3,
        use_ai_summary: bool = False,
        openai_client: Any = None,
    ) -> Dict[str, Any]:
        if not self._is_initialized():
            logger.warning("Memory system not initialized for proactive recall")
            return {"recalled_memories": [], "summary": "Memory system unavailable"}

        try:
            search_results = self.hierarchical_search(
                query=query,
                max_results=max_results * 2,
                max_per_system=max_results // 2,
            )

            recalled_memories = []
            for memory, score, source in search_results:
                if score >= min_relevance:
                    enhanced_memory = {
                        "content": self.extract_memory_content(memory, source),
                        "score": score,
                        "source": source,
                        "timestamp": self.extract_timestamp(memory, source),
                        "importance": self.extract_importance(memory, source),
                        "emotional_valence": self.extract_emotional_valence(memory, source),
                        "tags": self.extract_tags(memory, source),
                    }
                    recalled_memories.append(enhanced_memory)

                    if len(recalled_memories) >= max_results:
                        break

            summary = ""
            if recalled_memories:
                summary = self.generate_recall_summary(recalled_memories, query, use_ai_summary, openai_client)

            return {
                "recalled_memories": recalled_memories,
                "summary": summary,
                "query": query,
                "total_found": len(recalled_memories),
            }
        except Exception as exc:
            logger.error("Error in proactive recall: %s", exc)
            return {"recalled_memories": [], "summary": f"Recall failed: {str(exc)}"}

    def extract_memory_content(self, memory: Any, source: str) -> str:
        try:
            if source == "Episodic":
                return memory.detailed_content or memory.summary or str(memory)
            if source == "Semantic":
                if isinstance(memory, dict):
                    return f"{memory.get('subject', '')} {memory.get('predicate', '')} {memory.get('object', '')}"
                return str(memory)
            if isinstance(memory, dict):
                return memory.get("content", str(memory))
            if hasattr(memory, "content"):
                return memory.content
            return str(memory)
        except Exception:
            return str(memory)

    def extract_timestamp(self, memory: Any, source: str) -> Optional[datetime]:
        try:
            if source == "Episodic":
                return memory.timestamp
            if isinstance(memory, dict):
                timestamp = memory.get("timestamp")
                if isinstance(timestamp, str):
                    return datetime.fromisoformat(timestamp)
                return timestamp
            if hasattr(memory, "timestamp"):
                return memory.timestamp
        except Exception:
            pass
        return None

    def extract_importance(self, memory: Any, source: str) -> float:
        try:
            if source == "Episodic":
                return memory.importance
            if isinstance(memory, dict):
                return memory.get("importance", 0.5)
            if hasattr(memory, "importance"):
                return memory.importance
        except Exception:
            pass
        return 0.5

    def extract_emotional_valence(self, memory: Any, source: str) -> float:
        try:
            if source == "Episodic":
                return memory.emotional_valence
            if isinstance(memory, dict):
                return memory.get("emotional_valence", 0.0)
            if hasattr(memory, "emotional_valence"):
                return memory.emotional_valence
        except Exception:
            pass
        return 0.0

    def extract_tags(self, memory: Any, source: str) -> List[str]:
        try:
            if source == "Episodic":
                return memory.tags or []
            if isinstance(memory, dict):
                return memory.get("tags", [])
            if hasattr(memory, "tags"):
                return memory.tags
        except Exception:
            pass
        return []

    def generate_recall_summary(
        self,
        memories: List[Dict[str, Any]],
        query: str,
        use_ai: bool = False,
        openai_client: Any = None,
    ) -> str:
        if not memories:
            return "No relevant memories recalled."

        if use_ai and openai_client:
            try:
                return self.generate_ai_summary(memories, query, openai_client)
            except Exception as exc:
                logger.warning("AI summarization failed: %s, falling back to basic summary", exc)

        return self.generate_basic_summary(memories)

    def generate_basic_summary(self, memories: List[Dict[str, Any]]) -> str:
        source_counts: Dict[str, int] = {}
        total_importance = 0.0
        recent_count = 0

        for memory in memories:
            source = memory.get("source", "Unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
            total_importance += memory.get("importance", 0.5)

            timestamp = memory.get("timestamp")
            if timestamp and isinstance(timestamp, datetime):
                if (datetime.now() - timestamp).total_seconds() < 3600:
                    recent_count += 1

        avg_importance = total_importance / len(memories)

        summary_parts = [f"Recalled {len(memories)} relevant memories"]
        if source_counts:
            source_str = ", ".join([f"{count} from {source}" for source, count in source_counts.items()])
            summary_parts.append(f"from {source_str}")
        if recent_count > 0:
            summary_parts.append(f"including {recent_count} recent items")
        if avg_importance > 0.7:
            summary_parts.append("with high importance")
        elif avg_importance > 0.4:
            summary_parts.append("with moderate importance")

        return ". ".join(summary_parts) + "."

    def generate_ai_summary(self, memories: List[Dict[str, Any]], query: str, openai_client: Any) -> str:
        memory_texts = []
        for index, memory in enumerate(memories[:5]):
            content = memory.get("content", "")[:200]
            source = memory.get("source", "Unknown")
            importance = memory.get("importance", 0.5)
            memory_texts.append(f"{index + 1}. [{source}] {content} (importance: {importance:.1f})")

        memories_text = "\n".join(memory_texts)
        prompt = f"""
        Based on the user's query: \"{query}\"

        Here are the most relevant recalled memories:
        {memories_text}

        Please provide a concise, natural summary (2-3 sentences) of what these memories suggest about the user's context or interests. Focus on patterns, themes, or key insights rather than listing individual items.
        """

        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that summarizes memory recall results concisely and insightfully.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=150,
            )
            summary = response.choices[0].message.content.strip()
            return summary if summary else self.generate_basic_summary(memories)
        except Exception as exc:
            logger.error("AI summary generation failed: %s", exc)
            return self.generate_basic_summary(memories)