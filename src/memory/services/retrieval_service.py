from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..ltm import VectorLongTermMemory
from ..stm import VectorShortTermMemory


logger = logging.getLogger(__name__)


def _is_suppressed(memory: Any, source: str) -> bool:
    source_key = str(source).strip().lower()
    if source_key == "episodic":
        return bool(getattr(memory, "suppressed", False))
    if isinstance(memory, dict):
        if source_key == "semantic" and str(memory.get("belief_status", "")).lower() == "quarantined":
            return True
        return bool(memory.get("suppressed", False))
    return bool(getattr(memory, "suppressed", False))


class MemoryRetrievalService:
    def __init__(
        self,
        *,
        get_stm: Callable[[], Any],
        get_ltm: Callable[[], Any],
        get_episodic: Callable[[], Any],
        get_config: Callable[[], Any],
        increment_operation: Callable[[str], None],
        increment_error: Callable[[str], None],
    ) -> None:
        self._get_stm = get_stm
        self._get_ltm = get_ltm
        self._get_episodic = get_episodic
        self._get_config = get_config
        self._increment_operation = increment_operation
        self._increment_error = increment_error

    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        try:
            try:
                memory = self._get_stm().retrieve(memory_id)
                if memory is not None:
                    self._increment_operation("retrieve_stm")
                    return memory
            except Exception as exc:
                logger.warning("Error retrieving from STM for %s: %s", memory_id, exc)

            try:
                memory = self._get_ltm().retrieve(memory_id)
                if memory is not None:
                    self._increment_operation("retrieve_ltm")
                    return memory
            except Exception as exc:
                logger.warning("Error retrieving from LTM for %s: %s", memory_id, exc)

            self._increment_operation("retrieve_miss")
            return None
        except Exception as exc:
            logger.error("Error retrieving memory %s: %s", memory_id, exc)
            self._increment_error("retrieve")
            return None

    def search_memories(
        self,
        query: str,
        search_stm: bool = True,
        search_ltm: bool = True,
        search_episodic: bool = True,
        memory_types: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> List[Tuple[Any, float, str]]:
        results: List[Tuple[Any, float, str]] = []
        config = self._get_config()
        stm = self._get_stm()
        ltm = self._get_ltm()
        episodic = self._get_episodic()

        try:
            if search_stm:
                stm_results = stm.search(query, max_results=max_results // 2)
                logger.debug("STM search for '%s' returned: %s", query, stm_results)
                for memory, score in stm_results:
                    if not _is_suppressed(memory, "STM"):
                        results.append((memory, score, "STM"))

            if search_ltm:
                if config.use_vector_ltm and isinstance(ltm, VectorLongTermMemory):
                    ltm_results = ltm.search_semantic(
                        query=query,
                        memory_types=memory_types,
                        max_results=max_results // 2,
                    )
                    logger.debug("LTM search for '%s' returned: %s", query, ltm_results)
                    for memory in ltm_results:
                        if not _is_suppressed(memory, "LTM"):
                            results.append((memory, memory.get("similarity_score", 0.0), "LTM"))
                else:
                    ltm_results = getattr(ltm, "search_by_content", lambda **kwargs: [])(
                        query=query,
                        memory_types=memory_types,
                        max_results=max_results // 2,
                    )
                    logger.debug("LTM search for '%s' returned: %s", query, ltm_results)
                    for memory, score in ltm_results:
                        if not _is_suppressed(memory, "LTM"):
                            results.append((memory, score, "LTM"))

            if search_episodic:
                try:
                    episodic_results = episodic.search_memories(query=query, limit=max_results)
                    logger.debug("Episodic search for '%s' returned: %s", query, episodic_results)
                    for result in episodic_results:
                        if not _is_suppressed(result.memory, "Episodic"):
                            results.append((result.memory, result.relevance, "Episodic"))
                except Exception as exc:
                    logger.error("Error searching episodic memory for query '%s': %s", query, exc)
        except Exception as exc:
            logger.error("Error searching memories for query '%s': %s", query, exc)

        seen_content = set()
        unique_results = []
        for memory, score, source in sorted(results, key=lambda item: item[1], reverse=True):
            content_repr = ""
            if source == "Episodic":
                content_repr = repr(memory.detailed_content)
            elif isinstance(memory, dict):
                content_repr = repr(memory.get("content"))
            elif hasattr(memory, "content"):
                content_repr = repr(memory.content)

            if content_repr and content_repr not in seen_content:
                unique_results.append((memory, score, source))
                seen_content.add(content_repr)

        return unique_results[:max_results]

    def search_stm_semantic(
        self,
        query: str,
        max_results: int = 5,
        min_similarity: float = 0.5,
        min_activation: float = 0.0,
    ) -> List[Tuple[Any, float]]:
        del min_activation
        config = self._get_config()
        stm = self._get_stm()
        if config.use_vector_stm and isinstance(stm, VectorShortTermMemory):
            vector_results = stm.search_semantic(
                query=query,
                max_results=max_results,
                min_similarity=min_similarity,
            )
            return [(result.item, result.relevance_score) for result in vector_results]

        results = stm.search(query=query, max_results=max_results)
        output = []
        for result in results:
            if isinstance(result, tuple) and len(result) == 2:
                output.append((result[0], result[1]))
            elif isinstance(result, dict):
                output.append((result, 1.0))
            else:
                output.append((result, 1.0))
        return output