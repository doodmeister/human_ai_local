from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List

from ..ltm import VectorLongTermMemory
from ..stm import VectorShortTermMemory


logger = logging.getLogger(__name__)


class MemoryContextService:
    def __init__(
        self,
        *,
        get_stm: Callable[[], Any],
        get_ltm: Callable[[], Any],
        get_config: Callable[[], Any],
    ) -> None:
        self._get_stm = get_stm
        self._get_ltm = get_ltm
        self._get_config = get_config

    def get_context_for_query(
        self,
        query: str,
        max_stm_context: int = 5,
        max_ltm_context: int = 5,
        min_relevance: float = 0.3,
    ) -> Dict[str, List[Any]]:
        context = {"stm": [], "ltm": []}
        config = self._get_config()
        stm = self._get_stm()
        ltm = self._get_ltm()

        try:
            if config.use_vector_stm and isinstance(stm, VectorShortTermMemory):
                stm_results = stm.get_context_for_query(
                    query=query,
                    max_context_items=max_stm_context,
                    min_relevance=min_relevance,
                )
                context["stm"] = [result.item for result in stm_results]
            else:
                stm_results = stm.search(query=query, max_results=max_stm_context)
                context["stm"] = [item for item, score in stm_results if score >= min_relevance]

            if config.use_vector_ltm and isinstance(ltm, VectorLongTermMemory):
                ltm_results = ltm.search_semantic(
                    query=query,
                    max_results=max_ltm_context,
                    min_similarity=min_relevance,
                )
                context["ltm"] = [result for result in ltm_results]
            else:
                ltm_results = getattr(ltm, "search_by_content", lambda **kwargs: [])(
                    query=query,
                    max_results=max_ltm_context,
                )
                context["ltm"] = [record for record, score in ltm_results if score >= min_relevance]

            logger.debug(
                "Retrieved context: %s STM + %s LTM items",
                len(context["stm"]),
                len(context["ltm"]),
            )
            return context
        except Exception as exc:
            logger.error("Error getting context for query: %s", exc)
            return {"stm": [], "ltm": []}