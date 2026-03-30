from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional


class MemoryFactService:
    def __init__(self, *, get_semantic: Callable[[], Any]) -> None:
        self._get_semantic = get_semantic

    def store_fact(self, subject: str, predicate: str, object_val: Any, **kwargs: Any) -> str:
        return self._get_semantic().store_fact(subject, predicate, object_val, **kwargs)

    def find_facts(
        self,
        subject: Optional[str] = None,
        predicate: Optional[str] = None,
        object_val: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        return self._get_semantic().find_facts(subject, predicate, object_val)

    def delete_fact(self, subject: str, predicate: str, object_val: Any) -> bool:
        return self._get_semantic().delete_fact(subject, predicate, object_val)