from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid
import json

class ProceduralMemory:
    """
    Stores procedural memories (skills, routines, action sequences) in STM or LTM.
    Delegates storage to STM or LTM based on memory_type ('stm' or 'ltm').
    
    Compatible with both dict-based in-memory stores and VectorShortTermMemory/VectorLongTermMemory.
    """

    def __init__(self, stm=None, ltm=None):
        self.stm = stm
        self.ltm = ltm

    def _parse_content(self, raw_content) -> Optional[Dict[str, Any]]:
        """Parse content which may be a dict or JSON-encoded string."""
        if raw_content is None:
            return None
        if isinstance(raw_content, dict):
            return raw_content
        if isinstance(raw_content, str):
            try:
                return json.loads(raw_content)
            except (json.JSONDecodeError, ValueError):
                return None
        return None

    def _get_stm_item(self, proc_id: str):
        """Get item from STM using available interface."""
        if self.stm is None:
            return None
        # Try retrieve method (VectorShortTermMemory)
        if hasattr(self.stm, 'retrieve'):
            return self.stm.retrieve(proc_id)
        # Fallback to dict access (legacy in-memory)
        if hasattr(self.stm, 'items'):
            return self.stm.items.get(proc_id)
        return None

    def _get_all_stm_items(self) -> List[Any]:
        """Get all items from STM using available interface."""
        if self.stm is None:
            return []
        # Try get_all_memories method (VectorShortTermMemory)
        if hasattr(self.stm, 'get_all_memories'):
            return self.stm.get_all_memories()
        # Fallback to dict access (legacy in-memory)
        if hasattr(self.stm, 'items'):
            return list(self.stm.items.values())
        return []

    def _get_ltm_record(self, proc_id: str):
        """Get record from LTM using available interface."""
        if self.ltm is None:
            return None
        # Try retrieve method (VectorLongTermMemory)
        if hasattr(self.ltm, 'retrieve'):
            return self.ltm.retrieve(proc_id)
        # Fallback to dict access (legacy in-memory)
        if hasattr(self.ltm, 'memories'):
            return self.ltm.memories.get(proc_id)
        return None

    def _get_all_ltm_records(self) -> List[Any]:
        """Get all records from LTM using available interface."""
        if self.ltm is None:
            return []
        # Try get_all_memories method (non-standard; may exist)
        if hasattr(self.ltm, "get_all_memories"):
            return self.ltm.get_all_memories()

        # Prefer direct collection enumeration when available (vector LTM).
        collection = getattr(self.ltm, "collection", None)
        if collection is not None and hasattr(collection, "get"):
            try:
                result = collection.get(
                    where={"memory_type": "procedural"},
                    include=["documents", "metadatas"],
                )
                ids = result.get("ids") or []
                documents = result.get("documents") or []
                metadatas = result.get("metadatas") or []

                records: List[Dict[str, Any]] = []
                for i, doc in enumerate(documents):
                    meta = metadatas[i] if i < len(metadatas) else {}
                    memory_id = ids[i] if i < len(ids) else None
                    records.append(
                        {
                            "id": meta.get("memory_id", memory_id),
                            "content": doc,
                            "memory_type": meta.get("memory_type", "procedural"),
                            "tags": (meta.get("tags", "").split(",") if meta.get("tags") else []),
                        }
                    )
                return records
            except Exception:
                pass

        # Fallback: semantic search with procedural type filter.
        if hasattr(self.ltm, "search_semantic"):
            try:
                return self.ltm.search_semantic(
                    query="procedural",
                    max_results=250,
                    min_similarity=0.0,
                    memory_types=["procedural"],
                )
            except Exception:
                pass

        # Fallback to dict access (legacy in-memory)
        if hasattr(self.ltm, 'memories'):
            return list(self.ltm.memories.values())
        return []

    def _remove_stm_item(self, proc_id: str) -> bool:
        """Remove item from STM using available interface."""
        if self.stm is None:
            return False
        # Try remove_item method (VectorShortTermMemory)
        if hasattr(self.stm, 'remove_item'):
            return self.stm.remove_item(proc_id)
        # Fallback to dict access (legacy in-memory)
        if hasattr(self.stm, 'items') and proc_id in self.stm.items:
            del self.stm.items[proc_id]
            return True
        return False

    def _remove_ltm_record(self, proc_id: str) -> bool:
        """Remove record from LTM using available interface."""
        if self.ltm is None:
            return False
        # Try remove method (VectorLongTermMemory)
        if hasattr(self.ltm, 'remove'):
            return self.ltm.remove(proc_id)
        # Fallback to dict access (legacy in-memory)
        if hasattr(self.ltm, 'memories') and proc_id in self.ltm.memories:
            del self.ltm.memories[proc_id]
            return True
        return False

    def store(
        self,
        description: str,
        steps: List[str],
        tags: Optional[List[str]] = None,
        created_at: Optional[datetime] = None,
        memory_type: str = "stm",
        importance: float = 0.5,
        attention_score: float = 0.0,
        emotional_valence: float = 0.0,
        source: str = "procedural",
        associations: Optional[List[str]] = None,
    ) -> str:
        proc_id = str(uuid.uuid4())
        procedure = {
            "id": proc_id,
            "description": description,
            "steps": steps,
            "tags": tags or [],
            "created_at": created_at or datetime.now(),
            "last_used": None,
            "usage_count": 0,
            "strength": 0.1,
            "memory_type": "procedural",
        }
        if memory_type == "ltm":
            if self.ltm is None:
                raise ValueError("LTM instance not provided to ProceduralMemory")
            self.ltm.store(
                memory_id=proc_id,
                content=procedure,
                memory_type="procedural",
                importance=importance,
                emotional_valence=emotional_valence,
                source=source,
                tags=tags,
                associations=associations,
            )
        else:
            if self.stm is None:
                raise ValueError("STM instance not provided to ProceduralMemory")
            self.stm.store(
                memory_id=proc_id,
                content=procedure,
                importance=importance,
                attention_score=attention_score,
                emotional_valence=emotional_valence,
                associations=associations,
            )
        return proc_id

    def _extract_content(self, record_or_item) -> Optional[str]:
        """Extract raw content from a memory item or record dict.
        
        Handles:
        - MemoryItem objects (have .content attribute)
        - Dict records (have "content" key)
        - String content directly
        """
        if record_or_item is None:
            return None
        # Object with .content attribute (MemoryItem)
        if hasattr(record_or_item, 'content'):
            return record_or_item.content
        # Dict with "content" key (VectorLongTermMemory.retrieve result)
        if isinstance(record_or_item, dict) and "content" in record_or_item:
            return record_or_item["content"]
        # Assume it's already the content
        if isinstance(record_or_item, str):
            return record_or_item
        return None

    def retrieve(self, proc_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a procedural memory by ID."""
        # Check STM first, then LTM
        if self.stm:
            item = self._get_stm_item(proc_id)
            if item:
                raw_content = self._extract_content(item)
                content = self._parse_content(raw_content)
                if content and content.get("memory_type") == "procedural":
                    return content
        if self.ltm:
            record = self._get_ltm_record(proc_id)
            if record:
                raw_content = self._extract_content(record)
                content = self._parse_content(raw_content)
                if content and content.get("memory_type") == "procedural":
                    return content
        return None

    def search(
        self,
        query: str,
        *,
        memory_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for procedural memories matching a query.

        Args:
            query: Search query string.
            memory_type: Optional store selector: "stm", "ltm", or None for both.
            tags: Optional tag filter; all tags must be present.
            max_results: Maximum number of returned procedures.
        """

        def _has_tags(proc: Dict[str, Any]) -> bool:
            if not tags:
                return True
            proc_tags = set(proc.get("tags", []) or [])
            return all(t in proc_tags for t in tags)

        def _matches_query(proc: Dict[str, Any], q: str) -> bool:
            if not q:
                return True
            desc = (proc.get("description", "") or "").lower()
            steps = proc.get("steps", []) or []
            return q in desc or any(q in str(step).lower() for step in steps)

        normalized = (memory_type or "").strip().lower() if memory_type else None
        search_stm = normalized in (None, "", "stm", "both", "all")
        search_ltm = normalized in (None, "", "ltm", "both", "all")

        results: List[Dict[str, Any]] = []
        q = (query or "").lower()

        # Search STM (local substring filter)
        if search_stm:
            for item in self._get_all_stm_items():
                raw_content = self._extract_content(item)
                content = self._parse_content(raw_content)
                if not content or content.get("memory_type") != "procedural":
                    continue
                if not _has_tags(content):
                    continue
                if not _matches_query(content, q):
                    continue
                results.append(content)

        # Search LTM
        if search_ltm and self.ltm is not None:
            # Prefer semantic search using the *actual query* to avoid sampling issues.
            if hasattr(self.ltm, "search_semantic"):
                try:
                    semantic_results = self.ltm.search_semantic(
                        query=query or "procedural",
                        max_results=max(max_results * 5, 25),
                        min_similarity=0.0,
                        memory_types=["procedural"],
                    )
                    for record in semantic_results:
                        raw_content = self._extract_content(record)
                        content = self._parse_content(raw_content)
                        if not content or content.get("memory_type") != "procedural":
                            continue
                        if not _has_tags(content):
                            continue
                        # Keep substring match to satisfy deterministic test expectations.
                        if not _matches_query(content, q):
                            continue
                        results.append(content)
                except Exception:
                    # Fall back to broad-scan below.
                    pass

            if not results:
                for record in self._get_all_ltm_records():
                    raw_content = self._extract_content(record)
                    content = self._parse_content(raw_content)
                    if not content or content.get("memory_type") != "procedural":
                        continue
                    if not _has_tags(content):
                        continue
                    if not _matches_query(content, q):
                        continue
                    results.append(content)

        results.sort(key=lambda p: -p.get("strength", 0))
        return results[: max_results if max_results > 0 else 10]

    def use(self, proc_id: str) -> bool:
        """Mark a procedure as used, incrementing its usage count and strength.
        
        Updates are persisted back to the memory store.
        """
        proc = self.retrieve(proc_id)
        if not proc:
            return False
        proc["usage_count"] = proc.get("usage_count", 0) + 1
        proc["last_used"] = datetime.now()
        proc["strength"] = min(1.0, proc.get("strength", 0.1) + 0.1)
        
        # Re-store to persist updates
        # Determine which store has this procedure and re-store
        if self.stm:
            item = self._get_stm_item(proc_id)
            if item:
                # Re-store in STM
                self.stm.store(
                    memory_id=proc_id,
                    content=proc,
                    importance=proc.get("importance", 0.5)
                )
                return True
        if self.ltm:
            record = self._get_ltm_record(proc_id)
            if record:
                # Re-store in LTM
                self.ltm.store(
                    memory_id=proc_id,
                    content=proc,
                    memory_type="procedural"
                )
                return True
        return True

    def all_procedures(self) -> List[Dict[str, Any]]:
        """Get all procedural memories from both STM and LTM."""
        all_procs = []
        for item in self._get_all_stm_items():
            raw_content = self._extract_content(item)
            content = self._parse_content(raw_content)
            if content and content.get("memory_type") == "procedural":
                all_procs.append(content)
        for record in self._get_all_ltm_records():
            raw_content = self._extract_content(record)
            content = self._parse_content(raw_content)
            if content and content.get("memory_type") == "procedural":
                all_procs.append(content)
        return all_procs

    def delete(self, proc_id: str) -> bool:
        """Delete a procedural memory by ID."""
        deleted = False
        if self._remove_stm_item(proc_id):
            deleted = True
        if self._remove_ltm_record(proc_id):
            deleted = True
        return deleted

    def clear(self, *, memory_type: Optional[str] = None):
        """Remove procedural memories from STM and/or LTM.

        Args:
            memory_type: "stm", "ltm", or None for both.
        """
        # Get all procedure IDs first
        procs_to_delete = []
        normalized = (memory_type or "").strip().lower() if memory_type else None
        clear_stm = normalized in (None, "", "stm", "both", "all")
        clear_ltm = normalized in (None, "", "ltm", "both", "all")

        if clear_stm:
            for item in self._get_all_stm_items():
                raw_content = self._extract_content(item)
                content = self._parse_content(raw_content)
                if content and content.get("memory_type") == "procedural":
                    procs_to_delete.append(("stm", content.get("id")))

        if clear_ltm:
            for record in self._get_all_ltm_records():
                raw_content = self._extract_content(record)
                content = self._parse_content(raw_content)
                if content and content.get("memory_type") == "procedural":
                    procs_to_delete.append(("ltm", content.get("id")))
        
        # Delete each procedure
        for store_type, proc_id in procs_to_delete:
            if proc_id:
                if store_type == "stm":
                    self._remove_stm_item(proc_id)
                else:
                    self._remove_ltm_record(proc_id)