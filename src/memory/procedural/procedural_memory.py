from typing import Optional, Dict, List, Any
from datetime import datetime
import uuid

class ProceduralMemory:
    """
    Stores procedural memories (skills, routines, action sequences) in STM or LTM.
    Delegates storage to STM or LTM based on memory_type ('stm' or 'ltm').
    """

    def __init__(self, stm=None, ltm=None):
        self.stm = stm
        self.ltm = ltm

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

    def retrieve(self, proc_id: str) -> Optional[Dict[str, Any]]:
        # Check STM first, then LTM
        if self.stm:
            item = self.stm.items.get(proc_id)
            if item and getattr(item, 'content', None):
                content = item.content if hasattr(item, 'content') else item
                if isinstance(content, dict) and content.get("memory_type") == "procedural":
                    return content
        if self.ltm:
            record = self.ltm.memories.get(proc_id)
            if record and getattr(record, 'content', None):
                content = record.content if hasattr(record, 'content') else record
                if isinstance(content, dict) and content.get("memory_type") == "procedural":
                    return content
        return None

    def search(self, query: str) -> List[Dict[str, Any]]:
        results = []
        q = query.lower()
        # Search STM
        if self.stm:
            for item in self.stm.items.values():
                content = item.content if hasattr(item, 'content') else item
                if (
                    isinstance(content, dict)
                    and content.get("memory_type") == "procedural"
                    and (q in content["description"].lower() or any(q in step.lower() for step in content["steps"]))
                ):
                    results.append(content)
        # Search LTM
        if self.ltm:
            for record in self.ltm.memories.values():
                content = record.content if hasattr(record, 'content') else record
                if (
                    isinstance(content, dict)
                    and content.get("memory_type") == "procedural"
                    and (q in content["description"].lower() or any(q in step.lower() for step in content["steps"]))
                ):
                    results.append(content)
        results.sort(key=lambda p: -p.get("strength", 0))
        return results

    def use(self, proc_id: str) -> bool:
        proc = self.retrieve(proc_id)
        if not proc:
            return False
        proc["usage_count"] += 1
        proc["last_used"] = datetime.now()
        proc["strength"] = min(1.0, proc["strength"] + 0.1)
        # Optionally, update in STM or LTM
        return True

    def all_procedures(self) -> List[Dict[str, Any]]:
        all_procs = []
        if self.stm:
            for item in self.stm.items.values():
                content = item.content if hasattr(item, 'content') else item
                if isinstance(content, dict) and content.get("memory_type") == "procedural":
                    all_procs.append(content)
        if self.ltm:
            for record in self.ltm.memories.values():
                content = record.content if hasattr(record, 'content') else record
                if isinstance(content, dict) and content.get("memory_type") == "procedural":
                    all_procs.append(content)
        return all_procs

    def delete(self, proc_id: str) -> bool:
        deleted = False
        if self.stm and proc_id in self.stm.items:
            del self.stm.items[proc_id]
            deleted = True
        if self.ltm and proc_id in self.ltm.memories:
            del self.ltm.memories[proc_id]
            deleted = True
        return deleted

    def clear(self):
        if self.stm:
            to_delete = [pid for pid, item in self.stm.items.items() if isinstance(getattr(item, 'content', {}), dict) and getattr(item, 'content', {}).get("memory_type") == "procedural"]
            for pid in to_delete:
                del self.stm.items[pid]
        if self.ltm:
            to_delete = [pid for pid, record in self.ltm.memories.items() if isinstance(getattr(record, 'content', {}), dict) and getattr(record, 'content', {}).get("memory_type") == "procedural"]
            for pid in to_delete:
                del self.ltm.memories[pid]