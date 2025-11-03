"""
Prospective Memory System for Human-AI Cognition Framework

Stores and manages future intentions, reminders, and scheduled tasks.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
import uuid
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required for vector prospective memory. "
        "Install it with: pip install sentence-transformers"
    ) from e

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        "chromadb is required for vector prospective memory. "
        "Install it with: pip install chromadb"
    ) from e

class ProspectiveMemorySystem:
    """Base marker class for prospective memory implementations.
    
    This is NOT dead code or a useless placeholder. It serves as:
    1. A semantic base class allowing type hints and isinstance checks
    2. A marker distinguishing prospective memory from other memory types
    3. Foundation for future Protocol/ABC if multiple implementations diverge
    
    Current implementations:
    - `ProspectiveMemory`: Lightweight in-memory reminders (no vectors)
    - `ProspectiveMemoryVectorStore`: Full ChromaDB-backed with semantic search
    
    Both implementations are actively used:
    - ProspectiveMemory: For basic reminder functionality in chat API
    - Vector store: For context-aware prospective retrieval
    
    This class intentionally has no methods - subclasses define their own interfaces
    appropriate to their storage mechanisms. If a common Protocol is needed in the
    future, it can be added here.
    """
    pass

# ------------------- Lightweight In-Memory Prospective Memory -------------------

class InMemoryReminder:
    def __init__(self, content: str, due_ts: float, metadata: Optional[Dict[str, Any]] = None):
        self.id = str(uuid.uuid4())
        self.content = content
        self.due_ts = due_ts
        self.created_ts = datetime.now().timestamp()
        self.triggered_ts: Optional[float] = None
        self.metadata = metadata or {}

class ProspectiveMemory(ProspectiveMemorySystem):
    """Lightweight alternative (no vectors) for tests & fast integration.

    Provides:
      - add_reminder(content, due_in_seconds)
      - check_due() -> list of newly triggered reminders (one-shot)
      - upcoming(within_seconds)
      - list_reminders(include_triggered)
    Metrics:
      - prospective_reminders_created_total
      - prospective_reminders_triggered_total
    """
    def __init__(self):
        self._store: Dict[str, InMemoryReminder] = {}

    def add_reminder(self, content: str, due_in_seconds: float, metadata: Optional[Dict[str, Any]] = None) -> InMemoryReminder:
        due_ts = datetime.now().timestamp() + max(0.0, float(due_in_seconds))
        r = InMemoryReminder(content=content, due_ts=due_ts, metadata=metadata)
        self._store[r.id] = r
        try:
            from src.chat.metrics import metrics_registry
            metrics_registry.inc("prospective_reminders_created_total")
        except Exception:
            pass
        return r

    def list_reminders(self, include_triggered: bool = True) -> List[InMemoryReminder]:
        vals = list(self._store.values())
        if not include_triggered:
            vals = [r for r in vals if r.triggered_ts is None]
        return sorted(vals, key=lambda x: x.due_ts)

    def check_due(self, now: Optional[float] = None) -> List[InMemoryReminder]:
        now = now or datetime.now().timestamp()
        due: List[InMemoryReminder] = []
        for r in self._store.values():
            if r.triggered_ts is None and r.due_ts <= now:
                r.triggered_ts = now
                due.append(r)
                try:
                    from src.chat.metrics import metrics_registry
                    metrics_registry.inc("prospective_reminders_triggered_total")
                except Exception:
                    pass
        return sorted(due, key=lambda x: x.due_ts)

    def upcoming(self, within_seconds: float) -> List[InMemoryReminder]:
        horizon = datetime.now().timestamp() + max(0.0, within_seconds)
        return [r for r in self.list_reminders(include_triggered=False) if r.due_ts <= horizon]

    def to_dict(self, r: InMemoryReminder) -> Dict[str, Any]:
        return {
            "id": r.id,
            "content": r.content,
            "due_ts": r.due_ts,
            "due_in_seconds": max(0.0, r.due_ts - datetime.now().timestamp()),
            "created_ts": r.created_ts,
            "triggered_ts": r.triggered_ts,
            "metadata": r.metadata,
        }

    def export_all(self) -> List[Dict[str, Any]]:
        return [self.to_dict(r) for r in self.list_reminders(include_triggered=True)]

    def purge_triggered(self) -> int:
        """Remove all reminders that have already triggered.

        Returns
        -------
        int
            Number of reminders removed.
        """
        to_delete = [rid for rid, r in self._store.items() if r.triggered_ts is not None]
        for rid in to_delete:
            self._store.pop(rid, None)
        if to_delete:
            try:
                from src.chat.metrics import metrics_registry
                metrics_registry.inc("prospective_reminders_purged_total", len(to_delete))
            except Exception:
                pass
        return len(to_delete)

_pm_singleton: Optional[ProspectiveMemory] = None

def get_inmemory_prospective_memory() -> ProspectiveMemory:
    global _pm_singleton
    if _pm_singleton is None:
        _pm_singleton = ProspectiveMemory()
    return _pm_singleton

class ProspectiveMemoryVectorStore(ProspectiveMemorySystem):
    """Vector / persistent prospective memory using ChromaDB.

    Responsibilities:
      - Store reminders (intentions) with semantic embeddings
      - Retrieve / search reminders by semantic similarity
      - Detect due reminders based on `trigger_time` (ISO timestamp or epoch seconds)
      - Support completion and cleanup
    """
    def __init__(self, chroma_persist_dir: Optional[str] = None, collection_name: str = "prospective_reminders", embedding_model: str = "all-MiniLM-L6-v2"):
        self.chroma_persist_dir = Path(chroma_persist_dir or "data/memory_stores/chroma_prospective")
        self.collection_name = collection_name
        self.embedding_model: Optional[Any] = None
        # Initialize embedding model with GPU support if available
        try:
            import torch
            self.embedding_model = SentenceTransformer(embedding_model)
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")
        except Exception as e:
            raise RuntimeError(f"ProspectiveMemorySystem failed to load embedding model: {e}") from e

        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_chromadb()

    def _initialize_chromadb(self) -> None:
        try:
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                persist_directory=str(self.chroma_persist_dir)
            )
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_persist_dir),
                settings=settings
            )
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Prospective memory reminders (intentions, scheduled tasks)"}
                )
        except Exception as e:  # pragma: no cover - critical init path
            raise RuntimeError(f"ProspectiveMemoryVectorStore failed to initialize ChromaDB: {e}") from e

    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        if not self.embedding_model or not text:
            return None
        try:
            content_str = str(text)
            embedding = self.embedding_model.encode(content_str)
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            elif hasattr(embedding, '__iter__'):
                return [float(x) for x in embedding]
            else:
                return None
        except Exception:
            return None

    def store(self, description: str, trigger_time: Optional[str] = None, tags: Optional[List[str]] = None, memory_type: str = "ltm") -> str:
        """Store a new reminder.

        Parameters
        ----------
        description : str
            Human-readable reminder description / intention.
        trigger_time : Optional[str]
            ISO 8601 timestamp (or cron syntax / epoch seconds in later extensions). If None the
            reminder has no due semantics yet.
        tags : list[str] | None
            Optional tag list for organization / filtering.
        memory_type : str
            Reserved for future STM/LTM segmentation.

        Returns
        -------
        str
            Newly created reminder id or empty string on failure.
        """
        if not self.collection or not self.embedding_model:
            return ""
        reminder_id = str(uuid.uuid4())
        now = datetime.now().isoformat()
        embedding = self._generate_embedding(description)
        if not embedding:
            return ""
        metadata = {
            "description": description,
            "trigger_time": trigger_time,
            # Chroma metadata must be scalar types; serialize list to comma string
            "tags": ",".join(tags) if tags else "",
            "created_at": now,
            "completed": False,
            "completed_at": None,
            "memory_type": memory_type
        }
        self.collection.upsert(
            ids=[reminder_id],
            embeddings=[embedding],
            documents=[description],
            metadatas=[metadata]
        )
        return reminder_id

    def retrieve(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        if not self.collection:
            return None
        result = self.collection.get(ids=[reminder_id])
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        documents = result.get('documents') or []
        if ids and len(ids) > 0:
            meta = dict(metadatas[0]) if metadatas and len(metadatas) > 0 and metadatas[0] is not None else {}
            doc = str(documents[0]) if documents and len(documents) > 0 and documents[0] is not None else ""
            meta['reminder_id'] = reminder_id
            meta['reminder_text'] = doc
            return meta
        return None

    def delete(self, reminder_id: str) -> bool:
        if not self.collection:
            return False
        try:
            self.collection.delete(ids=[reminder_id])
            return True
        except Exception:
            return False

    def list_reminders(self, include_completed: bool = False) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        result = self.collection.get()
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        documents = result.get('documents') or []
        reminders = []
        for i, reminder_id in enumerate(ids):
            meta = dict(metadatas[i]) if metadatas and len(metadatas) > i and metadatas[i] is not None else {}
            doc = str(documents[i]) if documents and len(documents) > i and documents[i] is not None else ""
            if include_completed or not meta.get("completed", False):
                meta['reminder_id'] = reminder_id
                meta['reminder_text'] = doc
                reminders.append(meta)
        return reminders

    def search(self, query: Optional[str] = None, include_completed: bool = False) -> List[Dict[str, Any]]:
        if not self.collection:
            return []
        if query:
            embedding = self._generate_embedding(query)
            if not embedding:
                return []
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=10,
                include=["documents", "metadatas"]
            )
            found = []
            ids = results.get('ids', [[]])
            metadatas = results.get('metadatas', [[]])
            documents = results.get('documents', [[]])
            if ids and ids[0]:
                for i, reminder_id in enumerate(ids[0]):
                    meta = dict(metadatas[0][i]) if metadatas and len(metadatas[0]) > i and metadatas[0][i] is not None else {}
                    doc = str(documents[0][i]) if documents and len(documents[0]) > i and documents[0][i] is not None else ""
                    if include_completed or not meta.get("completed", False):
                        meta['reminder_id'] = reminder_id
                        meta['reminder_text'] = doc
                        found.append(meta)
            return found
        else:
            return self.list_reminders(include_completed=include_completed)

    def complete_reminder(self, reminder_id: str):
        if not self.collection:
            return
        result = self.collection.get(ids=[reminder_id])
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        if ids and len(ids) > 0:
            meta = dict(metadatas[0]) if metadatas and len(metadatas) > 0 and metadatas[0] is not None else {}
            meta['completed'] = True
            meta['completed_at'] = datetime.now().isoformat()
            self.collection.update(ids=[reminder_id], metadatas=[meta])

    def get_due_reminders(self, now: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Return reminders whose trigger_time is due (<= now) and not completed.

        Supports `trigger_time` stored as ISO string or numeric (epoch seconds).
        Invalid / unparsable trigger_time values are skipped.
        """
        if not self.collection:
            return []
        now = now or datetime.now()
        result = self.collection.get()
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        documents = result.get('documents') or []
        due: List[Dict[str, Any]] = []
        for i, reminder_id in enumerate(ids):
            meta = dict(metadatas[i]) if metadatas and len(metadatas) > i and metadatas[i] is not None else {}
            doc = str(documents[i]) if documents and len(documents) > i and documents[i] is not None else ""
            trigger_time = meta.get("trigger_time")
            completed = meta.get("completed", False)
            if not trigger_time or completed:
                continue
            try:
                # support ISO datetime string
                if isinstance(trigger_time, str):
                    due_time = datetime.fromisoformat(trigger_time)
                # support numeric timestamp
                elif isinstance(trigger_time, (int, float)):
                    due_time = datetime.fromtimestamp(float(trigger_time))
                else:
                    # unsupported format -- skip
                    continue
                if due_time <= now:
                    meta['reminder_id'] = reminder_id
                    meta['reminder_text'] = doc
                    due.append(meta)
            except Exception:
                # Skip malformed entries
                continue
        return due

    def clear(self) -> None:
        """Remove all reminders from the collection (best-effort)."""
        if not self.collection:
            return
        try:
            result = self.collection.get()
            ids = result.get('ids') or []
            if ids:
                self.collection.delete(ids=ids)
        except Exception:  # pragma: no cover - defensive
            pass
