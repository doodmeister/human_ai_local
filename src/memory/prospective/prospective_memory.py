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
    """
    Prospective Memory System using ChromaDB vector store.
    Stores and manages future intentions, reminders, and scheduled tasks persistently.
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

    def _initialize_chromadb(self):
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
        except Exception as e:
            raise RuntimeError(f"ProspectiveMemorySystem failed to initialize ChromaDB: {e}") from e

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
        """
        Store a new prospective memory (reminder/intention) in the vector store.
        Args:
            description: Reminder description
            trigger_time: ISO datetime string or cron
            tags: List of tags
            memory_type: For future use ("ltm"/"stm")
        Returns:
            Unique reminder ID
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
            "tags": tags or [],
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
        if not self.collection:
            return []
        now = now or datetime.now()
        result = self.collection.get()
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        documents = result.get('documents') or []
        due = []
        for i, reminder_id in enumerate(ids):
            meta = dict(metadatas[i]) if metadatas and len(metadatas) > i and metadatas[i] is not None else {}
            doc = str(documents[i]) if documents and len(documents) > i and documents[i] is not None else ""
            trigger_time = meta.get("trigger_time")
            completed = meta.get("completed", False)
            if trigger_time and not completed:
                try:
                    if isinstance(trigger_time, str):
                        due_time = datetime.fromisoformat(trigger_time)
                    else:
                        continue  # skip if not a string
                    if due_time <= now:
                        meta['reminder_id'] = reminder_id
                        meta['reminder_text'] = doc
                        due.append(meta)
                except Exception:
                    continue
        return due

    def clear(self):
        if not self.collection:
            return
        result = self.collection.get()
        if result.get('ids'):
            self.collection.delete(ids=result['ids'])
