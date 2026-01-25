"""
Prospective Memory System for Human-AI Cognition Framework

Stores and manages future intentions, reminders, and scheduled tasks.

This module provides a unified interface for prospective memory with two implementations:
1. InMemoryProspectiveMemory: Lightweight, no external dependencies
2. VectorProspectiveMemory: Full-featured with semantic search (requires sentence-transformers, chromadb)

The system gracefully degrades if optional dependencies are not available.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import uuid
from pathlib import Path


# ===============================================================================
# Data Models
# ===============================================================================

@dataclass
class Reminder:
    """
    Unified reminder data structure.
    
    Attributes:
        id: Unique identifier
        content: Reminder description/text
        due_time: When the reminder is due (datetime or None for unscheduled)
        created_at: When the reminder was created
        completed: Whether the reminder has been completed/triggered
        completed_at: When the reminder was completed
        tags: Optional tags for organization
        metadata: Additional key-value data
    """
    id: str
    content: str
    due_time: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed: bool = False
    completed_at: Optional[datetime] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Backward compatibility properties for old timestamp-based API
    @property
    def due_ts(self) -> Optional[float]:
        """Legacy: timestamp version of due_time."""
        return self.due_time.timestamp() if self.due_time else None
    
    @property
    def created_ts(self) -> float:
        """Legacy: timestamp version of created_at."""
        return self.created_at.timestamp()
    
    @property
    def triggered_ts(self) -> Optional[float]:
        """Legacy: timestamp version of completed_at."""
        return self.completed_at.timestamp() if self.completed_at else None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert reminder to dictionary format."""
        return {
            "id": self.id,
            "content": self.content,
            "due_time": self.due_time.isoformat() if self.due_time else None,
            "due_in_seconds": (self.due_time - datetime.now()).total_seconds() if self.due_time else None,
            "created_at": self.created_at.isoformat(),
            "completed": self.completed,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "tags": self.tags,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Reminder':
        """Create reminder from dictionary format."""
        return cls(
            id=data.get("id", str(uuid.uuid4())),
            content=data["content"],
            due_time=datetime.fromisoformat(data["due_time"]) if data.get("due_time") else None,
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else datetime.now(),
            completed=data.get("completed", False),
            completed_at=datetime.fromisoformat(data["completed_at"]) if data.get("completed_at") else None,
            tags=data.get("tags", []),
            metadata=data.get("metadata", {})
        )


# ===============================================================================
# Base Interface
# ===============================================================================

class ProspectiveMemorySystem(ABC):
    """
    Abstract base class for prospective memory implementations.
    
    Defines the common interface that all prospective memory backends must implement.
    This allows seamless switching between in-memory and persistent implementations.
    
    Implementations:
    - InMemoryProspectiveMemory: Lightweight, no external dependencies
    - VectorProspectiveMemory: Full-featured with semantic search
    """
    
    @abstractmethod
    def add_reminder(
        self,
        content: str,
        due_time: Optional[Union[datetime, float, int]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Reminder:
        """
        Add a new reminder.
        
        Args:
            content: Reminder description/text
            due_time: When the reminder is due (datetime, or float/int seconds from now for backward compat)
            tags: Optional tags for organization
            metadata: Additional key-value data
            
        Returns:
            The created Reminder object
        """
        pass
    
    @abstractmethod
    def get_reminder(self, reminder_id: str) -> Optional[Reminder]:
        """
        Get a specific reminder by ID.
        
        Args:
            reminder_id: Unique reminder identifier
            
        Returns:
            Reminder object or None if not found
        """
        pass
    
    @abstractmethod
    def get_due_reminders(self, now: Optional[datetime] = None) -> List[Reminder]:
        """
        Get all reminders that are currently due.
        
        Args:
            now: Reference time (defaults to current time)
            
        Returns:
            List of due reminders (not completed, due_time <= now)
        """
        pass
    
    @abstractmethod
    def get_upcoming(self, within: timedelta) -> List[Reminder]:
        """
        Get reminders due within a time window.
        
        Args:
            within: Time window (e.g., timedelta(hours=24))
            
        Returns:
            List of upcoming reminders
        """
        pass
    
    @abstractmethod
    def list_reminders(self, include_completed: bool = False, include_triggered: Optional[bool] = None) -> List[Reminder]:
        """
        List all reminders.
        
        Args:
            include_completed: Whether to include completed reminders
            include_triggered: Backward compat alias for include_completed
            
        Returns:
            List of reminders
        """
        pass
    
    @abstractmethod
    def search_reminders(self, query: str, limit: int = 10) -> List[Reminder]:
        """
        Search reminders by text query.
        
        For in-memory: simple text matching
        For vector store: semantic similarity search
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of matching reminders
        """
        pass
    
    @abstractmethod
    def complete_reminder(self, reminder_id: str) -> bool:
        """
        Mark a reminder as completed.
        
        Args:
            reminder_id: Reminder to complete
            
        Returns:
            True if successful, False if not found
        """
        pass
    
    @abstractmethod
    def delete_reminder(self, reminder_id: str) -> bool:
        """
        Delete a reminder.
        
        Args:
            reminder_id: Reminder to delete
            
        Returns:
            True if successful, False if not found
        """
        pass
    
    @abstractmethod
    def purge_completed(self) -> int:
        """
        Remove all completed reminders.
        
        Returns:
            Number of reminders purged
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Remove all reminders."""
        pass
    
    def check_due(self, now: Optional[datetime] = None) -> List[Reminder]:
        """
        Legacy compatibility method for checking due reminders.
        Alias for get_due_reminders().
        
        Args:
            now: Current time (defaults to datetime.now())
            
        Returns:
            List of due reminders
        """
        return self.get_due_reminders(now)
    
    def upcoming(self, within_seconds: float) -> List[Reminder]:
        """
        Legacy compatibility method for getting upcoming reminders.
        Converts seconds to timedelta and calls get_upcoming().
        
        Args:
            within_seconds: Time window in seconds
            
        Returns:
            List of upcoming reminders
        """
        return self.get_upcoming(within=timedelta(seconds=within_seconds))
    
    def purge_triggered(self) -> int:
        """
        Legacy compatibility method for purging triggered reminders.
        Alias for purge_completed().
        
        Returns:
            Number of reminders purged
        """
        return self.purge_completed()
    
    @staticmethod
    def to_dict(reminder: Reminder) -> Dict[str, Any]:
        """
        Legacy compatibility method for converting Reminder to dict.
        
        Args:
            reminder: Reminder object to convert
            
        Returns:
            Dictionary representation
        """
        return {
            "id": reminder.id,
            "content": reminder.content,
            "due_time": reminder.due_time.isoformat() if reminder.due_time else None,
            "tags": reminder.tags,
            "metadata": reminder.metadata,
            "completed": reminder.completed,
            "completed_at": reminder.completed_at.isoformat() if reminder.completed_at else None,
            "created_at": reminder.created_at.isoformat()
        }
    
    def _increment_metric(self, metric_name: str, value: int = 1) -> None:
        """
        Increment a metric counter (shared helper).
        
        Args:
            metric_name: Name of the metric
            value: Amount to increment
        """
        try:
            from src.memory.metrics import metrics_registry
            metrics_registry.inc(metric_name, value)
        except Exception:
            pass  # Metrics are optional


# ===============================================================================
# In-Memory Implementation
# ===============================================================================

class InMemoryProspectiveMemory(ProspectiveMemorySystem):
    """
    Lightweight in-memory prospective memory implementation.
    
    Features:
    - No external dependencies
    - Fast and simple
    - Not persistent across restarts
    - Simple text-based search
    
    Perfect for:
    - Testing
    - Development
    - Minimal deployments
    - Short-lived reminders
    """
    
    def __init__(self):
        """Initialize in-memory storage."""
        self._reminders: Dict[str, Reminder] = {}
    
    def add_reminder(
        self,
        content: str,
        due_time: Optional[Union[datetime, float, int]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Reminder:
        """Add a new reminder.
        
        Args:
            content: Reminder content/description
            due_time: When reminder is due (datetime or float seconds from now for backward compat)
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            
        Returns:
            Created reminder
        """
        # Backward compatibility: if due_time is a float/int, treat as seconds from now
        if isinstance(due_time, (int, float)):
            due_time = datetime.now() + timedelta(seconds=float(due_time))
        
        reminder = Reminder(
            id=str(uuid.uuid4()),
            content=content,
            due_time=due_time,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        self._reminders[reminder.id] = reminder
        self._increment_metric("prospective_reminders_created_total")
        
        return reminder
    
    def get_reminder(self, reminder_id: str) -> Optional[Reminder]:
        """Get a specific reminder by ID."""
        return self._reminders.get(reminder_id)
    
    def get_due_reminders(self, now: Optional[datetime] = None) -> List[Reminder]:
        """Get all reminders that are currently due."""
        now = now or datetime.now()
        
        due_reminders = []
        for reminder in self._reminders.values():
            if (not reminder.completed and 
                reminder.due_time and 
                reminder.due_time <= now):
                
                # Auto-complete on retrieval (one-shot trigger)
                if not reminder.completed:
                    reminder.completed = True
                    reminder.completed_at = now
                    self._increment_metric("prospective_reminders_triggered_total")
                
                due_reminders.append(reminder)
        
        return sorted(due_reminders, key=lambda r: r.due_time or datetime.min)
    
    def get_upcoming(self, within: timedelta) -> List[Reminder]:
        """Get reminders due within a time window."""
        horizon = datetime.now() + within
        
        upcoming = []
        for reminder in self._reminders.values():
            if (not reminder.completed and 
                reminder.due_time and 
                reminder.due_time <= horizon):
                upcoming.append(reminder)
        
        return sorted(upcoming, key=lambda r: r.due_time or datetime.min)
    
    def list_reminders(self, include_completed: bool = False, include_triggered: Optional[bool] = None) -> List[Reminder]:
        """List all reminders.
        
        Args:
            include_completed: Include completed reminders
            include_triggered: Backward compat alias for include_completed
        """
        # Backward compatibility: include_triggered overrides include_completed
        if include_triggered is not None:
            include_completed = include_triggered
        
        reminders = list(self._reminders.values())
        
        if not include_completed:
            reminders = [r for r in reminders if not r.completed]
        
        return sorted(reminders, key=lambda r: r.due_time or datetime.max)
    
    def search_reminders(self, query: str, limit: int = 10) -> List[Reminder]:
        """Search reminders by simple text matching."""
        query_lower = query.lower()
        
        matches = []
        for reminder in self._reminders.values():
            if query_lower in reminder.content.lower():
                matches.append(reminder)
            elif any(query_lower in tag.lower() for tag in reminder.tags):
                matches.append(reminder)
        
        return matches[:limit]
    
    def complete_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as completed."""
        reminder = self._reminders.get(reminder_id)
        if not reminder:
            return False
        
        if not reminder.completed:
            reminder.completed = True
            reminder.completed_at = datetime.now()
            self._increment_metric("prospective_reminders_completed_total")
        
        return True
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder."""
        if reminder_id in self._reminders:
            del self._reminders[reminder_id]
            self._increment_metric("prospective_reminders_deleted_total")
            return True
        return False
    
    def purge_completed(self) -> int:
        """Remove all completed reminders."""
        to_delete = [rid for rid, r in self._reminders.items() if r.completed]
        
        for rid in to_delete:
            del self._reminders[rid]
        
        if to_delete:
            self._increment_metric("prospective_reminders_purged_total", len(to_delete))
        
        return len(to_delete)
    
    def clear(self) -> None:
        """Remove all reminders."""
        count = len(self._reminders)
        self._reminders.clear()
        if count > 0:
            self._increment_metric("prospective_reminders_cleared_total", count)


# ===============================================================================
# Vector Store Implementation (with optional dependencies)
# ===============================================================================

class VectorProspectiveMemory(ProspectiveMemorySystem):
    """
    Full-featured prospective memory with semantic search.
    
    Features:
    - Semantic similarity search
    - Persistent storage (ChromaDB)
    - Vector embeddings
    - Advanced querying
    
    Requirements:
    - sentence-transformers
    - chromadb
    
    Perfect for:
    - Production deployments
    - Long-term memory
    - Semantic search capabilities
    - Large reminder sets
    """
    
    def __init__(
        self,
        persist_dir: Optional[str] = None,
        collection_name: str = "prospective_reminders",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize vector-backed prospective memory.
        
        Args:
            persist_dir: Directory for ChromaDB persistence
            collection_name: Name of the ChromaDB collection
            embedding_model: Sentence transformer model name
            
        Raises:
            ImportError: If required dependencies are not installed
            RuntimeError: If initialization fails
        """
        # Lazy imports - only fail when actually creating vector store
        try:
            from sentence_transformers import SentenceTransformer
            import chromadb
            from chromadb.config import Settings
        except ImportError as e:
            raise ImportError(
                "VectorProspectiveMemory requires sentence-transformers and chromadb. "
                "Install with: pip install sentence-transformers chromadb\n"
                "Or use InMemoryProspectiveMemory for a lightweight alternative."
            ) from e
        
        self.persist_dir = Path(persist_dir or "data/memory_stores/chroma_prospective")
        self.collection_name = collection_name
        
        # Initialize embedding model
        try:
            self.embedding_model = SentenceTransformer(embedding_model)
            # Use GPU if available
            try:
                import torch
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to("cuda")
            except ImportError:
                pass  # torch not available, use CPU
        except Exception as e:
            raise RuntimeError(f"Failed to load embedding model '{embedding_model}': {e}") from e
        
        # Initialize ChromaDB
        try:
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                persist_directory=str(self.persist_dir)
            )
            
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.persist_dir),
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
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}") from e
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text."""
        try:
            embedding = self.embedding_model.encode(str(text))
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            elif hasattr(embedding, '__iter__'):
                return [float(x) for x in embedding]
            else:
                raise ValueError("Unable to convert embedding to list")
        except Exception as e:
            raise RuntimeError(f"Failed to generate embedding: {e}") from e
    
    def _reminder_to_metadata(self, reminder: Reminder) -> Dict[str, Any]:
        """Convert reminder to ChromaDB metadata format."""
        return {
            "content": reminder.content,
            "due_time": reminder.due_time.isoformat() if reminder.due_time else "",
            "created_at": reminder.created_at.isoformat(),
            "completed": reminder.completed,
            "completed_at": reminder.completed_at.isoformat() if reminder.completed_at else "",
            "tags": ",".join(reminder.tags),  # ChromaDB requires scalar values
            # Store metadata as JSON string if needed
        }
    
    # Add helper to safely normalize whatever Chroma returns into a dict
    def _normalize_metadata(self, metadata: Any) -> Dict[str, Any]:
        """Ensure metadata is a plain dict. Accepts None or Mapping-like inputs."""
        if metadata is None:
            return {}
        try:
            # If metadata is already a dict-like mapping, make a shallow dict copy
            return dict(metadata)  # type: ignore[arg-type]
        except Exception:
            # Fallback to an empty dict on unexpected types
            return {}

    def _metadata_to_reminder(self, reminder_id: str, metadata: Dict[str, Any], document: str) -> Reminder:
        """Convert ChromaDB metadata to Reminder object."""
        return Reminder(
            id=reminder_id,
            content=metadata.get("content", document),
            due_time=datetime.fromisoformat(metadata["due_time"]) if metadata.get("due_time") else None,
            created_at=datetime.fromisoformat(metadata["created_at"]) if metadata.get("created_at") else datetime.now(),
            completed=metadata.get("completed", False),
            completed_at=datetime.fromisoformat(metadata["completed_at"]) if metadata.get("completed_at") else None,
            tags=metadata.get("tags", "").split(",") if metadata.get("tags") else [],
            metadata={}
        )
    
    def add_reminder(
        self,
        content: str,
        due_time: Optional[Union[datetime, float, int]] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Reminder:
        """Add a new reminder with vector embedding.
        
        Args:
            content: Reminder content/description
            due_time: When reminder is due (datetime or float seconds from now for backward compat)
            tags: Optional list of tags
            metadata: Optional metadata dictionary
            
        Returns:
            Created reminder
        """
        # Backward compatibility: if due_time is a float/int, treat as seconds from now
        if isinstance(due_time, (int, float)):
            due_time = datetime.now() + timedelta(seconds=float(due_time))
        
        reminder = Reminder(
            id=str(uuid.uuid4()),
            content=content,
            due_time=due_time,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Store in ChromaDB
        self.collection.upsert(
            ids=[reminder.id],
            embeddings=[embedding],
            documents=[content],
            metadatas=[self._reminder_to_metadata(reminder)]
        )
        
        self._increment_metric("prospective_reminders_created_total")
        
        return reminder
    
    def get_reminder(self, reminder_id: str) -> Optional[Reminder]:
        """Get a specific reminder by ID."""
        try:
            result = self.collection.get(ids=[reminder_id])

            ids = result.get('ids') or []
            if not ids:
                return None

            metadatas = result.get('metadatas') or []
            documents = result.get('documents') or []

            raw_meta = metadatas[0] if len(metadatas) > 0 else {}
            meta = self._normalize_metadata(raw_meta)
            document = documents[0] if len(documents) > 0 else ""

            return self._metadata_to_reminder(
                ids[0],
                meta,
                document
            )
        except Exception:
            return None
    
    def get_due_reminders(self, now: Optional[datetime] = None) -> List[Reminder]:
        """Get all reminders that are currently due."""
        now = now or datetime.now()
        
        try:
            result = self.collection.get()

            ids = result.get('ids') or []
            metadatas = result.get('metadatas') or []
            documents = result.get('documents') or []

            due_reminders = []
            for i, reminder_id in enumerate(ids):
                if i < len(metadatas):
                    raw_meta = metadatas[i]
                    metadata = self._normalize_metadata(raw_meta)
                    document = documents[i] if i < len(documents) else ""
                    
                    if metadata.get("completed", False):
                        continue

                    due_time_str = metadata.get("due_time") or ""
                    if not due_time_str:
                        continue

                    try:
                        due_time = datetime.fromisoformat(due_time_str)
                        if due_time <= now:
                            reminder = self._metadata_to_reminder(reminder_id, metadata, document)
                            due_reminders.append(reminder)

                            # Auto-complete on retrieval
                            self.complete_reminder(reminder_id)
                            self._increment_metric("prospective_reminders_triggered_total")
                    except (ValueError, TypeError):
                        continue  # Skip malformed dates

            return sorted(due_reminders, key=lambda r: r.due_time or datetime.min)
        except Exception:
            return []
    
    def get_upcoming(self, within: timedelta) -> List[Reminder]:
        """Get reminders due within a time window."""
        horizon = datetime.now() + within
        
        try:
            result = self.collection.get()

            ids = result.get('ids') or []
            metadatas = result.get('metadatas') or []
            documents = result.get('documents') or []

            upcoming = []
            for i, reminder_id in enumerate(ids):
                if i < len(metadatas):
                    raw_meta = metadatas[i]
                    metadata = self._normalize_metadata(raw_meta)
                    document = documents[i] if i < len(documents) else ""

                    if metadata.get("completed", False):
                        continue

                    due_time_str = metadata.get("due_time") or ""
                    if not due_time_str:
                        continue

                    try:
                        due_time = datetime.fromisoformat(due_time_str)
                        if due_time <= horizon:
                            reminder = self._metadata_to_reminder(reminder_id, metadata, document)
                            upcoming.append(reminder)
                    except (ValueError, TypeError):
                        continue

            return sorted(upcoming, key=lambda r: r.due_time or datetime.min)
        except Exception:
            return []
    
    def list_reminders(self, include_completed: bool = False, include_triggered: Optional[bool] = None) -> List[Reminder]:
        """List all reminders.
        
        Args:
            include_completed: Include completed reminders
            include_triggered: Backward compat alias for include_completed
        """
        # Backward compatibility: include_triggered overrides include_completed
        if include_triggered is not None:
            include_completed = include_triggered
        
        try:
            result = self.collection.get()

            ids = result.get('ids') or []
            metadatas = result.get('metadatas') or []
            documents = result.get('documents') or []

            reminders = []
            for i, reminder_id in enumerate(ids):
                if i < len(metadatas):
                    raw_meta = metadatas[i]
                    metadata = self._normalize_metadata(raw_meta)
                    document = documents[i] if i < len(documents) else ""

                    if not include_completed and metadata.get("completed", False):
                        continue

                    reminder = self._metadata_to_reminder(reminder_id, metadata, document)
                    reminders.append(reminder)

            return sorted(reminders, key=lambda r: r.due_time or datetime.max)
        except Exception:
            return []
    
    def search_reminders(self, query: str, limit: int = 10) -> List[Reminder]:
        """Search reminders by semantic similarity."""
        try:
            embedding = self._generate_embedding(query)

            result = self.collection.query(
                query_embeddings=[embedding],
                n_results=limit,
                include=["documents", "metadatas"]
            )

            reminders = []
            ids_groups = result.get('ids') or []
            metadatas_groups = result.get('metadatas') or []
            docs_groups = result.get('documents') or []

            if ids_groups and ids_groups[0]:
                group = ids_groups[0]
                meta_group = metadatas_groups[0] if metadatas_groups and metadatas_groups[0] else []
                doc_group = docs_groups[0] if docs_groups and docs_groups[0] else []

                for i, reminder_id in enumerate(group):
                    raw_meta = meta_group[i] if i < len(meta_group) else {}
                    metadata = self._normalize_metadata(raw_meta)
                    document = doc_group[i] if i < len(doc_group) else ""

                    # Skip completed
                    if metadata.get("completed", False):
                        continue

                    reminder = self._metadata_to_reminder(reminder_id, metadata, document)
                    reminders.append(reminder)

            return reminders
        except Exception:
            return []
    
    def complete_reminder(self, reminder_id: str) -> bool:
        """Mark a reminder as completed."""
        try:
            result = self.collection.get(ids=[reminder_id])

            ids = result.get('ids') or []
            if not ids:
                return False

            metadatas = result.get('metadatas') or []
            raw_meta = metadatas[0] if len(metadatas) > 0 else {}
            metadata = self._normalize_metadata(raw_meta)

            metadata['completed'] = True
            metadata['completed_at'] = datetime.now().isoformat()

            self.collection.update(
                ids=[reminder_id],
                metadatas=[metadata]
            )

            self._increment_metric("prospective_reminders_completed_total")
            return True
        except Exception:
            return False
    
    def delete_reminder(self, reminder_id: str) -> bool:
        """Delete a reminder."""
        try:
            self.collection.delete(ids=[reminder_id])
            self._increment_metric("prospective_reminders_deleted_total")
            return True
        except Exception:
            return False
    
    def purge_completed(self) -> int:
        """Remove all completed reminders."""
        try:
            result = self.collection.get()
            
            ids = result.get('ids') or []
            metadatas = result.get('metadatas') or []
            
            to_delete = []
            for i, reminder_id in enumerate(ids):
                if i < len(metadatas):
                    raw_meta = metadatas[i]
                    metadata = self._normalize_metadata(raw_meta)
                    if metadata.get("completed", False):
                        to_delete.append(reminder_id)
            
            if to_delete:
                self.collection.delete(ids=to_delete)
                self._increment_metric("prospective_reminders_purged_total", len(to_delete))
            
            return len(to_delete)
        except Exception:
            return 0
    
    def clear(self) -> None:
        """Remove all reminders."""
        try:
            result = self.collection.get()
            ids = result.get('ids', [])
            
            if ids:
                self.collection.delete(ids=ids)
                self._increment_metric("prospective_reminders_cleared_total", len(ids))
        except Exception:
            pass


# ===============================================================================
# Factory and Singleton
# ===============================================================================

_prospective_memory_singleton: Optional[ProspectiveMemorySystem] = None


def create_prospective_memory(
    use_vector: bool = False,
    **kwargs
) -> ProspectiveMemorySystem:
    """
    Factory function to create prospective memory instance.
    
    Args:
        use_vector: If True, try to create VectorProspectiveMemory,
                   falling back to InMemory if dependencies unavailable
        **kwargs: Additional arguments for the implementation
        
    Returns:
        ProspectiveMemorySystem instance
    """
    if use_vector:
        try:
            return VectorProspectiveMemory(**kwargs)
        except ImportError:
            # Fall back to in-memory if dependencies not available
            import warnings
            warnings.warn(
                "Vector prospective memory dependencies not available. "
                "Falling back to in-memory implementation. "
                "Install with: pip install sentence-transformers chromadb"
            )
            return InMemoryProspectiveMemory()
        except Exception as e:
            import warnings
            warnings.warn(f"Failed to initialize vector prospective memory: {e}. Using in-memory.")
            return InMemoryProspectiveMemory()
    else:
        return InMemoryProspectiveMemory()


def get_prospective_memory(use_vector: bool = False) -> ProspectiveMemorySystem:
    """
    Get or create singleton prospective memory instance.
    
    Args:
        use_vector: Whether to use vector-backed implementation
        
    Returns:
        Singleton ProspectiveMemorySystem instance
    """
    global _prospective_memory_singleton
    
    if _prospective_memory_singleton is None:
        _prospective_memory_singleton = create_prospective_memory(use_vector=use_vector)
    
    return _prospective_memory_singleton


def reset_prospective_memory() -> None:
    """Reset singleton (primarily for testing)."""
    global _prospective_memory_singleton
    _prospective_memory_singleton = None


# ===============================================================================
# Backward Compatibility Aliases
# ===============================================================================

# Alias for backward compatibility
ProspectiveMemory = InMemoryProspectiveMemory
ProspectiveMemoryVectorStore = VectorProspectiveMemory

# Backward compatible singleton
def get_inmemory_prospective_memory() -> InMemoryProspectiveMemory:
    """Get in-memory prospective memory (backward compatibility)."""
    return get_prospective_memory(use_vector=False)  # type: ignore
