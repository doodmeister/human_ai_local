"""
Enhanced Long-Term Memory (LTM) System with Vector Database Integration
Implements ChromaDB for semantic memory storage and retrieval
"""
from typing import TYPE_CHECKING, Dict, List, Set, Optional, Any, Tuple, Sequence  # Add Sequence
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import types  # <-- added

from ..base import BaseMemorySystem  # Add import for base class

if TYPE_CHECKING:
    # for mypy/pyright: use real type signature
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
else:
    try:
        from sentence_transformers import SentenceTransformer
        SENTENCE_TRANSFORMERS_AVAILABLE = True
    except ImportError:
        SENTENCE_TRANSFORMERS_AVAILABLE = False
        class SentenceTransformer:
            def __init__(self, *args, **kwargs):
                raise ImportError("sentence-transformers library is not installed")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = types.ModuleType("chromadb")  # ensure chromadb is always a module
    Settings = None  # type: ignore

from .long_term_memory import LTMRecord  # Import existing record structure

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector similarity search"""
    record: LTMRecord
    similarity_score: float
    distance: float
    
class VectorLongTermMemory(BaseMemorySystem):
    def add_feedback(self, memory_id: str, feedback_type: str, value: Any, comment: Optional[str] = None, user_id: Optional[str] = None):
        """Add user feedback to a memory record (vector LTM)."""
        from datetime import datetime
        record = self.memories.get(memory_id)
        if not record:
            raise KeyError(f"Memory ID {memory_id} not found.")
        # Ensure feedback field exists (for compatibility)
        if not hasattr(record, "feedback"):
            record.feedback = []
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "value": value,
            "comment": comment,
            "user_id": user_id
        }
        record.feedback.append(event)
        # Optionally update memory fields based on feedback type
        if feedback_type == "relevance":
            record.confidence = min(1.0, max(0.0, float(value) / 5.0))
        elif feedback_type == "importance":
            record.importance = min(1.0, max(0.0, float(value) / 5.0))
        elif feedback_type == "emotion":
            record.emotional_valence = float(value)
        # Save updated record (JSON backup)
        if self.enable_json_backup:
            self._save_memory_json(record)
        # Optionally update in ChromaDB (not supported for feedback fields)
        logger.debug(f"Added feedback to LTM record {memory_id}: {event}")

    def get_feedback(self, memory_id: str) -> list:
        """Return all feedback events for a memory."""
        record = self.memories.get(memory_id)
        if not record:
            raise KeyError(f"Memory ID {memory_id} not found.")
        return getattr(record, "feedback", [])

    def get_feedback_summary(self, memory_id: str) -> dict:
        """Return summary statistics for feedback on a memory."""
        record = self.memories.get(memory_id)
        if not record:
            raise KeyError(f"Memory ID {memory_id} not found.")
        summary = {}
        for event in getattr(record, "feedback", []):
            t = event["type"]
            summary.setdefault(t, []).append(event["value"])
        stats = {k: (sum(map(float, v))/len(v) if v else 0) for k, v in summary.items()}
        stats["count"] = len(getattr(record, "feedback", []))
        return stats
    """
    Enhanced Long-Term Memory with ChromaDB vector database integration
    Implements the unified memory interface.
    
    Features:
    - Semantic similarity search with embeddings
    - ChromaDB vector storage for fast retrieval
    - Backward compatibility with JSON storage
    - Batch operations for performance
    - Advanced search capabilities
    """
    
    def __init__(
        self, 
        storage_path: Optional[str] = None,
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "ltm_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_vector_db: bool = True,
        enable_json_backup: bool = True
    ):
        """
        Initialize Enhanced LTM system
        
        Args:
            storage_path: Path for JSON backup storage
            chroma_persist_dir: Path for ChromaDB persistence
            collection_name: Name of ChromaDB collection
            embedding_model: SentenceTransformer model name
            use_vector_db: Whether to use ChromaDB (fallback to JSON if False)
            enable_json_backup: Whether to maintain JSON backup files
        """
        # Storage paths
        self.storage_path = Path(storage_path or "data/memory_stores/ltm")
        self.chroma_persist_dir = Path(chroma_persist_dir or "data/memory_stores/chroma_ltm")
        self.collection_name = collection_name
        self.use_vector_db = use_vector_db and CHROMADB_AVAILABLE
        self.enable_json_backup = enable_json_backup
        
        # Create directories
        self.storage_path.mkdir(parents=True, exist_ok=True)
        if self.use_vector_db:
            self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory indices (for compatibility)
        self.memories: Dict[str, LTMRecord] = {}
        self.tags_index: Dict[str, Set[str]] = {}
        self.associations_index: Dict[str, List[str]] = {}
        
        # Initialize embedding model
        self.embedding_model = None
        if self.use_vector_db and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"Failed to load embedding model {embedding_model}: {e}")
                self.embedding_model = None
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        if self.use_vector_db:
            self._initialize_chromadb()
        
        # Load existing memories
        self._load_memories()
        
        logger.info(f"Enhanced LTM initialized with {len(self.memories)} memories")
        logger.info(f"Vector DB enabled: {self.use_vector_db}, JSON backup: {self.enable_json_backup}")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, falling back to JSON storage")
            self.use_vector_db = False
            return

        # build settings if available
        settings = None
        if Settings:
            try:
                # new API may expect persist_directory inside Settings
                settings = Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    persist_directory=str(self.chroma_persist_dir)
                )
            except TypeError:
                # fallback for older Settings signature
                settings = Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )

        # pick whichever client class exists
        client_cls = getattr(chromadb, "PersistentClient", None) or getattr(chromadb, "Client", None)
        if client_cls is None or settings is None:
            logger.warning("ChromaDB client class or Settings not available, falling back to JSON storage")
            self.use_vector_db = False
            return

        try:
            # initialize appropriate client
            if hasattr(chromadb, "PersistentClient"):
                self.chroma_client = client_cls(
                    path=str(self.chroma_persist_dir),
                    settings=settings
                )
            else:
                # new API
                self.chroma_client = client_cls(settings)
            # get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.info(f"Connected to existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Long-term memory storage with semantic search"}
                )
                logger.info(f"Created new ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            self.use_vector_db = False
            self.chroma_client = None
            self.collection = None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text content"""
        if not self.embedding_model or not text:
            return None
        
        try:
            # Convert content to string if needed
            content_str = str(text)
            embedding = self.embedding_model.encode(content_str)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def _content_to_text(self, content: Any) -> str:
        """Convert content to searchable text"""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Extract text from dictionary
            text_parts = []
            for key, value in content.items():
                if isinstance(value, (str, int, float)):
                    text_parts.append(f"{key}: {value}")
            return " ".join(text_parts)
        else:
            return str(content)
    
    def store(
        self,
        memory_id: str,
        content: Any,
        memory_type: str = "episodic",
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        source: str = "unknown",
        tags: Optional[List[str]] = None,
        associations: Optional[List[str]] = None
    ) -> bool:
        """
        Store memory with vector database integration (unified interface)
        
        Args:
            memory_id: Unique identifier
            content: Memory content
            memory_type: Type of memory ("episodic", "semantic", "procedural")
            importance: Importance score (0.0 to 1.0)
            emotional_valence: Emotional weight (-1.0 to 1.0)
            source: Source of the memory
            tags: Associated tags
            associations: Associated memory IDs
        
        Returns:
            True if stored successfully
        """
        # Update existing or create new record
        if memory_id in self.memories:
            record = self.memories[memory_id]
            record.content = content
            record.reinforce()
            record.update_access()
        else:
            record = LTMRecord(
                id=memory_id,
                content=content,
                memory_type=memory_type,
                encoding_time=datetime.now(),
                last_access=datetime.now(),
                importance=importance,
                emotional_valence=emotional_valence,
                source=source,
                tags=tags or [],
                associations=associations or []
            )
            self.memories[memory_id] = record
        
        # Update in-memory indices
        self._update_indices(record)
        
        # Store in ChromaDB if available
        if self.use_vector_db and self.collection is not None:
            self._store_in_chromadb(record)
        
        # Save JSON backup if enabled
        if self.enable_json_backup:
            self._save_memory_json(record)
        
        logger.debug(f"Stored LTM record {memory_id} (type: {memory_type})")
        return True
    
    def _store_in_chromadb(self, record: LTMRecord):
        """Store record in ChromaDB collection"""
        # Guard against missing collection
        if self.collection is None:
            logger.warning(f"ChromaDB collection unavailable, skipping storage for {record.id}")
            return
        
        try:
            content_text = self._content_to_text(record.content)
            embedding = self._generate_embedding(content_text)
            
            # Prepare metadata
            metadata = {
                "memory_type": record.memory_type,
                "importance": record.importance,
                "emotional_valence": record.emotional_valence,
                "source": record.source,
                "encoding_time": record.encoding_time.isoformat(),
                "last_access": record.last_access.isoformat(),
                "access_count": record.access_count,
                "confidence": record.confidence,
                "consolidation_count": record.consolidation_count,
                "tags": ",".join(record.tags) if record.tags else "",
                "associations": ",".join(record.associations) if record.associations else ""
            }
            
            # Check if already exists
            try:
                existing = self.collection.get(ids=[record.id])
                if existing['ids']:
                    # Update existing
                    self.collection.update(
                        ids=[record.id],
                        documents=[content_text],
                        metadatas=[metadata],
                        embeddings=[embedding] if embedding else None
                    )
                else:
                    # Add new
                    self.collection.add(
                        ids=[record.id],
                        documents=[content_text],
                        metadatas=[metadata],
                        embeddings=[embedding] if embedding else None
                    )
            except Exception:
                # Add new (fallback)
                self.collection.add(
                    ids=[record.id],
                    documents=[content_text],
                    metadatas=[metadata],
                    embeddings=[embedding] if embedding else None
                )
                
        except Exception as e:
            logger.error(f"Failed to store record {record.id} in ChromaDB: {e}")
    
    def retrieve(self, memory_id: str) -> Optional[dict]:
        """
        Retrieve memory by ID (unified interface)
        
        Args:
            memory_id: Unique identifier
        
        Returns:
            Memory content as dict, or None if not found
        """
        record = self.memories.get(memory_id)
        if record:
            record.update_access()
            # Save access update if needed
            if self.enable_json_backup:
                self._save_memory_json(record)
            return record.to_dict()
        return None
    
    def delete(self, memory_id: str) -> bool:
        """
        Delete memory by ID (unified interface)
        
        Args:
            memory_id: Unique identifier
        
        Returns:
            True if deleted, False otherwise
        """
        removed = False
        # Remove from in-memory store
        if memory_id in self.memories:
            record = self.memories.pop(memory_id)
            removed = True
            # Remove from indices
            for tag in record.tags:
                if tag in self.tags_index:
                    self.tags_index[tag].discard(memory_id)
                    if not self.tags_index[tag]:
                        del self.tags_index[tag]
            if memory_id in self.associations_index:
                del self.associations_index[memory_id]
        # Remove from ChromaDB
        if self.collection and removed:
            try:
                self.collection.delete(ids=[memory_id])
            except Exception as e:
                logger.error(f"Failed to remove {memory_id} from ChromaDB: {e}")
        # Remove JSON file
        if self.enable_json_backup and removed:
            json_file = self.storage_path / f"{memory_id}.json"
            if json_file.exists():
                try:
                    json_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove JSON file for {memory_id}: {e}")
        if removed:
            logger.debug(f"Removed memory {memory_id} from LTM")
        return removed

    def search(
        self,
        query: Optional[str] = None,
        **kwargs
    ) -> Sequence[dict]:
        """
        Search for memories (unified interface).
        If query is provided, perform semantic search; otherwise, support tag/content search via kwargs.
        
        Args:
            query: Optional search query
            **kwargs: Additional search parameters (tags, content_query, etc.)
        
        Returns:
            Sequence of matching memory dicts
        """
        # Semantic search if query is provided
        if query:
            results = self.search_semantic(query=query, max_results=kwargs.get('max_results', 10))
            return [r.record.to_dict() for r in results]
        # Tag-based search
        tags = kwargs.get('tags')
        if tags:
            tag_results = self.search_by_tags(tags, operator=kwargs.get('operator', 'OR'))
            return [r.to_dict() for r in tag_results]
        # Content-based search
        content_query = kwargs.get('content_query')
        if content_query:
            content_results = self.search_by_content(content_query)
            return [r[0].to_dict() for r in content_results]
        # Return all if no filter
        return [r.to_dict() for r in self.memories.values()]
    
    def search_semantic(
        self,
        query: str,
        max_results: int = 10,
        min_similarity: float = 0.5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0
    ) -> List[VectorSearchResult]:
        """
        Semantic search using vector similarity
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_similarity: Minimum similarity threshold
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold
        
        Returns:
            List of VectorSearchResult objects
        """
        if not self.use_vector_db or not self.collection:
            # Fallback to content-based search
            return self._fallback_semantic_search(query, max_results, memory_types, min_importance)
        
        try:
            # Build where clause for filtering
            where_clause = {}
            if memory_types:
                where_clause["memory_type"] = {"$in": memory_types}
            if min_importance > 0.0:
                where_clause["importance"] = {"$gte": min_importance}
            
            # Perform vector search
            results = self.collection.query(
                query_texts=[query],
                n_results=max_results,
                where=where_clause if where_clause else None
            )
            
            # === ADDED GUARD ===
            # If distances or ids are missing, fallback to simple search
            if not results.get('ids') or not results.get('distances') \
               or not results['ids'][0] or not results['distances'][0]:
                logger.warning("ChromaDB returned no distance data, falling back to content search")
                return self._fallback_semantic_search(query, max_results, memory_types, min_importance)
            # === END GUARD ===

            # Convert to VectorSearchResult objects
            search_results = []
            for i, memory_id in enumerate(results['ids'][0]):
                record = self.memories.get(memory_id)
                if record:
                    distance = results['distances'][0][i]
                    similarity = 1.0 - distance  # Convert distance to similarity
                    if similarity >= min_similarity:
                        search_results.append(VectorSearchResult(
                            record=record,
                            similarity_score=similarity,
                            distance=distance
                        ))
                        record.update_access()
            return search_results

        except Exception as e:
            logger.error(f"ChromaDB search failed: {e}")
            return self._fallback_semantic_search(query, max_results, memory_types, min_importance)
    
    def _fallback_semantic_search(
        self,
        query: str,
        max_results: int,
        memory_types: Optional[List[str]],
        min_importance: float
    ) -> List[VectorSearchResult]:
        """Fallback search using content matching"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for record in self.memories.values():
            # Filter by type and importance
            if memory_types and record.memory_type not in memory_types:
                continue
            if record.importance < min_importance:
                continue
            
            # Calculate simple relevance score
            content_str = str(record.content).lower()
            relevance = 0.0
            
            # Exact phrase match
            if query_lower in content_str:
                relevance += 1.0
            
            # Word overlap
            content_words = set(content_str.split())
            overlap = len(query_words.intersection(content_words))
            if overlap > 0:
                relevance += 0.5 * (overlap / len(query_words))
            
            # Boost by importance and access frequency
            relevance *= (1.0 + record.importance)
            relevance *= (1.0 + min(record.access_count / 10.0, 1.0))
            
            if relevance > 0:
                results.append(VectorSearchResult(
                    record=record,
                    similarity_score=relevance,
                    distance=1.0 - relevance
                ))
        
        # Sort by similarity and return top results
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:max_results]
    
    def search_by_content(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        max_results: int = 10
    ) -> List[Tuple[LTMRecord, float]]:
        """
        Compatibility method for existing search_by_content interface
        """
        vector_results = self.search_semantic(
            query=query,
            max_results=max_results,
            memory_types=memory_types,
            min_importance=min_importance
        )
        
        # Convert to old format for compatibility
        return [(result.record, result.similarity_score) for result in vector_results]
    
    def search_by_tags(self, tags: List[str], operator: str = "OR") -> List[LTMRecord]:
        """Search memories by tags (unchanged from original implementation)"""
        if operator == "OR":
            memory_ids = set()
            for tag in tags:
                if tag in self.tags_index:
                    memory_ids.update(self.tags_index[tag])
        else:  # AND
            memory_ids = None
            for tag in tags:
                if tag in self.tags_index:
                    tag_ids = set(self.tags_index[tag])
                    if memory_ids is None:
                        memory_ids = tag_ids
                    else:
                        memory_ids = memory_ids.intersection(tag_ids)
                else:
                    return []  # Tag not found, no results
            
            if memory_ids is None:
                memory_ids = set()
        
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]
    
    def get_status(self) -> Dict[str, Any]:
        """Get current LTM status including vector DB info"""
        status = {
            "total_memories": len(self.memories),
            "memory_types": list(set(record.memory_type for record in self.memories.values())),
            "total_tags": len(self.tags_index),
            "storage_path": str(self.storage_path),
            "vector_db_enabled": self.use_vector_db,
            "chroma_persist_dir": str(self.chroma_persist_dir) if self.use_vector_db else None,
            "collection_name": self.collection_name if self.use_vector_db else None,
            "embedding_model": getattr(self.embedding_model, 'model_name', None) if self.embedding_model else None,
            "json_backup_enabled": self.enable_json_backup
        }
        
        # Add ChromaDB stats if available
        if self.use_vector_db and self.collection:
            try:
                collection_count = self.collection.count()
                status["vector_db_count"] = collection_count
            except Exception as e:
                status["vector_db_error"] = str(e)
                return status
        
        # Ensure we return status on all code paths
        return status
    
    def consolidate_from_stm(self, stm_items: List[Any]) -> int:
        """
        Consolidate items from STM into LTM
        
        Args:
            stm_items: List of STM memory items
        
        Returns:
            Number of items consolidated
        """
        consolidated = 0
        for item in stm_items:
            # Determine if item should be consolidated based on importance and access
            if item.importance > 0.6 or item.access_count > 2:
                # Convert STM item to LTM record
                success = self.store(
                    memory_id=item.id,
                    content=item.content,
                    memory_type='episodic',
                    importance=item.importance,
                    tags=[],
                    associations=getattr(item, 'associations', [])
                )
                if success:
                    consolidated += 1
        
        logger.info(f"Consolidated {consolidated} items from STM to LTM")
        return consolidated
    
    def remove(self, memory_id: str) -> bool:
        """
        Remove memory from all storage systems
        
        Args:
            memory_id: ID of memory to remove
        
        Returns:
            True if removed successfully
        """
        removed = False
        
        # Remove from in-memory store
        if memory_id in self.memories:
            record = self.memories.pop(memory_id)
            removed = True
            
            # Remove from indices
            for tag in record.tags:
                if tag in self.tags_index:
                    self.tags_index[tag].discard(memory_id)
                    if not self.tags_index[tag]:
                        del self.tags_index[tag]
            
            if memory_id in self.associations_index:
                del self.associations_index[memory_id]
        
        # Remove from ChromaDB
        if self.collection and removed:
            try:
                self.collection.delete(ids=[memory_id])
            except Exception as e:
                logger.error(f"Failed to remove {memory_id} from ChromaDB: {e}")
        
        # Remove JSON file
        if self.enable_json_backup and removed:
            json_file = self.storage_path / f"{memory_id}.json"
            if json_file.exists():
                try:
                    json_file.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove JSON file for {memory_id}: {e}")
        
        if removed:
            logger.debug(f"Removed memory {memory_id} from LTM")
        
        return removed

    def search_associations(self, memory_id: str, max_depth: int = 2) -> List[LTMRecord]:
        """
        Search for memories associated with a given memory
        
        Args:
            memory_id: Starting memory ID
            max_depth: Maximum depth of association traversal
        
        Returns:
            List of associated LTMRecord objects
        """
        visited = set()
        result = []
        queue = [(memory_id, 0)]
        
        while queue:
            current_id, depth = queue.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            if current_id in self.associations_index:
                for assoc_id in self.associations_index[current_id]:
                    if assoc_id in self.memories and assoc_id not in visited:
                        result.append(self.memories[assoc_id])
                        if depth < max_depth:
                            queue.append((assoc_id, depth + 1))
        
        return result

    def get_memories_by_type(self, memory_type: str) -> List[LTMRecord]:
        """Get all memories of a specific type"""
        return [record for record in self.memories.values() if record.memory_type == memory_type]

    def get_recent_memories(self, hours: int = 24, max_results: int = 50) -> List[LTMRecord]:
        """Get recently accessed memories"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent = [
            record for record in self.memories.values()
            if record.last_access > cutoff_time
        ]
        # Sort by last access time, most recent first
        recent.sort(key=lambda x: x.last_access, reverse=True)
        return recent[:max_results]

    def get_important_memories(self, min_importance: float = 0.7, max_results: int = 50) -> List[LTMRecord]:
        """Get memories above importance threshold"""
        important = [
            record for record in self.memories.values()
            if record.importance >= min_importance
        ]
        # Sort by importance, highest first
        important.sort(key=lambda x: x.importance, reverse=True)
        return important[:max_results]

    def _update_indices(self, record: LTMRecord):
        """Update search indices for a memory record"""
        # Update tags index
        for tag in record.tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = set()
            self.tags_index[tag].add(record.id)
        
        # Update associations index
        self.associations_index[record.id] = record.associations[:]
    
    def _save_memory_json(self, record: LTMRecord):
        """Save individual memory record to JSON file"""
        if not self.enable_json_backup:
            return
            
        memory_file = self.storage_path / f"{record.id}.json"
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(record.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory {record.id} to JSON: {e}")
    
    def _load_memories(self):
        """Load memories from both JSON and ChromaDB"""
        # Load from JSON files first
        self._load_from_json()
        
        # If using ChromaDB, synchronize or load from there
        if self.use_vector_db and self.collection:
            self._load_from_chromadb()
    
    def _load_from_json(self):
        """Load memories from JSON files"""
        if not self.storage_path.exists():
            return
        
        loaded = 0
        for memory_file in self.storage_path.glob("*.json"):
            try:
                with open(memory_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                record = LTMRecord.from_dict(data)
                self.memories[record.id] = record
                self._update_indices(record)
                loaded += 1
                
            except Exception as e:
                logger.error(f"Failed to load memory from {memory_file}: {e}")
        if loaded > 0:
            logger.info(f"Loaded {loaded} memories from JSON files")
    
    def _load_from_chromadb(self):
        """Load memories from ChromaDB and sync with in-memory store"""
        if not self.collection:
            return
            
        try:
            # Get all records from ChromaDB
            all_results = self.collection.get()
            
            if all_results['ids']:
                for i, memory_id in enumerate(all_results['ids']):
                    if memory_id not in self.memories:
                        # Reconstruct record from ChromaDB metadata
                        metadata = all_results['metadatas'][i] if all_results['metadatas'] else {}
                        document = all_results['documents'][i] if all_results['documents'] else ""
                        
                        try:
                            # Safe type conversion helpers
                            def get_str(key: str, default: str) -> str:
                                val = metadata.get(key, default)
                                return str(val) if val is not None else default
                            
                            def get_int(key: str, default: int) -> int:
                                val = metadata.get(key, default)
                                try:
                                    return int(val) if val is not None else default
                                except (ValueError, TypeError):
                                    return default
                            
                            def get_float(key: str, default: float) -> float:
                                val = metadata.get(key, default)
                                try:
                                    return float(val) if val is not None else default
                                except (ValueError, TypeError):
                                    return default
                            
                            def get_list(key: str) -> List[str]:
                                val = metadata.get(key, '')
                                if isinstance(val, str) and val:
                                    return [tag.strip() for tag in val.split(',') if tag.strip()]
                                return []
                            
                            # Parse datetime safely
                            def get_datetime(key: str) -> datetime:
                                val = metadata.get(key)
                                if isinstance(val, str):
                                    try:
                                        return datetime.fromisoformat(val)
                                    except ValueError:
                                        pass
                                return datetime.now()
                            
                            record = LTMRecord(
                                id=memory_id,
                                content=document,  # Use document as content
                                memory_type=get_str('memory_type', 'episodic'),
                                encoding_time=get_datetime('encoding_time'),
                                last_access=get_datetime('last_access'),
                                access_count=get_int('access_count', 0),
                                importance=get_float('importance', 0.5),
                                emotional_valence=get_float('emotional_valence', 0.0),
                                confidence=get_float('confidence', 1.0),
                                source=get_str('source', 'unknown'),
                                tags=get_list('tags'),
                                associations=get_list('associations'),
                                consolidation_count=get_int('consolidation_count', 0)
                            )
                            
                            self.memories[memory_id] = record
                            self._update_indices(record)
                            
                        except Exception as e:
                            logger.error(f"Failed to reconstruct record {memory_id} from ChromaDB: {e}")
                
                logger.info(f"Synchronized with ChromaDB: {len(all_results['ids'])} records")
        
        except Exception as e:
            logger.error(f"Failed to load from ChromaDB: {e}")

    def save_all(self):
        """Save all memories to persistent storage"""
        if self.enable_json_backup:
            for record in self.memories.values():
                self._save_memory_json(record)
            logger.info(f"Saved {len(self.memories)} memories to JSON files")
        
        # ChromaDB saves automatically with persistence
        if self.use_vector_db:
            logger.info("ChromaDB persistence is automatic")
