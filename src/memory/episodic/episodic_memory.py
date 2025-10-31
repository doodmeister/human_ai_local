"""
Episodic Memory System for Human-AI Cognition Framework

Implements biologically-inspired episodic memory with rich temporal and contextual metadata.
Uses ChromaDB for semantic storage with cross-references to STM and LTM systems.

Key Features:
- Rich metadata (timestamp, context, emotional valence, associated_stm_ids, summary)
- Cross-references to STM and LTM memories  
- Temporal clustering and autobiographical organization
- Integration with memory consolidation pipeline
- Semantic search and temporal retrieval patterns
"""

from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
import uuid
import sys
import re
from collections import Counter
from ..base import BaseMemorySystem  # Add import for base class

chromadb = None
CHROMADB_AVAILABLE = False
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    pass

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)

# Import advanced search strategies (after logger initialization)
try:
    from .search_strategies import TieredSearchStrategy, SearchResult as StrategySearchResult
    ADVANCED_SEARCH_AVAILABLE = True
except ImportError as e:
    ADVANCED_SEARCH_AVAILABLE = False
    logger.warning(f"Advanced search strategies not available: {e}")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)

def _safe_first_list(val):
    if isinstance(val, list) and len(val) > 0 and isinstance(val[0], list):
        return val[0]
    return []

@dataclass
class EpisodicContext:
    """Rich contextual information for episodic memories"""
    location: Optional[str] = None
    emotional_state: float = 0.0  # -1.0 to 1.0
    cognitive_load: float = 0.0   # 0.0 to 1.0
    attention_focus: List[str] = field(default_factory=list)
    interaction_type: str = "conversation"  # conversation, reflection, consolidation, etc.
    participants: List[str] = field(default_factory=list)
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "location": self.location,
            "emotional_state": self.emotional_state,
            "cognitive_load": self.cognitive_load,
            "attention_focus": self.attention_focus,
            "interaction_type": self.interaction_type,
            "participants": self.participants,
            "environmental_factors": self.environmental_factors
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicContext":
        """Create from dictionary"""
        return cls(
            location=data.get("location"),
            emotional_state=data.get("emotional_state", 0.0),
            cognitive_load=data.get("cognitive_load", 0.0),
            attention_focus=data.get("attention_focus", []),
            interaction_type=data.get("interaction_type", "conversation"),
            participants=data.get("participants", []),
            environmental_factors=data.get("environmental_factors", {})
        )

@dataclass 
class EpisodicMemory:
    """Individual episodic memory record with rich metadata"""
    id: str
    summary: str  # Brief summary of the episode
    detailed_content: str  # Full content/narrative of the episode
    timestamp: datetime
    duration: timedelta = field(default_factory=lambda: timedelta(minutes=1))
    
    # Contextual information
    context: EpisodicContext = field(default_factory=EpisodicContext)
    
    # Memory system cross-references
    associated_stm_ids: List[str] = field(default_factory=list)
    associated_ltm_ids: List[str] = field(default_factory=list)
    source_memory_ids: List[str] = field(default_factory=list)  # Memories that contributed to this episode
    
    # Episodic characteristics
    importance: float = 0.5  # 0.0 to 1.0
    emotional_valence: float = 0.0  # -1.0 to 1.0 
    vividness: float = 0.5  # How clear/detailed the memory is (0.0 to 1.0)
    confidence: float = 0.8  # Confidence in memory accuracy (0.0 to 1.0)
    entropy: float = 0.2  # Entropy/uncertainty score (0.0 = certain, 1.0 = forgotten)
    
    # Autobiographical organization
    life_period: Optional[str] = None  # e.g., "work_conversation", "learning_session"
    episode_sequence: int = 0  # Sequence number within a period
    related_episodes: List[str] = field(default_factory=list)  # Related episode IDs
    
    # Access and consolidation tracking
    access_count: int = 0
    last_access: datetime = field(default_factory=datetime.now)
    consolidation_strength: float = 0.0  # How well consolidated this memory is
    rehearsal_count: int = 0  # How many times it's been rehearsed/recalled
    tags: List[str] = field(default_factory=list)  # Keywords for search
    
    # Provenance and recency tracking
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    episodic_source: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "summary": self.summary,
            "detailed_content": self.detailed_content,
            "timestamp": self.timestamp.isoformat(),
            "duration": self.duration.total_seconds(),
            "context": self.context.to_dict(),
            "associated_stm_ids": self.associated_stm_ids,
            "associated_ltm_ids": self.associated_ltm_ids,
            "source_memory_ids": self.source_memory_ids,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "vividness": self.vividness,
            "confidence": self.confidence,
            "life_period": self.life_period,
            "episode_sequence": self.episode_sequence,
            "related_episodes": self.related_episodes,
            "access_count": self.access_count,
            "last_access": self.last_access.isoformat(),
            "consolidation_strength": self.consolidation_strength,
            "rehearsal_count": self.rehearsal_count,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "source": self.source,
            "episodic_source": self.episodic_source,
            "entropy": self.entropy
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EpisodicMemory":
        """Create from dictionary"""
        context_data = data.get("context", {})
        context = EpisodicContext.from_dict(context_data) if context_data else EpisodicContext()
        
        return cls(
            id=data["id"],
            summary=data["summary"],
            detailed_content=data["detailed_content"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            duration=timedelta(seconds=data.get("duration", 60)),
            context=context,
            associated_stm_ids=data.get("associated_stm_ids", []),
            associated_ltm_ids=data.get("associated_ltm_ids", []),
            source_memory_ids=data.get("source_memory_ids", []),
            importance=data.get("importance", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
            vividness=data.get("vividness", 0.5),
            confidence=data.get("confidence", 0.8),
            life_period=data.get("life_period"),
            episode_sequence=data.get("episode_sequence", 0),
            related_episodes=data.get("related_episodes", []),
            access_count=data.get("access_count", 0),
            last_access=datetime.fromisoformat(data.get("last_access", datetime.now().isoformat())),
            consolidation_strength=data.get("consolidation_strength", 0.0),
            rehearsal_count=data.get("rehearsal_count", 0),
            tags=data.get("tags", []),
            created_at=datetime.fromisoformat(data.get("created_at", data.get("timestamp", datetime.now().isoformat()))),
            updated_at=datetime.fromisoformat(data.get("updated_at", data.get("last_access", datetime.now().isoformat()))),
            source=data.get("source"),
            episodic_source=data.get("episodic_source"),
            entropy=data.get("entropy", 0.2)
        )
    
    def update_access(self):
        """Update access tracking when memory is retrieved"""
        self.access_count += 1
        self.last_access = datetime.now()
        self.updated_at = self.last_access
        # Retrieval reduces entropy (increases certainty)
        self.entropy = max(0.0, self.entropy - 0.01)
    
    def rehearse(self, strength_increment: float = 0.1):
        """Rehearse the memory, strengthening consolidation and reducing entropy"""
        self.rehearsal_count += 1
        self.consolidation_strength = min(1.0, self.consolidation_strength + strength_increment)
        self.entropy = max(0.0, self.entropy - 0.02)
        self.update_access()
    
    def decay_entropy(self, amount: float = 0.01, nonlinear: bool = True, context: Optional[dict] = None):
        """Increase entropy (uncertainty) due to time/distraction, with nonlinear/sigmoid option."""
        # Contextual modulation
        mod = 1.0
        if context:
            # Example: higher cognitive load or low attention increases decay
            mod *= 1.0 + context.get("cognitive_load", 0.0)
            if context.get("attention", 1.0) < 0.5:
                mod *= 1.2
        base = self.entropy
        if nonlinear:
            # Sigmoid-like: dE = k * (1 / (1 + exp(-a*(E-b))))
            import math
            a = 6.0  # steepness
            b = 0.5  # midpoint
            k = amount * mod
            sigmoid = 1 / (1 + math.exp(-a * (base - b)))
            delta = k * sigmoid
        else:
            delta = amount * mod
        self.entropy = min(1.0, self.entropy + delta)

@dataclass
class EpisodicSearchResult:
    """Result from episodic memory search"""
    memory: EpisodicMemory
    relevance: float  # 0.0 to 1.0
    match_type: str  # "semantic", "temporal", "contextual", "cross_reference"
    search_metadata: Dict[str, Any] = field(default_factory=dict)

class EpisodicMemorySystem(BaseMemorySystem):
    """
    Episodic Memory System for autobiographical memory storage and retrieval
    Implements the unified memory interface.
    """
    
    def __init__(
        self,
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "episodic_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_json_backup: bool = True,
        storage_path: Optional[str] = None
    ):
        """
        Initialize Episodic Memory System
        
        Args:
            chroma_persist_dir: ChromaDB persistence directory
            collection_name: Name of ChromaDB collection
            embedding_model: SentenceTransformer model name
            enable_json_backup: Whether to maintain JSON backups
            storage_path: Path for JSON storage
        """
        # Storage configuration
        self.chroma_persist_dir = Path(chroma_persist_dir or "data/memory_stores/chroma_episodic")
        self.storage_path = Path(storage_path or "data/memory_stores/episodic")
        self.collection_name = collection_name
        self.enable_json_backup = enable_json_backup
          # Create directories
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer is not None:
            try:
                import torch
                self.embedding_model = SentenceTransformer(embedding_model)
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to("cuda")
                logger.info(f"Loaded embedding model: {embedding_model} (GPU: {torch.cuda.is_available()})")
            except Exception as e:
                logger.warning(f"Failed to load embedding model: {e}")
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        if CHROMADB_AVAILABLE:
            self._initialize_chromadb()
        
        # In-memory cache
        self._memory_cache: Dict[str, EpisodicMemory] = {}
        self._load_from_json_backup()
        
        # Initialize advanced search strategies
        self._search_strategy = None
        if ADVANCED_SEARCH_AVAILABLE:
            try:
                from .search_strategies import TieredSearchStrategy as TSS
                self._search_strategy = TSS()
                logger.info("Advanced search strategies initialized (BM25, TF-IDF, Enhanced Word Overlap)")
            except Exception as e:
                logger.warning(f"Failed to initialize advanced search strategies: {e}")
                self._search_strategy = None
        
        logger.info("Episodic Memory System initialized")
    
    def _initialize_chromadb(self):
        global chromadb
        """Initialize ChromaDB client and collection"""
        if not CHROMADB_AVAILABLE or chromadb is None:
            self.chroma_client = None
            self.collection = None
            return
            
        try:
            from chromadb.config import Settings
            # Use PersistentClient for on-disk storage, allowing resets for testing
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_persist_dir),
                settings=Settings(allow_reset=True)
            )
            
            # Get or create collection
            self.collection = self.chroma_client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "Episodic memories with rich contextual metadata"}
            )
            logger.info(f"Initialized ChromaDB collection: {self.collection_name} at {self.chroma_persist_dir}")
                
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB at {self.chroma_persist_dir}: {e}")
            self.chroma_client = None
            self.collection = None
        
        # Fallback initialization logic
        if not hasattr(self, 'chroma_client') or self.chroma_client is None:
            try:
                import chromadb
                self.chroma_client = chromadb.Client()
                sys.stdout.flush()
            except Exception:
                sys.stdout.flush()
        if self.collection is None and hasattr(self, 'chroma_client') and self.chroma_client is not None:
            try:
                sys.stdout.flush()
                self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
                sys.stdout.flush()
            except Exception:
                sys.stdout.flush()
    
    def _load_from_json_backup(self):
        """Load memories from JSON backup files"""
        if not self.enable_json_backup:
            return
            
        json_files = list(self.storage_path.glob("*.json"))
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    memory = EpisodicMemory.from_dict(data)
                    self._memory_cache[memory.id] = memory
            except Exception as e:
                logger.warning(f"Failed to load memory from {json_file}: {e}")
        
        logger.info(f"Loaded {len(self._memory_cache)} memories from JSON backup")
    
    def _save_to_json_backup(self, memory: EpisodicMemory):
        """Save memory to JSON backup file"""
        if not self.enable_json_backup:
            return
            
        try:
            json_file = self.storage_path / f"{memory.id}.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(memory.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save memory to JSON backup: {e}")
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text"""
        if not self.embedding_model:
            return None
            
        try:
            return self.embedding_model.encode(text).tolist()
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return None

    def store(
        self,
        detailed_content: str,
        context: Optional[EpisodicContext] = None,
        associated_stm_ids: Optional[List[str]] = None,
        associated_ltm_ids: Optional[List[str]] = None,
        source_memory_ids: Optional[List[str]] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        life_period: Optional[str] = None
    ) -> str:
        """
        Store a new episodic memory (unified interface)
        Returns the memory ID of the stored episode.
        """
        return self.store_memory(
            detailed_content=detailed_content,
            context=context,
            associated_stm_ids=associated_stm_ids,
            associated_ltm_ids=associated_ltm_ids,
            source_memory_ids=source_memory_ids,
            importance=importance,
            emotional_valence=emotional_valence,
            life_period=life_period
        )

    def retrieve(self, memory_id: str) -> Optional[dict]:
        """
        Retrieve a specific memory by ID (unified interface)
        Returns the memory as a dict, or None if not found.
        Also boosts rehearsal/consolidation strength and reduces entropy.
        """
        memory = self._memory_cache.get(memory_id)
        if memory:
            # Ensure all required attributes exist and are consistent
            if not hasattr(memory, 'consolidation_strength'):
                memory.consolidation_strength = getattr(memory, 'consolidation_strength', 0.5)
            if not hasattr(memory, 'rehearsal_count'):
                memory.rehearsal_count = getattr(memory, 'rehearsal_count', 0)
            if not hasattr(memory, 'access_count'):
                memory.access_count = getattr(memory, 'access_count', 0)
            if not hasattr(memory, 'importance'):
                memory.importance = getattr(memory, 'importance', 0.5)
            if not hasattr(memory, 'emotional_valence'):
                memory.emotional_valence = getattr(memory, 'emotional_valence', 0.0)
            if not hasattr(memory, 'life_period'):
                memory.life_period = getattr(memory, 'life_period', None)
            if not hasattr(memory, 'tags'):
                memory.tags = getattr(memory, 'tags', [])
            if not hasattr(memory, 'related_episodes'):
                memory.related_episodes = getattr(memory, 'related_episodes', [])
            if not hasattr(memory, 'associated_stm_ids'):
                memory.associated_stm_ids = getattr(memory, 'associated_stm_ids', [])
            if not hasattr(memory, 'associated_ltm_ids'):
                memory.associated_ltm_ids = getattr(memory, 'associated_ltm_ids', [])
            if not hasattr(memory, 'source_memory_ids'):
                memory.source_memory_ids = getattr(memory, 'source_memory_ids', [])
            if not hasattr(memory, 'context'):
                from .episodic_memory import EpisodicContext
                memory.context = EpisodicContext()
            memory.update_access()
            # Boost rehearsal/consolidation strength on retrieval
            memory.rehearsal_count += 1  # Increment rehearsal count
            memory.consolidation_strength = min(1.0, memory.consolidation_strength + 0.05)  # Boost consolidation
            # Optionally, increase confidence (reduce uncertainty/entropy)
            memory.confidence = min(1.0, memory.confidence + 0.01)
            self._save_to_json_backup(memory)  # Update backup with access info
            return memory.to_dict()
        return None

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory by ID (unified interface)
        Returns True if deleted, False otherwise.
        """
        removed = False
        if memory_id in self._memory_cache:
            del self._memory_cache[memory_id]
            removed = True
            # Remove from ChromaDB
            if self.collection is not None:
                try:
                    self.collection.delete(ids=[memory_id])
                except Exception as e:
                    logger.warning(f"Failed to remove episodic memory {memory_id} from ChromaDB: {e}")
            # Remove JSON file
            if self.enable_json_backup:
                json_file = self.storage_path / f"{memory_id}.json"
                if json_file.exists():
                    try:
                        json_file.unlink()
                    except Exception as e:
                        logger.warning(f"Failed to remove JSON file for episodic memory {memory_id}: {e}")
        return removed

    def search(self, query: Optional[str] = None, **kwargs) -> List[dict]:
        """
        Search for episodic memories (unified interface).
        Returns a list of matching memory dicts.
        """
        if not query:
            # Return all memories as dicts if no query
            return [m.to_dict() for m in self._memory_cache.values()]
        # Use search_memories for semantic/text search
        limit = kwargs.get('limit', 10)
        min_relevance = kwargs.get('min_relevance', 0.3)
        time_range = kwargs.get('time_range')
        life_period = kwargs.get('life_period')
        emotional_range = kwargs.get('emotional_range')
        importance_threshold = kwargs.get('importance_threshold', 0.0)
        results = self.search_memories(
            query=query,
            limit=limit,
            min_relevance=min_relevance,
            time_range=time_range,
            life_period=life_period,
            emotional_range=emotional_range,
            importance_threshold=importance_threshold
        )
        return [r.memory.to_dict() for r in results]
    
    def shutdown(self):
        """Shutdown the memory system and release resources."""
        logger.info("Shutting down Episodic Memory System.")
        if self.chroma_client:
            try:
                self.collection = None
                self.chroma_client.reset()
            except Exception as e:
                logger.error(f"Error shutting down ChromaDB client: {e}")
        self.chroma_client = None

    def _summarize_content(self, content: str, max_length: int = 128) -> str:
        """Generate a simple summary of the content."""
        sentences = content.split('.')
        return sentences[0] + '.' if sentences else content[:max_length]

    def _extract_tags(self, content: str, max_tags: int = 10) -> List[str]:
        """Extract keyword tags from the content."""
        words = re.findall(r'\b\w+\b', content.lower())
        stop_words = set(["the", "a", "an", "in", "on", "of", "for", "to", "and", "is", "are", "was", "were"])
        words = [word for word in words if word not in stop_words and len(word) > 2]
        return [word for word, _ in Counter(words).most_common(max_tags)]
    
    def store_memory(
        self,
        detailed_content: str,
        context: Optional[EpisodicContext] = None,
        associated_stm_ids: Optional[List[str]] = None,
        associated_ltm_ids: Optional[List[str]] = None,
        source_memory_ids: Optional[List[str]] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        life_period: Optional[str] = None
    ) -> str:
        """
        Store a new episodic memory
        
        Args:
            detailed_content: Full content/narrative
            context: Rich contextual information
            associated_stm_ids: Related STM memory IDs
            associated_ltm_ids: Related LTM memory IDs
            source_memory_ids: Source memories that contributed
            importance: Importance score (0.0-1.0)
            emotional_valence: Emotional valence (-1.0 to 1.0)
            life_period: Life period categorization
            
        Returns:
            Memory ID of stored episode
        """
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        # Automatically generate summary and tags
        summary = self._summarize_content(detailed_content)
        tags = self._extract_tags(detailed_content)

        # Create memory object
        memory = EpisodicMemory(
            id=memory_id,
            summary=summary,
            detailed_content=detailed_content,
            timestamp=datetime.now(),
            context=context or EpisodicContext(),
            associated_stm_ids=associated_stm_ids or [],
            associated_ltm_ids=associated_ltm_ids or [],
            source_memory_ids=source_memory_ids or [],
            importance=importance,
            emotional_valence=emotional_valence,
            life_period=life_period,
            tags=tags
        )
        
        # Store in cache
        self._memory_cache[memory_id] = memory
        
        # Store in ChromaDB
        if self.collection is not None:
            try:
                # Create searchable text combining summary and content
                searchable_text = f"{summary} {detailed_content}"
                embedding = self._generate_embedding(searchable_text)
                
                metadata = {
                    "summary": summary,
                    "timestamp": memory.timestamp.isoformat(),
                    "importance": importance,
                    "emotional_valence": emotional_valence,
                    "life_period": life_period or "general",
                    "interaction_type": memory.context.interaction_type,
                    "duration": memory.duration.total_seconds(),
                    "vividness": memory.vividness,
                    "confidence": memory.confidence,
                    "tags": ",".join(tags) # Store tags as a comma-separated string
                }
                if life_period:
                    metadata["life_period"] = life_period
                
                self.collection.add(
                    ids=[memory_id],
                    documents=[searchable_text],
                    embeddings=[embedding] if embedding else None,
                    metadatas=[metadata]
                )
                
            except Exception as e:
                logger.warning(f"Failed to store memory in ChromaDB: {e}")
        
        # Save JSON backup
        self._save_to_json_backup(memory)
        
        logger.info(f"Stored episodic memory: {memory_id}")
        return memory_id
    
    def retrieve_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Retrieve a specific memory by ID"""
        memory = self._memory_cache.get(memory_id)
        if memory:
            memory.update_access()
            self._save_to_json_backup(memory)  # Update backup with access info
        return memory
    
    def search_memories(
        self,
        query: str,
        limit: int = 10,
        min_relevance: float = 0.3,
        time_range: Optional[Tuple[datetime, datetime]] = None,
        life_period: Optional[str] = None,
        emotional_range: Optional[Tuple[float, float]] = None,
        importance_threshold: float = 0.0
    ) -> List[EpisodicSearchResult]:
        """
        Search episodic memories with multiple criteria
        
        Args:
            query: Search query
            limit: Maximum results to return
            min_relevance: Minimum relevance threshold
            time_range: Optional time range filter (start, end)
            life_period: Filter by life period
            emotional_range: Filter by emotional valence range
            importance_threshold: Minimum importance threshold
            
        Returns:
            List of search results sorted by relevance
        """
        results = []
        
        # ChromaDB semantic search
        if self.collection is not None:
            try:
                # Build where clause for filtering
                where_clause = {}
                if life_period:
                    where_clause["life_period"] = life_period
                if importance_threshold > 0:
                    where_clause["importance"] = {"$gte": importance_threshold}
                if emotional_range:
                    where_clause["emotional_valence"] = {
                        "$gte": emotional_range[0],
                        "$lte": emotional_range[1]
                    }
                # Run the query
                search_results = self.collection.query(
                    query_texts=[query],
                    n_results=limit * 2,  # Get more results for additional filtering
                    where=where_clause if where_clause else None
                )
                result_ids = _safe_first_list(search_results.get('ids'))
                distances = _safe_first_list(search_results.get('distances'))
                for i, memory_id in enumerate(result_ids):
                    distance = distances[i] if distances and i < len(distances) else 0.0
                    relevance = 1.0 - distance
                    if relevance < min_relevance:
                        continue
                    memory = self.retrieve_memory(memory_id)
                    if memory is None:
                        continue
                    results.append(EpisodicSearchResult(
                        memory=memory,
                        relevance=relevance,
                        match_type="semantic",
                        search_metadata={"chroma_distance": distance}
                    ))
                # Track metrics
                try:
                    from src.chat.metrics import metrics_registry
                    metrics_registry.inc("episodic_search_tier_semantic")
                except Exception:
                    pass
                return results[:limit]
            except Exception as e:
                logger.warning(f"ChromaDB search failed: {e}")
                # Fallback to advanced search strategies below
        
        # Advanced fallback: Use TieredSearchStrategy (BM25 → TF-IDF → Enhanced Word Overlap)
        if self._search_strategy is not None and self._memory_cache:
            try:
                # Prepare documents for search
                documents = {}
                doc_to_memory = {}
                for memory in self._memory_cache.values():
                    if life_period and memory.life_period != life_period:
                        continue
                    doc_text = f"{memory.summary} {memory.detailed_content}"
                    documents[memory.id] = doc_text
                    doc_to_memory[memory.id] = memory
                
                # Execute tiered search
                search_results = self._search_strategy.search(
                    query=query,
                    documents=documents,
                    limit=limit,
                    min_relevance=min_relevance
                )
                
                # Convert to EpisodicSearchResult
                results = []
                for search_result in search_results:
                    memory = doc_to_memory.get(search_result.doc_id)
                    if memory:
                        memory.update_access()
                        results.append(EpisodicSearchResult(
                            memory=memory,
                            relevance=search_result.relevance,
                            match_type=search_result.match_type,
                            search_metadata=search_result.metadata
                        ))
                
                if results:
                    logger.debug(f"Advanced search ({results[0].match_type}) returned {len(results)} results")
                    # Track metrics if available
                    try:
                        from src.chat.metrics import metrics_registry
                        metrics_registry.inc(f"episodic_search_tier_{results[0].match_type}")
                    except Exception:
                        pass
                    return results
                    
            except Exception as e:
                logger.error(f"Advanced search failed: {e}")
                # Fall through to basic fallback
        
        # Basic fallback: Substring match (legacy behavior)
        query_lower = query.lower()
        for memory in self._memory_cache.values():
            text_to_search = f"{memory.summary} {memory.detailed_content}".lower()
            if query_lower in text_to_search:
                if life_period and memory.life_period != life_period:
                    continue
                relevance = 0.6
                memory.update_access()
                results.append(EpisodicSearchResult(
                    memory=memory,
                    relevance=relevance,
                    match_type="text_match",
                    search_metadata={}
                ))
        
        # Track basic fallback metrics
        if results:
            try:
                from src.chat.metrics import metrics_registry
                metrics_registry.inc("episodic_search_tier_text_match_basic")
            except Exception:
                pass
        
        # Word overlap fallback if no results
        if not results:
            query_words = set(query_lower.split())
            for memory in self._memory_cache.values():
                text_words = set(f"{memory.summary} {memory.detailed_content}".lower().split())
                overlap = query_words & text_words
                if overlap:
                    if life_period and memory.life_period != life_period:
                        continue
                    relevance = 0.4 + 0.1 * len(overlap)
                    if relevance >= min_relevance:
                        results.append(EpisodicSearchResult(
                            memory=memory,
                            relevance=relevance,
                            match_type="word_overlap",
                            search_metadata={}
                        ))
        results.sort(key=lambda x: x.relevance, reverse=True)
        return results[:limit]

    def get_related_memories(self, memory_id: str, relationship_types: Optional[List[str]] = None, limit: int = 10) -> List['EpisodicSearchResult']:
        """
        Get related memories for a given memory ID
        
        Args:
            memory_id: The ID of the memory to find relations for
            relationship_types: List of relationship types to consider
            limit: Maximum number of results to return
            
        Returns:
            List of related EpisodicSearchResult objects
        """
        memory = self.retrieve_memory(memory_id)
        if not memory:
            logger.debug(f"Memory {memory_id} not found for related search.")
            return []
        relationship_types = relationship_types or ["temporal", "cross_reference", "semantic"]
        results = []
        # Explicit related episodes
        if "cross_reference" in relationship_types:
            for related_id in memory.related_episodes:
                related_memory = self.retrieve_memory(related_id)
                if related_memory:
                    logger.debug(f"Explicit related episode: {related_id}")
                    results.append(EpisodicSearchResult(
                        memory=related_memory,
                        relevance=0.9,
                        match_type="cross_reference",
                        search_metadata={"relationship": "explicit_related"}
                    ))
        # Cross-referenced STM/LTM memories
        if "cross_reference" in relationship_types:
            for mem in self._memory_cache.values():
                if mem.id == memory_id:
                    continue
                shared_stm = set(memory.associated_stm_ids) & set(mem.associated_stm_ids)
                if shared_stm:
                    logger.debug(f"Shared STM cross-reference: {mem.id} shared_stm={shared_stm}")
                    relevance = min(0.8, len(shared_stm) * 0.2)
                    results.append(EpisodicSearchResult(
                        memory=mem,
                        relevance=relevance,
                        match_type="cross_reference",
                        search_metadata={"shared_stm_ids": list(shared_stm)}
                    ))
                shared_ltm = set(memory.associated_ltm_ids) & set(mem.associated_ltm_ids)
                if shared_ltm:
                    logger.debug(f"Shared LTM cross-reference: {mem.id} shared_ltm={shared_ltm}")
                    relevance = min(0.8, len(shared_ltm) * 0.2)
                    results.append(EpisodicSearchResult(
                        memory=mem,
                        relevance=relevance,
                        match_type="cross_reference",
                        search_metadata={"shared_ltm_ids": list(shared_ltm)}
                    ))
        # Temporal proximity
        if "temporal" in relationship_types:
            time_window = timedelta(hours=2)
            for mem in self._memory_cache.values():
                if mem.id == memory_id:
                    continue
                time_diff = abs((memory.timestamp - mem.timestamp).total_seconds())
                if time_diff <= time_window.total_seconds():
                    logger.debug(f"Temporal relationship: {mem.id} time_diff={time_diff}")
                    relevance = 0.7 * (1.0 - time_diff / time_window.total_seconds())
                    results.append(EpisodicSearchResult(
                        memory=mem,
                        relevance=relevance,
                        match_type="temporal",
                        search_metadata={"time_diff_seconds": time_diff}
                    ))
        # Semantic similarity
        if "semantic" in relationship_types and self.collection:
            try:
                search_query = f"{memory.summary} {memory.detailed_content}"
                semantic_results = self.search_memories(
                    query=search_query,
                    limit=limit,
                    min_relevance=0.4
                )
                for result in semantic_results:
                    if result.memory.id != memory_id:
                        logger.debug(f"Semantic related: {result.memory.id} relevance={result.relevance}")
                        result.match_type = "semantic"
                        results.append(result)
            except Exception as e:
                logger.warning(f"Semantic search for related memories failed: {e}")
        # Remove duplicates and sort by relevance
        seen_ids = set()
        unique_results = []
        for result in results:
            if result.memory.id not in seen_ids:
                seen_ids.add(result.memory.id)
                unique_results.append(result)
        unique_results.sort(key=lambda x: x.relevance, reverse=True)
        logger.debug(f"get_related_memories returning {len(unique_results[:limit])} results.")
        return unique_results[:limit]

    def get_autobiographical_timeline(self, life_period: Optional[str] = None, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None, limit: int = 50) -> List['EpisodicMemory']:
        """
        Get memories organized as an autobiographical timeline
        
        Args:
            life_period: Optional life period to filter by
            start_date: Optional start date for filtering
            end_date: Optional end date for filtering
            limit: Maximum number of results to return
            
        Returns:
            List of EpisodicMemory objects sorted by timestamp
        """
        memories = list(self._memory_cache.values())
        if life_period:
            memories = [m for m in memories if m.life_period == life_period]
        if start_date:
            memories = [m for m in memories if m.timestamp >= start_date]
        if end_date:
            memories = [m for m in memories if m.timestamp <= end_date]
        memories.sort(key=lambda m: m.timestamp)
        logger.debug(f"get_autobiographical_timeline returning {len(memories[:limit])} results.")
        return memories[:limit]

    def consolidate_memory(self, memory_id: str, strength_increment: float = 0.1) -> bool:
        """
        Consolidate a memory by strengthening its consolidation

        Args:
            memory_id: The ID of the memory to consolidate
            strength_increment: The amount to increase the consolidation strength

        Returns:
            True if the memory was consolidated (strength increased), False otherwise
        """
        memory = self.retrieve_memory(memory_id)
        if not memory:
            logger.debug(f"consolidate_memory: memory {memory_id} not found.")
            return False
        before = memory.consolidation_strength
        memory.rehearse(strength_increment)
        after = memory.consolidation_strength
        logger.debug(f"consolidate_memory: {memory_id} strength {before} -> {after}")
        return after > before

    def get_memory_statistics(self) -> dict:
        """
        Get statistics about the memories in the system
        
        Returns:
            Dictionary containing various statistics about the memories
        """
        memories = list(self._memory_cache.values())
        stats = {
            "total_memories": len(memories),
            "memory_system_status": "active",
            "life_period_count": len(set(m.life_period for m in memories if m.life_period)),
        }
        if not memories:
            return stats
        import numpy as np
        importance = np.array([m.importance for m in memories])
        emotional = np.array([m.emotional_valence for m in memories])
        consolidation = np.array([m.consolidation_strength for m in memories])
        access = np.array([m.access_count for m in memories])
        stats["importance_stats"] = {"mean": float(importance.mean()), "min": float(importance.min()), "max": float(importance.max())}
        stats["emotional_stats"] = {
            "positive_memories": int((emotional > 0.1).sum()),
            "negative_memories": int((emotional < -0.1).sum()),
            "neutral_memories": int(((emotional >= -0.1) & (emotional <= 0.1)).sum()),
        }
        stats["consolidation_stats"] = {"mean": float(consolidation.mean()), "min": float(consolidation.min()), "max": float(consolidation.max())}
        stats["access_stats"] = {"mean": float(access.mean()), "min": int(access.min()), "max": int(access.max())}
        logger.debug(f"get_memory_statistics: {stats}")
        return stats

    def clear_memory(self, older_than: Optional[timedelta] = None, importance_threshold: Optional[float] = None):
        """
        Clear memories from the system based on criteria
        
        Args:
            older_than: Optional timedelta; if set, memories older than this will be removed
            importance_threshold: Optional importance threshold; if set, memories with importance below this will be removed
        """
        to_remove = []
        now = datetime.now()
        for mem_id, mem in self._memory_cache.items():
            if older_than and (now - mem.timestamp) > older_than:
                to_remove.append(mem_id)
            elif importance_threshold is not None and mem.importance < importance_threshold:
                to_remove.append(mem_id)
        for mem_id in to_remove:
            logger.debug(f"clear_memory: removing {mem_id}")
            self._memory_cache.pop(mem_id, None)
        logger.debug(f"clear_memory: removed {len(to_remove)} memories.")

    def get_consolidation_candidates(self, min_importance: float = 0.5, max_consolidation: float = 0.9, limit: int = 10) -> list:
        """
        Get memories that are candidates for consolidation based on importance and consolidation strength.
        """
        candidates = [
            m for m in self._memory_cache.values()
            if m.importance >= min_importance and m.consolidation_strength <= max_consolidation
        ]
        candidates.sort(key=lambda m: (-m.importance, m.consolidation_strength))
        logger.debug(f"get_consolidation_candidates returning {len(candidates[:limit])} candidates.")
        return candidates[:limit]

    def clear_all_memories(self):
        """Clear all episodic memories from the in-memory cache (for test isolation)."""
        self._memory_cache.clear()
        logger.debug("clear_all_memories: all in-memory episodic memories cleared.")

    def batch_consolidate_memories(self, min_importance: float = 0.5, max_consolidation: float = 0.9, limit: int = 20, strength_increment: float = 0.2, cluster: bool = True) -> dict:
        """
        Batch consolidate episodic memories (dream-state consolidation).
        Optionally cluster/merge similar memories for compression.
        Returns a summary dict.
        """
        candidates = self.get_consolidation_candidates(min_importance=min_importance, max_consolidation=max_consolidation, limit=limit)
        consolidated = []
        merged = []
        if cluster and candidates:
            # Simple clustering: group by similar summary (could use embedding similarity for more advanced)
            from collections import defaultdict
            import difflib
            clusters = defaultdict(list)
            for mem in candidates:
                found = False
                for key in clusters:
                    # Use difflib to check if summaries are similar
                    if difflib.SequenceMatcher(None, mem.summary, key).ratio() > 0.8:
                        clusters[key].append(mem)
                        found = True
                        break
                if not found:
                    clusters[mem.summary].append(mem)
            # Merge clusters with more than one memory
            for key, group in clusters.items():
                if len(group) > 1:
                    # Merge: concatenate details, average importance/valence, keep earliest timestamp
                    merged_content = "\n---\n".join([m.detailed_content for m in group])
                    avg_importance = sum(m.importance for m in group) / len(group)
                    avg_valence = sum(m.emotional_valence for m in group) / len(group)
                    earliest = min(m.timestamp for m in group)
                    merged_summary = key + f" (merged {len(group)})"
                    merged_id = self.store_memory(
                        detailed_content=merged_content,
                        importance=avg_importance,
                        emotional_valence=avg_valence,
                        context=group[0].context,
                        life_period=group[0].life_period
                    )
                    # Optionally, delete originals
                    for m in group:
                        self.delete(m.id)
                    merged.append(merged_id)
                else:
                    # Singletons: just reinforce
                    self.consolidate_memory(group[0].id, strength_increment=strength_increment)
                    consolidated.append(group[0].id)
        else:
            # No clustering, just reinforce all
            for mem in candidates:
                self.consolidate_memory(mem.id, strength_increment=strength_increment)
                consolidated.append(mem.id)
        return {
            "consolidated": consolidated,
            "merged": merged,
            "total_candidates": len(candidates),
            "clusters": len(merged),
        }
