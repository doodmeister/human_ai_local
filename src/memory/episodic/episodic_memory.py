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
            "tags": self.tags
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
            tags=data.get("tags", [])
        )
    
    def update_access(self):
        """Update access tracking when memory is retrieved"""
        self.access_count += 1
        self.last_access = datetime.now()
    
    def rehearse(self, strength_increment: float = 0.1):
        """Rehearse the memory, strengthening consolidation"""
        self.rehearsal_count += 1
        self.consolidation_strength = min(1.0, self.consolidation_strength + strength_increment)
        self.update_access()

@dataclass
class EpisodicSearchResult:
    """Result from episodic memory search"""
    memory: EpisodicMemory
    relevance: float  # 0.0 to 1.0
    match_type: str  # "semantic", "temporal", "contextual", "cross_reference"
    search_metadata: Dict[str, Any] = field(default_factory=dict)

class EpisodicMemorySystem:
    """
    Episodic Memory System for autobiographical memory storage and retrieval
    
    Features:
    - Rich contextual metadata storage
    - Cross-references to STM and LTM systems
    - Temporal and semantic search capabilities
    - Autobiographical organization and clustering
    - Integration with memory consolidation pipeline
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
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"Loaded embedding model: {embedding_model}")
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

    def get_memory(self, memory_id: str) -> Optional[EpisodicMemory]:
        """Retrieve a memory by its ID."""
        return self._memory_cache.get(memory_id)

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
                # Append all valid ChromaDB results to the results list
                for i, memory_id in enumerate(result_ids):
                    # If a metadata filter is used, treat all as relevant
                    if where_clause:
                        relevance = 1.0
                        distance = distances[i] if distances and i < len(distances) else 0.0
                    else:
                        distance = distances[i] if distances and i < len(distances) else 0.0
                        relevance = 1.0 - distance
                    if relevance < min_relevance:
                        continue
                    memory = self.retrieve_memory(memory_id)
                    if memory is None:
                        continue
                    # No extra filtering here; already filtered by ChromaDB
                    results.append(EpisodicSearchResult(
                        memory=memory,
                        relevance=relevance,
                        match_type="semantic",
                        search_metadata={"chroma_distance": distance}
                    ))
                return results[:limit]
                
            except Exception as e:
                logger.warning(f"ChromaDB search failed: {e}")
                # Fallback to cache search if ChromaDB fails
                results = []
                query_lower = query.lower()
                for memory in self._memory_cache.values():
                    # Simple text matching
                    text_to_search = f"{memory.summary} {memory.detailed_content}".lower()
                    if query_lower in text_to_search:
                        # Apply filters
                        if life_period and memory.life_period != life_period:
                            continue
                        if memory.importance < importance_threshold:
                            continue
                        if emotional_range and not (emotional_range[0] <= memory.emotional_valence <= emotional_range[1]):
                            continue
                        if time_range and not (time_range[0] <= memory.timestamp <= time_range[1]):
                            continue
                        # Simple relevance calculation
                        relevance = 0.7 if query_lower in memory.summary.lower() else 0.5
                        if relevance >= min_relevance:
                            memory.update_access()
                            results.append(EpisodicSearchResult(
                                memory=memory,
                                relevance=relevance,
                                match_type="text_match",
                                search_metadata={}
                            ))
                # Sort by relevance and limit results
                results.sort(key=lambda x: x.relevance, reverse=True)
                return results[:limit]
        # If ChromaDB is not available at all, fallback to cache search
        query_lower = query.lower()
        for memory in self._memory_cache.values():
            # Simple text matching
            text_to_search = f"{memory.summary} {memory.detailed_content}".lower()
            if query_lower in text_to_search:
                # Apply filters
                if life_period and memory.life_period != life_period:
                    continue
                if memory.importance < importance_threshold:
                    continue
                if emotional_range and not (emotional_range[0] <= memory.emotional_valence <= emotional_range[1]):
                    continue
                if time_range and not (time_range[0] <= memory.timestamp <= time_range[1]):
                    continue
                # Simple relevance calculation
                relevance = 0.7 if query_lower in memory.summary.lower() else 0.5
                if relevance >= min_relevance:
                    memory.update_access()
                    results.append(EpisodicSearchResult(
                        memory=memory,
                        relevance=relevance,
                        match_type="text_match",
                        search_metadata={}
                    ))
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.relevance, reverse=True)
        return results[:limit]
    
    def get_related_memories(
        self,
        memory_id: str,
        relationship_types: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[EpisodicSearchResult]:
        """
        Get memories related to a specific memory through various relationships
        
        Args:
            memory_id: ID of the reference memory
            relationship_types: Types of relationships to search for
            limit: Maximum results to return
            
        Returns:
            List of related memories
        """
        memory = self.retrieve_memory(memory_id)
        if not memory:
            return []
        
        relationship_types = relationship_types or ["temporal", "cross_reference", "semantic"]
        results = []
        
        # Explicit related episodes
        if "cross_reference" in relationship_types:
            for related_id in memory.related_episodes:
                related_memory = self.retrieve_memory(related_id)
                if related_memory:
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
                
                # Check for shared STM references
                shared_stm = set(memory.associated_stm_ids) & set(mem.associated_stm_ids)
                if shared_stm:
                    relevance = min(0.8, len(shared_stm) * 0.2)
                    results.append(EpisodicSearchResult(
                        memory=mem,
                        relevance=relevance,
                        match_type="cross_reference",
                        search_metadata={"shared_stm_ids": list(shared_stm)}
                    ))
                
                # Check for shared LTM references
                shared_ltm = set(memory.associated_ltm_ids) & set(mem.associated_ltm_ids)
                if shared_ltm:
                    relevance = min(0.8, len(shared_ltm) * 0.2)
                    results.append(EpisodicSearchResult(
                        memory=mem,
                        relevance=relevance,
                        match_type="cross_reference",
                        search_metadata={"shared_ltm_ids": list(shared_ltm)}
                    ))
        
        # Temporal proximity
        if "temporal" in relationship_types:
            time_window = timedelta(hours=2)  # 2-hour window
            for mem in self._memory_cache.values():
                if mem.id == memory_id:
                    continue
                
                time_diff = abs((memory.timestamp - mem.timestamp).total_seconds())
                if time_diff <= time_window.total_seconds():
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
        return unique_results[:limit]
    
    def get_autobiographical_timeline(
        self,
        life_period: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 50
    ) -> List[EpisodicMemory]:
        """
        Get memories organized as an autobiographical timeline
        
        Args:
            life_period: Filter by specific life period
            start_date: Start of time range
            end_date: End of time range
            limit: Maximum memories to return
            
        Returns:
            List of memories sorted chronologically
        """
        filtered_memories = []
        
        for memory in self._memory_cache.values():
            # Apply filters
            if life_period and memory.life_period != life_period:
                continue
            if start_date and memory.timestamp < start_date:
                continue
            if end_date and memory.timestamp > end_date:
                continue
            
            filtered_memories.append(memory)
        
        # Sort by timestamp
        filtered_memories.sort(key=lambda x: x.timestamp)
        
        # Update access for retrieved memories
        for memory in filtered_memories[:limit]:
            memory.update_access()
        
        return filtered_memories[:limit]
    
    def consolidate_memory(self, memory_id: str, strength_increment: float = 0.2) -> bool:
        """
        Consolidate a specific memory (typically called during dream cycles)
        
        Args:
            memory_id: ID of memory to consolidate
            strength_increment: How much to increase consolidation strength
            
        Returns:
            Success status
        """
        memory = self.retrieve_memory(memory_id)
        if not memory:
            return False
        
        # Increase consolidation strength
        memory.consolidation_strength = min(1.0, memory.consolidation_strength + strength_increment)
        
        # Increase vividness for important memories
        if memory.importance > 0.7:
            memory.vividness = min(1.0, memory.vividness + 0.1)
        
        # Save updated memory
        self._save_to_json_backup(memory)
        
        logger.debug(f"Consolidated memory {memory_id}: strength={memory.consolidation_strength:.2f}")
        return True
    
    def get_consolidation_candidates(
        self,
        min_importance: float = 0.4,
        max_consolidation: float = 0.8,
        limit: int = 20
    ) -> List[EpisodicMemory]:
        """
        Get memories that are candidates for consolidation
        
        Args:
            min_importance: Minimum importance threshold
            max_consolidation: Maximum current consolidation strength
            limit: Maximum candidates to return
            
        Returns:
            List of memories needing consolidation
        """
        candidates = []
        
        for memory in self._memory_cache.values():
            if (memory.importance >= min_importance and 
                memory.consolidation_strength <= max_consolidation):
                candidates.append(memory)
        
        # Sort by importance and access frequency
        candidates.sort(
            key=lambda x: (x.importance, x.access_count, x.rehearsal_count),
            reverse=True
        )
        
        return candidates[:limit]
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about the episodic memory system"""
        if not self._memory_cache:
            return {
                "total_memories": 0,
                "memory_system_status": "empty"
            }
        
        memories = list(self._memory_cache.values())
        
        # Basic counts
        total_memories = len(memories)
        life_periods = set(m.life_period for m in memories if m.life_period)
        
        # Time span
        timestamps = [m.timestamp for m in memories]
        earliest = min(timestamps)
        latest = max(timestamps)
        time_span = latest - earliest
        
        # Importance and emotional analysis
        importances = [m.importance for m in memories]
        emotional_valences = [m.emotional_valence for m in memories]
        consolidation_strengths = [m.consolidation_strength for m in memories]
        
        # Access patterns
        access_counts = [m.access_count for m in memories]
        rehearsal_counts = [m.rehearsal_count for m in memories]
        
        return {
            "total_memories": total_memories,
            "memory_system_status": "active",
            "time_span_days": time_span.days,
            "earliest_memory": earliest.isoformat(),
            "latest_memory": latest.isoformat(),
            "life_periods": list(life_periods),
            "life_period_count": len(life_periods),
            "importance_stats": {
                "mean": sum(importances) / len(importances),
                "min": min(importances),
                "max": max(importances)
            },
            "emotional_stats": {
                "mean_valence": sum(emotional_valences) / len(emotional_valences),
                "positive_memories": len([v for v in emotional_valences if v > 0.1]),
                "negative_memories": len([v for v in emotional_valences if v < -0.1]),
                "neutral_memories": len([v for v in emotional_valences if -0.1 <= v <= 0.1])
            },
            "consolidation_stats": {
                "mean_strength": sum(consolidation_strengths) / len(consolidation_strengths),
                "well_consolidated": len([s for s in consolidation_strengths if s > 0.7]),
                "needs_consolidation": len([s for s in consolidation_strengths if s < 0.3])
            },
            "access_stats": {
                "mean_access_count": sum(access_counts) / len(access_counts),
                "total_accesses": sum(access_counts),
                "frequently_accessed": len([c for c in access_counts if c > 5]),
                "mean_rehearsal_count": sum(rehearsal_counts) / len(rehearsal_counts)
            },
            "chromadb_available": self.collection is not None,
            "embedding_model_available": self.embedding_model is not None,
            "json_backup_enabled": self.enable_json_backup
        }
    
    def clear_memory(self, older_than: Optional[timedelta] = None, importance_threshold: Optional[float] = None):
        """
        Clear memories based on age or importance
        
        Args:
            older_than: Remove memories older than this duration
            importance_threshold: Remove memories below this importance
        """
        to_remove = []
        current_time = datetime.now()
        
        for memory_id, memory in self._memory_cache.items():
            should_remove = False
            
            if older_than and (current_time - memory.timestamp) > older_than:
                should_remove = True
            
            if importance_threshold and memory.importance < importance_threshold:
                should_remove = True
            
            if should_remove:
                to_remove.append(memory_id)
        
        # Remove from cache
        for memory_id in to_remove:
            del self._memory_cache[memory_id]
        
        # Remove from ChromaDB
        if self.collection and to_remove:
            try:
                self.collection.delete(ids=to_remove)
            except Exception as e:
                logger.warning(f"Failed to remove memories from ChromaDB: {e}")
        
        # Remove JSON backups
        if self.enable_json_backup:
            for memory_id in to_remove:
                json_file = self.storage_path / f"{memory_id}.json"
                if json_file.exists():
                    json_file.unlink()
        
        logger.info(f"Cleared {len(to_remove)} episodic memories")
