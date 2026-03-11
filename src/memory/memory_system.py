"""
Integrated Memory System for Human-AI Cognition Framework

This module provides a comprehensive memory management system that coordinates
between different types of memory systems (STM, LTM, Episodic, Semantic, 
Prospective, and Procedural) in a biologically-inspired cognitive architecture.

Features:
- Automatic STM to LTM consolidation based on importance and emotional valence
- Cross-system memory retrieval with semantic search capabilities
- Memory reinforcement and forgetting mechanisms
- Dream-state consolidation for episodic memories
- Integration with ChromaDB for persistent vector storage
- GPU-accelerated embedding generation
- Robust error handling and logging
- Production-grade performance optimization

Author: Human-AI Cognition Framework Team
Version: 2.0.0
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    Any, Dict, List, Optional, Tuple, TYPE_CHECKING, 
    Protocol, runtime_checkable
)

from .stm import VectorShortTermMemory, STMConfiguration
from .ltm import VectorLongTermMemory
from .episodic import EpisodicMemorySystem
from .prospective.prospective_memory import ProspectiveMemorySystem
from .procedural.procedural_memory import ProceduralMemory
from .services import (
    MemoryConsolidationService,
    MemoryContextService,
    MemoryFactService,
    MemoryProspectiveService,
    MemoryRecallService,
    MemoryRetrievalService,
    MemoryStatusService,
    MemoryStorageRouter,
)
from .runtime import MemorySubsystemInitializer

if TYPE_CHECKING:
    from .episodic.episodic_memory import EpisodicSearchResult
    from .semantic.semantic_memory import SemanticMemorySystem
else:
    SemanticMemorySystem = Any  # type: ignore

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create console handler with formatting if not already configured
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)


@dataclass
class MemorySystemConfig:
    """Configuration class for Memory System initialization."""
    stm_capacity: int = 100
    stm_decay_threshold: float = 0.1
    ltm_storage_path: Optional[str] = None
    consolidation_interval: int = 300  # seconds
    use_vector_ltm: bool = True
    use_vector_stm: bool = True
    chroma_persist_dir: Optional[str] = None
    embedding_model: str = "all-MiniLM-L6-v2"
    semantic_storage_path: Optional[str] = None
    max_concurrent_operations: int = 4
    lazy_embeddings: bool = True
    consolidation_threshold_importance: float = 0.6
    consolidation_threshold_emotional: float = 0.5
    consolidation_age_minutes: int = 30
    auto_process_prospective: bool = True
    prospective_process_interval: int = 60  # seconds
    # New flag: whether to use vector-backed prospective memory (Chroma) instead of lightweight in-memory singleton
    use_vector_prospective: bool = False


@dataclass
class MemoryOperationResult:
    """Result of a memory operation."""
    success: bool
    memory_id: Optional[str] = None
    error_message: Optional[str] = None
    operation_time: Optional[datetime] = field(default_factory=datetime.now)
    system_used: Optional[str] = None


@dataclass
class ConsolidationStats:
    """Statistics from memory consolidation process."""
    start_time: datetime
    end_time: Optional[datetime] = None
    consolidated_count: int = 0
    failed_count: int = 0
    errors: List[str] = field(default_factory=list)
    duration_seconds: float = 0.0
    
    @property
    def is_complete(self) -> bool:
        """Check if consolidation is complete."""
        return self.end_time is not None


@runtime_checkable
class MemorySearchable(Protocol):
    """Protocol for searchable memory systems."""
    
    def search(self, query: str, max_results: int = 10) -> List[Tuple[Any, float]]:
        """Search for memories matching the query."""
        ...


@runtime_checkable
class MemoryStorable(Protocol):
    """Protocol for storable memory systems."""
    
    def store(self, memory_id: str, content: Any, **kwargs) -> bool:
        """Store a memory."""
        ...
    
    def retrieve(self, memory_id: str) -> Optional[Any]:
        """Retrieve a memory by ID."""
        ...
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        ...


class MemorySystemError(Exception):
    """Base exception for memory system errors."""
    pass


class ConsolidationError(MemorySystemError):
    """Exception raised during memory consolidation."""
    pass


class SearchError(MemorySystemError):
    """Exception raised during memory search."""
    pass


class MemoryStorageError(MemorySystemError):
    """Exception raised during memory storage operations."""
    pass


class MemorySystem:
    """
    Central memory management system integrating multiple memory types.
    
    This class provides a unified interface for managing different types of memory
    systems in a biologically-inspired cognitive architecture. It handles:
    
    - Short-Term Memory (STM): Temporary storage with decay
    - Long-Term Memory (LTM): Persistent storage with vector search
    - Episodic Memory: Event-based memories with rich context
    - Semantic Memory: Factual knowledge as subject-predicate-object triples
    - Prospective Memory: Future intentions and reminders
    - Procedural Memory: Skills and action sequences
    
    Features:
    - Automatic consolidation from STM to LTM
    - Cross-system memory retrieval and search
    - Memory reinforcement and forgetting
    - Dream-state consolidation for episodic memories
    - Thread-safe operations with connection pooling
    - Comprehensive error handling and logging
    - Performance monitoring and optimization
    
    Thread Safety:
        This class is thread-safe and can be used concurrently from multiple threads.
        Memory operations are protected by locks where necessary.
    
    Performance:
        - Uses connection pooling for database operations
        - Implements lazy loading for memory systems
        - Provides async operations for batch processing
        - Caches frequently accessed memories
    """
    
    def __init__(self, config: Optional[MemorySystemConfig] = None):
        """
        Initialize the integrated memory system.
        
        Args:
            config: Configuration object. If None, uses default configuration.
            
        Raises:
            MemorySystemError: If initialization fails
        """
        self._config = config or MemorySystemConfig()
        self._initialized = False
        self._shutdown = False
        self._consolidation_lock = threading.RLock()
        self._operation_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=self._config.max_concurrent_operations)
        
        # Memory systems (initialized lazily)
        self._stm: Optional[VectorShortTermMemory] = None
        self._ltm: Optional[VectorLongTermMemory] = None
        self._episodic: Optional[EpisodicMemorySystem] = None
        self._semantic: Optional[SemanticMemorySystem] = None
        self._prospective: Optional[ProspectiveMemorySystem] = None
        self._procedural: Optional[ProceduralMemory] = None

        # Session tracking
        self._session_memories: List[Dict[str, Any]] = []
        self._last_consolidation = datetime.now()
        self._last_prospective_process = datetime.now()

        # Performance monitoring
        self._operation_counts: Dict[str, int] = {}
        self._error_counts: Dict[str, int] = {}
        self._start_time = datetime.now()
        self._status_service = MemoryStatusService(
            get_stm=lambda: self.stm,
            get_ltm=lambda: self.ltm,
            get_config=lambda: self._config,
            get_start_time=lambda: self._start_time,
            get_last_consolidation=lambda: self._last_consolidation,
            get_session_memories_count=lambda: len(self._session_memories),
            get_operation_counts=lambda: self._operation_counts.copy(),
            get_error_counts=lambda: self._error_counts.copy(),
            is_active=self.is_initialized,
            clear_session_memories=self._clear_session_memories,
        )
        self._fact_service = MemoryFactService(get_semantic=lambda: self.semantic)
        self._retrieval_service = MemoryRetrievalService(
            get_stm=lambda: self.stm,
            get_ltm=lambda: self.ltm,
            get_episodic=lambda: self.episodic,
            get_config=lambda: self._config,
            increment_operation=self._increment_operation_count,
            increment_error=self._increment_error_count,
        )
        self._context_service = MemoryContextService(
            get_stm=lambda: self.stm,
            get_ltm=lambda: self.ltm,
            get_config=lambda: self._config,
        )
        self._consolidation_service = MemoryConsolidationService(
            get_stm=lambda: self.stm,
            get_ltm=lambda: self.ltm,
            get_config=lambda: self._config,
            get_last_consolidation=lambda: self._last_consolidation,
            set_last_consolidation=self._set_last_consolidation,
            get_executor=lambda: self._executor,
            get_consolidation_lock=lambda: self._consolidation_lock,
        )
        self._prospective_service = MemoryProspectiveService(
            get_config=lambda: self._config,
            get_last_process=lambda: self._last_prospective_process,
            set_last_process=self._set_last_prospective_process,
            get_executor=lambda: self._executor,
            get_prospective=lambda: self.prospective,
            get_ltm=lambda: self.ltm,
        )
        self._recall_service = MemoryRecallService(
            is_initialized=self.is_initialized,
            get_stm=lambda: self.stm,
            get_ltm=lambda: self.ltm,
            get_episodic=lambda: self.episodic,
            get_semantic=lambda: self.semantic,
            use_vector_ltm=lambda: self._config.use_vector_ltm,
        )
        self._storage_router = MemoryStorageRouter(
            get_config=lambda: self._config,
            get_stm=lambda: self.stm,
            get_ltm=lambda: self.ltm,
            create_episodic_memory=self.create_episodic_memory,
            append_session_memory=self._append_session_memory,
            increment_operation=self._increment_operation_count,
            increment_error=self._increment_error_count,
            should_consolidate=self._should_consolidate,
            schedule_consolidation=self._schedule_consolidation,
            should_process_prospective=self._should_process_prospective,
            schedule_prospective_processing=self._schedule_prospective_processing,
        )

        # Initialize core systems
        try:
            self._initialize_systems()
            self._initialized = True
            logger.info("Memory system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize memory system: {e}")
            raise MemorySystemError(f"Initialization failed: {e}") from e
    
    def _initialize_systems(self) -> None:
        """Initialize all memory subsystems."""
        try:
            bundle = MemorySubsystemInitializer(self._config).initialize()
            self._stm = bundle.stm
            self._ltm = bundle.ltm
            self._episodic = bundle.episodic
            self._semantic = bundle.semantic
            self._prospective = bundle.prospective
            self._procedural = bundle.procedural
        except Exception as e:
            logger.error(f"Error initializing memory systems: {e}")
            raise
    
    @property
    def stm(self) -> VectorShortTermMemory:
        """Get the STM system."""
        if self._stm is None:
            raise MemorySystemError("STM not initialized")
        return self._stm
    
    @property
    def ltm(self) -> VectorLongTermMemory:
        """Get the LTM system."""
        if self._ltm is None:
            raise MemorySystemError("LTM not initialized")
        return self._ltm
    
    @property
    def episodic(self) -> EpisodicMemorySystem:
        """Get the episodic memory system."""
        if self._episodic is None:
            raise MemorySystemError("Episodic memory not initialized")
        return self._episodic
    
    @property
    def semantic(self) -> SemanticMemorySystem:
        """Get the semantic memory system."""
        if self._semantic is None:
            raise MemorySystemError("Semantic memory not initialized")
        return self._semantic
    
    @property
    def prospective(self) -> Any:
        """Get the prospective memory system.

        Provides a lazy safety net: if initialization path didn't set _prospective
        (e.g., early failure before assignment), return the in-memory singleton
        instead of raising to keep higher-level systems functioning.
        """
        if self._prospective is None:
            try:
                from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory
                self._prospective = get_inmemory_prospective_memory()
                logger.info("Prospective memory lazily initialized to in-memory singleton")
            except Exception as e:  # pragma: no cover - should not occur
                raise MemorySystemError(f"Prospective memory not initialized and lazy init failed: {e}") from e
        return self._prospective
    
    @property
    def procedural(self) -> ProceduralMemory:
        """Get the procedural memory system."""
        if self._procedural is None:
            raise MemorySystemError("Procedural memory not initialized")
        return self._procedural
    
    def is_initialized(self) -> bool:
        """Check if the memory system is fully initialized."""
        return self._initialized and not self._shutdown
    
    def shutdown(self) -> None:
        """Gracefully shutdown the memory system."""
        if self._shutdown:
            return
            
        logger.info("Shutting down memory system...")
        self._shutdown = True
        
        try:
            # Shutdown executor
            self._executor.shutdown(wait=True)
            
            # Shutdown memory systems that support it
            for system in [self._semantic, self._prospective]:
                if system and hasattr(system, 'shutdown'):
                    try:
                        system.shutdown()
                    except Exception as e:
                        logger.error(f"Error shutting down memory system: {e}")
            
            logger.info("Memory system shutdown complete")
        except Exception as e:
            logger.error(f"Error during memory system shutdown: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def store_memory(
        self,
        memory_id: str,
        content: Any,
        importance: float = 0.5,
        attention_score: float = 0.0,
        emotional_valence: float = 0.0,
        memory_type: str = "episodic",
        tags: Optional[List[str]] = None,
        associations: Optional[List[str]] = None,
        force_ltm: bool = False
    ) -> MemoryOperationResult:
        """
        Store memory in the appropriate system based on importance and emotional valence.
        
        This method implements intelligent memory routing based on cognitive principles:
        - High importance or emotional memories go to LTM
        - Regular memories go to STM with potential consolidation later
        - Creates episodic memories for significant events
        
        Args:
            memory_id: Unique memory identifier (must be non-empty string)
            content: Memory content (any serializable object)
            importance: Importance score (0.0 to 1.0)
            attention_score: Attention during encoding (0.0 to 1.0)
            emotional_valence: Emotional weight (-1.0 to 1.0)
            memory_type: Type for categorization (default: "episodic")
            tags: Optional tags for organization
            associations: Associated memory IDs
            force_ltm: Force storage in LTM regardless of importance
        
        Returns:
            MemoryOperationResult with success status and details
            
        Raises:
            MemorySystemError: If system is not initialized or shutdown
            ValueError: If parameters are invalid
        """
        if not self.is_initialized():
            raise MemorySystemError("Memory system not initialized or shutdown")
        
        # Input validation
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        
        if not (0.0 <= importance <= 1.0):
            raise ValueError("importance must be between 0.0 and 1.0")
        
        if not (0.0 <= attention_score <= 1.0):
            raise ValueError("attention_score must be between 0.0 and 1.0")
        
        if not (-1.0 <= emotional_valence <= 1.0):
            raise ValueError("emotional_valence must be between -1.0 and 1.0")
        
        if content is None:
            raise ValueError("content cannot be None")
        
        memory_id = memory_id.strip()
        tags = tags or []
        associations = associations or []
        
        start_time = datetime.now()
        
        try:
            with self._operation_lock:
                return self._storage_router.store_memory(
                    result_factory=MemoryOperationResult,
                    memory_id=memory_id,
                    content=content,
                    importance=importance,
                    attention_score=attention_score,
                    emotional_valence=emotional_valence,
                    memory_type=memory_type,
                    tags=tags,
                    associations=associations,
                    force_ltm=force_ltm,
                    operation_time=start_time,
                )
                    
        except Exception as e:
            logger.error(f"Error storing memory {memory_id}: {e}")
            self._error_counts['store'] = self._error_counts.get('store', 0) + 1
            return MemoryOperationResult(
                success=False,
                memory_id=memory_id,
                error_message=str(e),
                operation_time=start_time
            )
    
    def _determine_storage_system(
        self, 
        importance: float, 
        emotional_valence: float, 
        force_ltm: bool
    ) -> str:
        """Determine which storage system to use based on memory characteristics."""
        return self._storage_router.determine_storage_system(importance, emotional_valence, force_ltm)
    
    def _store_in_ltm(
        self, 
        memory_id: str, 
        content: Any, 
        memory_type: str, 
        importance: float, 
        tags: List[str], 
        associations: List[str]
    ) -> bool:
        """Store memory in LTM with error handling."""
        return self._storage_router.store_in_ltm(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            tags=tags,
            associations=associations,
        )
    
    def _store_in_stm(
        self, 
        memory_id: str, 
        content: Any, 
        importance: float, 
        attention_score: float, 
        emotional_valence: float
    ) -> bool:
        """Store memory in STM with error handling."""
        return self._storage_router.store_in_stm(
            memory_id=memory_id,
            content=content,
            importance=importance,
            attention_score=attention_score,
            emotional_valence=emotional_valence,
        )
    
    def _create_episodic_memory_for_storage(
        self, 
        memory_id: str, 
        content: Any, 
        importance: float, 
        emotional_valence: float,
        force_ltm: bool
    ) -> Optional[str]:
        """Create corresponding episodic memory for significant storage events."""
        return self._storage_router.create_episodic_memory_for_storage(
            memory_id=memory_id,
            content=content,
            importance=importance,
            emotional_valence=emotional_valence,
            force_ltm=force_ltm,
        )
    
    def _schedule_consolidation(self) -> None:
        """Schedule memory consolidation in a background thread."""
        self._consolidation_service.schedule_consolidation(self.consolidate_memories)
    
    def _should_process_prospective(self) -> bool:
        """Check if prospective memory processing should occur."""
        return self._prospective_service.should_process()
    
    def _schedule_prospective_processing(self) -> None:
        """Schedule prospective memory processing in a background thread."""
        self._prospective_service.schedule_processing()
    
    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve memory from any system with intelligent search order.
        
        Searches STM first (faster access), then LTM, with comprehensive
        error handling and logging.
        
        Args:
            memory_id: Memory identifier to retrieve
        
        Returns:
            Memory content if found, None otherwise
            
        Raises:
            MemorySystemError: If system not initialized
            ValueError: If memory_id is invalid
        """
        if not self.is_initialized():
            raise MemorySystemError("Memory system not initialized or shutdown")
        
        if not isinstance(memory_id, str) or not memory_id.strip():
            raise ValueError("memory_id must be a non-empty string")
        
        memory_id = memory_id.strip()
        
        return self._retrieval_service.retrieve_memory(memory_id)
    
    def store_fact(self, subject: str, predicate: str, object_val: Any) -> str:
        """Stores a new fact in the semantic memory system."""
        return self._fact_service.store_fact(subject, predicate, object_val)

    def find_facts(self, subject: Optional[str] = None, predicate: Optional[str] = None, object_val: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Finds facts in the semantic memory system."""
        return self._fact_service.find_facts(subject, predicate, object_val)

    def delete_fact(self, subject: str, predicate: str, object_val: Any) -> bool:
        """Deletes a fact from the semantic memory system."""
        return self._fact_service.delete_fact(subject, predicate, object_val)
    
    def search_memories(
        self,
        query: str,
        search_stm: bool = True,
        search_ltm: bool = True,
        search_episodic: bool = True,
        memory_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Tuple[Any, float, str]]:
        """
        Search across memory systems
        
        Args:
            query: Search query
            search_stm: Whether to search STM
            search_ltm: Whether to search LTM
            search_episodic: Whether to search Episodic Memory
            memory_types: Filter by memory types (LTM only)
            max_results: Maximum results
        
        Returns:
            List of (memory_object, relevance_score, source_system) tuples
        """
        return self._retrieval_service.search_memories(
            query=query,
            search_stm=search_stm,
            search_ltm=search_ltm,
            search_episodic=search_episodic,
            memory_types=memory_types,
            max_results=max_results,
        )
    
    def consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        """
        Consolidate memories from STM to LTM
        
        Args:
            force: Force consolidation regardless of timing
        
        Returns:
            Dict with consolidation statistics
        """
        return self._consolidation_service.consolidate_memories(force=force)
    
    def dream_state_consolidation(
        self,
        min_importance: float = 0.5,
        max_consolidation: float = 0.9,
        limit: int = 20,
        strength_increment: float = 0.2,
        cluster: bool = True
    ) -> dict:
        """
        Run batch dream-state consolidation on episodic memory (with optional clustering/merging).
        Returns a summary dict.
        """
        return self.episodic.batch_consolidate_memories(
            min_importance=min_importance,
            max_consolidation=max_consolidation,
            limit=limit,
            strength_increment=strength_increment,
            cluster=cluster
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive memory system status with performance metrics."""
        return self._status_service.get_status()
    
    def reset_session(self) -> None:
        """Reset session-specific memory tracking"""
        with self._operation_lock:
            self._status_service.reset_session()

    def _clear_session_memories(self) -> None:
        self._session_memories = []

    def _increment_operation_count(self, key: str) -> None:
        self._operation_counts[key] = self._operation_counts.get(key, 0) + 1

    def _increment_error_count(self, key: str) -> None:
        self._error_counts[key] = self._error_counts.get(key, 0) + 1

    def _append_session_memory(self, entry: Dict[str, Any]) -> None:
        self._session_memories.append(entry)
    
    def _should_consolidate(self) -> bool:
        """Check if automatic consolidation should occur"""
        return self._consolidation_service.should_consolidate()

    def _set_last_consolidation(self, value: datetime) -> None:
        self._last_consolidation = value

    def _set_last_prospective_process(self, value: datetime) -> None:
        self._last_prospective_process = value
    
    def search_stm_semantic(
        self,
        query: str,
        max_results: int = 5,
        min_similarity: float = 0.5,
        min_activation: float = 0.0
    ) -> List[Tuple[Any, float]]:
        """
        Semantic search in STM using vector similarity
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_similarity: Minimum similarity threshold
            min_activation: Minimum activation threshold
        
        Returns:
            List of (memory_item, relevance_score) tuples
        """
        return self._retrieval_service.search_stm_semantic(
            query=query,
            max_results=max_results,
            min_similarity=min_similarity,
            min_activation=min_activation,
        )
    
    def get_context_for_query(
        self,
        query: str,
        max_stm_context: int = 5,
        max_ltm_context: int = 5,
        min_relevance: float = 0.3
    ) -> Dict[str, List[Any]]:
        """
        Get relevant context from both STM and LTM for cognitive processing
        
        Args:
            query: Query to find context for
            max_stm_context: Maximum STM context items
            max_ltm_context: Maximum LTM context items  
            min_relevance: Minimum relevance threshold
        
        Returns:
            Dictionary with 'stm' and 'ltm' context lists
        """
        return self._context_service.get_context_for_query(
            query=query,
            max_stm_context=max_stm_context,
            max_ltm_context=max_ltm_context,
            min_relevance=min_relevance,
        )
    
    # Episodic Memory Methods
    def create_episodic_memory(
        self,
        summary: str,
        detailed_content: str,
        participants: Optional[List[str]] = None,
        location: Optional[str] = None,
        emotional_state: Optional[str] = None,
        cognitive_load: float = 0.5,
        stm_ids: Optional[List[str]] = None,
        ltm_ids: Optional[List[str]] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        life_period: Optional[str] = None
    ) -> Optional[str]:
        """
        Create a new episodic memory with rich contextual information
        
        Args:
            summary: Brief summary of the episode
            detailed_content: Detailed content/narrative of the episode
            participants: People/entities involved
            location: Physical or virtual location
            emotional_state: Emotional context
            cognitive_load: Cognitive load during episode (0.0-1.0)
            stm_ids: Associated STM memory IDs
            ltm_ids: Associated LTM memory IDs
            importance: Episode importance (0.0-1.0)
            emotional_valence: Emotional valence (-1.0 to 1.0)
            life_period: Life period categorization
          Returns:
            Episode ID if created successfully, None otherwise
        """
        if not hasattr(self, 'episodic') or self.episodic is None:
            logger.warning("Episodic memory system not initialized")
            return None
        
        try:            
            # Create context from provided information
            from .episodic.episodic_memory import EpisodicContext
            context = EpisodicContext(
                participants=participants or [],
                location=location or "",
                emotional_state=emotional_valence,  # Use emotional_valence as emotional_state
                cognitive_load=cognitive_load
            )
            
            episode_id = self.episodic.store_memory(
                detailed_content=detailed_content,
                context=context,
                associated_stm_ids=stm_ids or [],
                associated_ltm_ids=ltm_ids or [],
                importance=importance,
                emotional_valence=emotional_valence,
                life_period=life_period
            )
            return episode_id
        except Exception as e:
            logger.error(f"Error creating episodic memory: {e}")
            return None
    
    def search_episodic_memories(
        self,
        query: str,
        max_results: int = 10,
        min_similarity: float = 0.5,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        life_periods: Optional[List[str]] = None,
        min_importance: float = 0.0
    ) -> List['EpisodicSearchResult']:
        """
        Search episodic memories with contextual filters
        
        Args:
            query: Search query
            max_results: Maximum results to return
            min_similarity: Minimum similarity threshold
            date_range: Optional date range filter (start, end)
            life_periods: Optional life period filter
            min_importance: Minimum importance threshold
          Returns:
            List of episodic search results
        """
        if not hasattr(self, 'episodic') or self.episodic is None:
            logger.warning("Episodic memory system not initialized")
            return []
        
        try:
            results = self.episodic.search_memories(
                query=query,
                limit=max_results,
                min_relevance=min_similarity,
                time_range=date_range,
                life_period=life_periods[0] if life_periods and len(life_periods) > 0 else None,
                importance_threshold=min_importance
            )
            return results
        except Exception as e:
            logger.error(f"Error searching episodic memories: {e}")
            return []

    def get_cross_referenced_episodes(
        self,
        memory_id: str,
        memory_system: str = "both"  # "stm", "ltm", or "both"
    ) -> List[Any]:
        """
        Get episodic memories cross-referenced to STM/LTM memories
        
        Args:
            memory_id: STM or LTM memory ID
            memory_system: Which system to search ("stm", "ltm", "both")
        
        Returns:
            List of cross-referenced episodic memories
        """
        if not hasattr(self, 'episodic') or self.episodic is None:
            logger.warning("Episodic memory system not initialized")
            return []
            
        try:
            # Search for memories with cross-references to the given memory ID
            all_memories = []
            for memory in self.episodic._memory_cache.values():
                found = False
                
                if memory_system in ["stm", "both"] and memory_id in memory.associated_stm_ids:
                    found = True
                elif memory_system in ["ltm", "both"] and memory_id in memory.associated_ltm_ids:
                    found = True
                elif memory_system == "both" and memory_id in memory.source_memory_ids:
                    found = True
                
                if found:
                    all_memories.append(memory)
            
            return all_memories
        except Exception as e:
            logger.error(f"Error getting cross-referenced episodes: {e}")
            return []
    
    # Prospective memory API
    def add_prospective_reminder(self, description: str, due_time: datetime) -> str:
        """Add a new prospective reminder"""
        try:
            return self.prospective.store(
                description=description,
                trigger_time=due_time.isoformat(),
                tags=[],
                memory_type="ltm"
            )
        except Exception as e:
            logger.error(f"Failed to add prospective reminder: {e}")
            return ""
    
    def get_due_prospective_reminders(self, now: Optional[datetime] = None):
        """Get due prospective reminders"""
        return self.prospective.get_due_reminders(now)
    
    def list_prospective_reminders(self, include_completed: bool = False):
        """List all prospective reminders"""
        return self.prospective.list_reminders(include_completed)
    
    def complete_prospective_reminder(self, item_id: str):
        """Mark a prospective reminder as complete"""
        return self.prospective.complete_reminder(item_id)

    def hierarchical_search(
        self,
        query: str,
        max_results: int = 10,
        max_per_system: int = 5,
    ) -> List[Tuple[Any, float, str]]:
        """
        Perform a hierarchical search across all memory systems in a specific order.
        Order: STM -> LTM -> Episodic -> Semantic
        """
        return self._recall_service.hierarchical_search(
            query=query,
            max_results=max_results,
            max_per_system=max_per_system,
        )

    def proactive_recall(
        self,
        query: str,
        max_results: int = 5,
        min_relevance: float = 0.7,
        context_window: int = 3,
        use_ai_summary: bool = False,
        openai_client = None
    ) -> Dict[str, Any]:
        """
        Perform proactive recall of relevant memories based on current context.
        
        This method implements episodic memory proactive recall by:
        1. Searching across all memory systems for relevant content
        2. Prioritizing recent and emotionally salient memories
        3. Providing contextual summaries for recall
        
        Args:
            query: The current user input or context to recall against
            max_results: Maximum number of memories to recall
            min_relevance: Minimum relevance threshold for recall
            context_window: Number of recent memories to consider for context
            use_ai_summary: Whether to use AI (GPT-4.1) for enhanced summarization
            openai_client: OpenAI client instance for AI summarization
            
        Returns:
            Dictionary containing recalled memories with metadata
        """
        return self._recall_service.proactive_recall(
            query=query,
            max_results=max_results,
            min_relevance=min_relevance,
            context_window=context_window,
            use_ai_summary=use_ai_summary,
            openai_client=openai_client,
        )

    def _extract_memory_content(self, memory: Any, source: str) -> str:
        """Extract readable content from memory object."""
        return self._recall_service.extract_memory_content(memory, source)

    def _extract_timestamp(self, memory: Any, source: str) -> Optional[datetime]:
        """Extract timestamp from memory object."""
        return self._recall_service.extract_timestamp(memory, source)

    def _extract_importance(self, memory: Any, source: str) -> float:
        """Extract importance score from memory object."""
        return self._recall_service.extract_importance(memory, source)

    def _extract_emotional_valence(self, memory: Any, source: str) -> float:
        """Extract emotional valence from memory object."""
        return self._recall_service.extract_emotional_valence(memory, source)

    def _extract_tags(self, memory: Any, source: str) -> List[str]:
        """Extract tags from memory object."""
        return self._recall_service.extract_tags(memory, source)

    def _generate_recall_summary(self, memories: List[Dict], query: str, use_ai: bool = False, openai_client = None) -> str:
        """Generate a concise summary of recalled memories."""
        return self._recall_service.generate_recall_summary(memories, query, use_ai, openai_client)

    def _generate_basic_summary(self, memories: List[Dict]) -> str:
        """Generate a basic statistical summary of recalled memories."""
        return self._recall_service.generate_basic_summary(memories)

    def _generate_ai_summary(self, memories: List[Dict], query: str, openai_client) -> str:
        """Generate an AI-powered summary of recalled memories using GPT-4.1."""
        return self._recall_service.generate_ai_summary(memories, query, openai_client)
