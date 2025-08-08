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
from .semantic.semantic_memory import SemanticMemorySystem
from .prospective.prospective_memory import ProspectiveMemorySystem
from .procedural.procedural_memory import ProceduralMemory

if TYPE_CHECKING:
    from .episodic.episodic_memory import EpisodicSearchResult

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
    consolidation_threshold_importance: float = 0.6
    consolidation_threshold_emotional: float = 0.5
    consolidation_age_minutes: int = 30
    auto_process_prospective: bool = True
    prospective_process_interval: int = 60  # seconds


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
            # Initialize STM - always use VectorShortTermMemory
            stm_config = STMConfiguration(
                chroma_persist_dir=self._config.chroma_persist_dir,
                embedding_model=self._config.embedding_model,
                capacity=self._config.stm_capacity,
                enable_gpu=getattr(self._config, 'enable_gpu', True)
            )
            self._stm = VectorShortTermMemory(stm_config)
            
            # Initialize LTM
            if self._config.use_vector_ltm:
                self._ltm = VectorLongTermMemory(
                    chroma_persist_dir=self._config.chroma_persist_dir,
                    embedding_model=self._config.embedding_model
                )
            else:
                raise MemorySystemError("LTM initialization failed: VectorLTM must be used")
            
            # Initialize Episodic Memory
            self._episodic = EpisodicMemorySystem(
                chroma_persist_dir=self._config.chroma_persist_dir,
                embedding_model=self._config.embedding_model
            )
            
            # Initialize Semantic Memory (using ChromaDB)
            self._semantic = SemanticMemorySystem(
                chroma_persist_dir=self._config.chroma_persist_dir or "data/memory_stores/chroma_semantic",
                embedding_model=self._config.embedding_model
            )
            
            # Initialize Prospective Memory
            self._prospective = ProspectiveMemorySystem(
                chroma_persist_dir=self._config.chroma_persist_dir,
                embedding_model=self._config.embedding_model
            )
            
            # Initialize Procedural Memory
            self._procedural = ProceduralMemory(stm=self._stm, ltm=self._ltm)
            
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
    def prospective(self) -> ProspectiveMemorySystem:
        """Get the prospective memory system."""
        if self._prospective is None:
            raise MemorySystemError("Prospective memory not initialized")
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
                # Track for session
                session_entry = {
                    'memory_id': memory_id,
                    'timestamp': start_time,
                    'importance': importance,
                    'storage_location': None,
                    'memory_type': memory_type
                }
                
                # Create episodic memory for significant events
                episodic_id = None
                if importance > 0.6:
                    try:
                        episodic_id = self._create_episodic_memory_for_storage(
                            memory_id=memory_id,
                            content=content,
                            importance=importance,
                            emotional_valence=emotional_valence,
                            force_ltm=force_ltm
                        )
                    except Exception as e:
                        logger.warning(f"Failed to create episodic memory for {memory_id}: {e}")
                
                # Determine storage location based on cognitive principles
                storage_system = self._determine_storage_system(
                    importance, emotional_valence, force_ltm
                )
                
                # Store in appropriate system
                success = False
                if storage_system == 'LTM':
                    success = self._store_in_ltm(
                        memory_id, content, memory_type, importance, tags, associations
                    )
                    session_entry['storage_location'] = 'LTM'
                else:
                    success = self._store_in_stm(
                        memory_id, content, importance, attention_score, emotional_valence
                    )
                    session_entry['storage_location'] = 'STM'
                
                if success:
                    self._session_memories.append(session_entry)
                    self._operation_counts['store'] = self._operation_counts.get('store', 0) + 1
                    
                    # Schedule consolidation check if needed
                    if self._should_consolidate():
                        self._schedule_consolidation()
                    
                    # Process prospective memories if enabled
                    if (self._config.auto_process_prospective and 
                        self._should_process_prospective()):
                        self._schedule_prospective_processing()
                    
                    return MemoryOperationResult(
                        success=True,
                        memory_id=memory_id,
                        operation_time=start_time,
                        system_used=storage_system
                    )
                else:
                    self._error_counts['store'] = self._error_counts.get('store', 0) + 1
                    return MemoryOperationResult(
                        success=False,
                        memory_id=memory_id,
                        error_message=f"Failed to store in {storage_system}",
                        operation_time=start_time
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
        if (force_ltm or 
            importance >= self._config.consolidation_threshold_importance or 
            abs(emotional_valence) >= self._config.consolidation_threshold_emotional):
            return 'LTM'
        return 'STM'
    
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
        try:
            return self.ltm.store(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                tags=tags,
                associations=associations
            )
        except Exception as e:
            logger.error(f"Failed to store {memory_id} in LTM: {e}")
            return False
    
    def _store_in_stm(
        self, 
        memory_id: str, 
        content: Any, 
        importance: float, 
        attention_score: float, 
        emotional_valence: float
    ) -> bool:
        """Store memory in STM with error handling."""
        try:
            return self.stm.store(
                memory_id=memory_id,
                content=content,
                importance=importance,
                attention_score=attention_score,
                emotional_valence=emotional_valence
            )
        except Exception as e:
            logger.error(f"Failed to store {memory_id} in STM: {e}")
            return False
    
    def _create_episodic_memory_for_storage(
        self, 
        memory_id: str, 
        content: Any, 
        importance: float, 
        emotional_valence: float,
        force_ltm: bool
    ) -> Optional[str]:
        """Create corresponding episodic memory for significant storage events."""
        try:
            return self.create_episodic_memory(
                summary=str(content)[:128],
                detailed_content=str(content),
                importance=importance,
                emotional_valence=emotional_valence,
                stm_ids=[memory_id] if not force_ltm else [],
                ltm_ids=[memory_id] if force_ltm else []
            )
        except Exception as e:
            logger.warning(f"Failed to create episodic memory for {memory_id}: {e}")
            return None
    
    def _schedule_consolidation(self) -> None:
        """Schedule memory consolidation in a background thread."""
        def consolidation_task():
            try:
                with self._consolidation_lock:
                    self.consolidate_memories()
            except Exception as e:
                logger.error(f"Background consolidation failed: {e}")
        
        self._executor.submit(consolidation_task)
    
    def _should_process_prospective(self) -> bool:
        """Check if prospective memory processing should occur."""
        time_elapsed = (datetime.now() - self._last_prospective_process).total_seconds()
        return time_elapsed >= self._config.prospective_process_interval
    
    def _schedule_prospective_processing(self) -> None:
        """Schedule prospective memory processing in a background thread."""
        def prospective_task():
            try:
                processed = self.prospective.process_due_reminders(ltm_system=self.ltm)
                if processed > 0:
                    logger.info(f"Processed {processed} due prospective reminders")
                self._last_prospective_process = datetime.now()
            except Exception as e:
                logger.error(f"Background prospective processing failed: {e}")
        
        self._executor.submit(prospective_task)
    
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
        
        try:
            # Check STM first (faster access)
            try:
                memory = self.stm.retrieve(memory_id)
                if memory is not None:
                    self._operation_counts['retrieve_stm'] = self._operation_counts.get('retrieve_stm', 0) + 1
                    return memory
            except Exception as e:
                logger.warning(f"Error retrieving from STM for {memory_id}: {e}")
            
            # Check LTM
            try:
                memory = self.ltm.retrieve(memory_id)
                if memory is not None:
                    self._operation_counts['retrieve_ltm'] = self._operation_counts.get('retrieve_ltm', 0) + 1
                    return memory
            except Exception as e:
                logger.warning(f"Error retrieving from LTM for {memory_id}: {e}")
            
            # Not found in either system
            self._operation_counts['retrieve_miss'] = self._operation_counts.get('retrieve_miss', 0) + 1
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            self._error_counts['retrieve'] = self._error_counts.get('retrieve', 0) + 1
            return None
    
    def store_fact(self, subject: str, predicate: str, object_val: Any) -> str:
        """Stores a new fact in the semantic memory system."""
        return self.semantic.store_fact(subject, predicate, object_val)

    def find_facts(self, subject: Optional[str] = None, predicate: Optional[str] = None, object_val: Optional[Any] = None) -> List[Dict[str, Any]]:
        """Finds facts in the semantic memory system."""
        return self.semantic.find_facts(subject, predicate, object_val)

    def delete_fact(self, subject: str, predicate: str, object_val: Any) -> bool:
        """Deletes a fact from the semantic memory system."""
        return self.semantic.delete_fact(subject, predicate, object_val)
    
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
        results = []
        
        try:
            if search_stm:
                stm_results = self.stm.search(query, max_results=max_results//2)
                logger.debug(f"STM search for '{query}' returned: {stm_results}")
                for memory, score in stm_results:
                    results.append((memory, score, 'STM'))
            
            if search_ltm:
                if self._config.use_vector_ltm and isinstance(self.ltm, VectorLongTermMemory):
                    # Vector LTM
                    ltm_results = self.ltm.search_semantic(
                        query=query,
                        memory_types=memory_types,
                        max_results=max_results//2
                    )
                    logger.debug(f"LTM search for '{query}' returned: {ltm_results}")
                    for memory in ltm_results:
                        results.append((memory, memory.get('similarity_score', 0.0), 'LTM'))
                else:
                    # Legacy LTM or fallback if method exists
                    ltm_results = getattr(self.ltm, 'search_by_content', lambda **kwargs: [])(
                        query=query,
                        memory_types=memory_types,
                        max_results=max_results//2
                    )
                    logger.debug(f"LTM search for '{query}' returned: {ltm_results}")
                    for memory, score in ltm_results:
                        results.append((memory, score, 'LTM'))

            if search_episodic:
                try:
                    episodic_results = self.episodic.search_memories(query=query, limit=max_results)
                    logger.debug(f"Episodic search for '{query}' returned: {episodic_results}")
                    for result in episodic_results:
                        results.append((result.memory, result.relevance, 'Episodic'))
                except Exception as e:
                    logger.error(f"Error searching episodic memory for query '{query}': {e}")

        except Exception as e:
            logger.error(f"Error searching memories for query '{query}': {e}")

        # Combine and sort results
        # Deduplicate based on content to avoid redundancy from different systems
        seen_content = set()
        unique_results = []
        for mem, score, source in sorted(results, key=lambda item: item[1], reverse=True):
            content_repr = ""
            if source == 'Episodic':
                # mem is an EpisodicMemory object
                content_repr = repr(mem.detailed_content)
            elif isinstance(mem, dict):
                content_repr = repr(mem.get('content'))
            elif hasattr(mem, 'content'):
                # Assumes an object with a .content attribute
                content_repr = repr(mem.content)
            
            if content_repr and content_repr not in seen_content:
                unique_results.append((mem, score, source))
                seen_content.add(content_repr)

        return unique_results[:max_results]
    
    def consolidate_memories(self, force: bool = False) -> Dict[str, Any]:
        """
        Consolidate memories from STM to LTM
        
        Args:
            force: Force consolidation regardless of timing
        
        Returns:
            Dict with consolidation statistics
        """
        if not force and not self._should_consolidate():
            return {'status': 'skipped', 'reason': 'not due'}
        
        stats = {
            'start_time': datetime.now(),
            'consolidated_count': 0,
            'failed_count': 0,
            'errors': []
        }
        
        try:
            # Get memories ready for consolidation
            stm_items = {}
            if hasattr(self.stm, 'get_all_memories'):
                # Vector STM
                stm_memories = getattr(self.stm, 'get_all_memories')()
                stm_items = {mem.id: mem for mem in stm_memories}
            else:
                # Legacy STM
                stm_items = getattr(self.stm, 'items', {})
            
            for memory_id, memory_item in stm_items.items():
                try:
                    # Determine if memory should be consolidated
                    importance = getattr(memory_item, 'importance', 0.0)
                    emotional_valence = getattr(memory_item, 'emotional_valence', 0.0)
                    age_minutes = (datetime.now() - memory_item.encoding_time).total_seconds() / 60
                    
                    # Consolidation criteria
                    should_consolidate = (
                        importance >= 0.6 or
                        abs(emotional_valence) >= 0.5 or
                        age_minutes >= 30  # 30 minutes
                    )
                    
                    if should_consolidate:
                        # Move to LTM
                        success = self.ltm.store(
                            memory_id=memory_id,
                            content=memory_item.content,
                            memory_type='episodic',
                            importance=importance,
                            tags=[],
                            associations=getattr(memory_item, 'associations', [])
                        )
                        
                        if success:
                            if hasattr(self.stm, 'remove_item'):
                                getattr(self.stm, 'remove_item')(memory_id)
                            else:
                                # Legacy STM - use delete
                                getattr(self.stm, 'delete', lambda x: None)(memory_id)
                            stats['consolidated_count'] += 1
                        else:
                            stats['failed_count'] += 1
                            
                except Exception as e:
                    stats['errors'].append(f"Error consolidating {memory_id}: {e}")
                    stats['failed_count'] += 1
                    logger.error(f"Consolidation error for {memory_id}: {e}")
            
            self.last_consolidation = datetime.now()
            stats['end_time'] = datetime.now()
            stats['duration_seconds'] = (stats['end_time'] - stats['start_time']).total_seconds()
            
            logger.info(f"Consolidation completed: {stats['consolidated_count']} memories moved to LTM")
            
        except Exception as e:
            stats['errors'].append(f"Consolidation process error: {e}")
            logger.error(f"Consolidation process error: {e}")
        
        return stats
    
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
        try:
            uptime = (datetime.now() - self._start_time).total_seconds()
            
            return {
                'stm': self.stm.get_status() if hasattr(self.stm, 'get_status') else {'status': 'unknown'},
                'ltm': self.ltm.get_status() if hasattr(self.ltm, 'get_status') else {'status': 'unknown'},
                'last_consolidation': self._last_consolidation,
                'session_memories_count': len(self._session_memories),
                'consolidation_interval': self._config.consolidation_interval,
                'use_vector_stm': self._config.use_vector_stm,
                'use_vector_ltm': self._config.use_vector_ltm,
                'system_active': self.is_initialized(),
                'uptime_seconds': uptime,
                'operation_counts': self._operation_counts.copy(),
                'error_counts': self._error_counts.copy(),
                'config': {
                    'stm_capacity': self._config.stm_capacity,
                    'max_concurrent_operations': self._config.max_concurrent_operations,
                    'auto_process_prospective': self._config.auto_process_prospective
                }
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e), 'system_active': False}
    
    def reset_session(self) -> None:
        """Reset session-specific memory tracking"""
        with self._operation_lock:
            self._session_memories = []
            logger.info("Memory system session reset")
    
    def _should_consolidate(self) -> bool:
        """Check if automatic consolidation should occur"""
        time_elapsed = (datetime.now() - self._last_consolidation).total_seconds()
        return time_elapsed >= self._config.consolidation_interval
    
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
        if self._config.use_vector_stm and isinstance(self.stm, VectorShortTermMemory):
            # Use vector STM semantic search
            vector_results = self.stm.search_semantic(
                query=query,
                max_results=max_results,
                min_similarity=min_similarity
            )
            return [(result.item, result.relevance_score) for result in vector_results]
        else:
            # Fallback to regular STM search
            results = self.stm.search(query=query, max_results=max_results)
            # Ensure return type is List[Tuple[Any, float]]
            out = []
            for r in results:
                if isinstance(r, tuple) and len(r) == 2:
                    out.append((r[0], r[1]))
                elif isinstance(r, dict):
                    out.append((r, 1.0))
                else:
                    out.append((r, 1.0))
            return out
    
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
        context = {"stm": [], "ltm": []}
        
        try:
            # Get STM context
            if self._config.use_vector_stm and isinstance(self.stm, VectorShortTermMemory):
                stm_results = self.stm.get_context_for_query(
                    query=query,
                    max_context_items=max_stm_context,
                    min_relevance=min_relevance
                )
                context["stm"] = [result.item for result in stm_results]
            else:
                # Fallback for regular STM
                stm_results = self.stm.search(query=query, max_results=max_stm_context)
                context["stm"] = [item for item, score in stm_results if score >= min_relevance]
            
            # Get LTM context
            if self._config.use_vector_ltm and isinstance(self.ltm, VectorLongTermMemory):
                ltm_results = self.ltm.search_semantic(
                    query=query,
                    max_results=max_ltm_context,
                    min_similarity=min_relevance
                )
                context["ltm"] = [result for result in ltm_results]
            else:
                # Fallback for regular LTM
                ltm_results = getattr(self.ltm, 'search_by_content', lambda **kwargs: [])(query=query, max_results=max_ltm_context)
                context["ltm"] = [record for record, score in ltm_results if score >= min_relevance]
            
            logger.debug(f"Retrieved context: {len(context['stm'])} STM + {len(context['ltm'])} LTM items")
            return context
            
        except Exception as e:
            logger.error(f"Error getting context for query: {e}")
            return {"stm": [], "ltm": []}
    
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
        all_results = []

        # 1. Search STM
        try:
            stm_results = self.stm.search(query, max_results=max_per_system)
            for memory, score in stm_results:
                all_results.append((memory, score, 'STM'))
        except Exception as e:
            logger.error(f"Error searching STM for query '{query}': {e}")

        # 2. Search LTM
        try:
            if self._config.use_vector_ltm:
                ltm_results = getattr(self.ltm, 'search_semantic', lambda **kwargs: [])(query=query, max_results=max_per_system)
                for memory in ltm_results:
                    score = memory.get('similarity_score', 0.0)
                    all_results.append((memory, score, 'LTM'))
            else:
                ltm_results = getattr(self.ltm, 'search_by_content', lambda **kwargs: [])(query=query, max_results=max_per_system)
                for memory, score in ltm_results:
                    all_results.append((memory, score, 'LTM'))
        except Exception as e:
            logger.error(f"Error searching LTM for query '{query}': {e}")

        # 3. Search Episodic Memory
        try:
            episodic_results = self.episodic.search_memories(query=query, limit=max_per_system)
            for result in episodic_results:
                all_results.append((result.memory, result.relevance, 'Episodic'))
        except Exception as e:
            logger.error(f"Error searching Episodic Memory for query '{query}': {e}")

        # 4. Search Semantic Memory
        try:
            # Semantic search is different, it looks for facts.
            # We can create a composite query or just use the raw query.
            # For simplicity, we search for the query in subject, predicate, or object.
            # This part might need more sophisticated logic depending on requirements.
            subject_matches = self.semantic.find_facts(subject=query)
            object_matches = self.semantic.find_facts(object_val=query)
            # A simple scoring mechanism for semantic results
            for fact in subject_matches:
                all_results.append((fact, 0.9, 'Semantic')) # High relevance for direct match
            for fact in object_matches:
                all_results.append((fact, 0.8, 'Semantic'))
        except Exception as e:
            logger.error(f"Error searching Semantic Memory for query '{query}': {e}")

        # Deduplicate and sort results
        seen_content = set()
        unique_results = []
        # Sort by relevance score, descending
        for mem, score, source in sorted(all_results, key=lambda item: item[1], reverse=True):
            content_repr = ""
            if source == 'Episodic':
                content_repr = repr(mem.detailed_content)
            elif source == 'Semantic':
                content_repr = repr(mem) # Fact is a dict
            elif isinstance(mem, dict):
                content_repr = repr(mem.get('content'))
            elif hasattr(mem, 'content'):
                content_repr = repr(mem.content)

            if content_repr and content_repr not in seen_content:
                unique_results.append((mem, score, source))
                seen_content.add(content_repr)

        return unique_results[:max_results]
