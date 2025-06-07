"""
Integrated Memory System
Coordinates between STM, LTM, and other memory components
"""
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime
import logging
from .stm import ShortTermMemory, VectorShortTermMemory
from .ltm import LongTermMemory, VectorLongTermMemory
from .episodic import EpisodicMemorySystem

logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Central memory management system integrating STM and LTM.
    
    Features:
    - Automatic STM to LTM consolidation
    - Cross-system memory retrieval
    - Memory reinforcement and forgetting
    - Dream-state consolidation    """
    
    def __init__(
        self,
        stm_capacity: int = 100,  # Increased for cognitive system
        stm_decay_threshold: float = 0.1,
        ltm_storage_path: Optional[str] = None,
        consolidation_interval: int = 300,  # 5 minutes
        use_vector_ltm: bool = True,
        use_vector_stm: bool = True,  # New parameter for vector STM
        chroma_persist_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize integrated memory system
        
        Args:
            stm_capacity: STM capacity limit (100 for cognitive system)
            stm_decay_threshold: STM decay threshold
            ltm_storage_path: LTM storage path
            consolidation_interval: Auto-consolidation interval in seconds
            use_vector_ltm: Whether to use vector-based LTM with ChromaDB
            use_vector_stm: Whether to use vector-based STM with ChromaDB
            chroma_persist_dir: ChromaDB persistence directory
            embedding_model: SentenceTransformer model name
        """
        # Initialize STM with vector database support
        self.use_vector_stm = use_vector_stm
        if use_vector_stm:
            self.stm = VectorShortTermMemory(
                capacity=stm_capacity,
                decay_threshold=stm_decay_threshold,
                chroma_persist_dir=chroma_persist_dir,
                embedding_model=embedding_model,
                use_vector_db=True
            )
        else:
            self.stm = ShortTermMemory(capacity=stm_capacity, decay_threshold=stm_decay_threshold)
          # Initialize LTM with vector database support
        self.use_vector_ltm = use_vector_ltm
        if use_vector_ltm:
            self.ltm = VectorLongTermMemory(
                storage_path=ltm_storage_path,
                chroma_persist_dir=chroma_persist_dir,
                embedding_model=embedding_model,
                use_vector_db=True,
                enable_json_backup=True
            )
        else:
            self.ltm = LongTermMemory(storage_path=ltm_storage_path)
        
        # Initialize Episodic Memory (if enabled)
        episodic_persist_dir = f"{chroma_persist_dir}_episodic" if chroma_persist_dir else None
        self.episodic = EpisodicMemorySystem(
            chroma_persist_dir=episodic_persist_dir,
            embedding_model=embedding_model
        )
        
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = datetime.now()
        self.session_memories = []  # Track memories for this session
        
        logger.info(f"Integrated memory system initialized (Vector STM: {use_vector_stm}, Vector LTM: {use_vector_ltm})")
    
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
    ) -> bool:
        """
        Store memory in appropriate system (STM or LTM)
        
        Args:
            memory_id: Unique memory identifier
            content: Memory content
            importance: Importance score (0.0 to 1.0)
            attention_score: Attention during encoding
            emotional_valence: Emotional weight (-1.0 to 1.0)
            memory_type: Type for LTM storage
            tags: Tags for organization
            associations: Associated memory IDs
            force_ltm: Force storage in LTM regardless of importance
        
        Returns:
            True if stored successfully
        """
        # Track for session
        session_entry = {
            'memory_id': memory_id,
            'timestamp': datetime.now(),
            'importance': importance,
            'storage_location': None
        }
        
        try:
            # Determine storage location
            if force_ltm or importance >= 0.7 or abs(emotional_valence) >= 0.6:
                # Store in LTM
                success = self.ltm.store(
                    memory_id=memory_id,
                    content=content,
                    memory_type=memory_type,
                    importance=importance,
                    tags=tags or [],
                    associations=associations or []
                )
                session_entry['storage_location'] = 'LTM'
            else:
                # Store in STM
                success = self.stm.store(
                    memory_id=memory_id,
                    content=content,
                    importance=importance,
                    attention_score=attention_score,
                    emotional_valence=emotional_valence
                )
                session_entry['storage_location'] = 'STM'
            
            if success:
                self.session_memories.append(session_entry)
                
                # Check for automatic consolidation
                if self._should_consolidate():
                    self.consolidate_memories()
            
            return success
            
        except Exception as e:
            logger.error(f"Error storing memory {memory_id}: {e}")
            return False
    
    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve memory from any system
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            Memory content if found, None otherwise
        """
        try:
            # Check STM first (faster)
            memory = self.stm.retrieve(memory_id)
            if memory:
                return memory
            
            # Check LTM
            return self.ltm.retrieve(memory_id)
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    def search_memories(
        self,
        query: str,
        search_stm: bool = True,
        search_ltm: bool = True,
        memory_types: Optional[List[str]] = None,
        max_results: int = 10
    ) -> List[Tuple[Any, float, str]]:
        """
        Search across memory systems
        
        Args:
            query: Search query
            search_stm: Whether to search STM
            search_ltm: Whether to search LTM
            memory_types: Filter by memory types (LTM only)
            max_results: Maximum results
        
        Returns:
            List of (memory_object, relevance_score, source_system) tuples
        """
        results = []
        
        try:
            if search_stm:
                stm_results = self.stm.search(query, max_results=max_results//2)
                for memory, score in stm_results:
                    results.append((memory, score, 'STM'))
            
            if search_ltm:
                ltm_results = self.ltm.search_by_content(
                    query=query,
                    memory_types=memory_types,
                    max_results=max_results//2
                )
                for memory, score in ltm_results:
                    results.append((memory, score, 'LTM'))
            
            # Sort by relevance
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Error searching memories with query '{query}': {e}")
            return []
    
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
            stm_memories = self.stm.get_all_items()
            
            for memory_id, memory_item in stm_memories.items():
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
                            self.stm.remove_item(memory_id)
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
    
    def dream_state_consolidation(self, duration_minutes: int = 10) -> Dict[str, Any]:
        """
        Legacy dream-state consolidation mode (basic version)
        
        Note: This is kept for backward compatibility. 
        For advanced dream processing, use the DreamProcessor class.
        
        Args:
            duration_minutes: Duration of dream processing
        
        Returns:
            Dream state processing results
        """
        logger.info(f"Starting basic dream-state consolidation for {duration_minutes} minutes")
        
        # Basic consolidation with enhanced criteria
        stats = self.consolidate_memories(force=True)
          # Note: Memory reinforcement would require additional methods in LTM classes
        stats['reinforced_memories'] = 0
        stats['dream_duration_minutes'] = duration_minutes
        
        return stats
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive memory system status"""
        return {
            'stm': self.stm.get_status(),
            'ltm': self.ltm.get_status(),
            'last_consolidation': self.last_consolidation,
            'session_memories_count': len(self.session_memories),
            'consolidation_interval': self.consolidation_interval,
            'use_vector_stm': self.use_vector_stm,
            'use_vector_ltm': self.use_vector_ltm,
            'system_active': True
        }
    
    def reset_session(self) -> None:
        """Reset session-specific memory tracking"""
        self.session_memories = []
        logger.info("Memory system session reset")
    
    def _should_consolidate(self) -> bool:
        """Check if automatic consolidation should occur"""
        time_elapsed = (datetime.now() - self.last_consolidation).total_seconds()
        return time_elapsed >= self.consolidation_interval
    
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
        if self.use_vector_stm and isinstance(self.stm, VectorShortTermMemory):
            # Use vector STM semantic search
            vector_results = self.stm.search_semantic(
                query=query,
                max_results=max_results,
                min_similarity=min_similarity,
                min_activation=min_activation            )
            return [(result.item, result.relevance_score) for result in vector_results]
        else:
            # Fallback to regular STM search
            return self.stm.search(query=query, max_results=max_results, min_activation=min_activation)
    
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
            if self.use_vector_stm and isinstance(self.stm, VectorShortTermMemory):
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
            if self.use_vector_ltm and isinstance(self.ltm, VectorLongTermMemory):
                ltm_results = self.ltm.search_semantic(
                    query=query,
                    max_results=max_ltm_context,
                    min_similarity=min_relevance
                )
                context["ltm"] = [result.record for result in ltm_results]
            else:
                # Fallback for regular LTM
                ltm_results = self.ltm.search_by_content(query=query, max_results=max_ltm_context)
                context["ltm"] = [record for record, score in ltm_results if score >= min_relevance]
            
            logger.debug(f"Retrieved context: {len(context['stm'])} STM + {len(context['ltm'])} LTM items")
            return context
            
        except Exception as e:
            logger.error(f"Error getting context for query: {e}")
            return {"stm": [], "ltm": []}
    
    # Type annotations for memory components
    stm: Union[ShortTermMemory, VectorShortTermMemory]
    ltm: Union[LongTermMemory, VectorLongTermMemory]    # Episodic Memory Methods
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
                summary=summary,
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
    ) -> List[Any]:
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
            return [result.memory for result in results]
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
