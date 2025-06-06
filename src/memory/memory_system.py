"""
Integrated Memory System
Coordinates between STM, LTM, and other memory components
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import logging
from .stm import ShortTermMemory
from .ltm import LongTermMemory, VectorLongTermMemory

logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Central memory management system integrating STM and LTM.
    
    Features:
    - Automatic STM to LTM consolidation
    - Cross-system memory retrieval
    - Memory reinforcement and forgetting
    - Dream-state consolidation
    """
    
    def __init__(
        self,
        stm_capacity: int = 7,
        stm_decay_threshold: float = 0.1,
        ltm_storage_path: Optional[str] = None,
        consolidation_interval: int = 300,  # 5 minutes
        use_vector_ltm: bool = True,
        chroma_persist_dir: Optional[str] = None,
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """
        Initialize integrated memory system
        
        Args:
            stm_capacity: STM capacity limit
            stm_decay_threshold: STM decay threshold
            ltm_storage_path: LTM storage path
            consolidation_interval: Auto-consolidation interval in seconds
            use_vector_ltm: Whether to use vector-based LTM with ChromaDB
            chroma_persist_dir: ChromaDB persistence directory
            embedding_model: SentenceTransformer model name
        """
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
        
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = datetime.now()
        self.session_memories = []  # Track memories for this session
        
        logger.info(f"Integrated memory system initialized (Vector LTM: {use_vector_ltm})")
    
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
