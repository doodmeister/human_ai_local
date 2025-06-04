"""
Integrated Memory System
Coordinates between STM, LTM, and other memory components
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
from .stm import ShortTermMemory, MemoryItem
from .ltm import LongTermMemory, LTMRecord

logger = logging.getLogger(__name__)

class MemorySystem:
    """
    Integrated memory system coordinating STM, LTM, and consolidation processes
    
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
        consolidation_interval: int = 300  # 5 minutes
    ):
        """
        Initialize integrated memory system
        
        Args:
            stm_capacity: STM capacity limit
            stm_decay_threshold: STM decay threshold
            ltm_storage_path: LTM storage path
            consolidation_interval: Auto-consolidation interval in seconds
        """
        self.stm = ShortTermMemory(capacity=stm_capacity, decay_threshold=stm_decay_threshold)
        self.ltm = LongTermMemory(storage_path=ltm_storage_path)
        
        self.consolidation_interval = consolidation_interval
        self.last_consolidation = datetime.now()
        self.session_memories = []  # Track memories for this session
        
        logger.info("Integrated memory system initialized")
    
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
            "id": memory_id,
            "content": content,
            "timestamp": datetime.now(),
            "importance": importance,
            "stored_in": []
        }
        
        success = True
        
        # Always try STM first (working memory)
        if self.stm.store(
            memory_id=memory_id,
            content=content,
            importance=importance,
            attention_score=attention_score,
            emotional_valence=emotional_valence,
            associations=associations
        ):
            session_entry["stored_in"].append("stm")
            logger.debug(f"Stored {memory_id} in STM")
        
        # Store in LTM if important enough or forced
        if force_ltm or importance > 0.7 or attention_score > 0.8:
            if self.ltm.store(
                memory_id=memory_id,
                content=content,
                memory_type=memory_type,
                importance=importance,
                emotional_valence=emotional_valence,
                tags=tags,
                associations=associations
            ):
                session_entry["stored_in"].append("ltm")
                logger.debug(f"Stored {memory_id} in LTM")
        
        self.session_memories.append(session_entry)
        
        # Check for auto-consolidation
        self._check_auto_consolidation()
        
        return success
    
    def retrieve_memory(self, memory_id: str) -> Optional[Any]:
        """
        Retrieve memory from any system
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            Memory content if found, None otherwise
        """
        # Try STM first (faster access)
        stm_item = self.stm.retrieve(memory_id)
        if stm_item:
            logger.debug(f"Retrieved {memory_id} from STM")
            return stm_item
        
        # Try LTM
        ltm_record = self.ltm.retrieve(memory_id)
        if ltm_record:
            logger.debug(f"Retrieved {memory_id} from LTM")
            
            # Optionally promote frequently accessed LTM items back to STM
            if ltm_record.access_count > 3:
                self.stm.store(
                    memory_id=ltm_record.id,
                    content=ltm_record.content,
                    importance=ltm_record.importance,
                    emotional_valence=ltm_record.emotional_valence
                )
                logger.debug(f"Promoted {memory_id} from LTM to STM")
            
            return ltm_record
        
        logger.debug(f"Memory {memory_id} not found in any system")
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
        
        # Search STM
        if search_stm:
            stm_results = self.stm.search(query=query, max_results=max_results)
            for item, relevance in stm_results:
                results.append((item, relevance, "stm"))
        
        # Search LTM
        if search_ltm:
            ltm_results = self.ltm.search_by_content(
                query=query,
                memory_types=memory_types,
                max_results=max_results
            )
            for record, relevance in ltm_results:
                results.append((record, relevance, "ltm"))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def consolidate_memories(self, force: bool = False) -> Dict[str, int]:
        """
        Consolidate memories from STM to LTM
        
        Args:
            force: Force consolidation regardless of timing
        
        Returns:
            Dict with consolidation statistics
        """
        if not force:
            time_since_last = (datetime.now() - self.last_consolidation).total_seconds()
            if time_since_last < self.consolidation_interval:
                return {"consolidated": 0, "forgotten": 0, "reason": "too_soon"}
        
        # Apply decay to STM
        forgotten_ids = self.stm.decay_memories()
        
        # Get items for potential consolidation
        consolidation_candidates = []
        for item in self.stm.items.values():
            # Criteria for consolidation: high importance, multiple accesses, or emotional salience
            if (item.importance > 0.6 or 
                item.access_count > 2 or 
                abs(item.emotional_valence) > 0.5):
                consolidation_candidates.append(item)
        
        # Consolidate to LTM
        consolidated = self.ltm.consolidate_from_stm(consolidation_candidates)
        
        self.last_consolidation = datetime.now()
        
        logger.info(f"Memory consolidation: {consolidated} items to LTM, {len(forgotten_ids)} forgotten")
        
        return {
            "consolidated": consolidated,
            "forgotten": len(forgotten_ids),
            "forgotten_ids": forgotten_ids,
            "timestamp": self.last_consolidation
        }
    
    def enter_dream_state(self, duration_minutes: int = 5) -> Dict[str, Any]:
        """
        Enter dream-state consolidation mode
        
        Args:
            duration_minutes: Duration of dream processing
        
        Returns:
            Dream state processing results
        """
        logger.info(f"Entering dream state for {duration_minutes} minutes")
        
        # Perform intensive consolidation
        consolidation_results = self.consolidate_memories(force=True)
        
        # Strengthen important memories through replay
        important_memories = []
        for item in self.stm.items.values():
            if item.importance > 0.7:
                important_memories.append(item)
        
        # Simulate memory replay and strengthening
        replayed = 0
        for item in important_memories:
            item.importance = min(1.0, item.importance + 0.1)
            replayed += 1
          # Create associative links between related memories
        associations_created = self._create_dream_associations()
        
        results = {
            "duration_minutes": duration_minutes,
            "consolidation": consolidation_results,
            "memories_replayed": replayed,
            "associations_created": associations_created,
            "dream_timestamp": datetime.now()
        }
        
        logger.info(f"Dream state completed: {results}")
        return results
    
    def get_memory_status(self) -> Dict[str, Any]:
        """Get comprehensive memory system status"""
        stm_status = self.stm.get_status()
        ltm_status = self.ltm.get_status()
        
        return {
            "stm": stm_status,
            "ltm": ltm_status,
            "session_memories": len(self.session_memories),
            "last_consolidation": self.last_consolidation,
            "consolidation_interval": self.consolidation_interval,
            "total_capacity": stm_status["capacity"] + ltm_status["total_memories"]
        }
    
    def reset_session(self):
        """Reset session-specific memory tracking"""
        self.session_memories.clear()
        logger.info("Memory session reset")
    
    def _check_auto_consolidation(self):
        """Check if automatic consolidation should occur"""
        time_since_last = (datetime.now() - self.last_consolidation).total_seconds()
        
        if time_since_last >= self.consolidation_interval:
            self.consolidate_memories()
    
    def _create_dream_associations(self) -> int:
        """Create associative links between memories during dream state"""
        associations_created = 0
        
        # Simple implementation: link memories with similar content or timing
        stm_items = list(self.stm.items.values())
        
        for i, item1 in enumerate(stm_items):
            for item2 in stm_items[i+1:]:
                # Check for temporal proximity
                time_diff = abs((item1.encoding_time - item2.encoding_time).total_seconds())
                
                if time_diff < 3600:  # Within 1 hour
                    # Add mutual associations
                    if item2.id not in item1.associations:
                        item1.associations.append(item2.id)
                        associations_created += 1
                    
                    if item1.id not in item2.associations:
                        item2.associations.append(item1.id)
                        associations_created += 1
        
        return associations_created
