"""
Short-Term Memory (STM) System
Implements working memory with capacity limits and decay mechanisms
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import heapq
import logging

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Individual memory item in STM"""
    id: str
    content: Any
    encoding_time: datetime
    last_access: datetime
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    attention_score: float = 0.0
    emotional_valence: float = 0.0  # -1.0 to 1.0
    decay_rate: float = 0.1
    associations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived properties after initialization"""
        self.age_seconds = (datetime.now() - self.encoding_time).total_seconds()
        self.recency_score = max(0.0, 1.0 - (self.age_seconds / 3600))  # Decay over 1 hour
    
    def update_access(self):
        """Update access statistics when memory is retrieved"""
        self.last_access = datetime.now()
        self.access_count += 1
        # Boost importance slightly with access
        self.importance = min(1.0, self.importance + 0.05)
    
    def calculate_activation(self) -> float:
        """Calculate memory activation level (0.0 to 1.0)"""
        # Recency component
        age_hours = (datetime.now() - self.encoding_time).total_seconds() / 3600
        recency = max(0.0, 1.0 - (age_hours * self.decay_rate))
        
        # Frequency component
        frequency = min(1.0, self.access_count / 10.0)  # Normalize to max 10 accesses
        
        # Importance and attention components
        salience = (self.importance + self.attention_score) / 2.0
        
        # Combine components
        activation = (recency * 0.4) + (frequency * 0.3) + (salience * 0.3)
        return max(0.0, min(1.0, activation))

class ShortTermMemory:
    """
    Short-Term Memory system with capacity limits and decay
    
    Implements:
    - Limited capacity (7±2 items as per Miller's law)
    - Temporal decay of unused items
    - Importance-based retention
    - Associative retrieval
    """
    
    def __init__(self, capacity: int = 7, decay_threshold: float = 0.1):
        """
        Initialize STM system
        
        Args:
            capacity: Maximum number of items to retain
            decay_threshold: Activation threshold below which items are forgotten
        """
        self.capacity = capacity
        self.decay_threshold = decay_threshold
        self.items: Dict[str, MemoryItem] = {}
        self.access_order: List[str] = []  # For LRU eviction
        
        logger.info(f"STM initialized with capacity={capacity}, decay_threshold={decay_threshold}")
    
    def store(
        self,
        memory_id: str,
        content: Any,
        importance: float = 0.5,
        attention_score: float = 0.0,
        emotional_valence: float = 0.0,
        associations: Optional[List[str]] = None
    ) -> bool:
        """
        Store new item in STM
        
        Args:
            memory_id: Unique identifier for the memory
            content: Memory content (text, dict, etc.)
            importance: Importance score (0.0 to 1.0)
            attention_score: Attention paid during encoding
            emotional_valence: Emotional weight (-1.0 to 1.0)
            associations: List of associated memory IDs
        
        Returns:
            True if stored successfully, False if rejected
        """
        # Check if already exists
        if memory_id in self.items:
            # Update existing item
            item = self.items[memory_id]
            item.content = content
            item.importance = max(item.importance, importance)
            item.update_access()
            self._update_access_order(memory_id)
            return True
        
        # Create new memory item
        new_item = MemoryItem(
            id=memory_id,
            content=content,
            encoding_time=datetime.now(),
            last_access=datetime.now(),
            importance=importance,
            attention_score=attention_score,
            emotional_valence=emotional_valence,
            associations=associations or []
        )
        
        # Check capacity and evict if necessary
        if len(self.items) >= self.capacity:
            if not self._evict_least_important():
                # Could not evict anything, reject new item if it's not important enough
                min_existing_importance = min(item.importance for item in self.items.values())
                if importance <= min_existing_importance:
                    logger.debug(f"Rejected memory {memory_id} - insufficient importance")
                    return False
        
        # Store the new item
        self.items[memory_id] = new_item
        self.access_order.append(memory_id)
        
        logger.debug(f"Stored memory {memory_id} in STM (size: {len(self.items)})")
        return True
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve memory by ID
        
        Args:
            memory_id: Memory identifier
        
        Returns:
            MemoryItem if found, None otherwise
        """
        if memory_id not in self.items:
            return None
        
        item = self.items[memory_id]
        item.update_access()
        self._update_access_order(memory_id)
        
        logger.debug(f"Retrieved memory {memory_id} from STM")
        return item
    
    def search(
        self,
        query: str = "",
        min_activation: float = 0.0,
        max_results: int = 5,
        search_associations: bool = True
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search STM for relevant memories
        
        Args:
            query: Search query (basic string matching for now)
            min_activation: Minimum activation threshold
            max_results: Maximum number of results
            search_associations: Whether to include associatively linked memories
        
        Returns:
            List of (MemoryItem, relevance_score) tuples
        """
        results = []
        
        for item in self.items.values():
            activation = item.calculate_activation()
            
            if activation < min_activation:
                continue
            
            # Simple relevance calculation (can be enhanced with embeddings)
            relevance = activation
            
            if query:
                # Basic string matching (placeholder for semantic search)
                content_str = str(item.content).lower()
                query_lower = query.lower()
                
                if query_lower in content_str:
                    relevance += 0.3
                
                # Check for word overlap
                query_words = set(query_lower.split())
                content_words = set(content_str.split())
                overlap = len(query_words.intersection(content_words))
                if overlap > 0:
                    relevance += 0.1 * overlap
            
            if relevance > 0:
                results.append((item, relevance))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def decay_memories(self) -> List[str]:
        """
        Apply decay to all memories and remove those below threshold
        
        Returns:
            List of memory IDs that were forgotten
        """
        forgotten_ids = []
        
        for memory_id, item in list(self.items.items()):
            activation = item.calculate_activation()
            
            if activation < self.decay_threshold:
                forgotten_ids.append(memory_id)
                del self.items[memory_id]
                if memory_id in self.access_order:
                    self.access_order.remove(memory_id)
        
        if forgotten_ids:
            logger.debug(f"Forgot {len(forgotten_ids)} memories due to decay: {forgotten_ids}")
        
        return forgotten_ids
    
    def get_status(self) -> Dict[str, Any]:
        """Get current STM status"""
        if not self.items:
            return {
                "size": 0,
                "capacity": self.capacity,
                "utilization": 0.0,
                "avg_activation": 0.0,
                "oldest_memory": None,
                "newest_memory": None
            }
        
        activations = [item.calculate_activation() for item in self.items.values()]
        times = [item.encoding_time for item in self.items.values()]
        
        return {
            "size": len(self.items),
            "capacity": self.capacity,
            "utilization": len(self.items) / self.capacity,
            "avg_activation": sum(activations) / len(activations),
            "oldest_memory": min(times),
            "newest_memory": max(times)
        }
    
    def get_all_items(self) -> Dict[str, MemoryItem]:
        """
        Get all memory items in STM
        
        Returns:
            Dictionary of memory_id -> MemoryItem
        """
        return self.items.copy()
    
    def _evict_least_important(self) -> bool:
        """
        Evict the least important memory item
        
        Returns:
            True if an item was evicted, False if all items are too important
        """
        if not self.items:
            return False
        
        # Find item with lowest combined score (activation + importance)
        min_score = float('inf')
        evict_id = None
        
        for memory_id, item in self.items.items():
            score = item.calculate_activation() + item.importance
            if score < min_score:
                min_score = score
                evict_id = memory_id
        
        if evict_id and min_score < 1.0:  # Don't evict highly important items
            del self.items[evict_id]
            if evict_id in self.access_order:
                self.access_order.remove(evict_id)
            logger.debug(f"Evicted memory {evict_id} from STM (score: {min_score:.3f})")
            return True
        
        return False
    
    def _update_access_order(self, memory_id: str):
        """Update the access order for LRU tracking"""
        if memory_id in self.access_order:
            self.access_order.remove(memory_id)
        self.access_order.append(memory_id)
    
    def remove_item(self, memory_id: str) -> bool:
        """
        Remove a specific memory item from STM
        
        Args:
            memory_id: ID of the memory to remove
            
        Returns:
            True if item was removed, False if not found
        """
        if memory_id not in self.items:
            return False
        
        del self.items[memory_id]
        if memory_id in self.access_order:
            self.access_order.remove(memory_id)
        
        logger.debug(f"Removed memory {memory_id} from STM")
        return True

    def clear(self):
        """Clear all memories from STM"""
        self.items.clear()
        self.access_order.clear()
        logger.info("STM cleared")
