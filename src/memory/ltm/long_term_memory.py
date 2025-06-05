"""
Long-Term Memory (LTM) System
Implements persistent memory storage with semantic organization
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class LTMRecord:
    """Long-term memory record with rich metadata"""
    id: str
    content: Any
    memory_type: str  # "episodic", "semantic", "procedural"
    encoding_time: datetime
    last_access: datetime
    access_count: int = 0
    importance: float = 0.5
    emotional_valence: float = 0.0
    confidence: float = 1.0
    source: str = "unknown"
    tags: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)
    consolidation_count: int = 0  # Number of times reinforced
    
    def update_access(self):
        """Update access statistics"""
        self.last_access = datetime.now()
        self.access_count += 1
    
    def reinforce(self, strength: float = 0.1):
        """Reinforce memory through repetition"""
        self.consolidation_count += 1
        self.importance = min(1.0, self.importance + strength)
        self.confidence = min(1.0, self.confidence + strength * 0.5)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type,
            "encoding_time": self.encoding_time.isoformat(),
            "last_access": self.last_access.isoformat(),
            "access_count": self.access_count,
            "importance": self.importance,
            "emotional_valence": self.emotional_valence,
            "confidence": self.confidence,
            "source": self.source,
            "tags": self.tags,
            "associations": self.associations,
            "consolidation_count": self.consolidation_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LTMRecord":
        """Create LTMRecord from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=data["memory_type"],
            encoding_time=datetime.fromisoformat(data["encoding_time"]),
            last_access=datetime.fromisoformat(data["last_access"]),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
            confidence=data.get("confidence", 1.0),
            source=data.get("source", "unknown"),
            tags=data.get("tags", []),
            associations=data.get("associations", []),
            consolidation_count=data.get("consolidation_count", 0)
        )

class LongTermMemory:
    """
    Long-Term Memory system with persistent storage and semantic organization
    
    Features:
    - Persistent storage to disk
    - Memory types: episodic, semantic, procedural
    - Tag-based organization
    - Associative networks
    - Importance-based retention
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize LTM system
        
        Args:
            storage_path: Path for persistent storage (defaults to data/memory_stores/ltm)
        """
        self.storage_path = Path(storage_path or "data/memory_stores/ltm")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.memories: Dict[str, LTMRecord] = {}
        self.tags_index: Dict[str, List[str]] = {}  # tag -> [memory_ids]
        self.associations_index: Dict[str, List[str]] = {}  # memory_id -> [associated_ids]
        
        # Load existing memories
        self._load_from_disk()
        
        logger.info(f"LTM initialized with {len(self.memories)} memories at {self.storage_path}")
    
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
        Store memory in LTM
        
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
        # Update existing or create new
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
        
        # Update indices
        self._update_indices(record)
        
        # Persist to disk
        self._save_memory(record)
        
        logger.debug(f"Stored LTM record {memory_id} (type: {memory_type})")
        return True
    
    def retrieve(self, memory_id: str) -> Optional[LTMRecord]:
        """Retrieve memory by ID"""
        if memory_id not in self.memories:
            return None
        
        record = self.memories[memory_id]
        record.update_access()
        self._save_memory(record)  # Update access statistics on disk
        
        logger.debug(f"Retrieved LTM record {memory_id}")
        return record
    
    def search_by_content(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        max_results: int = 10
    ) -> List[Tuple[LTMRecord, float]]:
        """
        Search memories by content
        
        Args:
            query: Search query
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold
            max_results: Maximum results to return
        
        Returns:
            List of (LTMRecord, relevance_score) tuples
        """
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for record in self.memories.values():
            # Filter by type and importance
            if memory_types and record.memory_type not in memory_types:
                continue
            if record.importance < min_importance:
                continue
            
            # Calculate relevance score
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
                results.append((record, relevance))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]
    
    def search_by_tags(self, tags: List[str], operator: str = "OR") -> List[LTMRecord]:
        """
        Search memories by tags
        
        Args:
            tags: Tags to search for
            operator: "OR" (any tag) or "AND" (all tags)
        
        Returns:
            List of matching LTMRecord objects
        """
        if operator == "OR":
            memory_ids = set()
            for tag in tags:
                if tag in self.tags_index:
                    memory_ids.update(self.tags_index[tag])
        else:  # AND
            memory_ids = None
            for tag in tags:
                if tag in self.tags_index:
                    tag_memories = set(self.tags_index[tag])
                    if memory_ids is None:
                        memory_ids = tag_memories
                    else:
                        memory_ids = memory_ids.intersection(tag_memories)
                else:
                    return []  # Tag not found, no matches possible
            
            if memory_ids is None:
                memory_ids = set()
        
        return [self.memories[mid] for mid in memory_ids if mid in self.memories]
    
    def get_associations(self, memory_id: str, depth: int = 1) -> List[LTMRecord]:
        """
        Get associated memories
        
        Args:
            memory_id: Starting memory ID
            depth: Depth of association traversal
        
        Returns:
            List of associated LTMRecord objects
        """
        if memory_id not in self.memories:
            return []
        
        visited = set()
        to_visit = [(memory_id, 0)]
        associated = []
        
        while to_visit:
            current_id, current_depth = to_visit.pop(0)
            
            if current_id in visited or current_depth > depth:
                continue
            
            visited.add(current_id)
            
            if current_depth > 0:  # Don't include the starting memory
                if current_id in self.memories:
                    associated.append(self.memories[current_id])
            
            # Add associated memories to visit queue
            if current_id in self.associations_index:
                for assoc_id in self.associations_index[current_id]:
                    if assoc_id not in visited:
                        to_visit.append((assoc_id, current_depth + 1))
        
        return associated
    
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
                self.store(
                    memory_id=item.id,
                    content=item.content,
                    memory_type="episodic",  # Default for STM->LTM transfer
                    importance=item.importance,
                    emotional_valence=item.emotional_valence,
                    source="stm_consolidation",
                    associations=item.associations
                )
                consolidated += 1
        
        logger.info(f"Consolidated {consolidated} items from STM to LTM")
        return consolidated
    
    def get_status(self) -> Dict[str, Any]:
        """Get LTM status information"""
        if not self.memories:
            return {
                "total_memories": 0,
                "memory_types": {},
                "avg_importance": 0.0,
                "total_tags": 0,
                "storage_path": str(self.storage_path)
            }
        
        memory_types = {}
        for record in self.memories.values():
            memory_types[record.memory_type] = memory_types.get(record.memory_type, 0) + 1
        
        avg_importance = sum(r.importance for r in self.memories.values()) / len(self.memories)
        
        return {
            "total_memories": len(self.memories),
            "memory_types": memory_types,
            "avg_importance": avg_importance,
            "total_tags": len(self.tags_index),
            "storage_path": str(self.storage_path)
        }
    
    def _update_indices(self, record: LTMRecord):
        """Update search indices for a memory record"""
        # Update tags index
        for tag in record.tags:
            if tag not in self.tags_index:
                self.tags_index[tag] = []
            if record.id not in self.tags_index[tag]:
                self.tags_index[tag].append(record.id)
        
        # Update associations index
        self.associations_index[record.id] = record.associations[:]
    
    def _save_memory(self, record: LTMRecord):
        """Save individual memory record to disk"""
        memory_file = self.storage_path / f"{record.id}.json"
        try:
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(record.to_dict(), f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save memory {record.id}: {e}")
    
    def _load_from_disk(self):
        """Load all memories from disk storage"""
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
        
        logger.info(f"Loaded {loaded} memories from disk")
    
    def save_all(self):
        """Save all memories to disk"""
        for record in self.memories.values():
            self._save_memory(record)
        
        logger.info(f"Saved {len(self.memories)} memories to disk")
