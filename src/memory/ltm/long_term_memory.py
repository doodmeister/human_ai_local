"""
Long-Term Memory (LTM) System
Implements persistent memory storage with semantic organization
"""
import time
from typing import Dict, List, Optional, Any, Tuple, Sequence
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from ..base import BaseMemorySystem

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
    consolidated_at: Optional[datetime] = None  # When last consolidated from STM
    consolidation_source: str = "direct"  # "stm", "manual", "direct"
    
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
            "consolidation_count": self.consolidation_count,
            "consolidated_at": self.consolidated_at.isoformat() if self.consolidated_at else None,
            "consolidation_source": self.consolidation_source
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
            consolidation_count=data.get("consolidation_count", 0),
            consolidated_at=datetime.fromisoformat(data["consolidated_at"]) if data.get("consolidated_at") else None,
            consolidation_source=data.get("consolidation_source", "direct")
        )

class LongTermMemory(BaseMemorySystem):
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
        
        # Meta-cognitive tracking
        self._retrieval_stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "total_retrieval_time": 0.0,
            "search_queries": 0,
            "search_results_returned": 0,
            "last_reset": datetime.now()
        }
        
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
    
    def retrieve(self, memory_id: str) -> Optional[dict]:
        """Retrieve memory by ID with performance tracking"""
        start_time = time.time()
        
        self._retrieval_stats["total_retrievals"] += 1
        
        record = self.memories.get(memory_id)
        if record is None:
            self._retrieval_stats["failed_retrievals"] += 1
            retrieval_time = time.time() - start_time
            self._retrieval_stats["total_retrieval_time"] += retrieval_time
            return None
            
        record.update_access()
        self._save_memory(record)
        
        self._retrieval_stats["successful_retrievals"] += 1
        retrieval_time = time.time() - start_time
        self._retrieval_stats["total_retrieval_time"] += retrieval_time
        
        logger.debug(f"Retrieved LTM record {memory_id} in {retrieval_time:.4f}s")
        return record.to_dict()

    def delete(self, memory_id: str) -> bool:
        """Delete memory by ID"""
        if memory_id in self.memories:
            del self.memories[memory_id]
            logger.debug(f"Deleted LTM record {memory_id}")
            return True
        return False

    def search(self, query: Optional[str] = None, **kwargs) -> Sequence[dict | tuple]:
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
        min_relevance = kwargs.get('min_relevance', 0.0)
        max_results = kwargs.get('max_results', 5)
        results = []
        for record, relevance in self.search_by_content(query or '', min_relevance, max_results):
            results.append((record.to_dict(), relevance))
        return results
    
    def search_by_content(
        self,
        query: str,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
        max_results: int = 10
    ) -> List[Tuple[LTMRecord, float]]:
        """
        Search memories by content with performance tracking
        
        Args:
            query: Search query
            memory_types: Filter by memory types
            min_importance: Minimum importance threshold
            max_results: Maximum results to return
        
        Returns:
            List of (LTMRecord, relevance_score) tuples
        """
        start_time = time.time()
        self._retrieval_stats["search_queries"] += 1
        
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
            
            # Recency weighting (exponential decay based on time since last_access)
            now = datetime.now()
            seconds_since_access = (now - record.last_access).total_seconds()
            # Decay half-life: 1 day (86400s) by default
            half_life = 86400
            recency_weight = 0.5 ** (seconds_since_access / half_life)
            relevance *= (1.0 + recency_weight)  # Boost recent memories
            
            if relevance > 0:
                results.append((record, relevance))
        
        # Sort by relevance
        results.sort(key=lambda x: x[1], reverse=True)
        final_results = results[:max_results]
        
        # Update search statistics
        self._retrieval_stats["search_results_returned"] += len(final_results)
        search_time = time.time() - start_time
        logger.debug(f"Search '{query}' returned {len(final_results)} results in {search_time:.4f}s")
        
        return final_results
    
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
        Consolidate items from STM into LTM with emotional weighting
        
        Args:
            stm_items: List of STM memory items
        
        Returns:
            Number of items consolidated
        """
        consolidated = 0
        consolidation_time = datetime.now()
        
        for item in stm_items:
            # Calculate consolidation score with emotional weighting
            consolidation_score = self._calculate_consolidation_score(item)
            
            # Consolidation threshold with emotional consideration
            # Base threshold is lower for emotionally charged memories
            base_threshold = 0.5
            emotional_adjustment = abs(item.emotional_valence) * 0.3  # 0.0 to 0.3 reduction
            threshold = base_threshold - emotional_adjustment
            
            # Determine if item should be consolidated
            should_consolidate = (
                item.importance > 0.6 or  # High importance
                item.access_count > 2 or  # Frequently accessed
                consolidation_score > threshold  # Emotionally significant or high combined score
            )
            
            if should_consolidate:
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
                # Mark consolidation metadata
                if item.id in self.memories:
                    self.memories[item.id].consolidated_at = consolidation_time
                    self.memories[item.id].consolidation_source = "stm"
                    self._save_memory(self.memories[item.id])
                consolidated += 1
        
        logger.info(f"Consolidated {consolidated} items from STM to LTM at {consolidation_time} (emotional weighting enabled)")
        return consolidated
    
    def _calculate_consolidation_score(self, item: Any) -> float:
        """
        Calculate consolidation score considering multiple factors including emotion
        
        Args:
            item: STM memory item
            
        Returns:
            Consolidation score (0.0 to 1.0+)
        """
        # Base score from importance
        score = item.importance
        
        # Access frequency bonus (normalized)
        access_bonus = min(item.access_count / 10.0, 0.3)
        score += access_bonus
        
        # Emotional significance bonus
        # Both positive and negative emotions boost consolidation
        emotional_intensity = abs(item.emotional_valence)
        emotional_bonus = emotional_intensity * 0.4  # Up to 0.4 bonus
        score += emotional_bonus
        
        # Recency bonus (more recent items get slight boost)
        if hasattr(item, 'encoding_time'):
            hours_since_encoding = (datetime.now() - item.encoding_time).total_seconds() / 3600
            recency_bonus = max(0, 0.1 * (1 - hours_since_encoding / 24))  # Decays over 24 hours
            score += recency_bonus
        
        return min(score, 1.0)  # Cap at 1.0
    
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
    
    def get_recently_consolidated(self, hours: int = 24, consolidation_source: Optional[str] = None) -> List[LTMRecord]:
        """
        Get memories that were consolidated recently
        
        Args:
            hours: Hours back to search (default 24)
            consolidation_source: Filter by consolidation source ("stm", "manual", "direct")
        
        Returns:
            List of recently consolidated LTMRecord objects
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        results = []
        
        for record in self.memories.values():
            if record.consolidated_at and record.consolidated_at >= cutoff_time:
                if consolidation_source is None or record.consolidation_source == consolidation_source:
                    results.append(record)
        
        # Sort by consolidation time (most recent first)
        results.sort(key=lambda x: x.consolidated_at, reverse=True)
        return results
    
    def get_consolidation_stats(self) -> Dict[str, Any]:
        """
        Get consolidation statistics
        
        Returns:
            Dictionary with consolidation stats
        """
        stats = {
            "total_consolidated": 0,
            "consolidation_sources": {},
            "recent_24h": 0,
            "recent_7d": 0,
            "avg_consolidation_count": 0.0
        }
        
        now = datetime.now()
        cutoff_24h = now - timedelta(hours=24)
        cutoff_7d = now - timedelta(days=7)
        
        consolidation_counts = []
        
        for record in self.memories.values():
            if record.consolidated_at:
                stats["total_consolidated"] += 1
                
                # Count by source
                source = record.consolidation_source
                stats["consolidation_sources"][source] = stats["consolidation_sources"].get(source, 0) + 1
                
                # Count recent consolidations
                if record.consolidated_at >= cutoff_24h:
                    stats["recent_24h"] += 1
                if record.consolidated_at >= cutoff_7d:
                    stats["recent_7d"] += 1
            
            consolidation_counts.append(record.consolidation_count)
        
        if consolidation_counts:
            stats["avg_consolidation_count"] = sum(consolidation_counts) / len(consolidation_counts)
        
        return stats
    
    def get_metacognitive_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive meta-cognitive statistics for self-evaluation
        
        Returns:
            Dictionary with performance metrics and cognitive insights
        """
        stats = self._retrieval_stats.copy()
        
        # Calculate derived metrics
        if stats["total_retrievals"] > 0:
            stats["retrieval_success_rate"] = stats["successful_retrievals"] / stats["total_retrievals"]
            stats["retrieval_failure_rate"] = stats["failed_retrievals"] / stats["total_retrievals"]
            stats["avg_retrieval_time"] = stats["total_retrieval_time"] / stats["total_retrievals"]
        else:
            stats["retrieval_success_rate"] = 0.0
            stats["retrieval_failure_rate"] = 0.0
            stats["avg_retrieval_time"] = 0.0
        
        if stats["search_queries"] > 0:
            stats["avg_results_per_search"] = stats["search_results_returned"] / stats["search_queries"]
        else:
            stats["avg_results_per_search"] = 0.0
        
        # Memory health metrics
        total_memories = len(self.memories)
        if total_memories > 0:
            avg_access_count = sum(r.access_count for r in self.memories.values()) / total_memories
            avg_importance = sum(r.importance for r in self.memories.values()) / total_memories
            avg_confidence = sum(r.confidence for r in self.memories.values()) / total_memories
            
            # Memory freshness (average days since last access)
            now = datetime.now()
            total_days = sum((now - r.last_access).days for r in self.memories.values())
            avg_days_since_access = total_days / total_memories
            
            stats.update({
                "total_memories": total_memories,
                "avg_access_count": avg_access_count,
                "avg_importance": avg_importance,
                "avg_confidence": avg_confidence,
                "avg_days_since_access": avg_days_since_access,
                "memory_utilization": min(1.0, avg_access_count / 10.0),  # Normalized 0-1
            })
        else:
            stats.update({
                "total_memories": 0,
                "avg_access_count": 0.0,
                "avg_importance": 0.0,
                "avg_confidence": 0.0,
                "avg_days_since_access": 0.0,
                "memory_utilization": 0.0,
            })
        
        # Time since last reset
        time_since_reset = datetime.now() - stats["last_reset"]
        stats["hours_since_reset"] = time_since_reset.total_seconds() / 3600
        
        return stats
    
    def get_memory_health_report(self) -> Dict[str, Any]:
        """
        Generate a memory health report for debugging cognitive issues
        
        Returns:
            Dictionary with diagnostic information
        """
        now = datetime.now()
        
        # Categorize memories by access patterns
        never_accessed = []
        rarely_accessed = []  # < 3 times
        frequently_accessed = []  # >= 10 times
        stale_memories = []  # not accessed in 30+ days
        low_confidence = []  # confidence < 0.3
        
        for record in self.memories.values():
            if record.access_count == 0:
                never_accessed.append(record.id)
            elif record.access_count < 3:
                rarely_accessed.append(record.id)
            elif record.access_count >= 10:
                frequently_accessed.append(record.id)
            
            days_since_access = (now - record.last_access).days
            if days_since_access >= 30:
                stale_memories.append(record.id)
            
            if record.confidence < 0.3:
                low_confidence.append(record.id)
        
        # Memory type distribution
        type_distribution = {}
        for record in self.memories.values():
            type_distribution[record.memory_type] = type_distribution.get(record.memory_type, 0) + 1
        
        return {
            "memory_categories": {
                "never_accessed": len(never_accessed),
                "rarely_accessed": len(rarely_accessed),
                "frequently_accessed": len(frequently_accessed),
                "stale_memories": len(stale_memories),
                "low_confidence": len(low_confidence)
            },
            "memory_type_distribution": type_distribution,
            "potential_issues": {
                "high_stale_ratio": len(stale_memories) / max(1, len(self.memories)) > 0.5,
                "low_utilization": sum(r.access_count for r in self.memories.values()) / max(1, len(self.memories)) < 2,
                "confidence_degradation": len(low_confidence) / max(1, len(self.memories)) > 0.3
            },
            "recommendations": self._generate_recommendations(never_accessed, stale_memories, low_confidence)
        }
    
    def _generate_recommendations(self, never_accessed: List[str], stale_memories: List[str], low_confidence: List[str]) -> List[str]:
        """Generate recommendations based on memory health analysis"""
        recommendations = []
        
        # Get calculated stats first
        stats = self.get_metacognitive_stats()
        
        if len(never_accessed) > len(self.memories) * 0.3:
            recommendations.append("Consider reviewing memory storage criteria - many memories are never accessed")
        
        if len(stale_memories) > len(self.memories) * 0.4:
            recommendations.append("Run memory decay to reduce importance of stale memories")
        
        if len(low_confidence) > len(self.memories) * 0.2:
            recommendations.append("Review and reinforce low-confidence memories or consider removal")
        
        if stats["retrieval_failure_rate"] > 0.1:
            recommendations.append("High retrieval failure rate - check memory ID generation and storage")
        
        if stats["avg_results_per_search"] < 1.0:
            recommendations.append("Low search effectiveness - consider improving content indexing")
        
        return recommendations
    
    def reset_metacognitive_stats(self):
        """Reset meta-cognitive tracking statistics"""
        self._retrieval_stats = {
            "total_retrievals": 0,
            "successful_retrievals": 0,
            "failed_retrievals": 0,
            "total_retrieval_time": 0.0,
            "search_queries": 0,
            "search_results_returned": 0,
            "last_reset": datetime.now()
        }
        logger.info("Reset meta-cognitive statistics")
    
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
    
    def decay_memories(self, decay_rate: float = 0.01, half_life_days: float = 30.0, min_importance: float = 0.05, min_confidence: float = 0.1) -> int:
        """
        Decay importance and confidence for old, rarely accessed memories.
        Args:
            decay_rate: Base decay rate per call (default 1%)
            half_life_days: Half-life in days for exponential decay (default 30 days)
            min_importance: Minimum importance value
            min_confidence: Minimum confidence value
        Returns:
            Number of decayed memories
        """
        now = datetime.now()
        decayed = 0
        half_life_seconds = half_life_days * 86400
        for record in self.memories.values():
            seconds_since_access = (now - record.last_access).total_seconds()
            # Exponential decay factor based on time since last access
            decay_factor = 0.5 ** (seconds_since_access / half_life_seconds)
            # Only decay if not accessed recently (e.g., >1 day)
            if seconds_since_access > 86400:
                old_importance = record.importance
                old_confidence = record.confidence
                record.importance = max(min_importance, record.importance * (1 - decay_rate) * decay_factor)
                record.confidence = max(min_confidence, record.confidence * (1 - decay_rate/2) * decay_factor)
                if record.importance < old_importance or record.confidence < old_confidence:
                    self._save_memory(record)
                    decayed += 1
        logger.info(f"Decayed {decayed} LTM memories (rate={decay_rate}, half_life_days={half_life_days})")
        return decayed

    def find_cross_system_links(self, external_memory_id: str, system_type: str = "any") -> List[LTMRecord]:
        """
        Find LTM records linked to memories from other systems (STM, episodic)
        
        Args:
            external_memory_id: Memory ID from another system
            system_type: Type of system ("stm", "episodic", "any")
            
        Returns:
            List of linked LTMRecord objects
        """
        linked_records = []
        
        for record in self.memories.values():
            # Check associations list
            if external_memory_id in record.associations:
                linked_records.append(record)
                continue
            
            # Check content for embedded references (common pattern)
            content_str = str(record.content).lower()
            if external_memory_id.lower() in content_str:
                linked_records.append(record)
                continue
            
            # Check source field for consolidation links
            if (record.source == "stm_consolidation" and 
                system_type in ["stm", "any"] and 
                record.id == external_memory_id):
                linked_records.append(record)
        
        return linked_records
    
    def create_cross_system_link(self, ltm_memory_id: str, external_memory_id: str, link_type: str = "association") -> bool:
        """
        Create a link between an LTM memory and external system memory
        
        Args:
            ltm_memory_id: ID of LTM memory
            external_memory_id: ID of memory in another system
            link_type: Type of link ("association", "derivation", "reference")
            
        Returns:
            True if link created successfully
        """
        if ltm_memory_id not in self.memories:
            return False
        
        record = self.memories[ltm_memory_id]
        
        # Add to associations if not already present
        if external_memory_id not in record.associations:
            record.associations.append(external_memory_id)
            
            # Update associations index
            if ltm_memory_id not in self.associations_index:
                self.associations_index[ltm_memory_id] = []
            if external_memory_id not in self.associations_index[ltm_memory_id]:
                self.associations_index[ltm_memory_id].append(external_memory_id)
            
            # Save updated record
            self._save_memory(record)
            
            logger.debug(f"Created {link_type} link: LTM {ltm_memory_id} -> {external_memory_id}")
            return True
        
        return False
    
    def get_semantic_clusters(self, min_cluster_size: int = 2) -> Dict[str, List[str]]:
        """
        Identify semantic clusters across memory systems for cross-episode semanticization
        
        Args:
            min_cluster_size: Minimum number of memories to form a cluster
            
        Returns:
            Dictionary mapping cluster themes to memory IDs
        """
        # Group memories by common tags and content themes
        tag_clusters = {}
        content_clusters = {}
        
        # Cluster by tags
        for record in self.memories.values():
            for tag in record.tags:
                if tag not in tag_clusters:
                    tag_clusters[tag] = []
                tag_clusters[tag].append(record.id)
        
        # Cluster by content keywords (simple approach)
        for record in self.memories.values():
            content_words = str(record.content).lower().split()
            # Focus on meaningful words (length > 3)
            keywords = [word for word in content_words if len(word) > 3]
            
            for keyword in keywords:
                if keyword not in content_clusters:
                    content_clusters[keyword] = []
                if record.id not in content_clusters[keyword]:
                    content_clusters[keyword].append(record.id)
        
        # Filter clusters by minimum size
        significant_clusters = {}
        
        for tag, memory_ids in tag_clusters.items():
            if len(memory_ids) >= min_cluster_size:
                significant_clusters[f"tag:{tag}"] = memory_ids
        
        for keyword, memory_ids in content_clusters.items():
            if len(memory_ids) >= min_cluster_size:
                significant_clusters[f"content:{keyword}"] = memory_ids
        
        return significant_clusters
    
    def suggest_cross_system_associations(self, external_memories: List[Dict[str, Any]], system_type: str) -> List[Dict[str, Any]]:
        """
        Suggest potential associations between LTM and external system memories
        
        Args:
            external_memories: List of memory dictionaries from another system
            system_type: Type of external system ("stm", "episodic")
            
        Returns:
            List of suggested associations with confidence scores
        """
        suggestions = []
        
        for ext_memory in external_memories:
            ext_id = ext_memory.get("id", "")
            ext_content = str(ext_memory.get("content", "")).lower()
            ext_tags = ext_memory.get("tags", [])
            
            for ltm_record in self.memories.values():
                ltm_content = str(ltm_record.content).lower()
                
                # Calculate similarity score
                similarity_score = 0.0
                
                # Content word overlap
                ext_words = set(ext_content.split())
                ltm_words = set(ltm_content.split())
                word_overlap = 0  # Ensure always defined
                if ext_words and ltm_words:
                    word_overlap = len(ext_words.intersection(ltm_words))
                    similarity_score += word_overlap / max(len(ext_words), len(ltm_words))
                
                # Tag overlap
                tag_overlap = len(set(ext_tags).intersection(set(ltm_record.tags)))
                if tag_overlap > 0:
                    similarity_score += tag_overlap * 0.3
                
                # Emotional valence similarity (if available)
                valence_similarity = 0  # Ensure always defined
                if "emotional_valence" in ext_memory:
                    valence_diff = abs(ext_memory["emotional_valence"] - ltm_record.emotional_valence)
                    valence_similarity = 1 - valence_diff  # Closer valences = higher similarity
                    similarity_score += valence_similarity * 0.2
                
                # Only suggest if similarity is significant
                if similarity_score > 0.3:
                    suggestions.append({
                        "ltm_memory_id": ltm_record.id,
                        "external_memory_id": ext_id,
                        "system_type": system_type,
                        "confidence": min(similarity_score, 1.0),
                        "similarity_factors": {
                            "content_overlap": word_overlap,
                            "tag_overlap": tag_overlap,
                            "emotional_similarity": valence_similarity
                        }
                    })
        
        # Sort by confidence score
        suggestions.sort(key=lambda x: x["confidence"], reverse=True)
        return suggestions[:20]  # Return top 20 suggestions
