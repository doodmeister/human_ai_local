"""
Memory Query Interface - Task 8 (Part 2)

Executes memory queries parsed by MemoryQueryParser against actual memory systems.

Architecture:
    User Query → MemoryQueryParser → MemoryQueryInterface
    → (STM/LTM/Episodic retrieval) → MemoryQueryResponse
    
Usage:
    interface = MemoryQueryInterface(stm, ltm, episodic)
    
    # Execute query
    query_result = parser.parse_query("What do you remember about Python?")
    response = interface.execute_query(query_result)
    
    # Format for chat
    text = interface.format_response(response)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Any, Dict
import logging

from .memory_query_parser import MemoryQueryResult, QueryType, MemorySystem

logger = logging.getLogger(__name__)


@dataclass
class MemoryResult:
    """Single memory retrieval result."""
    memory_id: str
    content: str
    source_system: str  # 'stm', 'ltm', 'episodic'
    relevance: float  # 0-1
    timestamp: Optional[datetime] = None
    memory_type: Optional[str] = None  # 'fact', 'preference', 'event', etc.
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryQueryResponse:
    """Response from memory query execution."""
    results: List[MemoryResult]
    query_type: QueryType
    total_found: int
    systems_queried: List[str]
    execution_time_ms: float
    
    # Summary statistics
    stm_results: int = 0
    ltm_results: int = 0
    episodic_results: int = 0
    
    def is_empty(self) -> bool:
        """Check if no results found."""
        return len(self.results) == 0


class MemoryQueryInterface:
    """
    Executes memory queries against actual memory systems.
    
    Bridges parsed queries to STM/LTM/Episodic memory retrieval.
    """
    
    def __init__(
        self,
        stm: Optional[Any] = None,
        ltm: Optional[Any] = None,
        episodic: Optional[Any] = None
    ):
        """
        Initialize interface with memory systems.
        
        Args:
            stm: Short-term memory system
            ltm: Long-term memory system
            episodic: Episodic memory system
        """
        self.stm = stm
        self.ltm = ltm
        self.episodic = episodic
    
    def execute_query(self, query: MemoryQueryResult) -> MemoryQueryResponse:
        """
        Execute memory query against appropriate systems.
        
        Args:
            query: Parsed memory query
            
        Returns:
            MemoryQueryResponse with results
        """
        start_time = datetime.now()
        results = []
        systems_queried = []
        
        # Query each target system
        if MemorySystem.STM in query.target_systems and self.stm:
            stm_results = self._query_stm(query)
            results.extend(stm_results)
            systems_queried.append('stm')
        
        if MemorySystem.LTM in query.target_systems and self.ltm:
            ltm_results = self._query_ltm(query)
            results.extend(ltm_results)
            systems_queried.append('ltm')
        
        if MemorySystem.EPISODIC in query.target_systems and self.episodic:
            episodic_results = self._query_episodic(query)
            results.extend(episodic_results)
            systems_queried.append('episodic')
        
        # Sort by relevance
        results.sort(key=lambda r: r.relevance, reverse=True)
        
        # Apply limit
        if query.limit > 0:
            results = results[:query.limit]
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Count results by system
        stm_count = sum(1 for r in results if r.source_system == 'stm')
        ltm_count = sum(1 for r in results if r.source_system == 'ltm')
        episodic_count = sum(1 for r in results if r.source_system == 'episodic')
        
        return MemoryQueryResponse(
            results=results,
            query_type=query.query_type,
            total_found=len(results),
            systems_queried=systems_queried,
            execution_time_ms=execution_time,
            stm_results=stm_count,
            ltm_results=ltm_count,
            episodic_results=episodic_count
        )
    
    def _query_stm(self, query: MemoryQueryResult) -> List[MemoryResult]:
        """
        Query short-term memory.
        
        Args:
            query: Parsed query
            
        Returns:
            List of MemoryResult from STM
        """
        if not self.stm:
            return []
        
        results = []
        
        try:
            # Build search query
            search_text = ' '.join(query.search_terms) if query.search_terms else ''
            
            # For recent memory, get most recent items
            if query.query_type == QueryType.RECENT_MEMORY:
                # Get recent items without semantic search
                if hasattr(self.stm, 'get_all_memories'):
                    memories = self.stm.get_all_memories()
                    # Sort by timestamp, take most recent
                    memories = sorted(memories, key=lambda m: m.get('timestamp', datetime.min), reverse=True)
                    memories = memories[:query.limit]
                    
                    for mem in memories:
                        results.append(MemoryResult(
                            memory_id=mem.get('id', 'stm_unknown'),
                            content=mem.get('content', ''),
                            source_system='stm',
                            relevance=0.9,  # Recent items are highly relevant
                            timestamp=mem.get('timestamp'),
                            metadata=mem
                        ))
            
            # Semantic search
            elif search_text and hasattr(self.stm, 'query'):
                search_results = self.stm.query(
                    query_text=search_text,
                    n_results=query.limit * 2  # Get extras for filtering
                )
                
                if search_results and 'documents' in search_results:
                    documents = search_results.get('documents', [])
                    distances = search_results.get('distances', [])
                    metadatas = search_results.get('metadatas', [])
                    ids = search_results.get('ids', [])
                    
                    # Handle nested lists
                    if documents and isinstance(documents[0], list):
                        documents = documents[0]
                        distances = distances[0] if distances else []
                        metadatas = metadatas[0] if metadatas else []
                        ids = ids[0] if ids else []
                    
                    for i, doc in enumerate(documents):
                        if not doc:
                            continue
                        
                        # Convert distance to relevance (lower distance = higher relevance)
                        distance = distances[i] if i < len(distances) else 1.0
                        relevance = max(0.0, 1.0 - distance)
                        
                        if relevance < query.min_relevance:
                            continue
                        
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        mem_id = ids[i] if i < len(ids) else f'stm_{i}'
                        
                        results.append(MemoryResult(
                            memory_id=mem_id,
                            content=doc,
                            source_system='stm',
                            relevance=relevance,
                            timestamp=metadata.get('timestamp'),
                            memory_type=metadata.get('type'),
                            tags=metadata.get('tags', []),
                            metadata=metadata
                        ))
        
        except Exception as e:
            logger.error(f"Error querying STM: {e}", exc_info=True)
        
        return results
    
    def _query_ltm(self, query: MemoryQueryResult) -> List[MemoryResult]:
        """
        Query long-term memory.
        
        Args:
            query: Parsed query
            
        Returns:
            List of MemoryResult from LTM
        """
        if not self.ltm:
            return []
        
        results = []
        
        try:
            search_text = ' '.join(query.search_terms) if query.search_terms else ''
            
            if search_text and hasattr(self.ltm, 'query'):
                search_results = self.ltm.query(
                    query_text=search_text,
                    n_results=query.limit * 2
                )
                
                if search_results and 'documents' in search_results:
                    documents = search_results.get('documents', [])
                    distances = search_results.get('distances', [])
                    metadatas = search_results.get('metadatas', [])
                    ids = search_results.get('ids', [])
                    
                    # Handle nested lists
                    if documents and isinstance(documents[0], list):
                        documents = documents[0]
                        distances = distances[0] if distances else []
                        metadatas = metadatas[0] if metadatas else []
                        ids = ids[0] if ids else []
                    
                    for i, doc in enumerate(documents):
                        if not doc:
                            continue
                        
                        distance = distances[i] if i < len(distances) else 1.0
                        relevance = max(0.0, 1.0 - distance)
                        
                        if relevance < query.min_relevance:
                            continue
                        
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        mem_id = ids[i] if i < len(ids) else f'ltm_{i}'
                        
                        results.append(MemoryResult(
                            memory_id=mem_id,
                            content=doc,
                            source_system='ltm',
                            relevance=relevance,
                            timestamp=metadata.get('timestamp'),
                            memory_type=metadata.get('type'),
                            tags=metadata.get('tags', []),
                            metadata=metadata
                        ))
        
        except Exception as e:
            logger.error(f"Error querying LTM: {e}", exc_info=True)
        
        return results
    
    def _query_episodic(self, query: MemoryQueryResult) -> List[MemoryResult]:
        """
        Query episodic memory.
        
        Args:
            query: Parsed query
            
        Returns:
            List of MemoryResult from episodic
        """
        if not self.episodic:
            return []
        
        results = []
        
        try:
            search_text = ' '.join(query.search_terms) if query.search_terms else ''
            
            # Handle temporal queries
            if query.temporal_constraint:
                start_time, end_time = query.temporal_constraint.to_datetime_range()
                
                # Try temporal search if available
                if hasattr(self.episodic, 'search_by_timerange'):
                    memories = self.episodic.search_by_timerange(
                        start_time=start_time,
                        end_time=end_time,
                        query_text=search_text if search_text else None,
                        limit=query.limit * 2
                    )
                    
                    for mem in memories:
                        results.append(MemoryResult(
                            memory_id=mem.get('id', 'episodic_unknown'),
                            content=mem.get('summary', '') or mem.get('detailed_content', ''),
                            source_system='episodic',
                            relevance=mem.get('relevance', 0.7),
                            timestamp=mem.get('timestamp'),
                            memory_type='event',
                            tags=mem.get('tags', []),
                            metadata=mem
                        ))
            
            # Semantic search
            elif search_text and hasattr(self.episodic, 'query'):
                search_results = self.episodic.query(
                    query_text=search_text,
                    n_results=query.limit * 2
                )
                
                if search_results and 'documents' in search_results:
                    documents = search_results.get('documents', [])
                    distances = search_results.get('distances', [])
                    metadatas = search_results.get('metadatas', [])
                    ids = search_results.get('ids', [])
                    
                    # Handle nested lists
                    if documents and isinstance(documents[0], list):
                        documents = documents[0]
                        distances = distances[0] if distances else []
                        metadatas = metadatas[0] if metadatas else []
                        ids = ids[0] if ids else []
                    
                    for i, doc in enumerate(documents):
                        if not doc:
                            continue
                        
                        distance = distances[i] if i < len(distances) else 1.0
                        relevance = max(0.0, 1.0 - distance)
                        
                        if relevance < query.min_relevance:
                            continue
                        
                        metadata = metadatas[i] if i < len(metadatas) else {}
                        mem_id = ids[i] if i < len(ids) else f'episodic_{i}'
                        
                        results.append(MemoryResult(
                            memory_id=mem_id,
                            content=doc,
                            source_system='episodic',
                            relevance=relevance,
                            timestamp=metadata.get('timestamp'),
                            memory_type='event',
                            tags=metadata.get('tags', []),
                            metadata=metadata
                        ))
        
        except Exception as e:
            logger.error(f"Error querying episodic: {e}", exc_info=True)
        
        return results
    
    def format_response(self, response: MemoryQueryResponse, max_results: int = 5) -> str:
        """
        Format memory query response for chat display.
        
        Args:
            response: Memory query response
            max_results: Maximum results to display
            
        Returns:
            Formatted text for chat
        """
        if response.is_empty():
            return "I don't have any memories matching that query."
        
        lines = []
        
        # Header
        query_type_text = response.query_type.value.replace('_', ' ').title()
        lines.append(f"**{query_type_text} Results** ({response.total_found} found)")
        lines.append("")
        
        # Results
        for i, result in enumerate(response.results[:max_results], 1):
            # Format timestamp if available
            time_str = ""
            if result.timestamp:
                try:
                    # Handle both datetime objects and ISO strings
                    if isinstance(result.timestamp, str):
                        timestamp = datetime.fromisoformat(result.timestamp)
                    else:
                        timestamp = result.timestamp
                    time_str = f" *({timestamp.strftime('%b %d, %I:%M %p')})*"
                except (ValueError, AttributeError):
                    pass
            
            # Format content (truncate if long)
            content = result.content
            if len(content) > 200:
                content = content[:200] + "..."
            
            # Source indicator
            source_emoji = {
                'stm': '💭',  # Recent memory
                'ltm': '🧠',  # Long-term knowledge
                'episodic': '📅'  # Past event
            }.get(result.source_system, '📝')
            
            lines.append(f"{i}. {source_emoji} {content}{time_str}")
            
            # Add relevance if high confidence
            if result.relevance >= 0.8:
                lines.append(f"   *Relevance: {result.relevance:.0%}*")
            
            lines.append("")
        
        # Show if there are more results
        if response.total_found > max_results:
            remaining = response.total_found - max_results
            lines.append(f"*...and {remaining} more results*")
        
        # Summary statistics
        if len(response.systems_queried) > 1:
            stats = []
            if response.stm_results > 0:
                stats.append(f"{response.stm_results} recent")
            if response.ltm_results > 0:
                stats.append(f"{response.ltm_results} long-term")
            if response.episodic_results > 0:
                stats.append(f"{response.episodic_results} episodic")
            
            if stats:
                lines.append("")
                lines.append(f"*Sources: {', '.join(stats)}*")
        
        return "\n".join(lines)


# Factory function
def create_memory_query_interface(
    stm: Optional[Any] = None,
    ltm: Optional[Any] = None,
    episodic: Optional[Any] = None
) -> MemoryQueryInterface:
    """
    Create MemoryQueryInterface instance.
    
    Args:
        stm: Short-term memory system
        ltm: Long-term memory system
        episodic: Episodic memory system
        
    Returns:
        MemoryQueryInterface instance
    """
    return MemoryQueryInterface(stm=stm, ltm=ltm, episodic=episodic)
