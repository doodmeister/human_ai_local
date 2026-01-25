"""
Memory Query Parser - Task 8

Parses memory-related queries to extract search criteria and route to appropriate memory systems.

Architecture:
    User Query → MemoryQueryParser → MemoryQueryResult
    → (STM/LTM/Episodic retrieval) → Formatted Response
    
Usage:
    parser = MemoryQueryParser()
    
    # Parse query
    result = parser.parse_query("What do you remember about Python?")
    # result.query_type: 'semantic_search'
    # result.search_terms: ['python']
    # result.target_systems: ['stm', 'ltm']
    
    # Temporal queries
    result = parser.parse_query("When did we discuss the API design?")
    # result.query_type: 'temporal_search'
    # result.temporal_constraint: TimeConstraint(...)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Set, Dict, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Type of memory query."""
    SEMANTIC_SEARCH = "semantic_search"  # "What do you remember about X?"
    TEMPORAL_SEARCH = "temporal_search"  # "When did we discuss X?"
    EPISODIC_RECALL = "episodic_recall"  # "Tell me about our conversation yesterday"
    FACT_LOOKUP = "fact_lookup"  # "What's my preference for X?"
    RECENT_MEMORY = "recent_memory"  # "What did we just talk about?"
    MEMORY_STATS = "memory_stats"  # "What do you know about?"


class MemorySystem(Enum):
    """Target memory system."""
    STM = "stm"  # Short-term memory
    LTM = "ltm"  # Long-term memory
    EPISODIC = "episodic"  # Episodic memory
    PROSPECTIVE = "prospective"  # Prospective memory (reminders)
    PROCEDURAL = "procedural"  # Procedural memory (skills)


@dataclass
class TimeConstraint:
    """
    Temporal constraint for memory queries.
    
    Supports relative ("yesterday", "last week") and absolute ("on Monday") times.
    """
    time_type: str  # 'relative', 'absolute', 'range'
    
    # Relative constraints (e.g., "in the last hour")
    relative_amount: Optional[int] = None
    relative_unit: Optional[str] = None  # 'minutes', 'hours', 'days', 'weeks'
    
    # Absolute constraints (e.g., "on Monday")
    absolute_time: Optional[datetime] = None
    
    # Range constraints (e.g., "between Monday and Friday")
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_datetime_range(self) -> tuple[Optional[datetime], Optional[datetime]]:
        """
        Convert to datetime range (start, end).
        
        Returns:
            (start_time, end_time) where None means unbounded
        """
        now = datetime.now()
        
        if self.time_type == 'relative':
            if self.relative_unit and self.relative_amount:
                if self.relative_unit == 'minutes':
                    delta = timedelta(minutes=self.relative_amount)
                elif self.relative_unit == 'hours':
                    delta = timedelta(hours=self.relative_amount)
                elif self.relative_unit == 'days':
                    delta = timedelta(days=self.relative_amount)
                elif self.relative_unit == 'weeks':
                    delta = timedelta(weeks=self.relative_amount)
                else:
                    return (None, None)
                
                start = now - delta
                return (start, now)
        
        elif self.time_type == 'absolute':
            if self.absolute_time:
                # Return day range for absolute time
                start = self.absolute_time.replace(hour=0, minute=0, second=0, microsecond=0)
                end = start + timedelta(days=1)
                return (start, end)
        
        elif self.time_type == 'range':
            return (self.start_time, self.end_time)
        
        return (None, None)


@dataclass
class MemoryQueryResult:
    """
    Parsed memory query with extracted criteria.
    
    Contains all information needed to route and execute memory retrieval.
    """
    query_type: QueryType
    search_terms: List[str] = field(default_factory=list)
    target_systems: Set[MemorySystem] = field(default_factory=set)
    temporal_constraint: Optional[TimeConstraint] = None
    limit: int = 10
    min_relevance: float = 0.5
    
    # Additional filters
    tags: List[str] = field(default_factory=list)
    memory_types: List[str] = field(default_factory=list)  # 'fact', 'preference', 'event', etc.
    importance_threshold: Optional[float] = None
    
    # Original query for reference
    original_query: str = ""
    confidence: float = 0.8  # Confidence in parse
    
    def __str__(self) -> str:
        """Human-readable representation."""
        parts = [f"Query: {self.query_type.value}"]
        if self.search_terms:
            parts.append(f"Terms: {', '.join(self.search_terms)}")
        if self.target_systems:
            systems = [s.value for s in self.target_systems]
            parts.append(f"Systems: {', '.join(systems)}")
        if self.temporal_constraint:
            parts.append(f"Time: {self.temporal_constraint.time_type}")
        return " | ".join(parts)


class MemoryQueryParser:
    """
    Parses natural language memory queries.
    
    Extracts search terms, temporal constraints, and routes to appropriate memory systems.
    """
    
    # Temporal keywords and their mappings
    TEMPORAL_PATTERNS = {
        # Relative time
        r'(?:in|within|during)\s+(?:the\s+)?last\s+(\d+)\s+(minute|hour|day|week)s?': ('relative', 'last'),
        r'(?:in|within|during)\s+(?:the\s+)?past\s+(\d+)\s+(minute|hour|day|week)s?': ('relative', 'past'),
        r'recently|just now|lately': ('relative', 'recent'),
        r'yesterday': ('relative', 'yesterday'),
        r'today|this morning|this afternoon': ('relative', 'today'),
        r'(?:last|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)': ('absolute', 'weekday'),
        r'(?:last|this)\s+week': ('relative', 'week'),
        
        # Temporal query patterns
        r'when did (?:we|i)': ('temporal_search', 'when'),
        r'what time': ('temporal_search', 'what_time'),
    }
    
    # Query type patterns
    QUERY_TYPE_PATTERNS = {
        QueryType.TEMPORAL_SEARCH: [
            r'when did',
            r'what time',
            r'at what point',
            r'how long ago',
        ],
        QueryType.EPISODIC_RECALL: [
            r'tell me about (?:our|the) (?:conversation|discussion|talk)',
            r'what happened (?:when|during)',
            r'recall (?:our|the) (?:conversation|meeting)',
            r'(?:our|the) conversation about',
        ],
        QueryType.FACT_LOOKUP: [
            r'what(?:\'s| is) my (?:preference|setting|configuration)',
            r'what (?:is|are) my (?:preferences|settings)',
            r'how do i (?:prefer|like)',
        ],
        QueryType.RECENT_MEMORY: [
            r'what (?:did we|have we) (?:just|recently) (?:discuss|talked about|talk about)',
            r'what were we (?:just|recently) (?:discussing|talking about)',
            r'(?:remind me|tell me) what we (?:just|recently) (?:discussed|said)',
            r'\bjust (?:discuss|talked?|said)',  # Match 'just' as indicator
        ],
        QueryType.MEMORY_STATS: [
            r'what do you know',
            r'what have you learned',
            r'(?:show|tell) me what you (?:know|remember)',
            r'list (?:everything|what) you (?:know|remember)',
        ],
    }
    
    # Stop words to filter from search terms
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'about', 'as', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'under', 'again',
        'do', 'you', 'remember', 'recall', 'know', 'think', 'tell', 'me',
        'what', 'when', 'where', 'why', 'how', 'which', 'who', 'whom',
    }
    
    def __init__(self):
        """Initialize parser with compiled patterns."""
        self.temporal_patterns = {
            re.compile(pattern, re.IGNORECASE): info
            for pattern, info in self.TEMPORAL_PATTERNS.items()
        }
        
        self.query_type_patterns = {
            qtype: [re.compile(p, re.IGNORECASE) for p in patterns]
            for qtype, patterns in self.QUERY_TYPE_PATTERNS.items()
        }
    
    def parse_query(self, query: str) -> MemoryQueryResult:
        """
        Parse memory query and extract criteria.
        
        Args:
            query: Natural language query
            
        Returns:
            MemoryQueryResult with extracted information
        """
        query_lower = query.lower()
        
        # 1. Classify query type
        query_type = self._classify_query_type(query_lower)
        
        # 2. Extract temporal constraints
        temporal = self._extract_temporal_constraint(query_lower)
        
        # 3. Extract search terms
        search_terms = self._extract_search_terms(query_lower, query_type, temporal)
        
        # 4. Determine target systems
        target_systems = self._determine_target_systems(query_type, temporal, search_terms)
        
        # 5. Extract additional filters
        tags = self._extract_tags(query_lower)
        memory_types = self._extract_memory_types(query_lower)
        
        # 6. Set retrieval parameters
        limit = self._determine_limit(query_type)
        min_relevance = self._determine_min_relevance(query_type)
        
        return MemoryQueryResult(
            query_type=query_type,
            search_terms=search_terms,
            target_systems=target_systems,
            temporal_constraint=temporal,
            limit=limit,
            min_relevance=min_relevance,
            tags=tags,
            memory_types=memory_types,
            original_query=query,
            confidence=0.8  # Base confidence
        )
    
    def _classify_query_type(self, query: str) -> QueryType:
        """
        Classify the type of memory query.
        
        Args:
            query: Lowercased query string
            
        Returns:
            QueryType enum value
        """
        # Check each query type pattern
        for qtype, patterns in self.query_type_patterns.items():
            for pattern in patterns:
                if pattern.search(query):
                    return qtype
        
        # Default to semantic search
        return QueryType.SEMANTIC_SEARCH
    
    def _extract_temporal_constraint(self, query: str) -> Optional[TimeConstraint]:
        """
        Extract temporal constraint from query.
        
        Args:
            query: Lowercased query string
            
        Returns:
            TimeConstraint or None
        """
        for pattern, (time_type, subtype) in self.temporal_patterns.items():
            match = pattern.search(query)
            if match:
                if time_type == 'relative':
                    if subtype in ['last', 'past']:
                        # "last 5 days"
                        amount = int(match.group(1))
                        unit = match.group(2) + 's'  # Pluralize
                        return TimeConstraint(
                            time_type='relative',
                            relative_amount=amount,
                            relative_unit=unit
                        )
                    elif subtype == 'recent':
                        # "recently" → last hour
                        return TimeConstraint(
                            time_type='relative',
                            relative_amount=1,
                            relative_unit='hours'
                        )
                    elif subtype == 'yesterday':
                        return TimeConstraint(
                            time_type='relative',
                            relative_amount=1,
                            relative_unit='days'
                        )
                    elif subtype == 'today':
                        # Today = last 24 hours
                        now = datetime.now()
                        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
                        return TimeConstraint(
                            time_type='range',
                            start_time=start,
                            end_time=now
                        )
                    elif subtype == 'week':
                        return TimeConstraint(
                            time_type='relative',
                            relative_amount=1,
                            relative_unit='weeks'
                        )
                
                elif time_type == 'absolute':
                    # Absolute time (e.g., "last Monday")
                    # Simplified - would need more logic for actual dates
                    return TimeConstraint(
                        time_type='relative',
                        relative_amount=7,
                        relative_unit='days'
                    )
                
                elif time_type == 'temporal_search':
                    # Query is asking about time itself
                    return None
        
        return None
    
    def _extract_search_terms(
        self,
        query: str,
        query_type: QueryType,
        temporal: Optional[TimeConstraint]
    ) -> List[str]:
        """
        Extract search terms from query.
        
        Args:
            query: Lowercased query string
            query_type: Classified query type
            temporal: Temporal constraint if any
            
        Returns:
            List of search terms
        """
        # Remove common question patterns
        cleaned = query
        
        # Remove temporal phrases
        for pattern in self.temporal_patterns.keys():
            cleaned = pattern.sub('', cleaned)
        
        # Remove query type indicators
        patterns_to_remove = [
            r'what do you (?:remember|know|recall|think) about',
            r'do you (?:remember|recall)',
            r'tell me (?:about|what you know about)',
            r'remind me about',
            r'when did (?:we|i)\s+(?:discuss|talk about|mention|cover)',
            r'show me (?:what you know about|your knowledge of)',
            r'what(?:\'s| is) my (?:preference|setting) for',
        ]
        
        for pattern in patterns_to_remove:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        
        # Remove punctuation and split
        cleaned = re.sub(r'[?!.,;:]', '', cleaned)
        words = cleaned.split()
        
        # Filter stop words
        terms = [w for w in words if w not in self.STOP_WORDS and len(w) > 2]
        
        return terms
    
    def _determine_target_systems(
        self,
        query_type: QueryType,
        temporal: Optional[TimeConstraint],
        search_terms: List[str]
    ) -> Set[MemorySystem]:
        """
        Determine which memory systems to query.
        
        Args:
            query_type: Type of query
            temporal: Temporal constraint
            search_terms: Extracted search terms
            
        Returns:
            Set of MemorySystem values to query
        """
        systems = set()
        
        if query_type == QueryType.RECENT_MEMORY:
            # Recent queries → STM only
            systems.add(MemorySystem.STM)
        
        elif query_type == QueryType.EPISODIC_RECALL:
            # Episodic queries → Episodic + STM
            systems.add(MemorySystem.EPISODIC)
            systems.add(MemorySystem.STM)
        
        elif query_type == QueryType.FACT_LOOKUP:
            # Fact lookup → LTM (semantic facts)
            systems.add(MemorySystem.LTM)
        
        elif query_type == QueryType.TEMPORAL_SEARCH:
            # Temporal search → Episodic (has timestamps)
            systems.add(MemorySystem.EPISODIC)
            systems.add(MemorySystem.STM)
        
        elif query_type == QueryType.MEMORY_STATS:
            # Stats → All systems
            systems.add(MemorySystem.STM)
            systems.add(MemorySystem.LTM)
            systems.add(MemorySystem.EPISODIC)
        
        else:  # SEMANTIC_SEARCH
            # Default semantic search → STM + LTM
            systems.add(MemorySystem.STM)
            systems.add(MemorySystem.LTM)
            
            # Add episodic if temporal constraint
            if temporal:
                systems.add(MemorySystem.EPISODIC)
        
        return systems
    
    def _extract_tags(self, query: str) -> List[str]:
        """
        Extract tags from query.
        
        Args:
            query: Lowercased query string
            
        Returns:
            List of tags
        """
        # Look for hashtags or explicit tags
        tags = re.findall(r'#(\w+)', query)
        
        # Look for "tagged X" or "with tag X"
        tag_patterns = [
            r'tagged\s+(\w+)',
            r'with tag\s+(\w+)',
            r'tag:(\w+)',
        ]
        
        for pattern in tag_patterns:
            matches = re.findall(pattern, query)
            tags.extend(matches)
        
        return tags
    
    def _extract_memory_types(self, query: str) -> List[str]:
        """
        Extract memory types from query.
        
        Args:
            query: Lowercased query string
            
        Returns:
            List of memory types
        """
        types = []
        
        type_patterns = {
            'preference': r'\b(?:preferences?|prefer|like|favorite)\b',
            'fact': r'\b(?:facts?|information|data|knowledge)\b',
            'event': r'\b(?:events?|conversation|discussion|meeting|talk)\b',
            'goal': r'\b(?:goals?|objective|task|plan)\b',
            'decision': r'\b(?:decisions?|choice|pick|select)\b',
        }
        
        for mtype, pattern in type_patterns.items():
            if re.search(pattern, query, re.IGNORECASE):
                types.append(mtype)
        
        return types
    
    def _determine_limit(self, query_type: QueryType) -> int:
        """
        Determine result limit based on query type.
        
        Args:
            query_type: Type of query
            
        Returns:
            Result limit
        """
        if query_type == QueryType.RECENT_MEMORY:
            return 5  # Recent items only
        elif query_type == QueryType.MEMORY_STATS:
            return 20  # More results for stats
        else:
            return 10  # Default
    
    def _determine_min_relevance(self, query_type: QueryType) -> float:
        """
        Determine minimum relevance threshold.
        
        Args:
            query_type: Type of query
            
        Returns:
            Minimum relevance (0-1)
        """
        if query_type == QueryType.RECENT_MEMORY:
            return 0.3  # Lower threshold for recent
        elif query_type == QueryType.FACT_LOOKUP:
            return 0.7  # Higher threshold for facts
        else:
            return 0.5  # Default


# Factory function
def create_memory_query_parser() -> MemoryQueryParser:
    """
    Create MemoryQueryParser instance.
    
    Returns:
        MemoryQueryParser instance
    """
    return MemoryQueryParser()
