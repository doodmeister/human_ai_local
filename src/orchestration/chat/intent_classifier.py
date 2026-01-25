"""
Intent Classification for Chat-First Cognitive Integration

Analyzes user messages to understand intent and extract relevant entities.
Supports goal creation, memory queries, performance questions, and system status.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta


@dataclass
class Intent:
    """Represents classified user intent"""
    intent_type: str  # goal_creation, goal_query, memory_query, etc.
    confidence: float  # 0.0-1.0
    entities: Dict[str, any]  # extracted information
    original_message: str
    matched_pattern: Optional[str] = None


class IntentClassifier:
    """Classifies user intent from natural language"""
    
    # Intent patterns organized by category
    INTENT_PATTERNS = {
        'goal_creation': [
            # "I need to X by Friday"
            (r"(?:i need to|i want to|i have to|i must)\s+(.+?)(?:\s+by\s+(\w+day|tomorrow|today|next week))?(?:\.|$)", 0.85),
            # "Can you help me X"
            (r"(?:can you help me|help me|assist me with|assist me in)\s+(.+?)(?:\s+by\s+(\w+day|tomorrow|today))?(?:\.|$)", 0.80),
            # "By Friday, I need to X" - swap order of capture groups
            (r"by\s+(\w+day|tomorrow|today|next week)[,\s]+(?:i need to|i want to|i have to)\s+(.+?)(?:\.|$)", 0.85),
            # "I should X before Y" - more flexible pattern
            (r"i should\s+(.+?)\s+(?:before|by)\s+(.+?)(?:\.|$)", 0.80),
            # "Let's work on X"
            (r"(?:let['']s|we should|we need to)\s+(?:work on|tackle|do)\s+(.+)", 0.75),
        ],
        'goal_query': [
            # "How's my X goal?"
            (r"(?:how['']s|what['']s the status of|status of|update on)\s+(?:my|the)\s+(.+?)(?:\s+goal)?(?:\?|$)", 0.90),
            # "Am I making progress on X?"
            (r"(?:am i making progress|where are we|how are we doing) (?:on|with)\s+(.+?)(?:\?|$)", 0.85),
            # "What's happening with X?"
            (r"(?:what['']s happening with|tell me about|show me)\s+(?:my|the)\s+(.+?)(?:\s+goal)?(?:\?|$)", 0.80),
            # "Check on X"
            (r"(?:check on|status on|progress on)\s+(.+)", 0.85),
        ],
        'memory_query': [
            # "What do you remember about X?"
            (r"what do you (?:remember|know|recall|think) about\s+(.+?)(?:\?|$)", 0.95),
            # "Do you remember X?" / "Do you recall X?"
            (r"do you (?:remember|recall)\s+(.+?)(?:\?|$)", 0.95),
            # "Tell me about X" / "Tell me what you know about X"
            (r"tell me (?:about|what you know about)\s+(.+?)(?:\?|$)", 0.85),
            # "Remind me about X"
            (r"remind me about\s+(.+?)(?:\?|$)", 0.85),
            # "When did we discuss X?"
            (r"when did (?:we|i)\s+(?:discuss|talk about|mention|cover)\s+(.+?)(?:\?|$)", 0.90),
            # "Show me what you know about X"
            (r"show me (?:what you know about|your knowledge of)\s+(.+)", 0.90),
        ],
        'performance_query': [
            # "How are you performing?"
            (r"how (?:well )?(?:are you|am i) (?:doing|performing)(?:\?|$)", 0.95),
            # "Show me your stats" / "Show me your performance"
            (r"(?:show|give)\s+me\s+(?:your|my)\s+(?:performance|stats|metrics|accuracy|statistics)", 0.95),
            # "What's your success rate?"
            (r"what['']s your (?:accuracy|success rate|performance)(?:\?|$)", 0.95),
            # "How accurate are you?"
            (r"how (?:accurate|well) are you(?:\?|$)", 0.90),
            # "How well are you learning?"
            (r"how (?:well )?are you (?:learning|improving)(?:\?|$)", 0.90),
            # "Are you getting better?"
            (r"(?:are you|am i) getting (?:better|worse|more accurate)(?:\?|$)", 0.90),
        ],
        'system_status': [
            # "How are you feeling?"
            (r"how are you (?:feeling|doing)(?:\?|$)", 0.85),
            # "What's your status?"
            (r"(?:what['']s|show me) your (?:status|health|condition)(?:\?|$)", 0.95),
            # "What's your cognitive load?"
            (r"what['']s your (?:cognitive load|memory status|attention)(?:\?|$)", 0.95),
            # "Memory status" / "Cognitive status"
            (r"(?:memory|cognitive|attention|working memory) (?:status|health|capacity)(?:\?|$)", 0.95),
            # "Are you overloaded?"
            (r"are you (?:overloaded|overwhelmed|stressed|busy)(?:\?|$)", 0.90),
            # "System health"
            (r"(?:system|cognitive) health(?:\?|$)", 0.95),
        ],
    }
    
    def __init__(self):
        """Initialize intent classifier"""
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, float]]]:
        """Pre-compile regex patterns for performance"""
        compiled = {}
        for intent_type, patterns in self.INTENT_PATTERNS.items():
            compiled[intent_type] = [
                (re.compile(pattern, re.IGNORECASE), confidence)
                for pattern, confidence in patterns
            ]
        return compiled
    
    def classify(self, message: str) -> Intent:
        """
        Classify user message and extract entities.
        
        Args:
            message: User's natural language message
            
        Returns:
            Intent object with classification, confidence, and entities
        """
        message = message.strip()
        
        # Try each intent type
        best_match = None
        best_confidence = 0.0
        best_entities = {}
        best_pattern = None
        
        for intent_type, patterns in self.compiled_patterns.items():
            for pattern, base_confidence in patterns:
                match = pattern.search(message)
                if match:
                    # Extract entities based on intent type
                    entities = self._extract_entities(intent_type, match, message)
                    
                    # Adjust confidence based on entity quality
                    adjusted_confidence = self._adjust_confidence(
                        base_confidence, entities, message
                    )
                    
                    if adjusted_confidence > best_confidence:
                        best_confidence = adjusted_confidence
                        best_match = intent_type
                        best_entities = entities
                        best_pattern = pattern.pattern
        
        # Default to general_chat if no intent detected
        if best_match is None:
            return Intent(
                intent_type='general_chat',
                confidence=1.0,
                entities={},
                original_message=message
            )
        
        return Intent(
            intent_type=best_match,
            confidence=best_confidence,
            entities=best_entities,
            original_message=message,
            matched_pattern=best_pattern
        )
    
    def _extract_entities(self, intent_type: str, match: re.Match, message: str) -> Dict[str, Any]:
        """Extract relevant entities based on intent type"""
        entities = {}
        
        if intent_type == 'goal_creation':
            # Extract goal description and deadline
            groups = match.groups()
            
            # Check if this is the "By X, I need to Y" pattern (deadline first)
            if message.lower().strip().startswith('by '):
                # Deadline is first group, description is second
                if len(groups) >= 2 and groups[1]:
                    entities['deadline'] = groups[0].strip()
                    entities['goal_description'] = groups[1].strip()
                elif len(groups) >= 1:
                    entities['deadline'] = groups[0].strip()
            else:
                # Normal pattern: description first, deadline second
                if len(groups) >= 1:
                    entities['goal_description'] = groups[0].strip()
                if len(groups) >= 2 and groups[1]:
                    entities['deadline'] = groups[1].strip()
            
            # Check for priority indicators
            entities['priority'] = self._detect_priority(message)
            
            # Check for urgency indicators
            entities['urgent'] = self._detect_urgency(message)
        
        elif intent_type == 'goal_query':
            # Extract goal reference
            groups = match.groups()
            if groups and groups[0]:
                entities['goal_reference'] = groups[0].strip()
        
        elif intent_type == 'memory_query':
            # Extract query terms
            groups = match.groups()
            if groups and groups[0]:
                entities['query_term'] = groups[0].strip()
            
            # Detect temporal constraints
            entities['temporal_constraint'] = self._detect_temporal_constraint(message)
            
            # Detect query type (episodic vs semantic)
            entities['query_type'] = self._detect_memory_query_type(message)
        
        elif intent_type == 'performance_query':
            # Detect specific metric requests
            entities['metric_type'] = self._detect_metric_type(message)
        
        elif intent_type == 'system_status':
            # Detect specific system component
            entities['component'] = self._detect_system_component(message)
        
        return entities
    
    def _adjust_confidence(self, base_confidence: float, entities: Dict, message: str) -> float:
        """Adjust confidence based on entity quality and message characteristics"""
        confidence = base_confidence
        
        # Boost confidence if key entities are present and well-formed
        if entities.get('goal_description') and len(entities['goal_description']) > 10:
            confidence += 0.05
        
        if entities.get('deadline'):
            confidence += 0.05
        
        # Reduce confidence for very short or very long messages
        word_count = len(message.split())
        if word_count < 3:
            confidence -= 0.1
        elif word_count > 50:
            confidence -= 0.05
        
        # Boost confidence for question marks with query intents
        if '?' in message and any(q in entities for q in ['query_term', 'metric_type', 'component']):
            confidence += 0.05
        
        return min(1.0, max(0.0, confidence))
    
    def _detect_priority(self, message: str) -> str:
        """Detect priority indicators in message"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['critical', 'urgent', 'asap', 'immediately', 'emergency']):
            return 'high'
        elif any(word in message_lower for word in ['important', 'priority', 'soon']):
            return 'medium'
        elif any(word in message_lower for word in ['eventually', 'when possible', 'low priority']):
            return 'low'
        
        return 'medium'  # default
    
    def _detect_urgency(self, message: str) -> bool:
        """Detect if message indicates urgency"""
        message_lower = message.lower()
        urgency_words = ['urgent', 'asap', 'immediately', 'right now', 'critical', 'emergency']
        return any(word in message_lower for word in urgency_words)
    
    def _detect_temporal_constraint(self, message: str) -> Optional[str]:
        """Detect time-related constraints in memory queries"""
        message_lower = message.lower()
        
        temporal_patterns = [
            (r'last week', 'last_week'),
            (r'yesterday', 'yesterday'),
            (r'last month', 'last_month'),
            (r'recently', 'recent'),
            (r'this week', 'this_week'),
            (r'today', 'today'),
        ]
        
        for pattern, constraint in temporal_patterns:
            if re.search(pattern, message_lower):
                return constraint
        
        return None
    
    def _detect_memory_query_type(self, message: str) -> str:
        """Detect if query is episodic (conversations) or semantic (knowledge)"""
        message_lower = message.lower()
        
        episodic_indicators = ['discuss', 'talk about', 'conversation', 'said', 'told', 'when did']
        semantic_indicators = ['know about', 'understand', 'explain', 'what is', 'tell me about']
        
        episodic_score = sum(1 for ind in episodic_indicators if ind in message_lower)
        semantic_score = sum(1 for ind in semantic_indicators if ind in message_lower)
        
        if episodic_score > semantic_score:
            return 'episodic'
        elif semantic_score > episodic_score:
            return 'semantic'
        
        return 'general'
    
    def _detect_metric_type(self, message: str) -> str:
        """Detect specific performance metric being requested"""
        message_lower = message.lower()
        
        if 'accuracy' in message_lower or 'accurate' in message_lower or 'correct' in message_lower:
            return 'accuracy'
        elif 'success' in message_lower:
            return 'success_rate'
        elif 'time' in message_lower or 'speed' in message_lower:
            return 'time_accuracy'
        elif 'improve' in message_lower or 'learn' in message_lower:
            return 'improvement'
        elif 'confidence' in message_lower:
            return 'confidence'
        
        return 'general'
    
    def _detect_system_component(self, message: str) -> str:
        """Detect which system component is being queried"""
        message_lower = message.lower()
        
        if 'memory' in message_lower or 'stm' in message_lower or 'ltm' in message_lower:
            return 'memory'
        elif 'cognitive' in message_lower or 'load' in message_lower:
            return 'cognitive'
        elif 'attention' in message_lower:
            return 'attention'
        elif 'goal' in message_lower:
            return 'goals'
        
        return 'general'


def create_intent_classifier() -> IntentClassifier:
    """Factory function to create intent classifier"""
    return IntentClassifier()
