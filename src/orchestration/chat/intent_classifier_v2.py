"""
Enhanced Intent Classification with Multi-Intent Support

Improvements over v1:
- Multi-intent detection (messages can have multiple intents)
- Context-aware classification (uses conversation history)
- Ambiguity detection and resolution
- Better entity extraction with NER patterns
- Intent hierarchies for nuanced routing
- Confidence thresholds and fallback strategies
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Set
from datetime import datetime, timedelta
from collections import Counter


@dataclass
class IntentV2:
    """Enhanced intent representation with multi-intent support"""
    intent_type: str  # Primary intent
    confidence: float  # 0.0-1.0
    entities: Dict[str, Any]  # Extracted entities
    original_message: str
    matched_patterns: List[str] = field(default_factory=list)  # All matched patterns
    
    # Multi-intent support
    secondary_intents: List[Tuple[str, float]] = field(default_factory=list)  # [(type, confidence)]
    is_ambiguous: bool = False  # True if confidence spread is narrow
    ambiguity_score: float = 0.0  # 0.0 (clear) to 1.0 (very ambiguous)
    
    # Context awareness
    context_boost: float = 0.0  # Confidence boost from context
    conversation_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context from previous conversation turns"""
    recent_intents: List[str] = field(default_factory=list)  # Last N intents
    active_goals: Set[str] = field(default_factory=set)  # Currently active goals
    recent_topics: List[str] = field(default_factory=list)  # Recently discussed topics
    last_query_type: Optional[str] = None  # Last query type (memory/goal/performance)
    last_response_time: Optional[datetime] = None
    user_preferences: Dict[str, Any] = field(default_factory=dict)


class IntentClassifierV2:
    """
    Enhanced intent classifier with multi-intent and context awareness.
    
    Features:
    - Detects multiple intents in a single message
    - Uses conversation context for better classification
    - Detects and handles ambiguous intents
    - Improved entity extraction
    - Confidence thresholds and fallback logic
    """
    
    # Minimum confidence threshold for primary intent
    PRIMARY_INTENT_THRESHOLD = 0.60
    
    # Minimum confidence threshold for secondary intents
    SECONDARY_INTENT_THRESHOLD = 0.45
    
    # If top 2 intents are within this range, mark as ambiguous
    AMBIGUITY_RANGE = 0.15
    
    # Intent patterns with weights (higher = more specific)
    INTENT_PATTERNS = {
        'goal_creation': {
            'patterns': [
                # Explicit goal creation
                (r"(?:create|add|set)\s+(?:a\s+)?(?:new\s+)?goal\s+(?:to\s+)?(.+)", 0.95, 1.2),
                # "I need to X by Friday"
                (r"(?:i need to|i want to|i have to|i must|i should)\s+(.+?)(?:\s+by\s+(\w+day|tomorrow|today|next week|next month))?(?:\.|$)", 0.85, 1.0),
                # "Can you help me X"
                (r"(?:can you help me|help me|assist me (?:with|in))\s+(.+?)(?:\s+by\s+(\w+day|tomorrow|today))?(?:\.|$)", 0.80, 0.9),
                # "By Friday, I need to X"
                (r"by\s+(\w+day|tomorrow|today|next week)[,\s]+(?:i need to|i want to|i have to)\s+(.+?)(?:\.|$)", 0.85, 1.0),
                # "I should X before Y"
                (r"i should\s+(.+?)\s+(?:before|by)\s+(.+?)(?:\.|$)", 0.80, 0.9),
                # "Let's work on X"
                (r"(?:let['']s|we should|we need to)\s+(?:work on|tackle|do|start)\s+(.+)", 0.75, 0.8),
                # "My goal is to X"
                (r"my goal is to\s+(.+)", 0.90, 1.1),
            ],
            'keywords': ['goal', 'task', 'need', 'want', 'should', 'must', 'deadline', 'finish', 'complete'],
            'boost_words': ['urgent', 'important', 'asap', 'priority'],
        },
        'goal_query': {
            'patterns': [
                # Explicit goal query - LIST ALL
                (r"(?:show|get|list|what are)\s+(?:me\s+)?(?:my\s+)?(?:all\s+)?goals?(?:\?|$)", 0.95, 1.2),
                (r"what(?:'s| is| are)\s+(?:my|all)\s+(?:current\s+)?goals?(?:\?|$)", 0.95, 1.2),
                (r"check\s+(?:my|all)\s+goals?(?:\?|$)", 0.95, 1.2),
                (r"(?:what|show)\s+am\s+i\s+working\s+on(?:\?|$)", 0.90, 1.1),
                (r"do\s+i\s+have\s+any\s+(?:active\s+)?goals?(?:\?|$)", 0.90, 1.1),
                # Execution progress queries
                (r"what(?:'s| is)\s+(?:currently\s+)?(?:executing|running|in progress)(?:\?|$)", 0.95, 1.2),
                (r"(?:show|what is)\s+(?:execution|current|running)\s+progress(?:\?|$)", 0.95, 1.2),
                (r"what(?:'s| is)\s+(?:happening|being done|being executed)\s+(?:right now|currently|now)(?:\?|$)", 0.95, 1.2),
                (r"(?:what|which)\s+goals?\s+(?:are|is)\s+(?:being\s+)?(?:executed|running)(?:\?|$)", 0.90, 1.1),
                (r"(?:show|display|list)\s+(?:active\s+)?executions?(?:\?|$)", 0.90, 1.1),
                # "How's my X goal?"
                (r"(?:how['']s|what['']s the status of|status of|update on)\s+(?:my|the)\s+(.+?)(?:\s+goal)?(?:\?|$)", 0.90, 1.1),
                # "Am I making progress on X?"
                (r"(?:am i making progress|where are we|how are we doing) (?:on|with)\s+(.+?)(?:\?|$)", 0.85, 1.0),
                # "What's happening with X?"
                (r"(?:what['']s happening with|tell me about|show me)\s+(?:my|the)\s+(.+?)(?:\s+goal)?(?:\?|$)", 0.80, 0.9),
                # "Check on X"
                (r"(?:check on|status on|progress on)\s+(.+)", 0.85, 1.0),
                # "Did I finish X?"
                (r"(?:did i|have i)\s+(?:finish|complete|done)\s+(.+?)(?:\?|$)", 0.85, 1.0),
            ],
            'keywords': ['status', 'progress', 'update', 'check', 'goal', 'goals', 'how', 'doing', 'working', 'executing', 'running', 'execution'],
            'boost_words': ['finished', 'completed', 'done', 'ready', 'all', 'current', 'active', 'executing', 'running'],
        },
        'goal_update': {
            'patterns': [
                # Mark goal complete/done
                (r"(?:mark|set)\s+(?:my\s+)?(?:the\s+)?(.+?)\s+(?:goal|task\s+)?(?:as\s+)?(?:complete|done|finished)(?:\.|$)", 0.95, 1.2),
                (r"(?:i['']ve|i have)\s+(?:finished|completed|done)\s+(?:my\s+)?(?:the\s+)?(.+?)(?:\s+(?:goal|task))?(?:\.|$)", 0.90, 1.1),
                (r"(.+?)\s+(?:goal|task\s+)?is\s+(?:complete|done|finished)(?:\.|$)", 0.85, 1.0),
                # Cancel/delete goal
                (r"(?:cancel|delete|remove|drop)\s+(?:my\s+)?(?:the\s+)?(.+?)(?:\s+(?:goal|task))?(?:\.|$)", 0.95, 1.2),
                (r"(?:i don['']t need|forget about|nevermind|skip)\s+(?:the\s+)?(.+?)(?:\s+(?:goal|task))?(?:\.|$)", 0.85, 1.0),
                # Change priority
                (r"(?:change|set|update|make)\s+(?:my\s+)?(?:the\s+)?(.+?)\s+(?:(?:goal|task)\s+)?(?:priority\s+)?(?:to\s+)?(high|medium|low)(?:\.|$)", 0.90, 1.1),
                (r"(?:make|set)\s+(.+?)\s+(?:a\s+)?(high|medium|low)\s+priority(?:\.|$)", 0.85, 1.0),
                # Change deadline
                (r"(?:change|move|extend|update)\s+(?:my\s+)?(?:the\s+)?(.+?)\s+(?:(?:goal|task)\s+)?deadline\s+(?:to\s+)?(.+?)(?:\.|$)", 0.90, 1.1),
                (r"(?:extend|push back|postpone)\s+(?:my\s+)?(?:the\s+)?(.+?)(?:\s+(?:goal|task))?\s+(?:to|until)\s+(.+?)(?:\.|$)", 0.85, 1.0),
            ],
            'keywords': ['complete', 'done', 'finished', 'cancel', 'delete', 'remove', 'drop', 'priority', 'deadline', 'extend', 'postpone', 'skip'],
            'boost_words': ['urgent', 'high', 'low', 'medium', 'nevermind', 'forget'],
        },
        'memory_query': {
            'patterns': [
                # "What do you remember/know about X?"
                (r"what do you (?:remember|know|recall|think) about\s+(.+?)(?:\?|$)", 0.95, 1.2),
                # "Do you remember X?"
                (r"do you (?:remember|recall|know)\s+(.+?)(?:\?|$)", 0.95, 1.2),
                # "Tell me about X" (context-dependent)
                (r"tell me (?:about|what you know about)\s+(.+?)(?:\?|$)", 0.80, 0.8),
                # "Remind me about X"
                (r"remind me (?:about|of)\s+(.+?)(?:\?|$)", 0.90, 1.0),
                # "When did we discuss X?"
                (r"when did (?:we|i)\s+(?:discuss|talk about|mention|cover)\s+(.+?)(?:\?|$)", 0.90, 1.1),
                # "Show me what you know about X"
                (r"show me (?:what you know about|your knowledge of)\s+(.+)", 0.90, 1.1),
                # "What did we discuss/talk about X?"
                (r"what did (?:we|i)\s+(?:discuss|talk about|say about)\s+(.+?)(?:\?|$)", 0.90, 1.1),
                # "What did we just talk about?"
                (r"what did (?:we|i) just (?:talk about|discuss|say)(?:\?|$)", 0.90, 1.1),
                # "Recall X"
                (r"recall\s+(.+)", 0.85, 1.0),
            ],
            'keywords': ['remember', 'recall', 'know', 'discuss', 'talked', 'said', 'memory', 'forget'],
            'boost_words': ['when', 'where', 'who', 'what', 'how', 'why'],
        },
        'performance_query': {
            'patterns': [
                # "How are you performing?"
                (r"how (?:well )?(?:are you|am i) (?:doing|performing)(?:\?|$)", 0.95, 1.2),
                # "Show me your stats"
                (r"(?:show|give)\s+me\s+(?:your|my)\s+(?:performance|stats|metrics|accuracy|statistics)(?:\?|$)", 0.95, 1.2),
                # "What's your success rate?"
                (r"what['']s your (?:accuracy|success rate|performance|stats)(?:\?|$)", 0.95, 1.2),
                # "How accurate are you?"
                (r"how (?:accurate|well|good) are you(?:\?|$)", 0.90, 1.1),
                # "How well are you learning?"
                (r"how (?:well )?are you (?:learning|improving)(?:\?|$)", 0.90, 1.1),
                # "Are you getting better?" / "Are you improving?"
                (r"(?:are you|am i) (?:getting )?(?:better|worse|more accurate|improving|learning)(?:\?|$)", 0.90, 1.1),
                # "Show me your accuracy"
                (r"show (?:me )?(?:your )?(?:accuracy|metrics|performance)", 0.90, 1.1),
            ],
            'keywords': ['performance', 'accuracy', 'stats', 'metrics', 'success', 'learning', 'improving'],
            'boost_words': ['rate', 'score', 'percentage', 'improvement'],
        },
        'system_status': {
            'patterns': [
                # "How are you feeling?"
                (r"how are you (?:feeling|doing)(?:\?|$)", 0.85, 0.9),
                # "What's your status?"
                (r"(?:what['']s|show me) your (?:status|health|condition)(?:\?|$)", 0.95, 1.2),
                # "What's your cognitive load?"
                (r"what['']s your (?:cognitive load|memory status|attention|capacity)(?:\?|$)", 0.95, 1.2),
                # "Memory status" / "Cognitive status"
                (r"(?:memory|cognitive|attention|working memory) (?:status|health|capacity|load)(?:\?|$)", 0.95, 1.2),
                # "Are you overloaded?"
                (r"are you (?:overloaded|overwhelmed|stressed|busy)(?:\?|$)", 0.90, 1.1),
                # "System health"
                (r"(?:system|cognitive|mental) health(?:\?|$)", 0.95, 1.2),
                # "Check your status"
                (r"check your (?:status|health|load)", 0.90, 1.1),
                # "Give me a status report"
                (r"give me a (?:detailed |brief |quick )?status report", 0.90, 1.1),
            ],
            'keywords': ['status', 'health', 'load', 'capacity', 'feeling', 'system', 'cognitive'],
            'boost_words': ['overloaded', 'stressed', 'busy', 'overwhelmed'],
        },
        'reminder_request': {
            'patterns': [
                (r"remind me to\s+(?P<reminder_text>.+?)(?:\s+in\s+(?P<reminder_time>[\w\s]+))?(?:[.!?]|$)", 0.95, 1.2),
                (r"set (?:a )?reminder (?:for|to)\s+(?P<reminder_text>.+?)(?:[.!?]|$)", 0.90, 1.1),
                (r"(?:what|which) (?:are )?(?:my )?reminders?(?: do i have)?(?:\?|$)", 0.90, 1.0),
                (r"do i have any reminders(?:\?|$)", 0.90, 1.0),
                (r"list (?:my )?reminders?(?:\?|$)", 0.92, 1.05),
                (r"do i have any reminders due(?:\?|$)", 0.85, 0.95),
                (r"remind me in\s+(?P<reminder_time_only>[\w\s]+)\s+to\s+(?P<reminder_text_alt>.+?)(?:[.!?]|$)", 0.93, 1.15),
            ],
            'keywords': ['remind', 'reminder', 'reminders', 'due'],
            'boost_words': ['today', 'tonight', 'tomorrow', 'minutes', 'hours'],
        },
    }
    
    def __init__(self, context: Optional[ConversationContext] = None):
        """
        Initialize enhanced intent classifier.
        
        Args:
            context: Optional conversation context for context-aware classification
        """
        self.context = context or ConversationContext()
        self.compiled_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, float, float]]]:
        """Pre-compile regex patterns with confidence and weight"""
        compiled = {}
        for intent_type, config in self.INTENT_PATTERNS.items():
            compiled[intent_type] = [
                (re.compile(pattern, re.IGNORECASE), confidence, weight)
                for pattern, confidence, weight in config['patterns']
            ]
        return compiled
    
    def classify(self, message: str, use_context: bool = True) -> IntentV2:
        """
        Classify user message with multi-intent and context support.
        
        Args:
            message: User's natural language message
            use_context: Whether to use conversation context
            
        Returns:
            IntentV2 object with primary intent, secondary intents, and confidence
        """
        message = message.strip()
        
        # Collect all intent matches with their scores
        all_matches = []
        
        for intent_type, patterns in self.compiled_patterns.items():
            for pattern, base_confidence, weight in patterns:
                match = pattern.search(message)
                if match:
                    # Extract entities
                    entities = self._extract_entities(intent_type, match, message)
                    
                    # Calculate base score
                    adjusted_confidence = self._adjust_confidence(
                        base_confidence, entities, message, intent_type
                    )
                    
                    # Apply context boost if enabled
                    context_boost = 0.0
                    if use_context:
                        context_boost = self._calculate_context_boost(intent_type, message)
                    
                    # Final weighted score
                    final_score = (adjusted_confidence + context_boost) * weight
                    
                    all_matches.append({
                        'intent_type': intent_type,
                        'confidence': adjusted_confidence,
                        'context_boost': context_boost,
                        'final_score': final_score,
                        'entities': entities,
                        'pattern': pattern.pattern,
                        'weight': weight,
                    })
        
        # No matches - default to general_chat
        if not all_matches:
            return self._create_general_chat_intent(message)
        
        # Sort by final score
        all_matches.sort(key=lambda x: x['final_score'], reverse=True)
        
        # Get primary intent
        primary = all_matches[0]
        
        # Check if primary intent meets threshold
        if primary['confidence'] < self.PRIMARY_INTENT_THRESHOLD:
            return self._create_general_chat_intent(message)
        
        # Collect secondary intents
        secondary_intents = []
        matched_patterns = [primary['pattern']]
        
        for match in all_matches[1:]:
            if (match['confidence'] >= self.SECONDARY_INTENT_THRESHOLD and
                match['intent_type'] != primary['intent_type']):
                secondary_intents.append((match['intent_type'], match['confidence']))
                matched_patterns.append(match['pattern'])
        
        # Detect ambiguity
        is_ambiguous = False
        ambiguity_score = 0.0
        if len(all_matches) > 1:
            score_diff = primary['final_score'] - all_matches[1]['final_score']
            if score_diff < self.AMBIGUITY_RANGE:
                is_ambiguous = True
                ambiguity_score = max(0.0, 1.0 - (score_diff / self.AMBIGUITY_RANGE))
        
        # Update conversation context
        if use_context:
            self._update_context(primary['intent_type'], message, primary['entities'])
        
        return IntentV2(
            intent_type=primary['intent_type'],
            confidence=primary['confidence'],
            entities=primary['entities'],
            original_message=message,
            matched_patterns=matched_patterns,
            secondary_intents=secondary_intents[:3],  # Top 3 secondary intents
            is_ambiguous=is_ambiguous,
            ambiguity_score=ambiguity_score,
            context_boost=primary['context_boost'],
            conversation_context=self._get_context_summary(),
        )
    
    def classify_batch(self, messages: List[str]) -> List[IntentV2]:
        """Classify multiple messages with context awareness"""
        results = []
        for message in messages:
            intent = self.classify(message, use_context=True)
            results.append(intent)
        return results
    
    def _extract_entities(self, intent_type: str, match: re.Match, message: str) -> Dict[str, Any]:
        """Enhanced entity extraction with NER patterns"""
        entities = {}
        
        if intent_type == 'goal_creation':
            # Extract goal description and deadline
            groups = match.groups()
            
            # Handle "By X, I need to Y" pattern
            if message.lower().strip().startswith('by '):
                if len(groups) >= 2 and groups[1]:
                    entities['deadline'] = groups[0].strip()
                    entities['goal_description'] = groups[1].strip()
                elif len(groups) >= 1:
                    entities['deadline'] = groups[0].strip()
            else:
                # Normal pattern
                if len(groups) >= 1:
                    entities['goal_description'] = groups[0].strip()
                if len(groups) >= 2 and groups[1]:
                    entities['deadline'] = groups[1].strip()
            
            # Extract priority and urgency
            entities['priority'] = self._detect_priority(message)
            entities['urgent'] = self._detect_urgency(message)
            
            # Extract named entities (people, projects, etc.)
            entities['named_entities'] = self._extract_named_entities(message)
        
        elif intent_type == 'goal_query':
            groups = match.groups()
            if groups and groups[0]:
                entities['goal_reference'] = groups[0].strip()
            
            # Detect specific query type
            entities['query_subtype'] = self._detect_goal_query_subtype(message)
        
        elif intent_type == 'goal_update':
            groups = match.groups()
            # Extract goal reference (first group usually)
            if groups and groups[0]:
                entities['goal_reference'] = groups[0].strip()
            
            # Detect update action type
            entities['update_action'] = self._detect_goal_update_action(message)
            
            # Extract new values based on action type
            message_lower = message.lower()
            
            # Priority change - extract new priority
            if 'priority' in message_lower or any(p in message_lower for p in ['high', 'medium', 'low']):
                if 'high' in message_lower:
                    entities['new_priority'] = 'high'
                elif 'low' in message_lower:
                    entities['new_priority'] = 'low'
                else:
                    entities['new_priority'] = 'medium'
            
            # Deadline change - extract new deadline
            if len(groups) >= 2 and groups[1]:
                deadline_str = groups[1].strip()
                if deadline_str and deadline_str.lower() not in ['high', 'medium', 'low', 'done', 'complete', 'finished']:
                    entities['new_deadline'] = deadline_str
        
        elif intent_type == 'memory_query':
            groups = match.groups()
            if groups and groups[0]:
                entities['query_term'] = groups[0].strip()
            
            # Enhanced temporal and type detection
            entities['temporal_constraint'] = self._detect_temporal_constraint(message)
            entities['query_type'] = self._detect_memory_query_type(message)
            entities['memory_system'] = self._detect_target_memory_system(message)
        
        elif intent_type == 'performance_query':
            entities['metric_type'] = self._detect_metric_type(message)
            entities['time_range'] = self._detect_temporal_constraint(message)
        
        elif intent_type == 'system_status':
            entities['component'] = self._detect_system_component(message)
            entities['detail_level'] = self._detect_detail_level(message)
        elif intent_type == 'reminder_request':
            entities['reminder_action'] = self._detect_reminder_action(message)
            groups = match.groupdict()
            reminder_text = (
                groups.get('reminder_text')
                or groups.get('reminder_text_alt')
            )
            if reminder_text:
                entities['reminder_text'] = reminder_text.strip()
            time_hint = groups.get('reminder_time') or groups.get('reminder_time_only')
            offset = self._parse_time_offset_seconds(time_hint or message)
            if offset is not None:
                entities['reminder_offset_seconds'] = offset
            upcoming_window = self._parse_upcoming_window_seconds(message)
            if upcoming_window is not None:
                entities['reminder_upcoming_window_seconds'] = upcoming_window
        
        return entities
    
    def _adjust_confidence(self, base_confidence: float, entities: Dict, message: str, intent_type: str) -> float:
        """Enhanced confidence adjustment with keyword matching"""
        confidence = base_confidence
        
        # Boost for well-formed entities
        if entities.get('goal_description') and len(entities['goal_description']) > 10:
            confidence += 0.05
        
        if entities.get('deadline'):
            confidence += 0.05
        
        if entities.get('query_term') and len(entities['query_term']) > 3:
            confidence += 0.03
        
        # Message length adjustments
        word_count = len(message.split())
        if word_count < 3:
            confidence -= 0.15
        elif word_count > 50:
            confidence -= 0.05
        
        # Keyword matching boost
        config = self.INTENT_PATTERNS.get(intent_type, {})
        keywords = config.get('keywords', [])
        boost_words = config.get('boost_words', [])
        
        message_lower = message.lower()
        keyword_matches = sum(1 for kw in keywords if kw in message_lower)
        boost_matches = sum(1 for bw in boost_words if bw in message_lower)
        
        confidence += min(0.10, keyword_matches * 0.02)
        confidence += min(0.10, boost_matches * 0.03)
        
        # Question mark boost for query intents
        if '?' in message and intent_type in ['goal_query', 'memory_query', 'performance_query', 'system_status']:
            confidence += 0.05
        
        return min(1.0, max(0.0, confidence))
    
    def _calculate_context_boost(self, intent_type: str, message: str) -> float:
        """Calculate confidence boost based on conversation context"""
        boost = 0.0
        
        # Recent intent patterns
        if len(self.context.recent_intents) > 0:
            # Boost if similar to recent intents
            recent_count = self.context.recent_intents[-5:].count(intent_type)
            boost += recent_count * 0.02
        
        # Active goals boost goal queries
        if intent_type == 'goal_query' and len(self.context.active_goals) > 0:
            boost += 0.05
        
        # Recent memory queries boost follow-up memory queries
        if intent_type == 'memory_query' and self.context.last_query_type == 'memory_query':
            boost += 0.08
        
        # Topic continuity
        message_lower = message.lower()
        topic_matches = sum(1 for topic in self.context.recent_topics if topic.lower() in message_lower)
        boost += min(0.10, topic_matches * 0.03)
        
        return min(0.15, boost)  # Cap at 0.15
    
    def _update_context(self, intent_type: str, message: str, entities: Dict):
        """Update conversation context with new turn"""
        # Add intent to recent intents (keep last 10)
        self.context.recent_intents.append(intent_type)
        if len(self.context.recent_intents) > 10:
            self.context.recent_intents.pop(0)
        
        # Update last query type
        if intent_type in ['goal_query', 'memory_query', 'performance_query', 'system_status']:
            self.context.last_query_type = intent_type
        
        # Extract and add topics
        if entities.get('query_term'):
            self.context.recent_topics.append(entities['query_term'])
        if entities.get('goal_description'):
            self.context.recent_topics.append(entities['goal_description'])
        
        # Keep last 15 topics
        if len(self.context.recent_topics) > 15:
            self.context.recent_topics = self.context.recent_topics[-15:]
        
        # Update timestamp
        self.context.last_response_time = datetime.now()
    
    def _get_context_summary(self) -> Dict[str, Any]:
        """Get summary of conversation context"""
        return {
            'recent_intent_distribution': dict(Counter(self.context.recent_intents[-10:])),
            'active_goal_count': len(self.context.active_goals),
            'recent_topic_count': len(self.context.recent_topics),
            'last_query_type': self.context.last_query_type,
            'minutes_since_last_response': (
                (datetime.now() - self.context.last_response_time).total_seconds() / 60
                if self.context.last_response_time else None
            ),
        }
    
    def _create_general_chat_intent(self, message: str) -> IntentV2:
        """Create default general_chat intent"""
        return IntentV2(
            intent_type='general_chat',
            confidence=1.0,
            entities={},
            original_message=message,
            matched_patterns=[],
            conversation_context=self._get_context_summary(),
        )
    
    # Helper methods (reuse from v1 with enhancements)
    
    def _detect_priority(self, message: str) -> str:
        """Detect priority indicators"""
        message_lower = message.lower()
        if any(word in message_lower for word in ['critical', 'urgent', 'asap', 'immediately', 'emergency']):
            return 'high'
        elif any(word in message_lower for word in ['important', 'priority', 'soon']):
            return 'medium'
        elif any(word in message_lower for word in ['eventually', 'when possible', 'low priority', 'someday']):
            return 'low'
        return 'medium'
    
    def _detect_urgency(self, message: str) -> bool:
        """Detect urgency"""
        message_lower = message.lower()
        urgency_words = ['urgent', 'asap', 'immediately', 'right now', 'critical', 'emergency', 'quickly']
        return any(word in message_lower for word in urgency_words)
    
    def _detect_temporal_constraint(self, message: str) -> Optional[str]:
        """Enhanced temporal constraint detection"""
        message_lower = message.lower()
        
        temporal_patterns = [
            (r'last\s+(\d+)\s+(day|week|month|hour)s?', 'relative'),
            (r'past\s+(\d+)\s+(day|week|month|hour)s?', 'relative'),
            (r'yesterday', 'yesterday'),
            (r'last week', 'last_week'),
            (r'last month', 'last_month'),
            (r'recently', 'recent'),
            (r'this week', 'this_week'),
            (r'today', 'today'),
            (r'just now', 'just_now'),
            (r'earlier', 'earlier'),
        ]
        
        for pattern, constraint in temporal_patterns:
            if re.search(pattern, message_lower):
                return constraint
        
        return None
    
    def _detect_memory_query_type(self, message: str) -> str:
        """Detect memory query type"""
        message_lower = message.lower()
        
        episodic_indicators = ['discuss', 'talk about', 'conversation', 'said', 'told', 'when did', 'mention']
        semantic_indicators = ['know about', 'understand', 'explain', 'what is', 'tell me about', 'definition']
        
        episodic_score = sum(1 for ind in episodic_indicators if ind in message_lower)
        semantic_score = sum(1 for ind in semantic_indicators if ind in message_lower)
        
        if episodic_score > semantic_score:
            return 'episodic'
        elif semantic_score > episodic_score:
            return 'semantic'
        return 'general'
    
    def _detect_target_memory_system(self, message: str) -> str:
        """Detect which memory system to query"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['recent', 'just', 'moment ago', 'earlier', 'now']):
            return 'stm'
        elif any(word in message_lower for word in ['always', 'generally', 'usually', 'knowledge', 'fact']):
            return 'ltm'
        elif any(word in message_lower for word in ['when', 'discussion', 'conversation', 'event']):
            return 'episodic'
        
        return 'all'
    
    def _detect_metric_type(self, message: str) -> str:
        """Detect performance metric type"""
        message_lower = message.lower()
        
        if 'accuracy' in message_lower or 'accurate' in message_lower or 'correct' in message_lower:
            return 'accuracy'
        elif 'success' in message_lower:
            return 'success_rate'
        elif 'time' in message_lower or 'speed' in message_lower:
            return 'time_accuracy'
        elif 'improve' in message_lower or 'learn' in message_lower or 'better' in message_lower:
            return 'improvement'
        elif 'confidence' in message_lower:
            return 'confidence'
        
        return 'general'
    
    def _detect_system_component(self, message: str) -> str:
        """Detect system component"""
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
    
    def _detect_goal_query_subtype(self, message: str) -> str:
        """Detect specific type of goal query"""
        message_lower = message.lower()
        
        # List all goals - no specific goal mentioned
        list_all_patterns = [
            'show my goals', 'show me my goals', 'list my goals', 'list goals',
            'what are my goals', 'what goals', 'my goals', 'all goals',
            'check my goals', 'check goals', 'what am i working on',
            'do i have any goals', 'current goals', 'active goals'
        ]
        if any(pattern in message_lower for pattern in list_all_patterns):
            return 'list_all'
        
        # Execution progress queries
        execution_patterns = [
            'execution', 'executing', 'running', 'in progress',
            'what is happening', "what's happening", 'currently doing',
            'right now', 'being done', 'being executed'
        ]
        if any(pattern in message_lower for pattern in execution_patterns):
            return 'execution_progress'
        
        if any(word in message_lower for word in ['finished', 'complete', 'done']):
            return 'completion_status'
        elif any(word in message_lower for word in ['progress', 'how far', 'percentage']):
            return 'progress'
        elif any(word in message_lower for word in ['status', 'update', 'happening']):
            return 'status'
        elif any(word in message_lower for word in ['when', 'deadline', 'due']):
            return 'timeline'
        
        return 'general'
    
    def _detect_goal_update_action(self, message: str) -> str:
        """Detect the type of goal update action"""
        message_lower = message.lower()
        
        # Complete/done
        if any(word in message_lower for word in ['complete', 'done', 'finished', 'completed']):
            return 'complete'
        
        # Cancel/delete
        if any(word in message_lower for word in ['cancel', 'delete', 'remove', 'forget', 'nevermind', 'drop', 'skip']):
            return 'cancel'
        
        # Priority change
        if 'priority' in message_lower or any(
            f'{p} priority' in message_lower or f'priority {p}' in message_lower 
            for p in ['high', 'medium', 'low']
        ):
            return 'priority_change'
        
        # Deadline change
        if any(word in message_lower for word in ['deadline', 'extend', 'postpone', 'push back', 'move']):
            return 'deadline_change'
        
        return 'unknown'
    
    def _detect_detail_level(self, message: str) -> str:
        """Detect desired detail level"""
        message_lower = message.lower()
        
        if any(word in message_lower for word in ['detailed', 'full', 'complete', 'everything']):
            return 'detailed'
        elif any(word in message_lower for word in ['quick', 'brief', 'summary', 'short']):
            return 'brief'
        
        return 'normal'

    def _detect_reminder_action(self, message: str) -> str:
        """Determine whether the reminder intent is create, list, or due."""
        message_lower = message.lower()
        list_phrases = [
            "list",
            "show",
            "what reminders",
            "which reminders",
            "what are my reminders",
            "do i have",
        ]
        if any(phrase in message_lower for phrase in list_phrases):
            return 'list'
        if 'due' in message_lower or 'upcoming' in message_lower:
            return 'due'
        return 'create'

    def _parse_time_offset_seconds(self, text: Optional[str]) -> Optional[float]:
        """Parse relative time expressions into seconds."""
        if not text:
            return None
        lower = text.lower()
        match = re.search(r"in\s+(?P<num>\d+)\s+(?P<unit>minute|minutes|hour|hours|day|days)", lower)
        if not match:
            match = re.search(r"(?P<num>\d+)\s+(?P<unit>minute|minutes|hour|hours|day|days)\s+from now", lower)
        if not match:
            match = re.search(r"(?P<num>\d+)\s+(?P<unit>minute|minutes|hour|hours|day|days)", lower)
        if match:
            num = int(match.group('num'))
            unit = match.group('unit')
            multiplier = 60 if 'minute' in unit else 3600 if 'hour' in unit else 86400
            return float(num * multiplier)
        if 'tomorrow' in lower:
            return 86400.0
        if 'next week' in lower:
            return 7 * 86400.0
        if 'tonight' in lower:
            now = datetime.now()
            target = now.replace(hour=21, minute=0, second=0, microsecond=0)
            if target <= now:
                target += timedelta(days=1)
            return (target - now).total_seconds()
        return None

    def _parse_upcoming_window_seconds(self, message: str) -> Optional[float]:
        """Extract upcoming reminder window requests."""
        lower = message.lower()
        match = re.search(r"within\s+(?P<num>\d+)\s+(?P<unit>minute|minutes|hour|hours|day|days)", lower)
        if not match:
            return None
        num = int(match.group('num'))
        unit = match.group('unit')
        multiplier = 60 if 'minute' in unit else 3600 if 'hour' in unit else 86400
        return float(num * multiplier)
    
    def _extract_named_entities(self, message: str) -> List[str]:
        """Simple named entity extraction (capitalized words/phrases)"""
        # Simple heuristic: consecutive capitalized words (except first word)
        words = message.split()
        entities = []
        current_entity = []
        
        for i, word in enumerate(words):
            # Skip first word and common words
            if i == 0 or word.lower() in ['i', 'the', 'a', 'an', 'and', 'or', 'but']:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
                continue
            
            # Check if capitalized
            if word[0].isupper():
                current_entity.append(word)
            else:
                if current_entity:
                    entities.append(' '.join(current_entity))
                    current_entity = []
        
        if current_entity:
            entities.append(' '.join(current_entity))
        
        return entities


def create_intent_classifier_v2(context: Optional[ConversationContext] = None) -> IntentClassifierV2:
    """Factory function to create enhanced intent classifier"""
    return IntentClassifierV2(context=context)
