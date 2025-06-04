"""
Attention Mechanism - Manages cognitive focus and resource allocation
"""
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import logging
import math

logger = logging.getLogger(__name__)

@dataclass
class AttentionItem:
    """Individual item in attention focus"""
    id: str
    content: Any
    salience: float  # How attention-grabbing (0.0 to 1.0)
    activation: float  # Current activation level (0.0 to 1.0)
    created_at: datetime
    last_updated: datetime
    priority: float = 0.5  # Task priority (0.0 to 1.0)
    effort_required: float = 0.5  # Cognitive effort needed (0.0 to 1.0)
    duration_seconds: float = 0.0  # How long it's been in focus
    
    def update_activation(self, delta: float):
        """Update activation level"""
        self.activation = max(0.0, min(1.0, self.activation + delta))
        self.last_updated = datetime.now()
    
    def age_seconds(self) -> float:
        """Get age in seconds"""
        return (datetime.now() - self.created_at).total_seconds()

class AttentionMechanism:
    """
    Attention mechanism for cognitive resource allocation
    
    Features:
    - Salience-based attention allocation
    - Fatigue accumulation and recovery
    - Multi-item attention tracking
    - Resource competition modeling
    - Adaptive focus management
    """
    
    def __init__(
        self,
        max_attention_items: int = 7,  # Miller's magical number
        salience_threshold: float = 0.3,
        fatigue_decay_rate: float = 0.1,
        attention_recovery_rate: float = 0.05,
        novelty_boost: float = 0.2
    ):
        """
        Initialize attention mechanism
        
        Args:
            max_attention_items: Maximum items that can be in focus
            salience_threshold: Minimum salience to capture attention
            fatigue_decay_rate: Rate at which fatigue accumulates
            attention_recovery_rate: Rate of attention recovery during rest
            novelty_boost: Boost given to novel/unexpected inputs
        """
        self.max_attention_items = max_attention_items
        self.salience_threshold = salience_threshold
        self.fatigue_decay_rate = fatigue_decay_rate
        self.attention_recovery_rate = attention_recovery_rate
        self.novelty_boost = novelty_boost
        
        # Attention state
        self.focused_items: Dict[str, AttentionItem] = {}
        self.attention_history: List[Dict[str, Any]] = []
        
        # Fatigue and resource management
        self.current_fatigue = 0.0  # 0.0 to 1.0
        self.total_cognitive_load = 0.0
        self.attention_capacity = 1.0  # Total attention capacity
        
        # Performance tracking
        self.last_update = datetime.now()
        self.focus_switches = 0
        self.attention_episodes = []
        
        logger.info("Attention mechanism initialized")
    
    def allocate_attention(
        self,
        stimulus_id: str,
        content: Any,
        salience: float,
        novelty: float = 0.0,
        priority: float = 0.5,
        effort_required: float = 0.5
    ) -> Dict[str, Any]:
        """
        Allocate attention to a stimulus
        
        Args:
            stimulus_id: Unique identifier for the stimulus
            content: Stimulus content
            salience: Base salience score (0.0 to 1.0)
            novelty: Novelty score (0.0 to 1.0)
            priority: Task priority (0.0 to 1.0)
            effort_required: Cognitive effort needed (0.0 to 1.0)
        
        Returns:
            Dictionary with attention allocation results
        """
        # Calculate effective salience with novelty boost
        effective_salience = min(1.0, salience + (novelty * self.novelty_boost))
        
        # Check if stimulus meets attention threshold
        if effective_salience < self.salience_threshold:
            logger.debug(f"Stimulus {stimulus_id} below attention threshold ({effective_salience:.3f})")
            return {
                "allocated": False,
                "salience": effective_salience,
                "reason": "below_threshold"
            }
        
        # Check cognitive capacity
        if self._is_capacity_exceeded(effort_required):
            # Try to make room by removing low-priority items
            self._manage_capacity(priority, effort_required)
        
        # Calculate attention allocation
        attention_score = self._calculate_attention_score(
            effective_salience, priority, effort_required
        )
        
        # Create or update attention item
        if stimulus_id in self.focused_items:
            item = self.focused_items[stimulus_id]
            item.update_activation(attention_score * 0.1)
            item.priority = max(item.priority, priority)
        else:
            # Add new item to focus
            item = AttentionItem(
                id=stimulus_id,
                content=content,
                salience=effective_salience,
                activation=attention_score,
                created_at=datetime.now(),
                last_updated=datetime.now(),
                priority=priority,
                effort_required=effort_required
            )
            self.focused_items[stimulus_id] = item
            self.focus_switches += 1
        
        # Update cognitive load and fatigue
        self._update_cognitive_state(effort_required)
        
        # Log attention allocation
        self._log_attention_event(stimulus_id, effective_salience, attention_score)
        
        return {
            "allocated": True,
            "attention_score": attention_score,
            "effective_salience": effective_salience,
            "current_load": self.total_cognitive_load,
            "fatigue_level": self.current_fatigue,
            "items_in_focus": len(self.focused_items)
        }
    
    def update_attention_state(self, time_delta_seconds: Optional[float] = None) -> Dict[str, Any]:
        """
        Update attention state over time
        
        Args:
            time_delta_seconds: Time elapsed since last update
        
        Returns:
            Updated attention state information
        """
        if time_delta_seconds is None:
            time_delta_seconds = (datetime.now() - self.last_update).total_seconds()
        
        # Decay attention activations over time
        items_to_remove = []
        for item_id, item in self.focused_items.items():
            # Attention naturally decays
            decay_rate = 0.1 + (self.current_fatigue * 0.2)  # Fatigue increases decay
            decay = decay_rate * time_delta_seconds / 60.0  # Per minute
            
            item.update_activation(-decay)
            item.duration_seconds += time_delta_seconds
            
            # Remove items that fall below activation threshold
            if item.activation < 0.1:
                items_to_remove.append(item_id)
        
        # Remove items that lost attention
        for item_id in items_to_remove:
            self._remove_from_focus(item_id)
        
        # Recover from fatigue during low-load periods
        if self.total_cognitive_load < 0.3:
            recovery = self.attention_recovery_rate * time_delta_seconds / 60.0
            self.current_fatigue = max(0.0, self.current_fatigue - recovery)
        
        # Update cognitive load based on current focus
        self._recalculate_cognitive_load()
        
        self.last_update = datetime.now()
        
        return self.get_attention_status()
    
    def get_attention_focus(self) -> List[Dict[str, Any]]:
        """Get current attention focus items"""
        focus_items = []
        for item in sorted(self.focused_items.values(), 
                          key=lambda x: x.activation, reverse=True):
            focus_items.append({
                "id": item.id,
                "salience": item.salience,
                "activation": item.activation,
                "priority": item.priority,
                "effort_required": item.effort_required,
                "duration_seconds": item.duration_seconds,
                "age_seconds": item.age_seconds()
            })
        return focus_items
    
    def get_attention_status(self) -> Dict[str, Any]:
        """Get comprehensive attention status"""
        return {
            "focused_items": len(self.focused_items),
            "max_capacity": self.max_attention_items,
            "cognitive_load": self.total_cognitive_load,
            "fatigue_level": self.current_fatigue,
            "attention_capacity": self.attention_capacity,
            "available_capacity": max(0.0, self.attention_capacity - self.total_cognitive_load),
            "focus_switches": self.focus_switches,
            "salience_threshold": self.salience_threshold,
            "last_update": self.last_update,
            "current_focus": self.get_attention_focus()
        }
    
    def rest_attention(self, duration_minutes: float = 1.0) -> Dict[str, float]:
        """
        Simulate attention rest/recovery period
        
        Args:
            duration_minutes: Rest duration in minutes
        
        Returns:
            Recovery metrics
        """
        initial_fatigue = self.current_fatigue
        initial_load = self.total_cognitive_load
        
        # Accelerated recovery during rest
        recovery_amount = self.attention_recovery_rate * duration_minutes * 2.0
        self.current_fatigue = max(0.0, self.current_fatigue - recovery_amount)
        
        # Reduce cognitive load as attention naturally disperses
        load_reduction = 0.1 * duration_minutes
        self.total_cognitive_load = max(0.0, self.total_cognitive_load - load_reduction)
        
        # Some items may lose focus during rest
        items_lost = 0
        for item_id, item in list(self.focused_items.items()):
            if item.priority < 0.7:  # Low priority items fade during rest
                decay = 0.2 * duration_minutes
                item.update_activation(-decay)
                if item.activation < 0.2:
                    self._remove_from_focus(item_id)
                    items_lost += 1
        
        logger.info(f"Attention rest: {duration_minutes}min, "
                   f"fatigue {initial_fatigue:.3f}->{self.current_fatigue:.3f}, "
                   f"load {initial_load:.3f}->{self.total_cognitive_load:.3f}")
        
        return {
            "duration_minutes": duration_minutes,
            "fatigue_reduction": initial_fatigue - self.current_fatigue,
            "load_reduction": initial_load - self.total_cognitive_load,
            "items_lost_focus": items_lost
        }
    
    def _calculate_attention_score(
        self, 
        salience: float, 
        priority: float, 
        effort_required: float
    ) -> float:
        """Calculate overall attention score"""
        # Base score from salience and priority
        base_score = (salience * 0.6) + (priority * 0.4)
        
        # Adjust for current cognitive state
        fatigue_penalty = self.current_fatigue * 0.3
        load_penalty = (self.total_cognitive_load / self.attention_capacity) * 0.2
        
        # Effort consideration (harder tasks get less attention when fatigued)
        effort_penalty = effort_required * fatigue_penalty
        
        final_score = base_score - fatigue_penalty - load_penalty - effort_penalty
        return max(0.0, min(1.0, final_score))
    
    def _is_capacity_exceeded(self, additional_effort: float) -> bool:
        """Check if adding item would exceed cognitive capacity"""
        projected_load = self.total_cognitive_load + additional_effort
        return (len(self.focused_items) >= self.max_attention_items or 
                projected_load > self.attention_capacity)
    
    def _manage_capacity(self, new_priority: float, new_effort: float):
        """Manage attention capacity by removing low-priority items"""
        if not self.focused_items:
            return
        
        # Sort by priority and activation (lowest first)
        sorted_items = sorted(
            self.focused_items.items(),
            key=lambda x: (x[1].priority, x[1].activation)
        )
        
        # Remove lowest priority items until there's room
        for item_id, item in sorted_items:
            if (item.priority < new_priority and 
                len(self.focused_items) > 0):
                self._remove_from_focus(item_id)
                logger.debug(f"Removed {item_id} from focus for capacity management")
                
                # Check if we have enough capacity now
                if not self._is_capacity_exceeded(new_effort):
                    break
    
    def _update_cognitive_state(self, effort_required: float):
        """Update cognitive load and fatigue"""
        # Increase fatigue based on effort
        fatigue_increase = effort_required * self.fatigue_decay_rate * 0.1
        self.current_fatigue = min(1.0, self.current_fatigue + fatigue_increase)
        
        # Recalculate total cognitive load
        self._recalculate_cognitive_load()
    
    def _recalculate_cognitive_load(self):
        """Recalculate total cognitive load from current focus items"""
        total_load = 0.0
        for item in self.focused_items.values():
            # Load is effort weighted by activation
            item_load = item.effort_required * item.activation
            total_load += item_load
        
        self.total_cognitive_load = total_load
    
    def _remove_from_focus(self, item_id: str):
        """Remove item from attention focus"""
        if item_id in self.focused_items:
            item = self.focused_items[item_id]
            
            # Log attention episode
            episode = {
                "item_id": item_id,
                "start_time": item.created_at,
                "end_time": datetime.now(),
                "duration_seconds": item.duration_seconds,
                "final_activation": item.activation,
                "salience": item.salience,
                "priority": item.priority
            }
            self.attention_episodes.append(episode)
            
            del self.focused_items[item_id]
            logger.debug(f"Removed {item_id} from attention focus")
    
    def _log_attention_event(self, stimulus_id: str, salience: float, attention_score: float):
        """Log attention allocation event"""
        event = {
            "timestamp": datetime.now(),
            "stimulus_id": stimulus_id,
            "salience": salience,
            "attention_score": attention_score,
            "cognitive_load": self.total_cognitive_load,
            "fatigue_level": self.current_fatigue,
            "items_in_focus": len(self.focused_items)
        }
        self.attention_history.append(event)
        
        # Keep history manageable
        if len(self.attention_history) > 100:
            self.attention_history = self.attention_history[-50:]
