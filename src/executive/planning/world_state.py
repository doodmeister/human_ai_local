"""
World State Representation for GOAP

Represents the state of the world as key-value pairs for planning.
States are immutable snapshots that can be efficiently compared and hashed.
"""

from typing import Dict, Any, Set, Tuple
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldState:
    """
    Immutable world state represented as key-value pairs
    
    Used in GOAP planning to represent:
    - Current state (what is true now)
    - Goal state (what we want to be true)
    - Preconditions (what must be true for an action)
    - Effects (what becomes true after an action)
    
    Attributes:
        state: Immutable mapping of state variables to values
        
    Examples:
        >>> state = WorldState({'has_data': True, 'analyzed': False})
        >>> goal = WorldState({'analyzed': True})
        >>> state.satisfies(goal)
        False
    """
    
    _state: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        # Convert dict to frozenset of tuples for immutability and hashing
        # Handle unhashable values by converting to string
        hashable_items = []
        for k, v in self._state.items():
            try:
                hash(v)
                hashable_items.append((k, v))
            except TypeError:
                # Value is unhashable (list, dict, etc.), convert to string
                hashable_items.append((k, str(v)))
        
        object.__setattr__(self, '_hashable_state', frozenset(hashable_items))
    
    @property
    def state(self) -> Dict[str, Any]:
        """Public accessor for state dictionary"""
        return self._state
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get value for a state variable"""
        return self._state.get(key, default)
    
    def has(self, key: str) -> bool:
        """Check if state variable exists"""
        return key in self._state
    
    def set(self, key: str, value: Any) -> 'WorldState':
        """
        Create new WorldState with updated value (immutable)
        
        Args:
            key: State variable to set
            value: New value
            
        Returns:
            New WorldState with updated value
        """
        new_state = dict(self._state)
        new_state[key] = value
        return WorldState(new_state)
    
    def update(self, updates: Dict[str, Any]) -> 'WorldState':
        """
        Create new WorldState with multiple updates (immutable)
        
        Args:
            updates: Dictionary of updates to apply
            
        Returns:
            New WorldState with all updates applied
        """
        new_state = dict(self._state)
        new_state.update(updates)
        return WorldState(new_state)
    
    def satisfies(self, goal: 'WorldState') -> bool:
        """
        Check if this state satisfies a goal state
        
        A state satisfies a goal if all goal conditions are met.
        Goal may be partial (not all variables need to be specified).
        
        Args:
            goal: Goal state to check against
            
        Returns:
            True if all goal conditions are satisfied
            
        Examples:
            >>> current = WorldState({'a': 1, 'b': 2, 'c': 3})
            >>> goal = WorldState({'a': 1, 'c': 3})
            >>> current.satisfies(goal)
            True
        """
        for key, value in goal._state.items():
            if self.get(key) != value:
                return False
        return True
    
    def delta(self, other: 'WorldState') -> Dict[str, Tuple[Any, Any]]:
        """
        Calculate differences between this state and another
        
        Args:
            other: State to compare against
            
        Returns:
            Dictionary mapping keys to (this_value, other_value) tuples
            for all keys that differ
            
        Examples:
            >>> s1 = WorldState({'a': 1, 'b': 2})
            >>> s2 = WorldState({'a': 1, 'b': 3, 'c': 4})
            >>> s1.delta(s2)
            {'b': (2, 3), 'c': (None, 4)}
        """
        all_keys = set(self._state.keys()) | set(other._state.keys())
        differences = {}
        
        for key in all_keys:
            this_val = self.get(key)
            other_val = other.get(key)
            if this_val != other_val:
                differences[key] = (this_val, other_val)
        
        return differences
    
    def distance_to(self, goal: 'WorldState') -> int:
        """
        Calculate simple distance to goal state (number of differing values)
        
        This is a basic heuristic for A* search.
        
        Args:
            goal: Goal state
            
        Returns:
            Number of state variables that differ from goal
        """
        count = 0
        for key, value in goal._state.items():
            if self.get(key) != value:
                count += 1
        return count
    
    def keys(self) -> Set[str]:
        """Get all state variable keys"""
        return set(self._state.keys())
    
    def items(self) -> Dict[str, Any]:
        """Get state as dictionary (for iteration)"""
        return dict(self._state)
    
    def __len__(self) -> int:
        """Number of state variables"""
        return len(self._state)
    
    def __contains__(self, key: str) -> bool:
        """Check if state variable exists"""
        return key in self._state
    
    def __hash__(self) -> int:
        """Hash for use in sets and dicts (required for A* closed set)"""
        return hash(self._hashable_state)
    
    def __eq__(self, other: object) -> bool:
        """Equality comparison"""
        if not isinstance(other, WorldState):
            return False
        return self._state == other._state
    
    def __repr__(self) -> str:
        """String representation"""
        items = ', '.join(f'{k}={v}' for k, v in sorted(self._state.items()))
        return f'WorldState({{{items}}})'
    
    def __str__(self) -> str:
        """Human-readable string"""
        return repr(self)


def merge_states(base: WorldState, *overlays: WorldState) -> WorldState:
    """
    Merge multiple world states (later states override earlier)
    
    Args:
        base: Base state
        *overlays: States to overlay on base (right-to-left precedence)
        
    Returns:
        New WorldState with merged values
        
    Examples:
        >>> s1 = WorldState({'a': 1, 'b': 2})
        >>> s2 = WorldState({'b': 3, 'c': 4})
        >>> merge_states(s1, s2)
        WorldState({a=1, b=3, c=4})
    """
    merged = dict(base._state)
    for overlay in overlays:
        merged.update(overlay._state)
    return WorldState(merged)
