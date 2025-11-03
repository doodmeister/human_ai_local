"""
Prospective Memory Module

Unified interface for managing future intentions, reminders, and scheduled tasks.
"""

from .prospective_memory import (
    # Data models
    Reminder,
    
    # Base interface
    ProspectiveMemorySystem,
    
    # Implementations
    InMemoryProspectiveMemory,
    VectorProspectiveMemory,
    
    # Factory functions
    create_prospective_memory,
    get_prospective_memory,
    reset_prospective_memory,
    
    # Backward compatibility
    ProspectiveMemory,
    ProspectiveMemoryVectorStore,
    get_inmemory_prospective_memory,
)

__all__ = [
    "Reminder",
    "ProspectiveMemorySystem",
    "InMemoryProspectiveMemory",
    "VectorProspectiveMemory",
    "create_prospective_memory",
    "get_prospective_memory",
    "reset_prospective_memory",
    # Backward compatibility
    "ProspectiveMemory",
    "ProspectiveMemoryVectorStore",
    "get_inmemory_prospective_memory",
]
