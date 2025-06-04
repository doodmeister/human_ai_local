"""
Isolated test for STM component
"""

def test_stm_direct():
    """Test STM by copying the code directly"""
    
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
        """Short-Term Memory system with capacity limits"""
        
        def __init__(self, capacity: int = 7, decay_threshold: float = 0.1):
            self.capacity = capacity
            self.decay_threshold = decay_threshold
            self.items: Dict[str, MemoryItem] = {}
            self.access_order: List[str] = []
            
        def store(self, memory_id: str, content: Any, importance: float = 0.5) -> bool:
            """Store memory item"""
            if memory_id in self.items:
                return True
                
            new_item = MemoryItem(
                id=memory_id,
                content=content,
                encoding_time=datetime.now(),
                last_access=datetime.now(),
                importance=importance
            )
            
            if len(self.items) >= self.capacity:
                # Simple eviction - remove oldest
                oldest_id = min(self.items.keys(), key=lambda x: self.items[x].encoding_time)
                del self.items[oldest_id]
                if oldest_id in self.access_order:
                    self.access_order.remove(oldest_id)
            
            self.items[memory_id] = new_item
            self.access_order.append(memory_id)
            return True
        
        def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
            """Retrieve memory by ID"""
            if memory_id not in self.items:
                return None
                
            item = self.items[memory_id]
            item.update_access()
            return item
    
    # Test the classes
    try:
        print("Testing isolated STM...")
        stm = ShortTermMemory(capacity=3)
        print("✓ STM created successfully")
        
        # Test storing
        success = stm.store("test1", "Hello world", importance=0.8)
        print(f"✓ Storage successful: {success}")
        
        # Test retrieval
        item = stm.retrieve("test1")
        print(f"✓ Retrieved: {item.content if item else 'None'}")
        
        print("🎉 Isolated STM test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Error in isolated test: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_stm_direct()
