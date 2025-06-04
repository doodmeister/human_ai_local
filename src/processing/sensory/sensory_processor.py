"""
Minimal Sensory Processing Module for testing
"""

import numpy as np
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class SensoryInput:
    """Represents a sensory input with metadata"""
    content: str
    modality: str = 'text'
    timestamp: float = None
    source: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ProcessedSensoryData:
    """Represents processed sensory data with embeddings and scores"""
    input_data: SensoryInput
    embedding: np.ndarray
    entropy_score: float
    salience_score: float
    relevance_score: float
    filtered: bool = False
    processing_metadata: Optional[Dict] = None

class SensoryProcessor:
    """Minimal sensory processor for testing"""
    
    def __init__(self):
        self.processing_history = []
        
    def process_input(self, sensory_input: SensoryInput) -> ProcessedSensoryData:
        """Process a sensory input"""
        content = str(sensory_input.content) if sensory_input.content else ""
        
        # Simple scoring
        entropy_score = min(1.0, len(set(content)) / max(1, len(content)))
        salience_score = 0.5 + 0.3 * ('!' in content or '?' in content)
        relevance_score = min(1.0, len(content.split()) / 20.0)
        
        # Simple embedding (random for now)
        embedding = np.random.random(384).astype(np.float32)
        
        # Simple filtering
        filtered = entropy_score < 0.1 or len(content.strip()) == 0
        
        processed = ProcessedSensoryData(
            input_data=sensory_input,
            embedding=embedding,
            entropy_score=entropy_score,
            salience_score=salience_score,
            relevance_score=relevance_score,
            filtered=filtered
        )
        
        self.processing_history.append(processed)
        return processed
    
    def process_batch(self, inputs: List[SensoryInput]) -> List[ProcessedSensoryData]:
        """Process a batch of inputs"""
        return [self.process_input(inp) for inp in inputs]
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        if not self.processing_history:
            return {'total_processed': 0, 'filtered_count': 0}
        
        return {
            'total_processed': len(self.processing_history),
            'filtered_count': sum(1 for p in self.processing_history if p.filtered),
            'avg_scores': {
                'entropy': np.mean([p.entropy_score for p in self.processing_history]),
                'salience': np.mean([p.salience_score for p in self.processing_history]),
                'relevance': np.mean([p.relevance_score for p in self.processing_history])
            }
        }
