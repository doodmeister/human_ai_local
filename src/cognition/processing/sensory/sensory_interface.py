"""
Minimal Sensory Interface for testing
"""

from .sensory_processor import SensoryInput, SensoryProcessor, ProcessedSensoryData
import time
from typing import List, Dict, Optional, Union

class SensoryInterface:
    """High-level interface for sensory processing"""
    
    def __init__(self, processor: Optional[SensoryProcessor] = None):
        self.processor = processor or SensoryProcessor()
    
    def create_text_input(self, text: str, source: str = 'user') -> SensoryInput:
        """Create a text input"""
        return SensoryInput(
            content=text,
            modality='text',
            timestamp=time.time(),
            source=source
        )
    
    def process_user_input(self, text: str) -> ProcessedSensoryData:
        """Process user input"""
        sensory_input = self.create_text_input(text, 'user')
        return self.processor.process_input(sensory_input)
    
    def process_batch_inputs(self, inputs: List[Union[str, Dict]]) -> List[ProcessedSensoryData]:
        """Process batch inputs"""
        sensory_inputs = []
        for inp in inputs:
            if isinstance(inp, str):
                sensory_inputs.append(self.create_text_input(inp))
            else:
                content = inp.get('content', str(inp))
                sensory_inputs.append(self.create_text_input(content))
        
        return self.processor.process_batch(sensory_inputs)

class SensoryInputBuilder:
    """Builder for sensory inputs"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self._content = ""
        self._modality = "text"
        self._source = None
        return self
    
    def content(self, content):
        self._content = content
        return self
    
    def modality(self, modality):
        self._modality = modality
        return self
    
    def source(self, source):
        self._source = source
        return self
    
    def build(self) -> SensoryInput:
        return SensoryInput(
            content=self._content,
            modality=self._modality,
            source=self._source
        )

def quick_text_input(text: str, source: str = 'user') -> SensoryInput:
    """Quick text input creation"""
    return SensoryInput(content=text, modality='text', source=source)

def quick_process_text(text: str, processor: Optional[SensoryProcessor] = None) -> ProcessedSensoryData:
    """Quick text processing"""
    if processor is None:
        processor = SensoryProcessor()
    return processor.process_input(quick_text_input(text))
