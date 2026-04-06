"""
Minimal Sensory Interface for testing
"""

from .sensory_processor import SensoryInput, SensoryProcessor, ProcessedSensoryData
import time
from typing import Any, Dict, List, Optional, Union


_DEFAULT_QUICK_PROCESSOR: Optional[SensoryProcessor] = None

class SensoryInterface:
    """High-level interface for sensory processing"""
    
    def __init__(self, processor: Optional[SensoryProcessor] = None):
        self.processor = processor or SensoryProcessor()
    
    def create_input(
        self,
        content: str,
        modality: str = 'text',
        source: Optional[str] = 'user',
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SensoryInput:
        """Create a sensory input with explicit field control."""
        return SensoryInput(
            content=content,
            modality=modality,
            timestamp=time.time() if timestamp is None else timestamp,
            source=source,
            metadata=dict(metadata or {}),
        )

    def create_text_input(
        self,
        text: str,
        source: str = 'user',
        timestamp: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SensoryInput:
        """Create a text input."""
        return self.create_input(
            content=text,
            modality='text',
            source=source,
            timestamp=timestamp,
            metadata=metadata,
        )

    def _create_input_from_dict(self, raw_input: Dict[str, Any]) -> SensoryInput:
        content = raw_input.get('content', str(raw_input))
        metadata = dict(raw_input.get('metadata') or {})
        extra_metadata = {
            key: value
            for key, value in raw_input.items()
            if key not in {'content', 'modality', 'timestamp', 'source', 'metadata'}
        }
        for key, value in extra_metadata.items():
            metadata.setdefault(key, value)

        return self.create_input(
            content=content,
            modality=raw_input.get('modality', 'text'),
            source=raw_input.get('source', 'user'),
            timestamp=raw_input.get('timestamp'),
            metadata=metadata,
        )
    
    def process_user_input(self, text: str) -> ProcessedSensoryData:
        """Process user input"""
        sensory_input = self.create_text_input(text, 'user')
        return self.processor.process_input(sensory_input)
    
    def process_batch_inputs(self, inputs: List[Union[str, Dict[str, Any]]]) -> List[ProcessedSensoryData]:
        """Process batch inputs"""
        sensory_inputs = []
        for inp in inputs:
            if isinstance(inp, str):
                sensory_inputs.append(self.create_text_input(inp))
            else:
                sensory_inputs.append(self._create_input_from_dict(inp))
        
        return self.processor.process_batch(sensory_inputs)

class SensoryInputBuilder:
    """Builder for sensory inputs"""
    
    def __init__(self):
        self.reset()
    
    def reset(self) -> 'SensoryInputBuilder':
        self._content = ""
        self._modality = "text"
        self._source = 'user'
        self._timestamp = None
        self._metadata: Dict[str, Any] = {}
        return self
    
    def content(self, content: str) -> 'SensoryInputBuilder':
        self._content = content
        return self
    
    def modality(self, modality: str) -> 'SensoryInputBuilder':
        self._modality = modality
        return self
    
    def source(self, source: Optional[str]) -> 'SensoryInputBuilder':
        self._source = source
        return self

    def timestamp(self, timestamp: Optional[float]) -> 'SensoryInputBuilder':
        self._timestamp = timestamp
        return self

    def metadata(self, metadata: Optional[Dict[str, Any]]) -> 'SensoryInputBuilder':
        self._metadata = dict(metadata or {})
        return self
    
    def build(self) -> SensoryInput:
        return SensoryInput(
            content=self._content,
            modality=self._modality,
            timestamp=self._timestamp,
            source=self._source,
            metadata=dict(self._metadata),
        )


def _get_default_quick_processor() -> SensoryProcessor:
    global _DEFAULT_QUICK_PROCESSOR
    if _DEFAULT_QUICK_PROCESSOR is None:
        _DEFAULT_QUICK_PROCESSOR = SensoryProcessor()
    return _DEFAULT_QUICK_PROCESSOR


def quick_text_input(
    text: str,
    source: str = 'user',
    timestamp: Optional[float] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> SensoryInput:
    """Quick text input creation."""
    return SensoryInput(
        content=text,
        modality='text',
        timestamp=time.time() if timestamp is None else timestamp,
        source=source,
        metadata=dict(metadata or {}),
    )

def quick_process_text(text: str, processor: Optional[SensoryProcessor] = None) -> ProcessedSensoryData:
    """Quick text processing using the provided or shared default processor."""
    processor = processor or _get_default_quick_processor()
    return processor.process_input(quick_text_input(text))
