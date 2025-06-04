"""
Core configuration settings for the Human-AI Cognition Framework
"""
import os
from typing import Optional
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class MemoryConfig:
    """Configuration for memory systems"""
    # Short-Term Memory
    stm_capacity: int = 7  # Miller's magical number
    stm_decay_threshold: float = 0.1
    stm_decay_minutes: int = 60
    
    # Long-Term Memory
    ltm_storage_path: Optional[str] = None  # Will use default if None
    ltm_similarity_threshold: float = 0.7
    ltm_max_results: int = 10
    
    # Consolidation
    consolidation_interval_hours: int = 8
    consolidation_importance_threshold: float = 0.6

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms"""
    fatigue_decay_rate: float = 0.1
    attention_recovery_rate: float = 0.05
    salience_threshold: float = 0.5
    max_attention_items: int = 7  # Miller's magical number

@dataclass
class ProcessingConfig:
    """Configuration for cognitive processing"""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    entropy_threshold: float = 0.3
    batch_size: int = 32

@dataclass
class AWSConfig:
    """Configuration for AWS services"""
    region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    opensearch_endpoint: Optional[str] = None
    
@dataclass
class CognitiveConfig:
    """Main configuration class for the cognitive system"""
    # Core settings
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent)
    data_dir: Path = field(init=False)
    model_dir: Path = field(init=False)
    
    # Component configurations
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    
    # Runtime settings
    debug_mode: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    
    def __post_init__(self):
        """Initialize derived paths"""
        self.data_dir = self.project_root / "data"
        self.model_dir = self.project_root / "model"
        
        # Create directories if they don't exist
        self.data_dir.mkdir(exist_ok=True)
        self.model_dir.mkdir(exist_ok=True)
    
    @classmethod
    def from_env(cls) -> "CognitiveConfig":
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables if present
        if aws_region := os.getenv("AWS_REGION"):
            config.aws.region = aws_region
            
        if opensearch_endpoint := os.getenv("OPENSEARCH_ENDPOINT"):
            config.aws.opensearch_endpoint = opensearch_endpoint
            
        if embedding_model := os.getenv("EMBEDDING_MODEL"):
            config.processing.embedding_model = embedding_model
            
        return config

# Global configuration instance
config = CognitiveConfig.from_env()
