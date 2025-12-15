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
    use_vector_stm: bool = True  # Enable vector-based short-term memory

    # Long-Term Memory
    ltm_storage_path: Optional[str] = None  # Will use default if None
    chroma_persist_dir: Optional[str] = None
    ltm_similarity_threshold: float = 0.7
    ltm_max_results: int = 10
    use_vector_ltm: bool = True

    # Semantic Memory
    semantic_storage_path: Optional[str] = None # Will use default if None
    
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
    allow_dynamic_novelty: bool = False  # Enable per-call novelty boost override
    attention_decay_base: float = 0.1
    attention_decay_fatigue_scale: float = 0.2
    attention_decay_age_scale: float = 0.05

@dataclass
class AgentConfig:
    """Configuration for the agent's cognitive state"""
    fatigue_increase_rate: float = 0.01
    max_active_goals: int = 3
    max_context_turns: int = 10 # Max conversation turns to keep for proactive context
    max_concurrent_tasks: int = 5

@dataclass
class ProcessingConfig:
    """Configuration for cognitive processing"""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    entropy_threshold: float = 0.3
    batch_size: int = 32

@dataclass
class LLMConfig:
    """Configuration for Language Model providers"""
    provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "openai"))  # "openai" or "ollama"
    
    # OpenAI settings
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-nano"))
    
    # Ollama settings
    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    
    # Common settings
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 90

@dataclass
class AWSConfig:
    """Configuration for AWS services"""
    region: str = "us-east-1"
    bedrock_model: str = "anthropic.claude-3-sonnet-20240229-v1:0"
    opensearch_endpoint: Optional[str] = None

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    enabled: bool = field(default_factory=lambda: os.getenv("PERFORMANCE_OPTIMIZATION", "true").lower() == "true")
    
    # Batch processing
    enable_batch_optimization: bool = True
    dynamic_batch_sizing: bool = True
    max_batch_size: int = 64
    min_batch_size: int = 4
    target_memory_usage: float = 0.8
    
    # Memory efficiency
    enable_memory_pooling: bool = True
    enable_gradient_checkpointing: bool = True
    memory_cleanup_interval: int = 100
    tensor_cache_size: int = 1000
    
    # Neural training optimization
    enable_mixed_precision: bool = True
    enable_adaptive_learning_rate: bool = True
    enable_early_stopping: bool = True
    performance_monitoring_window: int = 50
    
    # Throughput optimization
    enable_parallel_processing: bool = True
    max_worker_threads: int = 4
    enable_embedding_cache: bool = True
    cache_ttl: int = 300
    
    # GPU optimization
    enable_gpu_acceleration: bool = True
    gpu_memory_fraction: float = 0.9
    enable_tensor_cores: bool = True
    
@dataclass
class ChatConfig:
    """Configuration for chat-based interactions"""
    max_recent_turns: int = 8
    max_context_items: int = 16
    stm_activation_min: float = 0.15
    ltm_similarity_threshold: float = 0.62
    retrieval_timeout_ms: int = 400
    fallback_min_overlap: float = 0.15
    streaming_enabled: bool = True
    performance_target_p95_ms: int = 1000
    scoring_version: str = "v1.0"
    preview_max_items: int = 8
    preview_max_content_chars: int = 120
    consolidation_salience_threshold: float = 0.55
    consolidation_valence_threshold: float = 0.60
    throughput_window_seconds: int = 60  # rolling window for throughput/rate metrics
    # Metacognition & adaptive settings
    metacog_turn_interval: int = 5
    metacog_snapshot_history_size: int = 50

    def to_dict(self) -> dict:
        return {
            "max_recent_turns": self.max_recent_turns,
            "max_context_items": self.max_context_items,
            "stm_activation_min": self.stm_activation_min,
            "ltm_similarity_threshold": self.ltm_similarity_threshold,
            "retrieval_timeout_ms": self.retrieval_timeout_ms,
            "fallback_min_overlap": self.fallback_min_overlap,
            "streaming_enabled": self.streaming_enabled,
            "performance_target_p95_ms": self.performance_target_p95_ms,
            "scoring_version": self.scoring_version,
            "preview_max_items": self.preview_max_items,
            "preview_max_content_chars": self.preview_max_content_chars,
            "consolidation_salience_threshold": self.consolidation_salience_threshold,
            "consolidation_valence_threshold": self.consolidation_valence_threshold,
            "throughput_window_seconds": self.throughput_window_seconds,
            "metacog_turn_interval": self.metacog_turn_interval,
            "metacog_snapshot_history_size": self.metacog_snapshot_history_size,
        }

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
    llm: LLMConfig = field(default_factory=LLMConfig)
    aws: AWSConfig = field(default_factory=AWSConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    chat: ChatConfig = field(default_factory=ChatConfig)
    
    def __post_init__(self):
        """Create data and model directories after initialization"""
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
    
    def validate(self) -> list[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List of validation warning messages (empty if all valid)
        """
        warnings = []
        
        # Memory config validation
        if self.memory.stm_capacity < 3 or self.memory.stm_capacity > 20:
            warnings.append(f"STM capacity {self.memory.stm_capacity} outside recommended range (3-20)")
        
        if not 0.0 <= self.memory.stm_decay_threshold <= 1.0:
            warnings.append(f"STM decay threshold {self.memory.stm_decay_threshold} must be 0.0-1.0")
        
        if not 0.0 <= self.memory.ltm_similarity_threshold <= 1.0:
            warnings.append(f"LTM similarity threshold {self.memory.ltm_similarity_threshold} must be 0.0-1.0")
        
        # Attention config validation
        if not 0.0 <= self.attention.salience_threshold <= 1.0:
            warnings.append(f"Salience threshold {self.attention.salience_threshold} must be 0.0-1.0")
        
        if self.attention.max_attention_items < 1:
            warnings.append(f"Max attention items must be >= 1")
        
        # Agent config validation
        if self.agent.max_active_goals < 1:
            warnings.append(f"Max active goals must be >= 1")
        
        # LLM config validation
        if self.llm.provider not in ("openai", "ollama"):
            warnings.append(f"Unknown LLM provider: {self.llm.provider}")
        
        if self.llm.provider == "openai" and not self.llm.openai_api_key:
            warnings.append("OpenAI API key not set (OPENAI_API_KEY)")
        
        if self.llm.temperature < 0.0 or self.llm.temperature > 2.0:
            warnings.append(f"LLM temperature {self.llm.temperature} outside typical range (0.0-2.0)")
        
        # Chat config validation
        if self.chat.max_context_items < 1:
            warnings.append(f"Max context items must be >= 1")
        
        if not 0.0 <= self.chat.salience_threshold <= 1.0:
            warnings.append(f"Chat salience threshold {self.chat.salience_threshold} must be 0.0-1.0")
        
        return warnings


# Global configuration instance
config = CognitiveConfig.from_env()

def get_global_config() -> CognitiveConfig:
    """Return the global cognitive configuration singleton."""
    return config

def get_chat_config() -> ChatConfig:
    """Return chat configuration (safe fallback)."""
    try:
        # If a global singleton (e.g., get_global_config()) exists, pull chat section.
        global_config = get_global_config()  # type: ignore
        return getattr(global_config, "chat", ChatConfig())
    except Exception:
        return ChatConfig()


def validate_config() -> tuple[bool, list[str]]:
    """
    Validate the global configuration.
    
    Returns:
        Tuple of (is_valid, list of warnings)
    """
    warnings = config.validate()
    return len(warnings) == 0, warnings
