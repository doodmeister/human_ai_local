"""
Enhanced Short-Term Memory (STM) System with Vector Database Integration

This module provides a production-grade implementation of short-term memory using ChromaDB
for semantic storage and retrieval while maintaining STM characteristics like capacity
limits, activation-based decay, and LRU eviction.

Key Features:
- Vector-based semantic storage and retrieval
- Capacity-limited STM with LRU eviction
- Activation-based forgetting mechanism
- Associative retrieval capabilities
- Comprehensive error handling and logging
- Type-safe implementation with full validation
- Performance optimizations and resource management

Author: AI Code Team
Version: 2.0.0
"""

import json
import logging
import threading
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

# Type imports
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer as SentenceTransformerType
    from chromadb import PersistentClient as ChromaClient
    from chromadb.config import Settings as ChromaSettings
else:
    SentenceTransformerType = None
    ChromaClient = None
    ChromaSettings = None

# Third-party imports with proper error handling
try:
    import torch
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    torch = None
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    chromadb = None
    Settings = None

# Local imports - Define MemoryItem here since we removed short_term_memory.py
@dataclass
class MemoryItem:
    """Represents a memory item with metadata"""
    id: str
    content: str
    encoding_time: datetime
    last_access: datetime
    access_count: int = 0
    importance: float = 0.5
    attention_score: float = 0.0
    emotional_valence: float = 0.0
    decay_rate: float = 0.1
    associations: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate the memory item after initialization"""
        if not self.id:
            raise ValueError("Memory ID cannot be empty")
        if not isinstance(self.encoding_time, datetime):
            raise ValueError("Encoding time must be a datetime object")
        if not isinstance(self.last_access, datetime):
            raise ValueError("Last access must be a datetime object")

# Configure logging
logger = logging.getLogger(__name__)


class VectorSTMError(Exception):
    """Base exception for Vector STM operations"""
    pass


class VectorSTMConfigError(VectorSTMError):
    """Configuration-related errors"""
    pass


class VectorSTMStorageError(VectorSTMError):
    """Storage operation errors"""
    pass


class VectorSTMRetrievalError(VectorSTMError):
    """Retrieval operation errors"""
    pass


@dataclass
class VectorMemoryResult:
    """Result from vector similarity search in STM"""
    item: MemoryItem
    similarity_score: float
    relevance_score: float
    distance: float = 0.0


@dataclass
class STMConfiguration:
    """Configuration for Vector STM system"""
    chroma_persist_dir: Optional[str] = None
    collection_name: str = "stm_memories"
    embedding_model: str = "all-MiniLM-L6-v2"
    max_decay_hours: int = 1
    capacity: int = 7
    min_activation_threshold: float = 0.2
    decay_rate: float = 0.1
    # Adaptive decay / activation extensions
    decay_mode: str = "linear"  # linear|exponential|power|sigmoid
    decay_lambda: float = 0.35
    decay_power_alpha: float = 0.8
    decay_sigmoid_k: float = 0.9
    decay_sigmoid_midpoint_hours: float = 0.5
    weight_recency: float = 0.4
    weight_frequency: float = 0.3
    weight_salience: float = 0.3
    enable_gpu: bool = True
    max_concurrent_operations: int = 4
    connection_timeout: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self):
        """Validate configuration parameters"""
        if self.capacity <= 0:
            raise VectorSTMConfigError("Capacity must be positive")
        if self.max_decay_hours <= 0:
            raise VectorSTMConfigError("Max decay hours must be positive")
        if not (0.0 <= self.min_activation_threshold <= 1.0):
            raise VectorSTMConfigError("Min activation threshold must be between 0 and 1")
        if not (0.0 <= self.decay_rate <= 1.0):
            raise VectorSTMConfigError("Decay rate must be between 0 and 1")
        if self.decay_mode not in {"linear", "exponential", "power", "sigmoid"}:
            raise VectorSTMConfigError("decay_mode must be one of linear|exponential|power|sigmoid")
        for attr in ("decay_lambda", "decay_power_alpha", "decay_sigmoid_k", "decay_sigmoid_midpoint_hours"):
            if getattr(self, attr) <= 0:
                raise VectorSTMConfigError(f"{attr} must be > 0")
        for w in (self.weight_recency, self.weight_frequency, self.weight_salience):
            if w < 0:
                raise VectorSTMConfigError("Activation weights must be non-negative")

    def activation_weights(self) -> Dict[str, float]:
        total = self.weight_recency + self.weight_frequency + self.weight_salience
        if total <= 0:
            return {"recency": 0.4, "frequency": 0.3, "salience": 0.3}
        return {
            "recency": self.weight_recency / total,
            "frequency": self.weight_frequency / total,
            "salience": self.weight_salience / total,
        }


class EmbeddingManager:
    """Manages embedding model lifecycle and operations"""
    
    def __init__(self, model_name: str, enable_gpu: bool = True):
        self.model_name = model_name
        self.enable_gpu = enable_gpu
        self._model: Optional[Any] = None
        self._lock = threading.Lock()
        self._device = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the embedding model with error handling"""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise VectorSTMConfigError(
                "sentence-transformers is required for vector memory systems. "
                "Install it with: pip install sentence-transformers"
            )
        
        try:
            # Import at runtime to avoid type checking issues
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
            
            # Configure device
            if self.enable_gpu and torch and torch.cuda.is_available():
                self._device = "cuda"
                self._model = self._model.to(self._device)
                logger.info(f"Embedding model '{self.model_name}' loaded on GPU")
            else:
                self._device = "cpu"
                logger.info(f"Embedding model '{self.model_name}' loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to initialize embedding model '{self.model_name}': {e}")
            raise VectorSTMConfigError(f"Failed to initialize embedding model: {e}") from e
    
    def encode(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text with error handling"""
        if not self._model or not text:
            return None
        
        try:
            with self._lock:
                embedding = self._model.encode(text.strip())
                
                if hasattr(embedding, 'tolist'):
                    return embedding.tolist()
                elif hasattr(embedding, '__iter__'):
                    return [float(x) for x in embedding]
                else:
                    return None
                    
        except Exception as e:
            logger.error(f"Failed to generate embedding for text: {e}")
            return None
    
    def get_device(self) -> str:
        """Get the device the model is running on"""
        return self._device or "cpu"
    
    def get_model_name(self) -> str:
        """Get the model name"""
        return self.model_name


class ChromaDBManager:
    """Manages ChromaDB client and collection operations"""
    
    def __init__(self, persist_dir: Path, collection_name: str, timeout: int = 30):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.timeout = timeout
        self._client = None
        self._collection = None
        self._lock = threading.Lock()
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection"""
        if not HAS_CHROMADB:
            raise VectorSTMConfigError(
                "chromadb is required for vector memory systems. "
                "Install it with: pip install chromadb"
            )
        
        try:
            # Ensure directory exists
            self.persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Import at runtime to avoid type checking issues
            import chromadb
            from chromadb.config import Settings
            
            # Configure ChromaDB settings
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                persist_directory=str(self.persist_dir)
            )
            
            # Initialize persistent client
            self._client = chromadb.PersistentClient(
                path=str(self.persist_dir),
                settings=settings
            )
            
            # Get or create collection
            self._initialize_collection()
            
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorSTMConfigError(f"Failed to initialize ChromaDB: {e}") from e
    
    def _initialize_collection(self):
        """Initialize or get existing collection"""
        if not self._client:
            raise VectorSTMConfigError("ChromaDB client not initialized")
            
        try:
            self._collection = self._client.get_collection(name=self.collection_name)
            logger.debug(f"Using existing ChromaDB collection: {self.collection_name}")
        except Exception:
            # Create new collection
            self._collection = self._client.create_collection(
                name=self.collection_name,
                metadata={"description": "Short-term memory vector storage"}
            )
            logger.info(f"Created new ChromaDB collection: {self.collection_name}")
    
    @contextmanager
    def get_collection(self):
        """Context manager for collection operations"""
        with self._lock:
            if not self._collection:
                raise VectorSTMStorageError("Collection not initialized")
            yield self._collection
    
    def reset_collection(self):
        """Reset the collection (for testing purposes)"""
        with self._lock:
            if self._client and self._collection:
                try:
                    self._client.delete_collection(name=self.collection_name)
                    self._initialize_collection()
                except Exception as e:
                    logger.error(f"Failed to reset collection: {e}")
                    raise VectorSTMStorageError(f"Failed to reset collection: {e}") from e


class MetadataValidator:
    """Validates and sanitizes metadata for ChromaDB storage"""
    
    @staticmethod
    def validate_memory_id(memory_id: str) -> str:
        """Validate and sanitize memory ID"""
        if not memory_id or not isinstance(memory_id, str):
            raise ValueError("Memory ID must be a non-empty string")
        
        # Remove invalid characters and limit length
        sanitized = ''.join(c for c in memory_id if c.isalnum() or c in '-_')
        if len(sanitized) > 255:
            sanitized = sanitized[:255]
        
        if not sanitized:
            raise ValueError("Memory ID contains no valid characters")
        
        return sanitized
    
    @staticmethod
    def validate_content(content: Any) -> str:
        """Validate and convert content to string"""
        if content is None:
            return ""
        
        if isinstance(content, str):
            return content.strip()
        elif isinstance(content, dict):
            return json.dumps(content, default=str, ensure_ascii=False)
        else:
            return str(content).strip()
    
    @staticmethod
    def validate_numeric_field(value: Any, field_name: str, min_val: float = 0.0, max_val: float = 1.0) -> float:
        """Validate numeric fields with bounds checking"""
        try:
            num_val = float(value) if value is not None else 0.0
            return max(min_val, min(max_val, num_val))
        except (ValueError, TypeError):
            logger.warning(f"Invalid {field_name} value: {value}, using default 0.0")
            return 0.0
    
    @staticmethod
    def validate_int_field(value: Any, field_name: str, default: int = 0) -> int:
        """Validate integer fields with safe conversion"""
        try:
            if value is None:
                return default
            return int(value)
        except (ValueError, TypeError):
            logger.warning(f"Invalid {field_name} value: {value}, using default {default}")
            return default
    
    @staticmethod
    def validate_associations(associations: Any) -> List[str]:
        """Validate and sanitize associations"""
        if not associations:
            return []
        
        if isinstance(associations, str):
            return [item.strip() for item in associations.split(',') if item.strip()]
        elif isinstance(associations, list):
            return [str(item).strip() for item in associations if item]
        else:
            return []
    
    @staticmethod
    def safe_metadata_to_dict(metadata: Any) -> Dict[str, Any]:
        """Safely convert ChromaDB metadata to Dict[str, Any]"""
        if metadata is None:
            return {}
        
        if isinstance(metadata, dict):
            return dict(metadata)
        
        try:
            # Handle ChromaDB Metadata type
            if hasattr(metadata, 'items'):
                return {str(k): v for k, v in metadata.items()}
            elif hasattr(metadata, '__dict__'):
                return dict(metadata.__dict__)
            else:
                return {}
        except Exception as e:
            logger.warning(f"Failed to convert metadata to dict: {e}")
            return {}
    
    @staticmethod
    def safe_list_access(items: Any, index: int, default: Any = None) -> Any:
        """Safely access list items with None checking"""
        if items is None:
            return default
        
        try:
            if hasattr(items, '__getitem__') and hasattr(items, '__len__'):
                if 0 <= index < len(items):
                    return items[index]
            return default
        except (IndexError, TypeError):
            return default


class VectorShortTermMemory:
    """
    Production-grade Short-Term Memory implementation using ChromaDB vector store.
    
    This implementation provides:
    - Vector-based semantic storage and retrieval
    - Capacity-limited STM with LRU eviction
    - Activation-based forgetting mechanism
    - Associative retrieval capabilities
    - Comprehensive error handling and logging
    - Type-safe implementation with full validation
    - Performance optimizations and resource management
    """
    
    def __init__(self, config: Optional[STMConfiguration] = None, *, disable_storage: bool = False):
        """
        Initialize Vector STM with configuration.
        
        Args:
            config: STM configuration object. If None, uses default configuration.
        """
        self.config = config or STMConfiguration()
        self._disable_storage = disable_storage
        if not self._disable_storage:
            self._validate_dependencies()
            self.persist_dir = Path(self.config.chroma_persist_dir or "data/memory_stores/chroma_stm")
            self.embedding_manager = EmbeddingManager(
                self.config.embedding_model,
                self.config.enable_gpu
            )
            self.chroma_manager = ChromaDBManager(
                self.persist_dir,
                self.config.collection_name,
                self.config.connection_timeout
            )
        else:
            # Minimal placeholders for test-mode activation calculations
            self.persist_dir = Path("/dev/null")
            # Lightweight stubs to satisfy attribute checks while disabling I/O
            class _NullEmbeddingManager:
                def encode(self, *_args, **_kwargs):
                    return [0.0]
                def get_model_name(self):
                    return "disabled"
                def get_device(self):
                    return "cpu"
            class _NullCollection:
                def __enter__(self): return self
                def __exit__(self, *a): return False
                def count(self): return 0
                def get(self, *args, **kwargs): return {"ids": [], "metadatas": [], "documents": [], "embeddings": []}
                def query(self, *args, **kwargs): return {"ids": [], "metadatas": [], "documents": [], "embeddings": [], "distances": []}
                def add(self, *args, **kwargs): return None
                def update(self, *args, **kwargs): return None
                def delete(self, *args, **kwargs): return None
                def upsert(self, *args, **kwargs): return None
            class _NullChromaManager:
                def get_collection(self):
                    return _NullCollection()
            self.embedding_manager = _NullEmbeddingManager()
            self.chroma_manager = _NullChromaManager()

        # Thread pool for concurrent operations
        if not self._disable_storage:
            self.executor = ThreadPoolExecutor(max_workers=self.config.max_concurrent_operations)
        else:
            class _NullFuture:
                def result(self, timeout=None): return None
                def done(self): return True
                def add_done_callback(self, fn):
                    try:
                        fn(self)
                    except Exception:
                        pass
            class _NullExecutor:
                def submit(self, *a, **k): return _NullFuture()
                def shutdown(self, wait=True): return None
            self.executor = _NullExecutor()

        # Performance metrics
        self._operation_count = 0
        self._error_count = 0

        logger.info(f"Vector STM initialized (capacity={self.config.capacity}, storage_disabled={self._disable_storage})")
    
    def _validate_dependencies(self):
        """Validate that required dependencies are available"""
        if not HAS_SENTENCE_TRANSFORMERS:
            raise VectorSTMConfigError(
                "sentence-transformers is required. Install with: pip install sentence-transformers"
            )
        
        if not HAS_CHROMADB:
            raise VectorSTMConfigError(
                "chromadb is required. Install with: pip install chromadb"
            )
    
    def _safe_metadata_to_memory_item(self, memory_id: str, metadata: Dict[str, Any], content: str = "") -> MemoryItem:
        """Convert ChromaDB metadata to MemoryItem with comprehensive validation"""
        try:
            # Handle datetime fields with validation
            now = datetime.now()
            
            encoding_time = now
            if metadata.get('encoding_time'):
                try:
                    encoding_time = datetime.fromisoformat(str(metadata['encoding_time']))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid encoding_time in metadata: {e}")
            
            last_access = encoding_time
            if metadata.get('last_access'):
                try:
                    last_access = datetime.fromisoformat(str(metadata['last_access']))
                except (ValueError, TypeError) as e:
                    logger.warning(f"Invalid last_access in metadata: {e}")
            
            return MemoryItem(
                id=MetadataValidator.validate_memory_id(metadata.get('memory_id', memory_id)),
                content=content or MetadataValidator.validate_content(metadata.get('content', '')),
                encoding_time=encoding_time,
                last_access=last_access,
                access_count=max(0, int(metadata.get('access_count', 0))),
                importance=MetadataValidator.validate_numeric_field(metadata.get('importance', 0.5), 'importance'),
                attention_score=MetadataValidator.validate_numeric_field(metadata.get('attention_score', 0.0), 'attention_score'),
                emotional_valence=MetadataValidator.validate_numeric_field(metadata.get('emotional_valence', 0.0), 'emotional_valence', -1.0, 1.0),
                decay_rate=MetadataValidator.validate_numeric_field(metadata.get('decay_rate', self.config.decay_rate), 'decay_rate'),
                associations=MetadataValidator.validate_associations(metadata.get('associations'))
            )
        except Exception as e:
            logger.error(f"Failed to convert metadata to MemoryItem: {e}")
            raise VectorSTMRetrievalError(f"Failed to convert metadata: {e}") from e
    
    def _memory_item_to_metadata(self, item: MemoryItem) -> Dict[str, Any]:
        """Convert MemoryItem to ChromaDB-safe metadata"""
        return {
            "memory_id": item.id,
            "content": item.content,
            "encoding_time": item.encoding_time.isoformat(),
            "last_access": item.last_access.isoformat(),
            "access_count": int(item.access_count),
            "importance": float(item.importance),
            "attention_score": float(item.attention_score),
            "emotional_valence": float(item.emotional_valence),
            "decay_rate": float(item.decay_rate),
            "associations": ",".join(item.associations) if item.associations else ""
        }
    
    def _calculate_activation(self, metadata: Dict[str, Any]) -> float:
        """Calculate activation score using configurable decay modes and adaptive weights.

        Components:
          recency   - time-based decay using selected mode
          frequency - normalized access count
          salience  - average of importance & attention
        Weights come from current configuration (normalized on access).
        """
        try:
            now = datetime.now()
            # Parse timestamps safely
            encoding_time = now
            if metadata.get('encoding_time'):
                try:
                    encoding_time = datetime.fromisoformat(str(metadata['encoding_time']))
                except (ValueError, TypeError):
                    pass

            age_hours = max(0.0, (now - encoding_time).total_seconds() / 3600)
            recency = self._compute_recency(age_hours, metadata)

            access_count = max(0, int(metadata.get('access_count', 0)))
            # Use a diminishing returns curve; 10 accesses -> ~1.0
            frequency = min(1.0, access_count / 10.0)

            importance = MetadataValidator.validate_numeric_field(metadata.get('importance', 0.5), 'importance')
            attention = MetadataValidator.validate_numeric_field(metadata.get('attention_score', 0.0), 'attention_score')
            salience = max(0.0, min(1.0, (importance + attention) / 2.0))

            w = self.config.activation_weights()
            activation = (recency * w['recency']) + (frequency * w['frequency']) + (salience * w['salience'])
            return max(0.0, min(1.0, activation))
        except Exception as e:
            logger.error(f"Failed to calculate activation: {e}")
            return 0.0

    # --- Adaptive activation weight API ---
    def set_activation_weights(self, *, recency: Optional[float] = None, frequency: Optional[float] = None, salience: Optional[float] = None) -> Dict[str, float]:
        """Dynamically adjust activation component weights.

        Args:
            recency: New weight for recency component
            frequency: New weight for frequency component
            salience: New weight for salience component

        Returns: Normalized weight dictionary actually applied.
        """
        updated = False
        if recency is not None:
            self.config.weight_recency = max(0.0, float(recency)); updated = True
        if frequency is not None:
            self.config.weight_frequency = max(0.0, float(frequency)); updated = True
        if salience is not None:
            self.config.weight_salience = max(0.0, float(salience)); updated = True
        if updated:
            logger.debug("STM activation weights updated: %s", self.config.activation_weights())
        return self.config.activation_weights()

    def get_activation_weights(self) -> Dict[str, float]:
        return self.config.activation_weights()

    # --- Decay / recency computation ---
    def _compute_recency(self, age_hours: float, metadata: Optional[Dict[str, Any]] = None) -> float:
        """Compute recency score (0-1) using configured decay mode.

        Modes:
          linear: 1 - age * decay_rate (floor at 0)
          exponential: exp(-lambda * age)
          power: (age + 1)^-alpha
          sigmoid: 1 / (1 + exp(k*(age-midpoint)))
        """
        try:
            mode = self.config.decay_mode
            decay_rate = self.config.decay_rate
            if metadata and 'decay_rate' in metadata:
                try:
                    decay_rate = float(metadata['decay_rate'])
                except (TypeError, ValueError):
                    pass

            if mode == 'linear':
                recency = max(0.0, 1.0 - (age_hours * decay_rate))
            elif mode == 'exponential':
                lam = max(1e-6, self.config.decay_lambda * decay_rate)
                recency = float(math.exp(-lam * age_hours))
            elif mode == 'power':
                alpha = max(1e-6, self.config.decay_power_alpha * (decay_rate + 1e-6))
                recency = float((age_hours + 1.0) ** (-alpha))
            elif mode == 'sigmoid':
                k = self.config.decay_sigmoid_k * (decay_rate + 1e-6)
                mid = self.config.decay_sigmoid_midpoint_hours
                recency = 1.0 / (1.0 + math.exp(k * (age_hours - mid)))
            else:
                recency = max(0.0, 1.0 - (age_hours * decay_rate))

            return max(0.0, min(1.0, recency))
        except Exception as e:
            logger.error(f"Failed to compute recency: {e}")
            return 0.0

    # Public helper primarily for tests / diagnostics without accessing private method
    def compute_activation_for_metadata(self, metadata: Dict[str, Any]) -> float:
        return self._calculate_activation(metadata)
    
    def _enforce_capacity_limit(self) -> List[str]:
        """Enforce capacity limit by removing least recently used items"""
        evicted_ids = []
        
        try:
            if self._disable_storage:
                return evicted_ids
            with self.chroma_manager.get_collection() as collection:
                result = collection.get()
                
                if not result.get('ids'):
                    return evicted_ids
                
                current_count = len(result['ids'])
                
                if current_count < self.config.capacity:
                    return evicted_ids
                
                # Find LRU items to evict
                items_to_evict = current_count - self.config.capacity + 1
                
                # Sort by last_access timestamp (oldest first)
                items_with_access = []
                for i, memory_id in enumerate(result['ids']):
                    metadatas = result.get('metadatas', [])
                    if metadatas and i < len(metadatas):
                        metadata = MetadataValidator.safe_metadata_to_dict(metadatas[i])
                    else:
                        metadata = {}
                    
                    try:
                        last_access_str = metadata.get('last_access', datetime.now().isoformat())
                        last_access = datetime.fromisoformat(str(last_access_str))
                    except (ValueError, TypeError):
                        last_access = datetime.now()
                    
                    items_with_access.append((memory_id, last_access))
                
                # Sort by last_access (oldest first)
                items_with_access.sort(key=lambda x: x[1])
                
                # Evict oldest items
                for memory_id, _ in items_with_access[:items_to_evict]:
                    evicted_ids.append(memory_id)
                
                if evicted_ids:
                    collection.delete(ids=evicted_ids)
                    logger.info(f"Evicted {len(evicted_ids)} items due to capacity limit")
                
        except Exception as e:
            logger.error(f"Failed to enforce capacity limit: {e}")
            raise VectorSTMStorageError(f"Failed to enforce capacity limit: {e}") from e
        
        return evicted_ids
    
    def store(
        self,
        memory_id: str,
        content: Any,
        importance: float = 0.5,
        attention_score: float = 0.0,
        emotional_valence: float = 0.0,
        associations: Optional[List[str]] = None
    ) -> bool:
        """
        Store a new memory item in the vector STM.
        
        Args:
            memory_id: Unique identifier for the memory
            content: Content to store
            importance: Importance score (0.0-1.0)
            attention_score: Attention score (0.0-1.0)
            emotional_valence: Emotional valence (-1.0-1.0)
            associations: List of associated terms
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Validate inputs
            memory_id = MetadataValidator.validate_memory_id(memory_id)
            content_str = MetadataValidator.validate_content(content)
            
            if not content_str:
                logger.warning(f"Empty content for memory {memory_id}")
                return False
            
            # Generate embedding
            embedding = self.embedding_manager.encode(content_str)
            if not embedding:
                logger.error(f"Failed to generate embedding for memory {memory_id}")
                return False
            
            # Enforce capacity limit before storing
            self._enforce_capacity_limit()
            
            # Prepare metadata
            now = datetime.now()
            
            # Check if memory already exists to preserve access count
            access_count = 0
            try:
                with self.chroma_manager.get_collection() as collection:
                    existing = collection.get(ids=[memory_id])
                    if existing.get('ids') and existing.get('metadatas'):
                        metadatas = existing.get('metadatas', [])
                        existing_meta = MetadataValidator.safe_metadata_to_dict(
                            MetadataValidator.safe_list_access(metadatas, 0, {})
                        )
                        access_count = MetadataValidator.validate_int_field(
                            existing_meta.get('access_count', 0), 'access_count', 0
                        )
            except Exception:
                pass  # New memory, access_count stays 0
            
            metadata = {
                "memory_id": memory_id,
                "content": content_str,
                "encoding_time": now.isoformat(),
                "last_access": now.isoformat(),
                "access_count": access_count,
                "importance": MetadataValidator.validate_numeric_field(importance, 'importance'),
                "attention_score": MetadataValidator.validate_numeric_field(attention_score, 'attention_score'),
                "emotional_valence": MetadataValidator.validate_numeric_field(emotional_valence, 'emotional_valence', -1.0, 1.0),
                "decay_rate": self.config.decay_rate,
                "associations": ",".join(MetadataValidator.validate_associations(associations))
            }
            
            # Store in ChromaDB
            with self.chroma_manager.get_collection() as collection:
                collection.upsert(
                    ids=[memory_id],
                    embeddings=[embedding],
                    documents=[content_str],
                    metadatas=[metadata]
                )
            
            self._operation_count += 1
            logger.debug(f"Successfully stored memory {memory_id}")
            return True
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to store memory {memory_id}: {e}")
            return False
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """
        Retrieve a specific memory by ID and update access tracking.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            MemoryItem if found, None otherwise
        """
        try:
            memory_id = MetadataValidator.validate_memory_id(memory_id)
            
            with self.chroma_manager.get_collection() as collection:
                result = collection.get(ids=[memory_id])
                
                if not result.get('ids'):
                    return None
                
                # Get metadata and content
                metadatas = result.get('metadatas', [])
                documents = result.get('documents', [])
                
                metadata = MetadataValidator.safe_metadata_to_dict(
                    MetadataValidator.safe_list_access(metadatas, 0, {})
                )
                content = MetadataValidator.safe_list_access(documents, 0, "")
                
                # Update access tracking
                now = datetime.now()
                access_count = MetadataValidator.validate_int_field(
                    metadata.get('access_count', 0), 'access_count', 0
                ) + 1
                
                metadata['last_access'] = now.isoformat()
                metadata['access_count'] = access_count
                
                # Re-upsert to update metadata
                embedding = self.embedding_manager.encode(content)
                if embedding:
                    collection.upsert(
                        ids=[memory_id],
                        embeddings=[embedding],
                        documents=[content],
                        metadatas=[metadata]
                    )
                
                # Convert to MemoryItem
                memory_item = self._safe_metadata_to_memory_item(memory_id, metadata, content)
                
                self._operation_count += 1
                return memory_item
                
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to retrieve memory {memory_id}: {e}")
            return None
    
    def search_semantic(
        self,
        query: str,
        max_results: int = 5,
        min_similarity: float = 0.3
    ) -> List[VectorMemoryResult]:
        """
        Perform semantic search using vector similarity.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of VectorMemoryResult objects
        """
        try:
            if not query.strip():
                return []
            
            # Generate query embedding
            query_embedding = self.embedding_manager.encode(query.strip())
            if not query_embedding:
                logger.warning(f"Failed to generate embedding for query: {query}")
                return []
            
            # Search in ChromaDB
            with self.chroma_manager.get_collection() as collection:
                results = collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(max_results * 2, 50),  # Get more results for filtering
                    include=["documents", "distances", "metadatas"]
                )
            
            vector_results = []
            
            # Process results
            if (results.get('ids') and results['ids'] and 
                results.get('distances') and results.get('metadatas') and results.get('documents')):
                
                for i, memory_id in enumerate(results['ids'][0]):
                    try:
                        # Calculate similarity from distance
                        distances = results.get('distances', [])
                        distance = float(MetadataValidator.safe_list_access(
                            MetadataValidator.safe_list_access(distances, 0, []), i, 1.0))
                        similarity = max(0.0, 1.0 - distance)
                        
                        if similarity >= min_similarity:
                            # Get metadata and content
                            metadatas = results.get('metadatas', [])
                            documents = results.get('documents', [])
                            
                            metadata = MetadataValidator.safe_metadata_to_dict(
                                MetadataValidator.safe_list_access(
                                    MetadataValidator.safe_list_access(metadatas, 0, []), i, {})
                            )
                            content = MetadataValidator.safe_list_access(
                                MetadataValidator.safe_list_access(documents, 0, []), i, "")
                            
                            # Convert to MemoryItem
                            item = self._safe_metadata_to_memory_item(memory_id, metadata, content)
                            
                            # Calculate relevance score (combine similarity with activation)
                            activation = self._calculate_activation(metadata)
                            relevance = (similarity * 0.7) + (activation * 0.3)
                            
                            vector_results.append(VectorMemoryResult(
                                item=item,
                                similarity_score=similarity,
                                relevance_score=relevance,
                                distance=distance
                            ))
                    except Exception as e:
                        logger.warning(f"Failed to process search result {i}: {e}")
                        continue
            
            # Sort by relevance score
            vector_results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            self._operation_count += 1
            return vector_results[:max_results]
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Semantic search failed for query '{query}': {e}")
            return []
    
    def search_associative(
        self,
        association: str,
        max_results: int = 5
    ) -> List[VectorMemoryResult]:
        """
        Search for memories with direct associations.
        
        Args:
            association: Association term to search for
            max_results: Maximum number of results
            
        Returns:
            List of VectorMemoryResult objects
        """
        try:
            if not association.strip():
                return []
            
            association_lower = association.strip().lower()
            matches = []
            
            with self.chroma_manager.get_collection() as collection:
                result = collection.get()
                
                if not result.get('ids'):
                    return []
                
                for i, memory_id in enumerate(result['ids']):
                    try:
                        metadatas = result.get('metadatas', [])
                        documents = result.get('documents', [])
                        
                        metadata = MetadataValidator.safe_metadata_to_dict(
                            MetadataValidator.safe_list_access(metadatas, i, {})
                        )
                        content = MetadataValidator.safe_list_access(documents, i, "")
                        
                        # Check associations
                        associations = MetadataValidator.validate_associations(metadata.get('associations'))
                        
                        if any(association_lower == assoc.lower() for assoc in associations):
                            item = self._safe_metadata_to_memory_item(memory_id, metadata, content)
                            
                            matches.append(VectorMemoryResult(
                                item=item,
                                similarity_score=1.0,
                                relevance_score=1.0,
                                distance=0.0
                            ))
                    except Exception as e:
                        logger.warning(f"Failed to process associative search result {i}: {e}")
                        continue
            
            # Sort by last access (most recent first)
            matches.sort(key=lambda x: x.item.last_access, reverse=True)
            
            self._operation_count += 1
            return matches[:max_results]
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Associative search failed for '{association}': {e}")
            return []
    
    def search(
        self,
        query: str = "",
        max_results: int = 5
    ) -> List[Tuple[MemoryItem, float]]:
        """
        General search interface for backward compatibility.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            List of (MemoryItem, relevance_score) tuples
        """
        if not query.strip():
            return []
        
        vector_results = self.search_semantic(query, max_results)
        return [(result.item, result.relevance_score) for result in vector_results]
    
    def decay_memories(self, min_activation: Optional[float] = None) -> List[str]:
        """
        Remove memories with low activation scores.
        
        Args:
            min_activation: Minimum activation threshold (uses config default if None)
            
        Returns:
            List of evicted memory IDs
        """
        if min_activation is None:
            min_activation = self.config.min_activation_threshold
        
        try:
            evicted_ids = []
            
            with self.chroma_manager.get_collection() as collection:
                result = collection.get()
                
                if not result.get('ids'):
                    return evicted_ids
                
                for i, memory_id in enumerate(result['ids']):
                    try:
                        metadatas = result.get('metadatas', [])
                        metadata = MetadataValidator.safe_metadata_to_dict(
                            MetadataValidator.safe_list_access(metadatas, i, {})
                        )
                        
                        activation = self._calculate_activation(metadata)
                        
                        if activation < min_activation:
                            evicted_ids.append(memory_id)
                            
                    except Exception as e:
                        logger.warning(f"Failed to process decay for memory {memory_id}: {e}")
                        continue
                
                # Remove low-activation memories
                if evicted_ids:
                    collection.delete(ids=evicted_ids)
                    logger.info(f"Decay process evicted {len(evicted_ids)} low-activation memories")
            
            self._operation_count += 1
            return evicted_ids
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Memory decay process failed: {e}")
            return []
    
    def remove_item(self, memory_id: str) -> bool:
        """
        Remove a specific memory item.
        
        Args:
            memory_id: Memory identifier
            
        Returns:
            bool: True if successful, False if item doesn't exist or other error
        """
        try:
            memory_id = MetadataValidator.validate_memory_id(memory_id)
            
            with self.chroma_manager.get_collection() as collection:
                # First check if the item exists
                existing = collection.get(ids=[memory_id])
                if not existing or not existing['ids']:
                    logger.debug(f"Memory {memory_id} not found for removal")
                    return False
                
                # Item exists, proceed with deletion
                collection.delete(ids=[memory_id])
            
            self._operation_count += 1
            logger.debug(f"Successfully removed memory {memory_id}")
            return True
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to remove memory {memory_id}: {e}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all memories from the STM.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with self.chroma_manager.get_collection() as collection:
                result = collection.get()
                if result.get('ids'):
                    collection.delete(ids=result['ids'])
            
            self._operation_count += 1
            logger.info("Successfully cleared all memories")
            return True
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to clear memories: {e}")
            return False
    
    def get_all_memories(self) -> List[MemoryItem]:
        """
        Get all memories from the STM.
        
        Returns:
            List of MemoryItem objects
        """
        try:
            memories = []
            
            with self.chroma_manager.get_collection() as collection:
                result = collection.get()
                
                if not result.get('ids'):
                    return memories
                
                for i, memory_id in enumerate(result['ids']):
                    try:
                        metadatas = result.get('metadatas', [])
                        documents = result.get('documents', [])
                        
                        metadata = MetadataValidator.safe_metadata_to_dict(
                            MetadataValidator.safe_list_access(metadatas, i, {})
                        )
                        content = MetadataValidator.safe_list_access(documents, i, "")
                        
                        memory_item = self._safe_metadata_to_memory_item(memory_id, metadata, content)
                        memories.append(memory_item)
                        
                    except Exception as e:
                        logger.warning(f"Failed to process memory {memory_id}: {e}")
                        continue
            
            self._operation_count += 1
            return memories
            
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to get all memories: {e}")
            return []
    
    def get_context_for_query(
        self,
        query: str,
        max_context_items: int = 10,
        min_relevance: float = 0.3
    ) -> List[VectorMemoryResult]:
        """
        Get relevant context memories for a query.
        
        Args:
            query: Query string
            max_context_items: Maximum number of context items
            min_relevance: Minimum relevance threshold
            
        Returns:
            List of VectorMemoryResult objects
        """
        results = self.search_semantic(
            query=query,
            max_results=max_context_items,
            min_similarity=min_relevance
        )
        
        context_results = [r for r in results if r.relevance_score >= min_relevance]
        
        logger.debug(f"Found {len(context_results)} context items for query: {query[:50]}...")
        return context_results
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status information about the STM.
        
        Returns:
            Dict with status information
        """
        try:
            status = {
                "vector_db_enabled": True,
                "collection_name": self.config.collection_name,
                "embedding_model": self.embedding_manager.get_model_name(),
                "embedding_device": self.embedding_manager.get_device(),
                "chroma_persist_dir": str(self.persist_dir),
                "capacity": self.config.capacity,
                "max_decay_hours": self.config.max_decay_hours,
                "min_activation_threshold": self.config.min_activation_threshold,
                "operation_count": self._operation_count,
                "error_count": self._error_count,
                "error_rate": self._error_count / max(1, self._operation_count)
            }
            
            # Get collection statistics
            try:
                with self.chroma_manager.get_collection() as collection:
                    result = collection.get()
                    
                    if result.get('ids'):
                        count = len(result['ids'])
                        status["vector_db_count"] = count
                        
                        if count > 0:
                            # Calculate statistics
                            activations = []
                            importances = []
                            attentions = []
                            
                            metadatas = result.get('metadatas', [])
                            for i, _ in enumerate(metadatas or []):
                                metadata = MetadataValidator.safe_metadata_to_dict(
                                    MetadataValidator.safe_list_access(metadatas, i, {})
                                )
                                if metadata:
                                    activations.append(self._calculate_activation(metadata))
                                    importances.append(MetadataValidator.validate_numeric_field(metadata.get('importance', 0.5), 'importance'))
                                    attentions.append(MetadataValidator.validate_numeric_field(metadata.get('attention_score', 0.0), 'attention_score'))
                            
                            if activations:
                                status["avg_activation"] = sum(activations) / len(activations)
                                status["min_activation"] = min(activations)
                                status["max_activation"] = max(activations)
                            
                            if importances:
                                status["avg_importance"] = sum(importances) / len(importances)
                                
                            if attentions:
                                status["avg_attention"] = sum(attentions) / len(attentions)
                    else:
                        status["vector_db_count"] = 0
                        
            except Exception as e:
                status["vector_db_error"] = str(e)
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
            return {"error": str(e)}
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup"""
        self.shutdown()
    
    def shutdown(self):
        """Shutdown the STM system and cleanup resources"""
        try:
            if getattr(self, 'executor', None) is not None and hasattr(self.executor, 'shutdown'):
                self.executor.shutdown(wait=True)
            logger.info("Vector STM shut down successfully")
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
