"""
Enhanced Short-Term Memory (STM) System with Vector Database Integration
Implements ChromaDB for semantic memory storage and retrieval while maintaining STM characteristics
"""
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import types

# Import ChromaDB and SentenceTransformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    class SentenceTransformer:
        def __init__(self, *args, **kwargs):
            raise ImportError("sentence-transformers library is not installed")

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = types.ModuleType("chromadb")
    Settings = None

from .short_term_memory import ShortTermMemory, MemoryItem

logger = logging.getLogger(__name__)

@dataclass
class VectorMemoryResult:
    """Result from vector similarity search in STM"""
    item: MemoryItem
    similarity_score: float
    relevance_score: float
    distance: float = 0.0

class VectorShortTermMemory(ShortTermMemory):
    """
    Enhanced Short-Term Memory with ChromaDB vector database integration
    
    Features:
    - Semantic similarity search with embeddings
    - ChromaDB vector storage for fast retrieval  
    - Maintains STM characteristics (capacity, decay, recency)
    - Backward compatibility with existing STM interface
    - Fast vector search for context assembly
    """
    
    def __init__(
        self,
        capacity: int = 100,  # Increased from 7 for cognitive system
        decay_threshold: float = 0.1,
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "stm_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        use_vector_db: bool = True,
        max_decay_hours: int = 1  # STM decay over 1 hour
    ):
        """
        Initialize Enhanced STM system
        
        Args:
            capacity: Maximum number of items (100 for cognitive system)
            decay_threshold: Activation threshold below which items are forgotten
            chroma_persist_dir: Path for ChromaDB persistence
            collection_name: Name of ChromaDB collection
            embedding_model: SentenceTransformer model name
            use_vector_db: Whether to use ChromaDB for semantic search
            max_decay_hours: Maximum hours before memories decay completely
        """
        # Initialize base STM
        super().__init__(capacity=capacity, decay_threshold=decay_threshold)
        
        # Vector database configuration
        self.chroma_persist_dir = Path(chroma_persist_dir or "data/memory_stores/chroma_stm")
        self.collection_name = collection_name
        self.use_vector_db = use_vector_db and CHROMADB_AVAILABLE
        self.max_decay_hours = max_decay_hours
        
        # Create directories
        if self.use_vector_db:
            self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedding model
        self.embedding_model = None
        if self.use_vector_db and SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model)
                logger.info(f"STM loaded embedding model: {embedding_model}")
            except Exception as e:
                logger.warning(f"STM failed to load embedding model {embedding_model}: {e}")
                self.embedding_model = None
        
        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        if self.use_vector_db:
            self._initialize_chromadb()
        
        logger.info(f"Enhanced Vector STM initialized - capacity: {capacity}, vector_db: {self.use_vector_db}")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection for STM"""
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available for STM, falling back to basic search")
            self.use_vector_db = False
            return

        # Build settings if available
        settings = None
        if Settings:
            try:
                settings = Settings(
                    allow_reset=True,
                    anonymized_telemetry=False,
                    persist_directory=str(self.chroma_persist_dir)
                )
            except TypeError:
                settings = Settings(
                    allow_reset=True,
                    anonymized_telemetry=False
                )

        # Get client class
        client_cls = getattr(chromadb, "PersistentClient", None) or getattr(chromadb, "Client", None)
        if client_cls is None or settings is None:
            logger.warning("ChromaDB client class or Settings not available for STM")
            self.use_vector_db = False
            return

        try:
            # Initialize client
            if hasattr(chromadb, "PersistentClient"):
                self.chroma_client = client_cls(
                    path=str(self.chroma_persist_dir),
                    settings=settings
                )
            else:
                self.chroma_client = client_cls(settings)
            
            # Get or create collection
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
                logger.debug(f"STM using existing ChromaDB collection: {self.collection_name}")
            except Exception:
                # Create new collection
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Short-term memory vector storage"}
                )
                logger.info(f"STM created new ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"STM failed to initialize ChromaDB: {e}")
            self.use_vector_db = False
            self.chroma_client = None
            self.collection = None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text content"""
        if not self.embedding_model or not text:
            return None
        
        try:
            content_str = str(text)
            embedding = self.embedding_model.encode(content_str)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except Exception as e:
            logger.error(f"STM failed to generate embedding: {e}")
            return None
    
    def _content_to_text(self, content: Any) -> str:
        """Convert content to searchable text"""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, default=str)
        else:
            return str(content)
    
    def _safe_metadata(self, item: MemoryItem) -> Dict[str, Any]:
        """Convert MemoryItem to ChromaDB-safe metadata"""
        return {
            "memory_id": item.id,
            "encoding_time": item.encoding_time.isoformat(),
            "last_access": item.last_access.isoformat(), 
            "access_count": int(item.access_count),
            "importance": float(item.importance),
            "attention_score": float(item.attention_score),
            "emotional_valence": float(item.emotional_valence),
            "decay_rate": float(item.decay_rate),
            "associations": ",".join(item.associations) if item.associations else ""
        }
    
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
        Store new item in STM with vector database integration
        
        Enhanced with semantic indexing while maintaining STM behavior
        """
        # Store in base STM first
        success = super().store(
            memory_id=memory_id,
            content=content,
            importance=importance,
            attention_score=attention_score,
            emotional_valence=emotional_valence,
            associations=associations
        )
        
        if not success:
            return False
        
        # Add to vector database if enabled
        if self.use_vector_db and self.collection and memory_id in self.items:
            try:
                item = self.items[memory_id]
                text_content = self._content_to_text(content)
                embedding = self._generate_embedding(text_content)
                
                if embedding:
                    # Store in ChromaDB
                    self.collection.upsert(
                        ids=[memory_id],
                        embeddings=[embedding],
                        documents=[text_content],
                        metadatas=[self._safe_metadata(item)]
                    )
                    logger.debug(f"STM stored {memory_id} in vector database")
                
            except Exception as e:
                logger.error(f"STM failed to store {memory_id} in vector database: {e}")
                # Continue without vector storage
        
        return True
    
    def search_semantic(
        self,
        query: str,
        max_results: int = 5,
        min_similarity: float = 0.5,
        min_activation: float = 0.0
    ) -> List[VectorMemoryResult]:
        """
        Semantic search in STM using vector similarity
        
        Args:
            query: Search query
            max_results: Maximum number of results
            min_similarity: Minimum similarity threshold
            min_activation: Minimum activation threshold (STM-specific)
        
        Returns:
            List of VectorMemoryResult objects
        """
        if not self.use_vector_db or not self.collection or not query:
            return self._fallback_semantic_search(query, max_results, min_activation)
        
        try:
            # Generate query embedding
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return self._fallback_semantic_search(query, max_results, min_activation)
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results * 2, 50),  # Get more results to filter
                include=["documents", "distances", "metadatas"]
            )
            
            vector_results = []
            
            if results['ids'][0]:  # Check if results exist
                for i, memory_id in enumerate(results['ids'][0]):
                    # Check if memory still exists in STM (might have decayed)
                    if memory_id not in self.items:
                        continue
                    
                    item = self.items[memory_id]
                    
                    # Check activation threshold (STM-specific)
                    activation = item.calculate_activation()
                    if activation < min_activation:
                        continue
                    
                    # Calculate similarity score
                    distance = results['distances'][0][i]
                    similarity = max(0.0, 1.0 - distance)
                    
                    if similarity >= min_similarity:
                        # Combine similarity with STM activation for relevance
                        relevance = (similarity * 0.7) + (activation * 0.3)
                        
                        vector_results.append(VectorMemoryResult(
                            item=item,
                            similarity_score=similarity,
                            relevance_score=relevance,
                            distance=distance
                        ))
            
            # Sort by relevance score and return top results
            vector_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return vector_results[:max_results]
            
        except Exception as e:
            logger.error(f"STM semantic search failed: {e}")
            return self._fallback_semantic_search(query, max_results, min_activation)
    
    def _fallback_semantic_search(
        self,
        query: str,
        max_results: int,
        min_activation: float
    ) -> List[VectorMemoryResult]:
        """Fallback semantic search using basic text matching"""
        results = []
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        for item in self.items.values():
            activation = item.calculate_activation()
            if activation < min_activation:
                continue
            
            # Basic text similarity
            content_str = self._content_to_text(item.content).lower()
            content_words = set(content_str.split())
            
            # Calculate similarity
            if query_lower in content_str:
                similarity = 0.8
            else:
                overlap = len(query_words.intersection(content_words))
                total_words = len(query_words.union(content_words))
                similarity = overlap / total_words if total_words > 0 else 0.0
            
            if similarity > 0:
                relevance = (similarity * 0.7) + (activation * 0.3)
                results.append(VectorMemoryResult(
                    item=item,
                    similarity_score=similarity,
                    relevance_score=relevance
                ))
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        return results[:max_results]
    
    def search(
        self,
        query: str = "",
        min_activation: float = 0.0,
        max_results: int = 5,
        search_associations: bool = True
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Enhanced search with semantic capabilities (backward compatible)
        """
        if query and self.use_vector_db:
            # Use semantic search
            vector_results = self.search_semantic(
                query=query,
                max_results=max_results,
                min_activation=min_activation
            )
            # Convert to old format for compatibility
            return [(result.item, result.relevance_score) for result in vector_results]
        else:
            # Use base STM search
            return super().search(
                query=query,
                min_activation=min_activation,
                max_results=max_results,
                search_associations=search_associations
            )
    
    def decay_memories(self) -> List[str]:
        """
        Apply decay and remove items from both STM and vector database
        """
        forgotten_ids = super().decay_memories()
        
        # Remove from vector database
        if forgotten_ids and self.use_vector_db and self.collection:
            try:
                self.collection.delete(ids=forgotten_ids)
                logger.debug(f"STM removed {len(forgotten_ids)} decayed memories from vector database")
            except Exception as e:
                logger.error(f"STM failed to remove decayed memories from vector database: {e}")
        
        return forgotten_ids
    
    def remove_item(self, memory_id: str) -> bool:
        """
        Remove item from both STM and vector database
        """
        success = super().remove_item(memory_id)
        
        if success and self.use_vector_db and self.collection:
            try:
                self.collection.delete(ids=[memory_id])
                logger.debug(f"STM removed {memory_id} from vector database")
            except Exception as e:
                logger.error(f"STM failed to remove {memory_id} from vector database: {e}")
        
        return success
    
    def clear(self):
        """Clear all memories from STM and vector database"""
        super().clear()
        
        if self.use_vector_db and self.collection:
            try:
                # Delete all documents in collection
                result = self.collection.get()
                if result['ids']:
                    self.collection.delete(ids=result['ids'])
                logger.debug("STM cleared vector database")
            except Exception as e:
                logger.error(f"STM failed to clear vector database: {e}")
    
    def get_context_for_query(
        self,
        query: str,
        max_context_items: int = 10,
        min_relevance: float = 0.3
    ) -> List[VectorMemoryResult]:
        """
        Get relevant context memories for a query (optimized for cognitive processing)
        
        Args:
            query: Query to find context for
            max_context_items: Maximum context items to return
            min_relevance: Minimum relevance threshold
        
        Returns:
            List of relevant memories for context building
        """
        results = self.search_semantic(
            query=query,
            max_results=max_context_items,
            min_similarity=min_relevance,
            min_activation=0.1  # Low threshold for context
        )
        
        # Filter by relevance and return
        context_results = [r for r in results if r.relevance_score >= min_relevance]
        
        logger.debug(f"STM found {len(context_results)} context items for query: {query[:50]}...")
        return context_results
    
    def get_status(self) -> Dict[str, Any]:
        """Get enhanced STM status including vector database info"""
        base_status = super().get_status()
        
        # Add vector database information
        base_status.update({
            "vector_db_enabled": self.use_vector_db,
            "collection_name": self.collection_name if self.use_vector_db else None,
            "embedding_model": getattr(self.embedding_model, 'model_name', None) if self.embedding_model else None,
            "chroma_persist_dir": str(self.chroma_persist_dir) if self.use_vector_db else None
        })
        
        # Add ChromaDB stats if available
        if self.use_vector_db and self.collection:
            try:
                collection_info = self.collection.get()
                base_status["vector_db_count"] = len(collection_info['ids']) if collection_info['ids'] else 0
            except Exception as e:
                base_status["vector_db_error"] = str(e)
        
        return base_status
