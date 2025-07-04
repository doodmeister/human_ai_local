"""
Enhanced Short-Term Memory (STM) System with Vector Database Integration
Implements ChromaDB for semantic memory storage and retrieval while maintaining STM characteristics
"""
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import types

# Import required dependencies for vector-only memory
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required for vector memory systems. "
        "Install it with: pip install sentence-transformers"
    ) from e

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        "chromadb is required for vector memory systems. "
        "Install it with: pip install chromadb"
    ) from e

from .short_term_memory import MemoryItem

logger = logging.getLogger(__name__)

@dataclass
class VectorMemoryResult:
    """Result from vector similarity search in STM"""
    item: MemoryItem
    similarity_score: float
    relevance_score: float
    distance: float = 0.0

class VectorShortTermMemory:
    """
    Short-Term Memory using only ChromaDB vector store (no flat file or in-memory fallback).
    All store, retrieve, and search operations are performed directly on ChromaDB.
    """
    def __init__(
        self,
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "stm_memories",
        embedding_model: str = "all-MiniLM-L6-v2",
        max_decay_hours: int = 1
    ):
        self.chroma_persist_dir = Path(chroma_persist_dir or "data/memory_stores/chroma_stm")
        self.collection_name = collection_name
        self.max_decay_hours = max_decay_hours
        self.embedding_model: Optional[Any] = None
        
        # Initialize embedding model with GPU support if available
        try:
            import torch
            self.embedding_model = SentenceTransformer(embedding_model)
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")
            logger.info(f"STM loaded embedding model: {embedding_model} (GPU: {torch.cuda.is_available()})")
        except Exception as e:
            logger.error(f"STM failed to load embedding model {embedding_model}: {e}")
            raise RuntimeError(f"Vector STM requires a working embedding model: {e}") from e
        
        # Initialize ChromaDB (required for vector-only STM)
        self.chroma_client = None
        self.collection = None
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_chromadb()
        
        logger.info(f"Vector STM initialized (vector-only, no fallback)")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection for STM"""
        try:
            # Build settings for ChromaDB
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                persist_directory=str(self.chroma_persist_dir)
            )
            
            # Initialize persistent client
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_persist_dir),
                settings=settings
            )
            
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
            raise RuntimeError(f"Vector STM requires ChromaDB to be properly initialized: {e}") from e
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text content"""
        if not self.embedding_model or not text:
            return None
        
        try:
            content_str = str(text)
            embedding = self.embedding_model.encode(content_str)
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            elif hasattr(embedding, '__iter__'):
                return [float(x) for x in embedding]
            else:
                return None
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
    
    def _safe_str(self, value: Any, default: str = "") -> str:
        """Safely convert value to string"""
        if value is None:
            return default
        return str(value)
    
    def _safe_int(self, value: Any, default: int = 0) -> int:
        """Safely convert value to int"""
        if value is None:
            return default
        try:
            return int(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Safely convert value to float"""
        if value is None:
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    def _safe_associations(self, value: Any) -> List[str]:
        """Safely convert associations value to list of strings"""
        if value is None or value == "":
            return []
        if isinstance(value, list):
            return [str(item) for item in value if item]
        if isinstance(value, str):
            return [item.strip() for item in value.split(',') if item.strip()]
        return []
    
    def _metadata_to_memory_item(self, memory_id: str, meta: Any, content: str = "") -> MemoryItem:
        """Convert ChromaDB metadata to MemoryItem with type safety"""
        from datetime import datetime
        
        # Convert meta to dict safely
        if not isinstance(meta, dict):
            meta = {}
        
        # Handle datetime fields safely
        encoding_time = datetime.now()
        last_access = datetime.now()
        
        if meta.get('encoding_time'):
            try:
                encoding_time = datetime.fromisoformat(self._safe_str(meta['encoding_time']))
            except (ValueError, TypeError):
                pass
                
        if meta.get('last_access'):
            try:
                last_access = datetime.fromisoformat(self._safe_str(meta['last_access']))
            except (ValueError, TypeError):
                pass
        
        return MemoryItem(
            id=self._safe_str(meta.get('memory_id', memory_id)),
            content=content or self._safe_str(meta.get('content', '')),
            encoding_time=encoding_time,
            last_access=last_access,
            access_count=self._safe_int(meta.get('access_count', 0)),
            importance=self._safe_float(meta.get('importance', 0.5)),
            attention_score=self._safe_float(meta.get('attention_score', 0.0)),
            emotional_valence=self._safe_float(meta.get('emotional_valence', 0.0)),
            decay_rate=self._safe_float(meta.get('decay_rate', 0.1)),
            associations=self._safe_associations(meta.get('associations'))
        )
    
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
        Store new item in STM (vector store only)
        """
        if not self.collection or not self.embedding_model:
            logger.error("ChromaDB or embedding model not available for STM store.")
            return False
        try:
            from datetime import datetime
            text_content = self._content_to_text(content)
            embedding = self._generate_embedding(text_content)
            if not embedding:
                logger.error("Failed to generate embedding for STM store.")
                return False
            
            # Store comprehensive metadata
            now = datetime.now()
            metadata = {
                "memory_id": memory_id,
                "content": str(content),  # Store content in metadata too
                "encoding_time": now.isoformat(),
                "last_access": now.isoformat(),
                "access_count": 0,
                "importance": float(importance),
                "attention_score": float(attention_score),
                "emotional_valence": float(emotional_valence),
                "associations": ",".join(associations) if associations else ""
            }
            self.collection.upsert(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[text_content],
                metadatas=[metadata]
            )
            logger.debug(f"STM stored {memory_id} in vector database")
            return True
        except Exception as e:
            logger.error(f"STM failed to store {memory_id} in vector database: {e}")
            return False
    
    def search_semantic(
        self,
        query: str,
        max_results: int = 5,
        min_similarity: float = 0.3
    ) -> List[VectorMemoryResult]:
        """
        Semantic search in STM using vector similarity (ChromaDB only)
        """
        if not self.collection or not self.embedding_model or not query:
            return []
        try:
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results * 2, 50),
                include=["documents", "distances", "metadatas"]
            )
            vector_results = []
            
            # Check if we have valid results
            if (results.get('ids') and results['ids'] and len(results['ids'][0]) > 0 and
                results.get('distances') and results.get('metadatas') and results.get('documents')):
                
                for i, memory_id in enumerate(results['ids'][0]):
                    # Safely get distance
                    distance = 0.0
                    if results['distances'] and len(results['distances'][0]) > i:
                        distance = float(results['distances'][0][i])
                    
                    similarity = max(0.0, 1.0 - distance)
                    if similarity >= min_similarity:
                        # Safely get metadata and content
                        meta = {}
                        if results['metadatas'] and len(results['metadatas'][0]) > i:
                            meta = results['metadatas'][0][i] or {}
                        
                        content = ""
                        if results['documents'] and len(results['documents'][0]) > i:
                            content = str(results['documents'][0][i] or "")
                        
                        # Use safe metadata conversion
                        item = self._metadata_to_memory_item(memory_id, meta, content)
                        vector_results.append(VectorMemoryResult(
                            item=item,
                            similarity_score=similarity,
                            relevance_score=similarity,
                            distance=distance
                        ))
            
            vector_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return vector_results[:max_results]
        except Exception as e:
            logger.error(f"STM semantic search failed: {e}")
            return []
    
    def _fallback_semantic_search(
        self,
        query: str,
        max_results: int,
        min_activation: float
    ) -> List[VectorMemoryResult]:
        # Fallback search is not supported in vector-only STM
        return []
    
    def search(
        self,
        query: str = "",
        max_results: int = 5
    ) -> List[Tuple[MemoryItem, float]]:
        """
        Search STM using only ChromaDB vector store.
        """
        if not query:
            return []
        vector_results = self.search_semantic(query=query, max_results=max_results)
        return [(result.item, result.relevance_score) for result in vector_results]
    
    def decay_memories(self) -> List[str]:
        """
        Decay logic is not supported in vector-only STM (no in-memory state).
        """
        logger.info("STM decay_memories: not supported in vector-only mode.")
        return []
    
    def remove_item(self, memory_id: str) -> bool:
        """
        Remove item from ChromaDB vector store.
        """
        if not self.collection:
            return False
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug(f"STM removed {memory_id} from vector database")
            return True
        except Exception as e:
            logger.error(f"STM failed to remove {memory_id} from vector database: {e}")
            return False
    
    def clear(self):
        """Clear all memories from ChromaDB vector store."""
        if not self.collection:
            return
        try:
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
        Get relevant context memories for a query (vector store only)
        """
        results = self.search_semantic(
            query=query,
            max_results=max_context_items,
            min_similarity=min_relevance
        )
        context_results = [r for r in results if r.relevance_score >= min_relevance]
        logger.debug(f"STM found {len(context_results)} context items for query: {query[:50]}...")
        return context_results
    
    def get_all_memories(self) -> List[MemoryItem]:
        """Get all memories from the vector store as MemoryItem objects"""
        if not self.collection:
            return []
            
        try:
            # Get all items from the collection
            result = self.collection.get()
            memories = []
            
            if result.get('ids'):
                for i, memory_id in enumerate(result['ids']):
                    # Safely get metadata
                    meta = {}
                    if (result.get('metadatas') and 
                        result['metadatas'] is not None and 
                        len(result['metadatas']) > i):
                        meta = result['metadatas'][i] or {}
                    
                    # Use safe metadata conversion
                    memory_item = self._metadata_to_memory_item(memory_id, meta)
                    memories.append(memory_item)
            
            return memories
            
        except Exception as e:
            logger.error(f"Error retrieving all memories from STM: {e}")
            return []
    
    def retrieve(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory by ID"""
        if not self.collection:
            return None
            
        try:
            result = self.collection.get(ids=[memory_id])
            
            if result.get('ids') and len(result['ids']) > 0:
                # Safely get metadata
                meta = {}
                if (result.get('metadatas') and 
                    result['metadatas'] is not None and 
                    len(result['metadatas']) > 0):
                    meta = result['metadatas'][0] or {}
                
                # Use safe metadata conversion
                memory_item = self._metadata_to_memory_item(memory_id, meta)
                return memory_item
            
            return None
            
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id} from STM: {e}")
            return None

    def get_status(self) -> Dict[str, Any]:
        """Get STM status (vector store only)"""
        # Get embedding model info
        embedding_model_info = None
        if self.embedding_model:
            # Try to get model name from various possible attributes
            try:
                embedding_model_info = (
                    getattr(self.embedding_model, '_model_name', None) or
                    getattr(self.embedding_model, 'model_name', None) or
                    "all-MiniLM-L6-v2"  # Default fallback since we know this is what we load
                )
            except Exception:
                embedding_model_info = "all-MiniLM-L6-v2"
        
        status = {
            "vector_db_enabled": True,
            "collection_name": self.collection_name,
            "embedding_model": embedding_model_info,
            "chroma_persist_dir": str(self.chroma_persist_dir)
        }
        if self.collection:
            try:
                collection_info = self.collection.get()
                status["vector_db_count"] = len(collection_info['ids']) if collection_info['ids'] else 0
            except Exception as e:
                status["vector_db_error"] = str(e)
        return status
