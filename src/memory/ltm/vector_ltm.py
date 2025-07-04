"""
Enhanced Long-Term Memory (LTM) System with Vector Database Integration
Implements ChromaDB for semantic memory storage and retrieval (Vector-Only)
"""
from typing import Dict, List, Optional, Any, Sequence
from datetime import datetime
from dataclasses import dataclass
import json
import logging
from pathlib import Path
import types

from ..base import BaseMemorySystem

# Import ChromaDB and SentenceTransformers with fallback
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = types.ModuleType("chromadb")
    Settings = None

logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector similarity search in LTM"""
    record: dict
    similarity_score: float
    distance: float

class VectorLongTermMemory(BaseMemorySystem):
    """
    Vector-only Long-Term Memory using ChromaDB
    All store, retrieve, and search operations are performed directly on ChromaDB.
    No in-memory storage or JSON backup.
    """
    
    def __init__(
        self,
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "ltm_memories",
        embedding_model: str = "all-MiniLM-L6-v2"
    ):
        """Initialize Vector-only LTM system (ChromaDB only)"""
        self.chroma_persist_dir = Path(chroma_persist_dir or "data/memory_stores/chroma_ltm")
        self.collection_name = collection_name
        self.embedding_model = None
        if SENTENCE_TRANSFORMERS_AVAILABLE and SentenceTransformer:
            try:
                import torch
                self.embedding_model = SentenceTransformer(embedding_model)
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to("cuda")
                logger.info(f"LTM loaded embedding model: {embedding_model} (GPU: {torch.cuda.is_available()})")
            except Exception as e:
                logger.warning(f"LTM failed to load embedding model {embedding_model}: {e}")
                self.embedding_model = None
        self.chroma_client = None
        self.collection = None
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_chromadb()
        logger.info("Vector LTM initialized (vector-only, no fallback)")
    
    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection"""
        if not CHROMADB_AVAILABLE:
            logger.error("ChromaDB not available for LTM")
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
            logger.error("ChromaDB client class or Settings not available for LTM")
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
                logger.info(f"LTM connected to existing ChromaDB collection: {self.collection_name}")
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Long-term memory vector storage"}
                )
                logger.info(f"LTM created new ChromaDB collection: {self.collection_name}")
            
        except Exception as e:
            logger.error(f"LTM failed to initialize ChromaDB: {e}")
            self.chroma_client = None
            self.collection = None
    
    def _generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for text content"""
        if not self.embedding_model or not text:
            return None
        
        try:
            content_str = str(text)
            embedding = self.embedding_model.encode(content_str)
            if hasattr(embedding, 'tolist'):
                return embedding.tolist()
            else:
                return [float(x) for x in embedding]
        except Exception as e:
            logger.error(f"LTM failed to generate embedding: {e}")
            return None
    
    def _content_to_text(self, content: Any) -> str:
        """Convert content to searchable text"""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            return json.dumps(content, default=str)
        else:
            return str(content)
    
    def store(
        self,
        memory_id: str,
        content: Any,
        memory_type: str = "episodic",
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        source: str = "unknown",
        tags: Optional[List[str]] = None,
        associations: Optional[List[str]] = None
    ) -> bool:
        """
        Store memory in ChromaDB only (vector-only LTM)
        """
        if not self.collection or not self.embedding_model:
            logger.error("ChromaDB or embedding model not available for LTM store.")
            return False
        try:
            content_text = self._content_to_text(content)
            embedding = self._generate_embedding(content_text)
            if not embedding:
                logger.error("Failed to generate embedding for LTM store.")
                return False
            metadata = {
                "memory_id": memory_id,
                "memory_type": memory_type,
                "importance": float(importance),
                "emotional_valence": float(emotional_valence),
                "source": source,
                "encoding_time": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat(),
                "tags": ",".join(tags) if tags else "",
                "associations": ",".join(associations) if associations else ""
            }
            self.collection.upsert(
                ids=[memory_id],
                embeddings=[embedding],
                documents=[content_text],
                metadatas=[metadata]
            )
            logger.debug(f"LTM stored {memory_id} (type: {memory_type}) in vector DB")
            return True
        except Exception as e:
            logger.error(f"LTM failed to store {memory_id} in vector DB: {e}")
            return False
    
    def retrieve(self, memory_id: str) -> Optional[dict]:
        """Retrieve memory from ChromaDB only"""
        if not self.collection:
            return None
        try:
            result = self.collection.get(ids=[memory_id])
            if result['ids'] and len(result['ids']) > 0:
                meta = result['metadatas'][0]
                doc = result['documents'][0]
                return {
                    "id": meta.get("memory_id", memory_id),
                    "content": doc,
                    "memory_type": meta.get("memory_type", "episodic"),
                    "encoding_time": meta.get("encoding_time"),
                    "last_access": meta.get("last_access"),
                    "importance": float(meta.get("importance", 0.5)),
                    "emotional_valence": float(meta.get("emotional_valence", 0.0)),
                    "source": meta.get("source", "unknown"),
                    "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                    "associations": meta.get("associations", "").split(",") if meta.get("associations") else []
                }
            return None
        except Exception as e:
            logger.error(f"LTM failed to retrieve {memory_id} from vector DB: {e}")
            return None
    
    def delete(self, memory_id: str) -> bool:
        """Remove memory from ChromaDB only"""
        if not self.collection:
            return False
        try:
            self.collection.delete(ids=[memory_id])
            logger.debug(f"LTM removed memory {memory_id} from vector DB")
            return True
        except Exception as e:
            logger.error(f"LTM failed to remove {memory_id} from vector DB: {e}")
            return False
    
    def search(
        self,
        query: Optional[str] = None,
        **kwargs
    ) -> Sequence[dict]:
        """Search for memories (vector-only LTM)"""
        if query:
            results = self.search_semantic(query=query, max_results=kwargs.get('max_results', 10))
            return results
        return []
    
    def search_semantic(
        self,
        query: str,
        max_results: int = 10,
        min_similarity: float = 0.5,
        memory_types: Optional[List[str]] = None,
        min_importance: float = 0.0
    ) -> List[dict]:
        """Semantic search using vector similarity (ChromaDB only)"""
        if not self.collection or not self.embedding_model:
            return []
        try:
            # Build where clause for filtering
            where_clause = {}
            if memory_types:
                where_clause["memory_type"] = {"$in": memory_types}
            if min_importance > 0.0:
                where_clause["importance"] = {"$gte": min_importance}
            
            query_embedding = self._generate_embedding(query)
            if not query_embedding:
                return []
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=max_results,
                where=where_clause if where_clause else None,
                include=["documents", "distances", "metadatas"]
            )
            
            search_results = []
            if results['ids'][0]:
                for i, memory_id in enumerate(results['ids'][0]):
                    distance = results['distances'][0][i]
                    similarity = max(0.0, 1.0 - distance)
                    if similarity >= min_similarity:
                        meta = results['metadatas'][0][i]
                        doc = results['documents'][0][i]
                        search_results.append({
                            "id": meta.get("memory_id", memory_id),
                            "content": doc,
                            "memory_type": meta.get("memory_type", "episodic"),
                            "encoding_time": meta.get("encoding_time"),
                            "last_access": meta.get("last_access"),
                            "importance": float(meta.get("importance", 0.5)),
                            "emotional_valence": float(meta.get("emotional_valence", 0.0)),
                            "source": meta.get("source", "unknown"),
                            "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                            "associations": meta.get("associations", "").split(",") if meta.get("associations") else [],
                            "similarity_score": similarity,
                            "distance": distance
                        })
            
            search_results.sort(key=lambda x: x["similarity_score"], reverse=True)
            return search_results
        except Exception as e:
            logger.error(f"LTM semantic search failed: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get current LTM status (vector DB only)"""
        status = {
            "vector_db_enabled": True,
            "collection_name": self.collection_name,
            "embedding_model": getattr(self.embedding_model, 'model_name', None) if self.embedding_model else None,
            "chroma_persist_dir": str(self.chroma_persist_dir)
        }
        if self.collection:
            try:
                collection_info = self.collection.get()
                status["vector_db_count"] = len(collection_info['ids']) if collection_info['ids'] else 0
            except Exception as e:
                status["vector_db_error"] = str(e)
        return status
    
    def consolidate_from_stm(self, stm_items: List[Any]) -> int:
        """Consolidate items from STM into LTM (vector DB only)"""
        consolidated = 0
        for item in stm_items:
            if getattr(item, "importance", 0.0) > 0.6 or getattr(item, "access_count", 0) > 2:
                item_id = getattr(item, "id", None)
                item_content = getattr(item, "content", None)
                if item_id and item_content:
                    success = self.store(
                        memory_id=str(item_id),
                        content=item_content,
                        memory_type='episodic',
                        importance=getattr(item, "importance", 0.5),
                        tags=[],
                        associations=getattr(item, 'associations', [])
                    )
                    if success:
                        consolidated += 1
        logger.info(f"LTM consolidated {consolidated} items from STM")
        return consolidated
    
    def remove(self, memory_id: str) -> bool:
        """Remove is an alias for delete in vector-only LTM"""
        return self.delete(memory_id)
    
    def clear(self):
        """Clear all memories from ChromaDB vector store"""
        if not self.collection:
            return
        try:
            result = self.collection.get()
            if result['ids']:
                self.collection.delete(ids=result['ids'])
            logger.debug("LTM cleared vector database")
        except Exception as e:
            logger.error(f"LTM failed to clear vector database: {e}")
