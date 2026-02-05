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
    def suggest_cross_system_associations(self, external_memories: List[Dict[str, Any]], system_type: str) -> List[Dict[str, Any]]:
        """Suggest potential associations between LTM and external system memories."""
        if not self.collection:
            return []
        suggestions = []
        try:
            result = self.collection.get()
            for ext_memory in external_memories:
                ext_id = ext_memory.get("id", "")
                ext_content = str(ext_memory.get("content", "")).lower()
                ext_tags = ext_memory.get("tags", [])
                for i, memory_id in enumerate(result['ids']):
                    meta = result['metadatas'][i]
                    doc = result['documents'][i]
                    ltm_content = str(doc).lower()
                    similarity_score = 0.0
                    ext_words = set(ext_content.split())
                    ltm_words = set(ltm_content.split())
                    word_overlap = 0
                    if ext_words and ltm_words:
                        word_overlap = len(ext_words.intersection(ltm_words))
                        similarity_score += word_overlap / max(len(ext_words), len(ltm_words))
                    tag_overlap = len(set(ext_tags).intersection(set(meta.get('tags', '').split(','))))
                    if tag_overlap > 0:
                        similarity_score += tag_overlap * 0.3
                    valence_similarity = 0
                    if "emotional_valence" in ext_memory:
                        valence_diff = abs(float(ext_memory["emotional_valence"]) - float(meta.get("emotional_valence", 0.0)))
                        valence_similarity = 1 - valence_diff
                        similarity_score += valence_similarity * 0.2
                    if similarity_score > 0.3:
                        suggestions.append({
                            "ltm_memory_id": memory_id,
                            "external_memory_id": ext_id,
                            "system_type": system_type,
                            "confidence": min(similarity_score, 1.0),
                            "similarity_factors": {
                                "content_overlap": word_overlap,
                                "tag_overlap": tag_overlap,
                                "emotional_similarity": valence_similarity
                            }
                        })
            suggestions.sort(key=lambda x: x["confidence"], reverse=True)
            return suggestions[:20]
        except Exception as e:
            logger.error(f"LTM cross-system association suggestion failed: {e}")
            return []
    def get_semantic_clusters(self, min_cluster_size: int = 2) -> Dict[str, List[str]]:
        """Identify semantic clusters by tags and content keywords."""
        if not self.collection:
            return {}
        try:
            result = self.collection.get()
            tag_clusters = {}
            content_clusters = {}
            for i, memory_id in enumerate(result['ids']):
                meta = result['metadatas'][i]
                tags = meta.get('tags', '').split(',') if meta.get('tags') else []
                for tag in tags:
                    if tag:
                        tag_clusters.setdefault(tag, []).append(memory_id)
                doc = result['documents'][i]
                content_words = str(doc).lower().split()
                keywords = [word for word in content_words if len(word) > 3]
                for keyword in keywords:
                    content_clusters.setdefault(keyword, []).append(memory_id)
            significant_clusters = {}
            for tag, memory_ids in tag_clusters.items():
                if len(memory_ids) >= min_cluster_size:
                    significant_clusters[f"tag:{tag}"] = memory_ids
            for keyword, memory_ids in content_clusters.items():
                if len(memory_ids) >= min_cluster_size:
                    significant_clusters[f"content:{keyword}"] = memory_ids
            return significant_clusters
        except Exception as e:
            logger.error(f"LTM semantic clustering failed: {e}")
            return {}
    def decay_memories(self, decay_rate: float = 0.01, half_life_days: float = 30.0, min_importance: float = 0.05, min_confidence: float = 0.1) -> int:
        """Decay importance and confidence for old, rarely accessed memories in ChromaDB."""
        if not self.collection:
            return 0
        from datetime import datetime
        now = datetime.now()
        decayed = 0
        half_life_seconds = half_life_days * 86400
        try:
            result = self.collection.get()
            for i, memory_id in enumerate(result['ids']):
                meta = result['metadatas'][i]
                last_access = meta.get('last_access')
                if last_access:
                    seconds_since_access = (now - datetime.fromisoformat(last_access)).total_seconds()
                else:
                    seconds_since_access = 0
                if seconds_since_access > 86400:
                    decay_factor = 0.5 ** (seconds_since_access / half_life_seconds)
                    old_importance = float(meta.get('importance', 0.5))
                    old_confidence = float(meta.get('confidence', 1.0))
                    new_importance = max(min_importance, old_importance * (1 - decay_rate) * decay_factor)
                    new_confidence = max(min_confidence, old_confidence * (1 - decay_rate/2) * decay_factor)
                    if new_importance < old_importance or new_confidence < old_confidence:
                        meta['importance'] = new_importance
                        meta['confidence'] = new_confidence
                        self.collection.update(ids=[memory_id], metadatas=[meta])
                        decayed += 1
            logger.info(f"Decayed {decayed} LTM memories (rate={decay_rate}, half_life_days={half_life_days})")
            return decayed
        except Exception as e:
            logger.error(f"LTM decay failed: {e}")
            return 0
    def get_memory_health_report(self) -> dict:
        """Generate a memory health report using ChromaDB metadata."""
        if not self.collection:
            return {}
        try:
            result = self.collection.get()
            now = datetime.now()
            never_accessed = []
            rarely_accessed = []
            frequently_accessed = []
            stale_memories = []
            low_confidence = []
            type_distribution = {}
            for i, memory_id in enumerate(result['ids']):
                meta = result['metadatas'][i]
                access_count = float(meta.get('access_count', 0))
                confidence = float(meta.get('confidence', 1.0))
                last_access = meta.get('last_access')
                if last_access:
                    days_since_access = (now - datetime.fromisoformat(last_access)).days
                else:
                    days_since_access = 0
                if access_count == 0:
                    never_accessed.append(memory_id)
                elif access_count < 3:
                    rarely_accessed.append(memory_id)
                elif access_count >= 10:
                    frequently_accessed.append(memory_id)
                if days_since_access >= 30:
                    stale_memories.append(memory_id)
                if confidence < 0.3:
                    low_confidence.append(memory_id)
                mtype = meta.get('memory_type', 'unknown')
                type_distribution[mtype] = type_distribution.get(mtype, 0) + 1
            total = len(result['ids'])
            return {
                "memory_categories": {
                    "never_accessed": len(never_accessed),
                    "rarely_accessed": len(rarely_accessed),
                    "frequently_accessed": len(frequently_accessed),
                    "stale_memories": len(stale_memories),
                    "low_confidence": len(low_confidence)
                },
                "memory_type_distribution": type_distribution,
                "potential_issues": {
                    "high_stale_ratio": len(stale_memories) / max(1, total) > 0.5,
                    "low_utilization": sum(float(result['metadatas'][i].get('access_count', 0)) for i in range(total)) / max(1, total) < 2,
                    "confidence_degradation": len(low_confidence) / max(1, total) > 0.3
                },
                "recommendations": []
            }
        except Exception as e:
            logger.error(f"LTM health report failed: {e}")
            return {}
    def create_cross_system_link(self, ltm_memory_id: str, external_memory_id: str, link_type: str = "association") -> bool:
        """Create a link between an LTM memory and external system memory (association in metadata)."""
        if not self.collection:
            return False
        rec = self.retrieve(ltm_memory_id)
        if not rec:
            return False
        associations = rec.get("associations", [])
        if isinstance(associations, str):
            associations = [a for a in associations.split(",") if a]
        if external_memory_id not in associations:
            associations.append(external_memory_id)
            meta = rec.copy()
            # Sanitize all metadata fields to allowed types for ChromaDB
            for key in list(meta.keys()):
                v = meta[key]
                if isinstance(v, list):
                    meta[key] = ",".join(str(x) for x in v) if v else ""
                elif not isinstance(v, (str, int, float, bool)) and v is not None:
                    meta[key] = str(v)
            meta["associations"] = ",".join(str(x) for x in associations if x)
            self.collection.update(
                ids=[ltm_memory_id],
                metadatas=[meta]
            )
            logger.debug(f"Created {link_type} link: LTM {ltm_memory_id} -> {external_memory_id}")
            return True
        return False

    def find_cross_system_links(self, external_memory_id: str, system_type: str = "any") -> List[dict]:
        """Find LTM records linked to memories from other systems (by association or content)."""
        if not self.collection:
            return []
        # Use get() with where clause for metadata-only filtering
        where = {"associations": {"$in": [external_memory_id]}}
        results = self.collection.get(where=where)
        matches = []
        if results['ids']:
            for i, memory_id in enumerate(results['ids']):
                meta = results['metadatas'][i]
                doc = results['documents'][i]
                matches.append({
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
                })
        return matches
    def add_feedback(self, memory_id: str, feedback_type: str, value: Any, comment: Optional[str] = None, user_id: Optional[str] = None):
        """Add user feedback to a memory record (stored in ChromaDB metadata as JSON string)."""
        if not self.collection:
            return
        rec = self.retrieve(memory_id)
        if not rec:
            return
        import json
        feedback = rec.get("feedback", [])
        # Defensive: always ensure feedback is a list
        if isinstance(feedback, str):
            try:
                feedback = json.loads(feedback)
            except Exception:
                feedback = []
        elif not isinstance(feedback, list):
            feedback = []
        from datetime import datetime
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": feedback_type,
            "value": value,
            "comment": comment,
            "user_id": user_id
        }
        feedback.append(event)
        meta = rec.copy()
        meta["feedback"] = json.dumps(feedback) if feedback else "[]"
        # Update confidence, importance, or emotional_valence if feedback_type matches
        if feedback_type in ("confidence", "emotion"):
            try:
                meta["confidence"] = float(value)
            except Exception:
                pass
        if feedback_type == "emotion":
            try:
                meta["emotional_valence"] = float(value)
            except Exception:
                pass
        if feedback_type == "importance":
            meta["importance"] = 1.0
        # Sanitize all metadata fields to allowed types for ChromaDB
        for key in list(meta.keys()):
            v = meta[key]
            if isinstance(v, list):
                meta[key] = ",".join(str(x) for x in v) if v else ""
            elif not isinstance(v, (str, int, float, bool)) and v is not None:
                meta[key] = str(v)
        self.collection.update(
            ids=[memory_id],
            metadatas=[meta]
        )

    def get_feedback(self, memory_id: str) -> list:
        """Return all feedback events for a memory."""
        rec = self.retrieve(memory_id)
        if not rec:
            return []
        return rec.get("feedback", [])

    def get_feedback_summary(self, memory_id: str) -> dict:
        """Return summary statistics for feedback on a memory."""
        rec = self.retrieve(memory_id)
        if not rec:
            return {}
        summary = {}
        for event in rec.get("feedback", []):
            t = event["type"]
            summary.setdefault(t, []).append(event["value"])
        stats = {k: (sum(map(float, v))/len(v) if v else 0) for k, v in summary.items()}
        stats["count"] = len(rec.get("feedback", []))
        return stats
    def search_by_tags(self, tags: List[str], operator: str = "OR") -> List[dict]:
        """
        Search memories by tags using ChromaDB metadata filtering.
        Args:
            tags: List of tags to search for
            operator: "OR" (any tag) or "AND" (all tags)
        Returns:
            List of matching memory dicts
        """
        if not self.collection:
            return []
        try:
            # Fetch all records (could be optimized with a where clause if ChromaDB supports substring search)
            results = self.collection.get()
            matches = []
            for i, memory_id in enumerate(results['ids']):
                meta = results['metadatas'][i]
                doc = results['documents'][i]
                tag_list = meta.get("tags", "").split(",") if meta.get("tags") else []
                if operator == "OR":
                    if any(tag in tag_list for tag in tags):
                        matches.append({
                            "id": meta.get("memory_id", memory_id),
                            "content": doc,
                            "memory_type": meta.get("memory_type", "episodic"),
                            "encoding_time": meta.get("encoding_time"),
                            "last_access": meta.get("last_access"),
                            "importance": float(meta.get("importance", 0.5)),
                            "emotional_valence": float(meta.get("emotional_valence", 0.0)),
                            "source": meta.get("source", "unknown"),
                            "tags": tag_list,
                            "associations": meta.get("associations", "").split(",") if meta.get("associations") else []
                        })
                else:  # AND
                    if all(tag in tag_list for tag in tags):
                        matches.append({
                            "id": meta.get("memory_id", memory_id),
                            "content": doc,
                            "memory_type": meta.get("memory_type", "episodic"),
                            "encoding_time": meta.get("encoding_time"),
                            "last_access": meta.get("last_access"),
                            "importance": float(meta.get("importance", 0.5)),
                            "emotional_valence": float(meta.get("emotional_valence", 0.0)),
                            "source": meta.get("source", "unknown"),
                            "tags": tag_list,
                            "associations": meta.get("associations", "").split(",") if meta.get("associations") else []
                        })
            return matches
        except Exception as e:
            logger.error(f"LTM tag search failed: {e}")
            return []

    def get_associations(self, memory_id: str, depth: int = 1) -> List[dict]:
        """
        Traverse associations for a memory up to a given depth.
        Args:
            memory_id: Starting memory ID
            depth: Depth of association traversal
        Returns:
            List of associated memory dicts
        """
        if not self.collection:
            return []
        visited = set()
        to_visit = [(memory_id, 0)]
        associated = []
        while to_visit:
            current_id, current_depth = to_visit.pop(0)
            if current_id in visited or current_depth > depth:
                continue
            visited.add(current_id)
            if current_depth > 0:
                rec = self.retrieve(current_id)
                if rec:
                    associated.append(rec)
            # Get associations for this memory
            rec = self.retrieve(current_id)
            if rec and rec.get("associations"):
                for assoc_id in rec["associations"]:
                    if assoc_id and assoc_id not in visited:
                        to_visit.append((assoc_id, current_depth + 1))
        return associated
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
            # Always store tags and associations as comma-separated strings
            tags_str = ",".join(tags) if tags else ""
            associations_str = ",".join(associations) if associations else ""
            metadata = {
                "memory_id": memory_id,
                "memory_type": memory_type,
                "importance": float(importance),
                "emotional_valence": float(emotional_valence),
                "source": source,
                "encoding_time": datetime.now().isoformat(),
                "last_access": datetime.now().isoformat(),
                "tags": tags_str,
                "associations": associations_str
            }
            # Sanitize all metadata fields to allowed types for ChromaDB
            for key in list(metadata.keys()):
                v = metadata[key]
                if isinstance(v, list):
                    metadata[key] = ",".join(str(x) for x in v) if v else ""
                elif not isinstance(v, (str, int, float, bool)) and v is not None:
                    metadata[key] = str(v)
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
        """Retrieve memory from ChromaDB only, parse feedback JSON if present."""
        if not self.collection:
            return None
        try:
            result = self.collection.get(ids=[memory_id])
            if result['ids'] and len(result['ids']) > 0:
                meta = result['metadatas'][0]
                doc = result['documents'][0]
                import json
                feedback = meta.get("feedback", None)
                if feedback is not None:
                    if isinstance(feedback, str):
                        try:
                            feedback = json.loads(feedback)
                        except Exception:
                            feedback = []
                    elif not isinstance(feedback, list):
                        feedback = []
                else:
                    feedback = []
                return {
                    "id": meta.get("memory_id", memory_id),
                    "content": doc,
                    "memory_type": meta.get("memory_type", "episodic"),
                    "encoding_time": meta.get("encoding_time"),
                    "last_access": meta.get("last_access"),
                    "importance": float(meta.get("importance", 0.5)),
                    "emotional_valence": float(meta.get("emotional_valence", 0.0)),
                    "confidence": float(meta["confidence"]) if "confidence" in meta and meta["confidence"] is not None else None,
                    "source": meta.get("source", "unknown"),
                    "tags": meta.get("tags", "").split(",") if meta.get("tags") else [],
                    "associations": meta.get("associations", "").split(",") if meta.get("associations") else [],
                    "feedback": feedback
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
            # Phase 7: utility-based learning law.
            # Benefit: importance + access evidence (retrieval success proxy).
            # Cost: fixed small write cost.
            try:
                from src.learning.learning_law import clamp01, utility_score
            except Exception:  # pragma: no cover
                def clamp01(x: Any) -> float:  # type: ignore[no-redef]
                    try:
                        return float(x)
                    except (TypeError, ValueError):
                        return 0.0

                def utility_score(*, benefit: Any, cost: Any, benefit_weight: float = 1.0, cost_weight: float = 1.0) -> float:  # type: ignore[no-redef]
                    return (benefit_weight * clamp01(benefit)) - (cost_weight * clamp01(cost))

            imp = float(getattr(item, "importance", 0.0) or 0.0)
            access = float(getattr(item, "access_count", 0) or 0.0)
            benefit = clamp01(max(imp, min(1.0, access / 3.0)))
            cost = 0.10
            u = utility_score(benefit=benefit, cost=cost)

            # Preserve prior thresholds by using a conservative utility gate.
            if u >= 0.50:
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
