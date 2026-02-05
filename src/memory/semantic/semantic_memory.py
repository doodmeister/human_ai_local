
from typing import Dict, Any, Optional, List, Sequence
import uuid
from ..base import BaseMemorySystem

# Import ChromaDB and embedding model
try:
    from sentence_transformers import SentenceTransformer
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required for vector semantic memory. "
        "Install it with: pip install sentence-transformers"
    ) from e

try:
    import chromadb
    from chromadb.config import Settings
except ImportError as e:
    raise ImportError(
        "chromadb is required for vector semantic memory. "
        "Install it with: pip install chromadb"
    ) from e


class SemanticMemorySystem(BaseMemorySystem):

    def shutdown(self):
        """Shutdown ChromaDB client to release file handles (for test cleanup)."""
        try:
            if getattr(self, "chroma_client", None) is not None:
                # Chroma shares a process-wide "System" instance per identifier.
                # Clearing the system cache is the supported teardown mechanism used by Chroma's
                # own tests to allow re-initialization against fresh temp directories.
                if hasattr(self.chroma_client, "clear_system_cache"):
                    self.chroma_client.clear_system_cache()
        finally:
            self.collection = None
            self.chroma_client = None
    """
    Semantic memory system using ChromaDB vector store.
    Stores and retrieves factual knowledge as triples (subject, predicate, object).
    """

    def __init__(self, chroma_persist_dir: Optional[str] = None, collection_name: str = "semantic_facts", embedding_model: str = "all-MiniLM-L6-v2"):
        """
        Initializes the SemanticMemorySystem with ChromaDB.
        Args:
            chroma_persist_dir (str): Directory for ChromaDB persistence.
            collection_name (str): ChromaDB collection name.
            embedding_model (str): SentenceTransformer model name.
        """
        from pathlib import Path
        self.chroma_persist_dir = Path(chroma_persist_dir or "data/memory_stores/chroma_semantic")
        self.collection_name = collection_name
        self.embedding_model: Optional[Any] = None

        # Initialize embedding model with GPU support if available
        try:
            import torch
            self.embedding_model = SentenceTransformer(embedding_model)
            if torch.cuda.is_available():
                self.embedding_model = self.embedding_model.to("cuda")
        except Exception as e:
            raise RuntimeError(f"SemanticMemorySystem failed to load embedding model: {e}") from e

        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self._initialize_chromadb()


    def _initialize_chromadb(self):
        """Initialize ChromaDB client and collection for semantic memory."""
        try:
            settings = Settings(
                allow_reset=True,
                anonymized_telemetry=False,
                persist_directory=str(self.chroma_persist_dir)
            )
            self.chroma_client = chromadb.PersistentClient(
                path=str(self.chroma_persist_dir),
                settings=settings
            )
            try:
                self.collection = self.chroma_client.get_collection(name=self.collection_name)
            except Exception:
                self.collection = self.chroma_client.create_collection(
                    name=self.collection_name,
                    metadata={"description": "Semantic memory facts (triples)"}
                )
        except Exception as e:
            raise RuntimeError(f"SemanticMemorySystem failed to initialize ChromaDB: {e}") from e


    def _generate_embedding(self, text: str) -> Optional[List[float]]:
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
        except Exception:
            return None

    def store_fact(self, subject: str, predicate: str, object_val: Any) -> str:
        """
        Stores a new fact in the vector store.
        Returns the unique fact ID.
        """
        if not self.collection or not self.embedding_model:
            return ""
        fact_id = str(uuid.uuid4())
        fact_text = f"{subject} {predicate} {object_val}"
        embedding = self._generate_embedding(fact_text)
        if not embedding:
            return ""
        metadata = {
            "subject": subject.lower(),
            "predicate": predicate.lower(),
            "object": str(object_val)
        }
        self.collection.upsert(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[fact_text],
            metadatas=[metadata]
        )
        return fact_id


    def retrieve_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves a fact by its unique ID from ChromaDB."""
        if not self.collection:
            return None
        result = self.collection.get(ids=[fact_id])
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        documents = result.get('documents') or []
        if ids and len(ids) > 0:
            meta = dict(metadatas[0]) if metadatas and len(metadatas) > 0 and metadatas[0] is not None else {}
            doc = str(documents[0]) if documents and len(documents) > 0 and documents[0] is not None else ""
            meta['fact_id'] = fact_id
            meta['fact_text'] = doc
            return meta
        return None


    def find_facts(self, subject: Optional[str] = None, predicate: Optional[str] = None, object_val: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Finds all facts matching a given subject, predicate, and/or object in ChromaDB.
        Normalizes subject and predicate to lowercase, and object_val to string for comparison.
        """
        if not self.collection:
            return []
        # Normalize query values
        norm_subject = subject.lower() if subject else None
        norm_predicate = predicate.lower() if predicate else None
        norm_object = str(object_val) if object_val is not None else None
        result = self.collection.get()
        ids = result.get('ids') or []
        metadatas = result.get('metadatas') or []
        documents = result.get('documents') or []
        facts = []
        for i, fact_id in enumerate(ids):
            meta = dict(metadatas[i]) if metadatas and len(metadatas) > i and metadatas[i] is not None else {}
            doc = str(documents[i]) if documents and len(documents) > i and documents[i] is not None else ""
            match = True
            if norm_subject and meta.get("subject") != norm_subject:
                match = False
            if norm_predicate and meta.get("predicate") != norm_predicate:
                match = False
            if norm_object is not None and meta.get("object") != norm_object:
                match = False
            if match:
                meta['fact_id'] = fact_id
                meta['fact_text'] = doc
                facts.append(meta)
        return facts


    def store(self, *args, **kwargs) -> str:
        """
        Store a new fact (triple) in the vector store. Returns the unique fact ID.
        """
        if args and len(args) == 3:
            subject, predicate, object_val = args
        else:
            subject = kwargs.get('subject')
            predicate = kwargs.get('predicate')
            object_val = kwargs.get('object_val')
        if not subject or not predicate:
            raise ValueError("Both subject and predicate are required to store a fact.")
        return self.store_fact(subject, predicate, object_val)


    def retrieve(self, memory_id: str) -> Optional[dict]:
        """
        Retrieve a fact by its unique ID (memory_id) from ChromaDB.
        """
        return self.retrieve_fact(memory_id)


    def delete(self, memory_id: str) -> bool:
        """
        Delete a fact by its unique ID (memory_id) from ChromaDB.
        Returns True if deleted, False otherwise.
        """
        if not self.collection:
            return False
        try:
            self.collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False


    def delete_fact(self, subject: str, predicate: str, object_val: Any) -> bool:
        """
        Delete a fact from the semantic memory system by triple.
        Returns True if fact was found and deleted, False otherwise.
        """
        # Find matching facts and delete them
        facts = self.find_facts(subject, predicate, object_val)
        deleted = False
        for fact in facts:
            fact_id = fact.get('fact_id')
            if fact_id:
                self.delete(fact_id)
                deleted = True
        return deleted


    def search(self, query: Optional[str] = None, **kwargs) -> Sequence[dict | tuple]:
        """
        Search for facts. If query is None, use kwargs for subject/predicate/object_val.
        Returns a sequence of matching fact dicts.
        """
        if not self.collection:
            return []
        if query:
            # Semantic search using embedding
            embedding = self._generate_embedding(query)
            if not embedding:
                return []
            results = self.collection.query(
                query_embeddings=[embedding],
                n_results=10,
                include=["documents", "metadatas"]
            )
            found = []
            ids = results.get('ids', [[]])
            metadatas = results.get('metadatas', [[]])
            documents = results.get('documents', [[]])
            if ids and ids[0]:
                for i, fact_id in enumerate(ids[0]):
                    meta = dict(metadatas[0][i]) if metadatas and len(metadatas[0]) > i and metadatas[0][i] is not None else {}
                    doc = str(documents[0][i]) if documents and len(documents[0]) > i and documents[0][i] is not None else ""
                    meta['fact_id'] = fact_id
                    meta['fact_text'] = doc
                    found.append(meta)
            return found
        else:
            subject = kwargs.get('subject')
            predicate = kwargs.get('predicate')
            object_val = kwargs.get('object_val')
            return self.find_facts(subject, predicate, object_val)


    def clear(self):
        """Clears the entire semantic memory collection (for testing)."""
        if not self.collection:
            return
        result = self.collection.get()
        if result.get('ids'):
            self.collection.delete(ids=result['ids'])
