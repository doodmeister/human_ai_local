
from datetime import datetime, timezone
import json
from typing import Dict, Any, Optional, List, Sequence
import threading
import uuid
from ..base import BaseMemorySystem
from ..schema.contradiction import evaluate_belief_revision, merge_source_history

# sentence_transformers lazy-loaded to avoid 10-30s startup (torch/sklearn/pandas)
SentenceTransformer = None


def _ensure_sentence_transformers():
    global SentenceTransformer
    if SentenceTransformer is not None:
        return
    try:
        from sentence_transformers import SentenceTransformer as _ST
        SentenceTransformer = _ST
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

    def __init__(
        self,
        chroma_persist_dir: Optional[str] = None,
        collection_name: str = "semantic_facts",
        embedding_model: str = "all-MiniLM-L6-v2",
        lazy_embeddings: bool = True
    ):
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
        self._embedding_model_name = embedding_model
        self._lazy_embeddings = lazy_embeddings
        self._embedding_lock = threading.Lock()

        # Initialize embedding model with GPU support if available
        if not self._lazy_embeddings:
            if not self._ensure_embedding_model():
                raise RuntimeError("SemanticMemorySystem failed to load embedding model")

        # Initialize ChromaDB
        self.chroma_client = None
        self.collection = None
        self._last_revision_event: Optional[Dict[str, Any]] = None
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
        if not text:
            return None
        if not self._ensure_embedding_model():
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

    def _ensure_embedding_model(self) -> bool:
        if self.embedding_model is not None:
            return True
        try:
            _ensure_sentence_transformers()
        except ImportError:
            return False
        if SentenceTransformer is None:
            return False
        with self._embedding_lock:
            if self.embedding_model is not None:
                return True
            try:
                import torch
                self.embedding_model = SentenceTransformer(self._embedding_model_name)
                if torch.cuda.is_available():
                    self.embedding_model = self.embedding_model.to("cuda")
                return True
            except Exception:
                self.embedding_model = None
                return False

    def _serialize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        serialized: Dict[str, Any] = {}
        for key, value in metadata.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                serialized[key] = value
                continue
            serialized[key] = json.dumps(value, sort_keys=True)
        return serialized

    def _decode_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        decoded = dict(metadata)
        for key, value in list(decoded.items()):
            if not key.endswith("_json") or not isinstance(value, str):
                continue
            try:
                decoded[key[:-5]] = json.loads(value)
            except json.JSONDecodeError:
                continue
        return decoded

    def _upsert_fact_record(self, fact_id: str, fact_text: str, metadata: Dict[str, Any]) -> str:
        embedding = self._generate_embedding(fact_text)
        if not embedding:
            return ""
        self.collection.upsert(
            ids=[fact_id],
            embeddings=[embedding],
            documents=[fact_text],
            metadatas=[self._serialize_metadata(metadata)],
        )
        return fact_id

    def _set_last_revision_event(self, event: Optional[Dict[str, Any]]) -> None:
        self._last_revision_event = dict(event) if event else None

    def get_last_revision_event(self) -> Optional[Dict[str, Any]]:
        event = getattr(self, "_last_revision_event", None)
        return dict(event) if isinstance(event, dict) else None

    @staticmethod
    def _clamp_unit(value: Any, default: float = 0.0) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            numeric = default
        return max(0.0, min(1.0, numeric))

    def _apply_quarantine_weakening(
        self,
        metadata: Dict[str, Any],
        *,
        candidate_source: str | None,
    ) -> Dict[str, Any]:
        weakened = dict(metadata)
        if str(weakened.get("belief_status", "")).lower() != "quarantined":
            return weakened

        source_key = str(candidate_source or "").strip().lower()
        confidence_delta = 0.16 if source_key == "explicit_user_correction" else 0.08
        importance_delta = 0.10 if source_key == "explicit_user_correction" else 0.05

        weakened["confidence"] = max(
            0.1,
            self._clamp_unit(weakened.get("confidence"), default=0.55) - confidence_delta,
        )
        weakened["importance"] = max(
            0.1,
            self._clamp_unit(weakened.get("importance"), default=0.6) - importance_delta,
        )
        weakened["revision_feedback"] = (
            "superseded_by_explicit_user_correction"
            if source_key == "explicit_user_correction"
            else "superseded_by_higher_weighted_belief"
        )
        weakened["last_access"] = datetime.now(timezone.utc).isoformat()
        return weakened

    def _default_confidence(self, source: str) -> float:
        if source == "explicit_user_correction":
            return 0.9
        if source == "user_assertion":
            return 0.82
        if source == "chat_capture":
            return 0.72
        if source == "inferred":
            return 0.5
        return 0.65

    def _prepare_fact_metadata(
        self,
        *,
        subject: str,
        predicate: str,
        object_value: Any,
        fact_text: str,
        metadata: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        merged = dict(metadata or {})
        source = str(merged.get("source") or "semantic")
        confidence = float(merged.get("confidence", self._default_confidence(source)))
        importance = float(merged.get("importance", 0.6))
        prepared: Dict[str, Any] = {
            "subject": str(subject).strip().lower(),
            "predicate": str(predicate).strip().lower(),
            "object": str(object_value).strip(),
            "source": source,
            "confidence": max(0.0, min(1.0, confidence)),
            "importance": max(0.0, min(1.0, importance)),
            "fact_text": fact_text,
            "encoding_time": str(merged.get("encoding_time") or now),
            "last_access": str(merged.get("last_access") or now),
            "belief_status": str(merged.get("belief_status") or "active"),
            "support_count": int(merged.get("support_count") or 1),
        }
        prepared.update(merged)
        prepared["subject"] = str(subject).strip().lower()
        prepared["predicate"] = str(predicate).strip().lower()
        prepared["object"] = str(object_value).strip()
        prepared["fact_text"] = fact_text
        prepared["source"] = source
        prepared["confidence"] = max(0.0, min(1.0, float(prepared.get("confidence", confidence))))
        prepared["importance"] = max(0.0, min(1.0, float(prepared.get("importance", importance))))
        prepared["support_count"] = int(prepared.get("support_count") or 1)
        return prepared

    def store_fact(
        self,
        subject: str,
        predicate: str,
        object_val: Any,
        *,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        fact_id: Optional[str] = None,
    ) -> str:
        """
        Stores a new fact in the vector store.
        Returns the unique fact ID.
        """
        if not self.collection or not self._ensure_embedding_model():
            return ""
        candidate_fact_id = str(fact_id or uuid.uuid4())
        fact_text = str(content or f"{subject} {predicate} {object_val}").strip()
        self._set_last_revision_event(None)
        prepared = self._prepare_fact_metadata(
            subject=subject,
            predicate=predicate,
            object_value=object_val,
            fact_text=fact_text,
            metadata=metadata,
        )
        existing_facts = self.find_facts(subject=subject, predicate=predicate)
        decision = evaluate_belief_revision(
            subject=prepared["subject"],
            predicate=prepared["predicate"],
            object_value=prepared["object"],
            candidate_fact_id=candidate_fact_id,
            candidate_source=prepared.get("source"),
            candidate_confidence=prepared.get("confidence"),
            existing_facts=existing_facts,
            relationship_target=prepared.get("relationship_target"),
        )

        for existing in existing_facts:
            existing_id = str(existing.get("fact_id") or existing.get("id") or "")
            if not existing_id:
                continue
            updates = decision.fact_updates.get(existing_id)
            if not updates:
                continue
            merged_existing = dict(existing)
            merged_existing.update(updates)
            merged_existing = self._apply_quarantine_weakening(
                merged_existing,
                candidate_source=prepared.get("source"),
            )
            if "source" in merged_existing:
                merged_existing["source_history_json"] = json.dumps(
                    merge_source_history(merged_existing.get("source_history_json"), merged_existing.get("source")),
                    sort_keys=True,
                )
            existing_text = str(
                merged_existing.get("fact_text")
                or f"{merged_existing.get('subject', '')} {merged_existing.get('predicate', '')} {merged_existing.get('object', '')}"
            ).strip()
            if not self._upsert_fact_record(existing_id, existing_text, merged_existing):
                return ""

        if decision.candidate_status == "merged" and decision.merge_into_fact_id:
            merged_fact = next(
                (fact for fact in existing_facts if str(fact.get("fact_id") or fact.get("id") or "") == decision.merge_into_fact_id),
                None,
            )
            if merged_fact is not None:
                merged_meta = dict(merged_fact)
                merged_meta.update(decision.fact_updates.get(decision.merge_into_fact_id, {}))
                merged_meta["source_history_json"] = json.dumps(
                    merge_source_history(merged_meta.get("source_history_json"), prepared.get("source")),
                    sort_keys=True,
                )
                merged_text = str(merged_meta.get("fact_text") or fact_text)
                if not self._upsert_fact_record(decision.merge_into_fact_id, merged_text, merged_meta):
                    return ""
            self._set_last_revision_event(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "candidate_fact_id": candidate_fact_id,
                    "stored_fact_id": decision.merge_into_fact_id,
                    "candidate_status": decision.candidate_status,
                    "revision_reason": decision.rationale,
                    "source": prepared.get("source"),
                    "conflicting_fact_ids": list(decision.conflicting_fact_ids),
                    "winning_fact_id": decision.winning_fact_id,
                    "merge_into_fact_id": decision.merge_into_fact_id,
                }
            )
            return decision.merge_into_fact_id

        prepared.update(
            {
                "contradiction_set_id": decision.contradiction_set_id,
                "belief_status": decision.candidate_status,
                "confidence": decision.candidate_confidence,
                "source_weight": decision.source_weight,
                "revision_reason": decision.rationale,
                "source_history_json": json.dumps(
                    merge_source_history(prepared.get("source_history_json"), prepared.get("source")),
                    sort_keys=True,
                ),
            }
        )
        if decision.conflicting_fact_ids:
            prepared["conflicting_fact_ids_json"] = json.dumps(decision.conflicting_fact_ids, sort_keys=True)
        if decision.winning_fact_id:
            prepared["winning_fact_id"] = decision.winning_fact_id
        if decision.candidate_status == "quarantined":
            prepared["quarantine_reason"] = decision.rationale
        else:
            prepared.pop("quarantine_reason", None)

        stored_id = self._upsert_fact_record(candidate_fact_id, fact_text, prepared)
        if stored_id:
            self._set_last_revision_event(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "candidate_fact_id": candidate_fact_id,
                    "stored_fact_id": stored_id,
                    "candidate_status": decision.candidate_status,
                    "revision_reason": decision.rationale,
                    "source": prepared.get("source"),
                    "conflicting_fact_ids": list(decision.conflicting_fact_ids),
                    "winning_fact_id": decision.winning_fact_id,
                    "merge_into_fact_id": decision.merge_into_fact_id,
                }
            )
        return stored_id

    def apply_recall_feedback(
        self,
        fact_id: str,
        *,
        outcome: str = "reinforce",
        importance_delta: float = 0.0,
        confidence_delta: float = 0.0,
        note: Optional[str] = None,
    ) -> bool:
        fact = self.retrieve_fact(fact_id)
        if not fact:
            return False

        updated = dict(fact)
        updated.pop("fact_id", None)
        updated.pop("fact_text", None)
        updated["last_access"] = datetime.now(timezone.utc).isoformat()
        updated["importance"] = max(
            0.0,
            min(1.0, float(updated.get("importance", 0.6)) + float(importance_delta)),
        )
        updated["confidence"] = max(
            0.0,
            min(1.0, float(updated.get("confidence", 0.65)) + float(confidence_delta)),
        )
        updated["recall_feedback"] = str(outcome)
        if note:
            updated["recall_feedback_note"] = str(note)

        stored_id = self._upsert_fact_record(fact_id, str(fact.get("fact_text") or ""), updated)
        if stored_id:
            self._set_last_revision_event(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "candidate_fact_id": fact_id,
                    "stored_fact_id": stored_id,
                    "candidate_status": updated.get("belief_status", "active"),
                    "revision_reason": f"recall_{outcome}",
                    "source": updated.get("source"),
                    "conflicting_fact_ids": [],
                    "winning_fact_id": updated.get("winning_fact_id"),
                    "merge_into_fact_id": None,
                }
            )
        return bool(stored_id)


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
            meta = self._decode_metadata(meta)
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
            meta = self._decode_metadata(meta)
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
                if not meta.get("suppressed") and str(meta.get("belief_status", "active")).lower() != "quarantined":
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
        return self.store_fact(
            subject,
            predicate,
            object_val,
            content=kwargs.get("content"),
            metadata=kwargs.get("metadata"),
            fact_id=kwargs.get("fact_id"),
        )


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
                    meta = self._decode_metadata(meta)
                    doc = str(documents[0][i]) if documents and len(documents[0]) > i and documents[0][i] is not None else ""
                    meta['fact_id'] = fact_id
                    meta['fact_text'] = doc
                    if not meta.get("suppressed") and str(meta.get("belief_status", "active")).lower() != "quarantined":
                        found.append(meta)
            return found
        else:
            subject = kwargs.get('subject')
            predicate = kwargs.get('predicate')
            object_val = kwargs.get('object_val')
            return self.find_facts(subject, predicate, object_val)

    def apply_forgetting_policy(
        self,
        *,
        min_importance: float = 0.3,
        min_confidence: float = 0.35,
        min_access_count: int = 1,
        min_age_days: float = 14.0,
    ) -> Dict[str, int]:
        del min_access_count
        if not self.collection:
            return {"suppressed": 0, "protected": 0}

        now = datetime.now(timezone.utc)
        min_age_seconds = float(min_age_days) * 86400.0
        suppressed = 0
        protected = 0

        result = self.collection.get()
        ids = result.get("ids") or []
        metadatas = result.get("metadatas") or []
        documents = result.get("documents") or []

        for i, fact_id in enumerate(ids):
            meta = dict(metadatas[i]) if len(metadatas) > i and metadatas[i] is not None else {}
            meta = self._decode_metadata(meta)
            doc = str(documents[i]) if len(documents) > i and documents[i] is not None else ""

            is_anchor = bool(
                str(meta.get("source", "")).lower() == "explicit_user_correction"
                or (str(meta.get("subject", "")).lower() == "user" and float(meta.get("confidence", 0.0) or 0.0) >= 0.8)
                or meta.get("relationship_target")
            )
            if is_anchor:
                protected += 1
                meta["suppressed"] = False
                meta.pop("suppression_reason", None)
                self._upsert_fact_record(fact_id, doc, meta)
                continue

            last_access = meta.get("last_access") or meta.get("encoding_time")
            age_seconds = 0.0
            if last_access:
                try:
                    age_seconds = (now - datetime.fromisoformat(str(last_access))).total_seconds()
                except Exception:
                    age_seconds = 0.0

            importance = float(meta.get("importance", 0.6) or 0.6)
            confidence = float(meta.get("confidence", 0.65) or 0.65)
            if age_seconds >= min_age_seconds and importance <= min_importance and confidence <= min_confidence:
                if not meta.get("suppressed"):
                    suppressed += 1
                meta["suppressed"] = True
                meta["suppressed_at"] = now.isoformat()
                meta["suppression_reason"] = "low_value_decay"
                self._upsert_fact_record(fact_id, doc, meta)

        return {"suppressed": suppressed, "protected": protected}


    def clear(self):
        """Clears the entire semantic memory collection (for testing)."""
        if not self.collection:
            return
        result = self.collection.get()
        if result.get('ids'):
            self.collection.delete(ids=result['ids'])
