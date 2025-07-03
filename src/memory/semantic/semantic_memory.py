import json
import os
from typing import Dict, Any, Optional, List, Sequence
import uuid
from ..base import BaseMemorySystem

class SemanticMemorySystem(BaseMemorySystem):
    """
    Semantic memory system implementing the unified memory interface.
    Stores and retrieves factual knowledge as triples (subject, predicate, object).
    """

    def __init__(self, storage_path: str):
        """
        Initializes the SemanticMemorySystem.

        Args:
            storage_path (str): The path to the JSON file for storing the knowledge base.
        """
        self.storage_path = storage_path
        self.knowledge_base: Dict[str, Dict[str, Any]] = self._load_kb()

    def _load_kb(self) -> Dict[str, Dict[str, Any]]:
        """Loads the knowledge base from the JSON file."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError:
                    return {}
        return {}

    def _save_kb(self):
        """Saves the knowledge base to the JSON file."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, 'w', encoding='utf-8') as f:
            json.dump(self.knowledge_base, f, indent=4)

    def store_fact(self, subject: str, predicate: str, object_val: Any) -> str:
        """
        Stores a new fact in the knowledge base.

        Args:
            subject (str): The subject of the fact.
            predicate (str): The predicate or relationship.
            object_val (Any): The object or value of the fact.

        Returns:
            str: The unique ID of the stored fact.
        """
        fact_id = str(uuid.uuid4())
        self.knowledge_base[fact_id] = {
            "subject": subject.lower(),
            "predicate": predicate.lower(),
            "object": object_val
        }
        self._save_kb()
        return fact_id

    def retrieve_fact(self, fact_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves a fact by its unique ID.

        Args:
            fact_id (str): The ID of the fact to retrieve.

        Returns:
            Optional[Dict[str, Any]]: The fact dictionary or None if not found.
        """
        return self.knowledge_base.get(fact_id)

    def find_facts(self, subject: Optional[str] = None, predicate: Optional[str] = None, object_val: Optional[Any] = None) -> List[Dict[str, Any]]:
        """
        Finds all facts matching a given subject, predicate, and/or object.

        Args:
            subject (Optional[str]): The subject to search for.
            predicate (Optional[str]): The predicate to search for.
            object_val (Optional[Any]): The object to search for.

        Returns:
            List[Dict[str, Any]]: A list of matching facts.
        """
        results = []
        
        # Normalize search terms to lowercase for comparison
        search_subject = subject.lower() if subject else None
        search_predicate = predicate.lower() if predicate else None
        
        for fact_id, fact in self.knowledge_base.items():
            match = True
            
            if search_subject and fact.get("subject") != search_subject:
                match = False
            if search_predicate and fact.get("predicate") != search_predicate:
                match = False
            if object_val is not None and fact.get("object") != object_val:
                match = False
            
            if match:
                # Include the fact_id in the result
                result = fact.copy()
                result["fact_id"] = fact_id
                results.append(result)
        
        return results

    def store(self, *args, **kwargs) -> str:
        """
        Store a new fact (triple) in the knowledge base.
        Returns the unique fact ID.
        """
        # Accepts (subject, predicate, object_val) as positional or keyword args
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
        Retrieve a fact by its unique ID (memory_id).
        Returns the fact dict or None if not found.
        """
        return self.retrieve_fact(memory_id)

    def delete(self, memory_id: str) -> bool:
        """
        Delete a fact by its unique ID (memory_id).
        Returns True if deleted, False otherwise.
        """
        if memory_id in self.knowledge_base:
            del self.knowledge_base[memory_id]
            self._save_kb()
            return True
        return False

    def delete_fact(self, subject: str, predicate: str, object_val: Any) -> bool:
        """
        Delete a fact from the semantic memory system
    
        Args:
            subject: The subject of the fact
            predicate: The predicate/relationship
            object_val: The object value
    
        Returns:
            True if fact was found and deleted, False otherwise
        """
        try:
            # Find matching facts
            matching_facts = []
            for fact_id, fact in self.knowledge_base.items():
                if (fact["subject"] == subject and 
                    fact["predicate"] == predicate and 
                    fact["object"] == object_val):
                    matching_facts.append(fact_id)
        
            # Delete matching facts
            deleted_count = 0
            for fact_id in matching_facts:
                del self.knowledge_base[fact_id]
                deleted_count += 1
        
            # Save changes if any facts were deleted
            if deleted_count > 0:
                self._save_kb()
                return True
            else:
                return False
            
        except Exception:
            return False

    def search(self, query: Optional[str] = None, **kwargs) -> Sequence[dict | tuple]:
        """
        Search for facts. If query is None, use kwargs for subject/predicate/object_val.
        Returns a sequence of matching fact dicts.
        """
        if query:
            # Text-based search across all fact content
            results = []
            query_lower = query.lower()
            for fact_id, fact in self.knowledge_base.items():
                subject_match = query_lower in str(fact.get("subject", "")).lower()
                predicate_match = query_lower in str(fact.get("predicate", "")).lower()
                object_match = query_lower in str(fact.get("object", "")).lower()
                if subject_match or predicate_match or object_match:
                    result = fact.copy()
                    result["fact_id"] = fact_id
                    results.append(result)
            return results
        else:
            subject = kwargs.get('subject')
            predicate = kwargs.get('predicate')
            object_val = kwargs.get('object_val')
            return self.find_facts(subject, predicate, object_val)

    def clear(self):
        """Clears the entire knowledge base (for testing)."""
        self.knowledge_base = {}
        self._save_kb()
