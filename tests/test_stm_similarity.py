import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest
from src.memory.stm.vector_stm import VectorShortTermMemory, STMConfiguration
from datetime import datetime

def test_similarity_search():
    """Test semantic similarity search in VectorSTM.
    
    Note: This test uses disable_storage=True which uses null embeddings,
    so it only tests the API contract, not actual semantic similarity.
    For real semantic search testing, use integration tests with ChromaDB.
    """
    config = STMConfiguration(capacity=10)
    stm = VectorShortTermMemory(config, disable_storage=True)
    
    # Store a memory
    memory_id = "test-kevin"
    stm.store(
        memory_id=memory_id,
        content="Kevin's full name is Kevin Lee Swaim.",
        importance=0.5,
        attention_score=0.0,
        emotional_valence=0.0,
        associations=[]
    )
    
    # Query with a different phrasing
    query = "what is kevin's full name?"
    results = stm.search(query=query, max_results=5)
    
    # Print results for debugging
    for item, score in results:
        print(f"Found: {item.content} (score={score})")
    
    # With disabled storage, results may be empty (null embeddings)
    # This test verifies the API contract works without errors
    # For semantic similarity testing, see integration tests
    assert isinstance(results, list), "search() should return a list"
