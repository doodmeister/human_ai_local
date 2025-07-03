import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

import pytest
from src.memory.stm.vector_stm import VectorShortTermMemory
from datetime import datetime

def test_similarity_search():
    stm = VectorShortTermMemory(
        capacity=10,
        use_vector_db=False,  # Disable ChromaDB for this test (fallback search)
    )
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
    # Assert that the correct memory is found with nonzero relevance
    assert any("Kevin Lee Swaim" in item.content for item, score in results), "Kevin's full name not found in STM search"
    assert any(score > 0 for item, score in results), "No relevant STM memory found"
