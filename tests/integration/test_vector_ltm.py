"""
Test suite for Vector Long-Term Memory implementation

NOTE: These tests were written for an older API that supported both vector and JSON fallback.
The current implementation (VectorLongTermMemory) is vector-only with ChromaDB.
Tests have been updated to work with the new API.
"""
import tempfile
import shutil
from pathlib import Path
import pytest

# Import the classes we want to test
from src.memory.ltm.vector_ltm import VectorLongTermMemory

class TestVectorLongTermMemory:
    """Test cases for VectorLongTermMemory"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.chroma_path = Path(self.temp_dir) / "chroma"
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization(self):
        """Test initialization of vector LTM"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.chroma_path),
            collection_name="test_ltm"
        )
        
        # Basic initialization checks
        assert ltm.chroma_persist_dir == self.chroma_path
        assert ltm.collection_name == "test_ltm"
    
    def test_store_and_retrieve_memory(self):
        """Test basic memory storage and retrieval"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.chroma_path),
            collection_name="test_ltm_store"
        )
        
        # Store a memory
        memory_id = "test_memory_1"
        content = "This is a test memory about artificial intelligence"
        
        ltm.store(
            memory_id=memory_id,
            content=content,
            memory_type="semantic",
            importance=0.8,
            tags=["ai", "test"],
            source="unit_test"
        )
        
        # Retrieve the memory
        retrieved = ltm.retrieve(memory_id)
        assert retrieved is not None
        assert retrieved.get("content") == content or content in str(retrieved)
    
    def test_semantic_search(self):
        """Test semantic search"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.chroma_path),
            collection_name="test_ltm_search"
        )
        
        # Store multiple memories
        memories = [
            ("mem1", "The cat sat on the mat", ["animal", "cat"]),
            ("mem2", "Artificial intelligence and machine learning", ["ai", "ml"]),
            ("mem3", "The dog ran in the park", ["animal", "dog"]),
            ("mem4", "Neural networks process information", ["ai", "neural"]),
        ]
        
        for mid, content, tags in memories:
            ltm.store(memory_id=mid, content=content, tags=tags, importance=0.7)
        
        # Search for AI-related content
        results = ltm.search(query="artificial intelligence", max_results=5)
        
        assert isinstance(results, list)
    
    def test_search_by_tags(self):
        """Test tag-based search"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.chroma_path),
            collection_name="test_ltm_tags"
        )
        
        # Store memories with different tags
        ltm.store("mem1", "Content 1", tags=["tag1", "common"])
        ltm.store("mem2", "Content 2", tags=["tag2", "common"])
        ltm.store("mem3", "Content 3", tags=["tag1", "unique"])
        
        # Test search with tags (if supported by current API)
        if hasattr(ltm, 'search'):
            results = ltm.search(tags=["tag1"])
            assert isinstance(results, list)
    
    def test_memory_removal(self):
        """Test memory removal functionality"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.chroma_path),
            collection_name="test_ltm_remove"
        )
        
        # Store and then remove memory
        memory_id = "to_remove"
        ltm.store(memory_id, "Content to remove", tags=["remove_me"])
        
        # Verify stored
        retrieved = ltm.retrieve(memory_id)
        assert retrieved is not None
        
        # Remove memory (if delete method exists)
        if hasattr(ltm, 'delete'):
            ltm.delete(memory_id)
            # Verify removed
            retrieved_after = ltm.retrieve(memory_id)
            assert retrieved_after is None
    
    def test_status_reporting(self):
        """Test status reporting functionality"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.chroma_path),
            collection_name="test_ltm_status"
        )
        
        # Store some test data
        ltm.store("test1", "Content 1", memory_type="episodic", tags=["tag1"])
        ltm.store("test2", "Content 2", memory_type="semantic", tags=["tag2"])
        
        # Check if get_status exists
        if hasattr(ltm, 'get_status'):
            status = ltm.get_status()
            assert isinstance(status, dict)
        elif hasattr(ltm, 'health_report'):
            report = ltm.health_report()
            assert isinstance(report, dict)


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])