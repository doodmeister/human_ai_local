"""
Test suite for Vector Long-Term Memory implementation
"""
import tempfile
import shutil
from pathlib import Path

# Import the classes we want to test
from src.memory.ltm.vector_ltm import VectorLongTermMemory, VectorSearchResult
from src.memory.ltm.long_term_memory import LTMRecord

class TestVectorLongTermMemory:
    """Test cases for VectorLongTermMemory"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = Path(self.temp_dir) / "ltm"
        self.chroma_path = Path(self.temp_dir) / "chroma"
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_initialization_fallback_mode(self):
        """Test initialization when ChromaDB is not available"""
        # Force fallback mode
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False  # Force JSON-only mode
        )
        
        assert ltm.use_vector_db is False
        assert ltm.enable_json_backup is True
        assert len(ltm.memories) == 0
        assert ltm.storage_path.exists()
    
    def test_store_and_retrieve_memory(self):
        """Test basic memory storage and retrieval"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False  # Use JSON for this test
        )
        
        # Store a memory
        memory_id = "test_memory_1"
        content = "This is a test memory about artificial intelligence"
        
        success = ltm.store(
            memory_id=memory_id,
            content=content,
            memory_type="semantic",
            importance=0.8,
            tags=["ai", "test"],
            source="unit_test"
        )
        
        assert success is True
        assert memory_id in ltm.memories
        
        # Retrieve the memory
        retrieved = ltm.retrieve(memory_id)
        assert retrieved is not None
        assert retrieved["content"] == content
        assert retrieved["memory_type"] == "semantic"
        assert retrieved["importance"] == 0.8
        assert "ai" in retrieved["tags"]
        assert "test" in retrieved["tags"]
        assert retrieved["source"] == "unit_test"
    
    def test_semantic_search_fallback(self):
        """Test semantic search in fallback mode"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False
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
        results = ltm.search_semantic("artificial intelligence", max_results=5)
        
        assert len(results) > 0
        assert all(isinstance(r, VectorSearchResult) for r in results)
        
        # Should find AI-related memories
        ai_memories = [r for r in results if "ai" in r.record.tags]
        assert len(ai_memories) > 0
    
    def test_search_by_content_compatibility(self):
        """Test backward compatibility of search_by_content"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False
        )
        
        # Store test memory
        ltm.store("test1", "Machine learning algorithms", tags=["ml"], importance=0.9)
        ltm.store("test2", "Cooking recipes", tags=["food"], importance=0.5)
        
        # Test compatibility method
        results = ltm.search_by_content("machine learning")
        
        assert len(results) > 0
        assert all(isinstance(r, tuple) and len(r) == 2 for r in results)
        assert all(isinstance(r[0], LTMRecord) for r in results)
        assert all(isinstance(r[1], float) for r in results)
    
    def test_search_by_tags(self):
        """Test tag-based search"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False
        )
        
        # Store memories with different tags
        ltm.store("mem1", "Content 1", tags=["tag1", "common"])
        ltm.store("mem2", "Content 2", tags=["tag2", "common"])
        ltm.store("mem3", "Content 3", tags=["tag1", "unique"])
        
        # Test OR search
        results_or = ltm.search_by_tags(["tag1", "tag2"], operator="OR")
        assert len(results_or) == 3
        
        # Test AND search
        results_and = ltm.search_by_tags(["tag1", "common"], operator="AND")
        assert len(results_and) == 1
        assert results_and[0].id == "mem1"
    
    def test_memory_filtering(self):
        """Test memory filtering by type and importance"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False
        )
        
        # Store different types of memories
        ltm.store("ep1", "Episode 1", memory_type="episodic", importance=0.3)
        ltm.store("sem1", "Semantic 1", memory_type="semantic", importance=0.8)
        ltm.store("proc1", "Procedure 1", memory_type="procedural", importance=0.9)
        
        # Test type filtering
        episodic = ltm.get_memories_by_type("episodic")
        assert len(episodic) == 1
        assert episodic[0].memory_type == "episodic"
        
        # Test importance filtering
        important = ltm.get_important_memories(min_importance=0.7)
        assert len(important) == 2  # sem1 and proc1
        
        # Test recent memories (should include all since just created)
        recent = ltm.get_recent_memories(hours=1)
        assert len(recent) == 3
    
    def test_memory_removal(self):
        """Test memory removal functionality"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False
        )
        
        # Store and then remove memory
        memory_id = "to_remove"
        ltm.store(memory_id, "Content to remove", tags=["remove_me"])
        
        assert memory_id in ltm.memories
        assert "remove_me" in ltm.tags_index
        
        # Remove memory
        success = ltm.remove(memory_id)
        assert success is True
        assert memory_id not in ltm.memories
        
        # Check that tags index is cleaned up
        assert "remove_me" not in ltm.tags_index
    
    def test_status_reporting(self):
        """Test status reporting functionality"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False
        )
        
        # Store some test data
        ltm.store("test1", "Content 1", memory_type="episodic", tags=["tag1"])
        ltm.store("test2", "Content 2", memory_type="semantic", tags=["tag2"])
        
        status = ltm.get_status()
        
        assert isinstance(status, dict)
        assert status["total_memories"] == 2
        assert status["vector_db_enabled"] is False
        assert status["json_backup_enabled"] is True
        assert len(status["memory_types"]) == 2
        assert "episodic" in status["memory_types"]
        assert "semantic" in status["memory_types"]
    
    def test_stm_consolidation(self):
        """Test STM to LTM consolidation"""
        ltm = VectorLongTermMemory(
            storage_path=str(self.storage_path),
            chroma_persist_dir=str(self.chroma_path),
            use_vector_db=False
        )
        
        # Mock STM items
        class MockSTMItem:
            def __init__(self, id, content, importance, emotional_valence, access_count, associations):
                self.id = id
                self.content = content
                self.importance = importance
                self.emotional_valence = emotional_valence
                self.access_count = access_count
                self.associations = associations
        
        stm_items = [
            MockSTMItem("stm1", "Important memory", 0.8, 0.5, 3, []),
            MockSTMItem("stm2", "Less important", 0.3, 0.0, 1, []),
            MockSTMItem("stm3", "Frequently accessed", 0.5, 0.2, 5, [])
        ]
        
        # Consolidate (should consolidate stm1 and stm3 based on criteria)
        consolidated = ltm.consolidate_from_stm(stm_items)
        
        assert consolidated == 2  # stm1 (importance > 0.6) and stm3 (access_count > 2)
        assert "stm1" in ltm.memories
        assert "stm3" in ltm.memories
        assert "stm2" not in ltm.memories

if __name__ == "__main__":
    # Run tests
    test = TestVectorLongTermMemory()
    test.setup_method()
    
    try:
        print("Running Vector LTM tests...")
        
        test.test_initialization_fallback_mode()
        print("✓ Initialization test passed")
        
        test.test_store_and_retrieve_memory()
        print("✓ Store/retrieve test passed")
        
        test.test_semantic_search_fallback()
        print("✓ Semantic search test passed")
        
        test.test_search_by_content_compatibility()
        print("✓ Content search compatibility test passed")
        
        test.test_search_by_tags()
        print("✓ Tag search test passed")
        
        test.test_memory_filtering()
        print("✓ Memory filtering test passed")
        
        test.test_memory_removal()
        print("✓ Memory removal test passed")
        
        test.test_status_reporting()
        print("✓ Status reporting test passed")
        
        test.test_stm_consolidation()
        print("✓ STM consolidation test passed")
        
        print("\n🎉 All Vector LTM tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        test.teardown_method()
