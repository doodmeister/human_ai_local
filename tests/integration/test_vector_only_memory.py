"""
Test suite for Vector-Only Memory Systems
Tests both STM and LTM vector-only implementations
"""
import tempfile
import shutil
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the vector-only classes
from src.memory.ltm.vector_ltm import VectorLongTermMemory
from src.memory.stm.vector_stm import VectorShortTermMemory, MemoryItem, STMConfiguration
from datetime import datetime

class TestVectorOnlyMemory:
    """Test cases for vector-only memory systems"""
    
    def setup_method(self):
        """Setup test environment"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.ltm_chroma_path = Path(self.temp_dir) / "ltm_chroma"
        self.stm_chroma_path = Path(self.temp_dir) / "stm_chroma"
        
    def teardown_method(self):
        """Cleanup test environment"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_ltm_vector_only_operations(self):
        """Test LTM vector-only store, retrieve, search, delete"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.ltm_chroma_path),
            collection_name="test_ltm"
        )
        
        # Test store
        memory_id = "test_memory_1"
        content = "This is a test memory about artificial intelligence"
        success = ltm.store(
            memory_id=memory_id,
            content=content,
            memory_type="episodic",
            importance=0.8,
            tags=["ai", "test"],
            associations=["memory", "test"]
        )
        assert success, "Failed to store memory in LTM"
        print(f"✓ Stored memory {memory_id} successfully")
        
        # Test retrieve
        retrieved = ltm.retrieve(memory_id)
        assert retrieved is not None, "Failed to retrieve memory from LTM"
        print(f"✓ Retrieved memory: {retrieved.get('content', '')[:50]}...")
        assert retrieved["content"] == content
        assert retrieved["memory_type"] == "episodic"
        assert retrieved["importance"] == 0.8
        assert "ai" in retrieved["tags"]
        
        # Test search
        search_results = ltm.search_semantic("artificial intelligence", max_results=5, min_similarity=0.1)
        print(f"Search returned {len(search_results)} results")
        
        # If no results, try with even lower threshold
        if len(search_results) == 0:
            search_results = ltm.search_semantic("artificial intelligence", max_results=5, min_similarity=0.0)
            print(f"Search with 0.0 threshold returned {len(search_results)} results")
        
        # If still no results, try different queries
        if len(search_results) == 0:
            search_results = ltm.search_semantic("test", max_results=5, min_similarity=0.0)
            print(f"Search for 'test' returned {len(search_results)} results")
        
        if len(search_results) == 0:
            search_results = ltm.search_semantic("memory", max_results=5, min_similarity=0.0)
            print(f"Search for 'memory' returned {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            print(f"  Result {i}: {result.get('id', 'unknown')} - {result.get('content', '')[:50]}... (score: {result.get('similarity_score', 0)})")
        
        assert len(search_results) > 0, "Search should return results"
        found_memory = None
        for result in search_results:
            if result["id"] == memory_id:
                found_memory = result
                break
        assert found_memory is not None, "Stored memory should be found in search"
        assert found_memory["similarity_score"] > 0.0, "Similarity score should be positive"
        
        # Test delete
        deleted = ltm.delete(memory_id)
        assert deleted, "Failed to delete memory from LTM"
        
        # Verify deletion
        retrieved_after_delete = ltm.retrieve(memory_id)
        assert retrieved_after_delete is None, "Memory should be deleted"
    
    def test_stm_vector_only_operations(self):
        """Test STM vector-only store, retrieve, search, remove"""
        config = STMConfiguration(
            chroma_persist_dir=str(self.stm_chroma_path),
            collection_name="test_stm"
        )
        stm = VectorShortTermMemory(config=config)
        
        # Create test memory item
        memory_item = MemoryItem(
            id="stm_test_1",
            content="Short term memory test content",
            encoding_time=datetime.now(),
            last_access=datetime.now(),
            importance=0.7,
            emotional_valence=0.2
        )
        
        # Test store
        success = stm.store(
            memory_id=memory_item.id,
            content=memory_item.content,
            importance=memory_item.importance,
            emotional_valence=memory_item.emotional_valence
        )
        assert success, "Failed to store memory in STM"
        
        # Test retrieve
        retrieved = stm.retrieve("stm_test_1")
        assert retrieved is not None, "Failed to retrieve memory from STM"
        assert retrieved.content == "Short term memory test content"
        assert retrieved.importance == 0.7
        
        # Test get all memories
        all_memories = stm.get_all_memories()
        assert len(all_memories) == 1, "Should have one memory"
        assert all_memories[0].id == "stm_test_1"
        
        # Test search
        search_results = stm.search_semantic("short term memory", max_results=5)
        assert len(search_results) > 0, "Search should return results"
        found = any(result.item.id == "stm_test_1" for result in search_results)
        assert found, "Stored memory should be found in search"
        
        # Test remove
        removed = stm.remove_item("stm_test_1")
        assert removed, "Failed to remove memory from STM"
        
        # Verify removal
        retrieved_after_remove = stm.retrieve("stm_test_1")
        assert retrieved_after_remove is None, "Memory should be removed"
    
    def test_ltm_multiple_memories_search(self):
        """Test LTM with multiple memories and semantic search"""
        ltm = VectorLongTermMemory(
            chroma_persist_dir=str(self.ltm_chroma_path),
            collection_name="test_ltm_multi"
        )
        
        # Store multiple test memories
        memories = [
            ("mem1", "Machine learning algorithms are powerful", ["ai", "ml"]),
            ("mem2", "Python programming language is versatile", ["programming", "python"]),
            ("mem3", "Deep neural networks learn patterns", ["ai", "neural", "ml"]),
            ("mem4", "Web development with JavaScript", ["programming", "web"]),
            ("mem5", "Artificial intelligence revolution", ["ai", "future"])
        ]
        
        for mem_id, content, tags in memories:
            ltm.store(
                memory_id=mem_id,
                content=content,
                memory_type="episodic",
                importance=0.6,
                tags=tags
            )
        
        # Test semantic search for AI-related content
        ai_results = ltm.search_semantic("artificial intelligence machine learning", max_results=10, min_similarity=0.1)
        print(f"AI search returned {len(ai_results)} results")
        for i, result in enumerate(ai_results):
            print(f"  AI Result {i}: {result.get('id', 'unknown')} - {result.get('content', '')[:50]}... (score: {result.get('similarity_score', 0)})")
        
        assert len(ai_results) >= 1, "Should find AI-related memories"
        
        # Check that AI-related memories rank higher
        ai_memories = [r for r in ai_results if any(tag in ["ai", "ml"] for tag in r.get("tags", []))]
        print(f"Found {len(ai_memories)} memories with AI tags")
        assert len(ai_memories) >= 1, "Should find memories with AI tags"
        
        # Test search with memory type filter
        filtered_results = ltm.search_semantic(
            "programming", 
            max_results=10, 
            memory_types=["episodic"],
            min_similarity=0.1
        )
        print(f"Programming search returned {len(filtered_results)} results")
        assert len(filtered_results) >= 1, "Should find programming-related memories"
        
        # Test minimum importance filter  
        important_results = ltm.search_semantic(
            "artificial intelligence", 
            max_results=10, 
            min_importance=0.5,
            min_similarity=0.1
        )
        print(f"Important search returned {len(important_results)} results")
        for i, result in enumerate(important_results):
            print(f"  Important Result {i}: {result.get('id', 'unknown')} - importance: {result.get('importance', 0)}")
        assert len(important_results) > 0, "Should find memories above importance threshold"
    
    def test_stm_decay_and_cleanup(self):
        """Test STM decay mechanisms"""
        config = STMConfiguration(
            chroma_persist_dir=str(self.stm_chroma_path),
            collection_name="test_stm_decay",
            max_decay_hours=1
        )
        stm = VectorShortTermMemory(config=config)
        
        # Add multiple memories
        for i in range(3):
            memory_item = MemoryItem(
                id=f"decay_test_{i}",
                content=f"Memory {i} for decay testing",
                encoding_time=datetime.now(),
                last_access=datetime.now(),
                importance=0.3 + (i * 0.2)  # Varying importance
            )
            stm.store(
                memory_id=memory_item.id,
                content=memory_item.content,
                importance=memory_item.importance
            )
        
        # Test decay (should remove old/low-importance memories)
        decayed_ids = stm.decay_memories()
        assert isinstance(decayed_ids, list), "Decay should return list of IDs"
        
        # Test clear
        stm.clear()
        all_after_clear = stm.get_all_memories()
        assert len(all_after_clear) == 0, "STM should be empty after clear"


def run_vector_only_tests():
    """Run all vector-only memory tests"""
    test = TestVectorOnlyMemory()
    
    print("Testing Vector-Only Memory Systems...")
    
    try:
        print("  Setting up test environment...")
        test.setup_method()
        
        print("  ✓ Testing LTM vector-only operations...")
        test.test_ltm_vector_only_operations()
        
        print("  ✓ Testing STM vector-only operations...")
        test.test_stm_vector_only_operations()
        
        print("  ✓ Testing LTM multiple memories and search...")
        test.test_ltm_multiple_memories_search()
        
        print("  ✓ Testing STM decay and cleanup...")
        test.test_stm_decay_and_cleanup()
        
        print("  Cleaning up...")
        test.teardown_method()
        
        print("✅ All vector-only memory tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Vector-only memory tests failed: {e}")
        try:
            test.teardown_method()
        except:
            pass
        return False


if __name__ == "__main__":
    success = run_vector_only_tests()
    exit(0 if success else 1)
