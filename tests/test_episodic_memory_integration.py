"""
Test Episodic Memory System Integration

Comprehensive tests for the newly implemented episodic memory system,
including all major functionality and integration with the cognitive architecture.
"""

import pytest
import sys
import os
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from memory.episodic.episodic_memory import (
    EpisodicMemorySystem,
    EpisodicMemory,
    EpisodicContext,
    EpisodicSearchResult
)

def test_episodic_context():
    """Test EpisodicContext dataclass functionality"""
    print("Testing EpisodicContext...")
    
    # Create context with some data
    context = EpisodicContext(
        location="test_environment",
        emotional_state=0.7,
        cognitive_load=0.5,
        attention_focus=["memory", "testing"],
        interaction_type="test",
        participants=["tester", "ai"],
        environmental_factors={"noise_level": "low", "temperature": "comfortable"}
    )
    
    # Test serialization
    context_dict = context.to_dict()
    assert context_dict["location"] == "test_environment"
    assert context_dict["emotional_state"] == 0.7
    assert "memory" in context_dict["attention_focus"]
    
    # Test deserialization
    restored_context = EpisodicContext.from_dict(context_dict)
    assert restored_context.location == context.location
    assert restored_context.emotional_state == context.emotional_state
    assert restored_context.attention_focus == context.attention_focus
    
    print("✓ EpisodicContext tests passed")

def test_episodic_memory():
    """Test EpisodicMemory dataclass functionality"""
    print("Testing EpisodicMemory...")
    
    # Create test memory
    context = EpisodicContext(
        location="test_location",
        emotional_state=0.5,
        interaction_type="test_conversation"
    )
    
    memory = EpisodicMemory(
        id="test_memory_1",
        summary="Test conversation about episodic memory",
        detailed_content="A detailed conversation about implementing episodic memory systems in AI",
        timestamp=datetime.now(),
        context=context,
        associated_stm_ids=["stm_1", "stm_2"],
        associated_ltm_ids=["ltm_1"],
        importance=0.8,
        emotional_valence=0.3,
        life_period="development_phase"
    )
    
    # Test access tracking
    initial_access_count = memory.access_count
    memory.update_access()
    assert memory.access_count == initial_access_count + 1
    
    # Test rehearsal
    initial_consolidation = memory.consolidation_strength
    memory.rehearse(0.2)
    assert memory.consolidation_strength > initial_consolidation
    assert memory.rehearsal_count == 1
    
    # Test serialization
    memory_dict = memory.to_dict()
    assert memory_dict["id"] == "test_memory_1"
    assert memory_dict["summary"] == "Test conversation about episodic memory"
    assert memory_dict["importance"] == 0.8
    
    # Test deserialization
    restored_memory = EpisodicMemory.from_dict(memory_dict)
    assert restored_memory.id == memory.id
    assert restored_memory.summary == memory.summary
    assert restored_memory.importance == memory.importance
    assert restored_memory.context.location == "test_location"
    
    print("✓ EpisodicMemory tests passed")

def test_episodic_memory_system_initialization():
    """Test EpisodicMemorySystem initialization"""
    print("Testing EpisodicMemorySystem initialization...")
    
    # Test with custom paths
    test_chroma_dir = "test_data/chroma_episodic"
    test_storage_dir = "test_data/episodic_json"
    
    system = EpisodicMemorySystem(
        chroma_persist_dir=test_chroma_dir,
        collection_name="test_episodic",
        enable_json_backup=True,
        storage_path=test_storage_dir
    )
    
    # Check initialization
    assert system.collection_name == "test_episodic"
    assert system.enable_json_backup == True
    assert system.chroma_persist_dir.name == "chroma_episodic"
    assert system.storage_path.name == "episodic_json"
    
    # Check directories were created
    assert system.chroma_persist_dir.exists()
    assert system.storage_path.exists()
    
    print("✓ EpisodicMemorySystem initialization tests passed")

def test_memory_storage_and_retrieval():
    """Test storing and retrieving episodic memories"""
    print("Testing memory storage and retrieval...")
    
    system = EpisodicMemorySystem(
        chroma_persist_dir="test_data/chroma_episodic_storage",
        collection_name="test_storage",
        enable_json_backup=True,
        storage_path="test_data/episodic_storage"
    )
    
    # Create test context
    context = EpisodicContext(
        location="test_lab",
        emotional_state=0.6,
        cognitive_load=0.4,
        interaction_type="development_session",
        participants=["developer", "ai_system"]
    )
    
    # Store a memory
    memory_id = system.store_memory(
        summary="Development session on episodic memory",
        detailed_content="Implemented and tested the episodic memory system with rich contextual metadata and cross-references to other memory systems.",
        context=context,
        associated_stm_ids=["stm_dev_1", "stm_dev_2"],
        associated_ltm_ids=["ltm_memory_systems"],
        importance=0.9,
        emotional_valence=0.7,
        life_period="system_development"
    )
    
    assert memory_id is not None
    print(f"✓ Stored memory with ID: {memory_id}")
    
    # Retrieve the memory
    retrieved_memory = system.retrieve_memory(memory_id)
    assert retrieved_memory is not None
    assert retrieved_memory.summary == "Development session on episodic memory"
    assert retrieved_memory.importance == 0.9
    assert retrieved_memory.context.location == "test_lab"
    assert "stm_dev_1" in retrieved_memory.associated_stm_ids
    
    print("✓ Memory storage and retrieval tests passed")

def test_memory_search():
    """Test memory search functionality"""
    print("Testing memory search...")
    
    system = EpisodicMemorySystem(
        chroma_persist_dir="test_data/chroma_episodic_search",
        collection_name="test_search",
        enable_json_backup=True,
        storage_path="test_data/episodic_search"
    )
    
    # Store multiple test memories
    memories_data = [
        {
            "summary": "Discussion about neural networks",
            "content": "Deep conversation about LSHN and DPAD neural architectures",
            "importance": 0.8,
            "valence": 0.5,
            "period": "research_phase"
        },
        {
            "summary": "Memory consolidation analysis", 
            "content": "Analyzed dream-state memory consolidation patterns",
            "importance": 0.7,
            "valence": 0.3,
            "period": "analysis_phase"
        },
        {
            "summary": "Attention mechanism debugging",
            "content": "Fixed attention fatigue and resource allocation issues",
            "importance": 0.6,
            "valence": -0.2,
            "period": "development_phase"
        }
    ]
    
    memory_ids = []
    for data in memories_data:
        memory_id = system.store_memory(
            summary=data["summary"],
            detailed_content=data["content"],
            importance=data["importance"],
            emotional_valence=data["valence"],
            life_period=data["period"]
        )
        memory_ids.append(memory_id)
    
    # Test semantic search
    results = system.search_memories("neural networks", limit=5, min_relevance=0.1)
    assert len(results) > 0
    
    # Should find the neural networks discussion
    neural_found = any("neural networks" in r.memory.summary.lower() for r in results)
    assert neural_found, "Should find memory about neural networks"
    
    # Test filtered search by importance
    important_results = system.search_memories(
        "memory", 
        limit=5, 
        importance_threshold=0.7,
        min_relevance=0.1
    )
    
    for result in important_results:
        assert result.memory.importance >= 0.7
    
    # Test search by life period
    dev_results = system.search_memories(
        "development",
        life_period="development_phase",
        limit=5,
        min_relevance=0.1
    )
    
    for result in dev_results:
        assert result.memory.life_period == "development_phase"
    
    print(f"✓ Found {len(results)} memories in semantic search")
    print(f"✓ Found {len(important_results)} important memories")
    print(f"✓ Found {len(dev_results)} development phase memories")
    print("✓ Memory search tests passed")

def test_related_memories():
    """Test related memory functionality"""
    print("Testing related memories...")
    
    system = EpisodicMemorySystem(
        chroma_persist_dir="test_data/chroma_episodic_related",
        collection_name="test_related", 
        enable_json_backup=True,
        storage_path="test_data/episodic_related"
    )
    
    # Store memories with shared references
    base_time = datetime.now()
    
    memory1_id = system.store_memory(
        summary="Initial memory system design",
        detailed_content="Designed the architecture for STM and LTM integration",
        associated_stm_ids=["stm_design_1", "stm_design_2"],
        associated_ltm_ids=["ltm_architecture"],
        importance=0.8,
        life_period="design_phase"
    )
    
    # Store related memory shortly after with shared references
    system._memory_cache[memory1_id].timestamp = base_time
    
    memory2_id = system.store_memory(
        summary="Memory system implementation",
        detailed_content="Implemented the memory systems with cross-references",
        associated_stm_ids=["stm_design_2", "stm_impl_1"], # Shared stm_design_2
        associated_ltm_ids=["ltm_architecture"], # Shared ltm_architecture
        importance=0.9,
        life_period="implementation_phase"
    )
    
    # Set timestamp for temporal relationship
    system._memory_cache[memory2_id].timestamp = base_time + timedelta(minutes=30)
    
    # Test getting related memories
    related = system.get_related_memories(
        memory1_id,
        relationship_types=["cross_reference", "temporal"],
        limit=10
    )
    
    assert len(related) > 0
    
    # Should find memory2 due to shared references
    related_ids = [r.memory.id for r in related]
    assert memory2_id in related_ids
    
    # Check relationship types
    cross_ref_found = any(r.match_type == "cross_reference" for r in related)
    temporal_found = any(r.match_type == "temporal" for r in related)
    
    print(f"✓ Found {len(related)} related memories")
    print(f"✓ Cross-reference relationships: {cross_ref_found}")
    print(f"✓ Temporal relationships: {temporal_found}")
    print("✓ Related memories tests passed")

def test_autobiographical_timeline():
    """Test autobiographical timeline functionality"""
    print("Testing autobiographical timeline...")
    
    system = EpisodicMemorySystem(
        chroma_persist_dir="test_data/chroma_episodic_timeline",
        collection_name="test_timeline",
        enable_json_backup=True,
        storage_path="test_data/episodic_timeline"
    )
    
    # Store memories across different time periods
    base_time = datetime.now() - timedelta(hours=3)
    
    timeline_memories = [
        ("Project initialization", "Started the cognitive architecture project", 0),
        ("Memory design phase", "Designed STM and LTM systems", 60),
        ("Implementation phase", "Implemented core memory systems", 120),
        ("Testing phase", "Created comprehensive test suite", 180)
    ]
    
    for i, (summary, content, minutes_offset) in enumerate(timeline_memories):
        memory_id = system.store_memory(
            summary=summary,
            detailed_content=content,
            importance=0.7 + i * 0.1,
            life_period="project_development"
        )
        
        # Set specific timestamps
        system._memory_cache[memory_id].timestamp = base_time + timedelta(minutes=minutes_offset)
    
    # Get autobiographical timeline
    timeline = system.get_autobiographical_timeline(
        life_period="project_development",
        limit=10
    )
    
    assert len(timeline) == 4
    
    # Check chronological order
    for i in range(1, len(timeline)):
        assert timeline[i-1].timestamp <= timeline[i].timestamp
    
    # Check content
    assert timeline[0].summary == "Project initialization"
    assert timeline[-1].summary == "Testing phase"
    
    print(f"✓ Retrieved {len(timeline)} memories in chronological order")
    print("✓ Autobiographical timeline tests passed")

def test_memory_consolidation():
    """Test memory consolidation functionality"""
    print("Testing memory consolidation...")
    
    system = EpisodicMemorySystem(
        chroma_persist_dir="test_data/chroma_episodic_consolidation",
        collection_name="test_consolidation",
        enable_json_backup=True,
        storage_path="test_data/episodic_consolidation"
    )
    
    # Store memories with different importance levels
    memory_id = system.store_memory(
        summary="Important learning session",
        detailed_content="Deep learning about cognitive architectures",
        importance=0.8,
        emotional_valence=0.6
    )
    
    # Get initial consolidation state
    memory = system.retrieve_memory(memory_id)
    initial_consolidation = memory.consolidation_strength
    
    # Consolidate the memory
    success = system.consolidate_memory(memory_id, strength_increment=0.3)
    assert success == True
    
    # Check consolidation increase
    updated_memory = system.retrieve_memory(memory_id)
    assert updated_memory.consolidation_strength > initial_consolidation
    
    # Test getting consolidation candidates
    candidates = system.get_consolidation_candidates(
        min_importance=0.5,
        max_consolidation=0.9,
        limit=10
    )
    
    assert len(candidates) > 0
    
    # All candidates should meet criteria
    for candidate in candidates:
        assert candidate.importance >= 0.5
        assert candidate.consolidation_strength <= 0.9
    
    print(f"✓ Consolidated memory, strength increased to {updated_memory.consolidation_strength:.2f}")
    print(f"✓ Found {len(candidates)} consolidation candidates")
    print("✓ Memory consolidation tests passed")

def test_memory_statistics():
    """Test memory statistics functionality"""
    print("Testing memory statistics...")
    
    system = EpisodicMemorySystem(
        chroma_persist_dir="test_data/chroma_episodic_stats",
        collection_name="test_stats",
        enable_json_backup=True,
        storage_path="test_data/episodic_stats"
    )
    
    # Store diverse memories for statistics
    for i in range(5):
        system.store_memory(
            summary=f"Test memory {i+1}",
            detailed_content=f"Content for test memory {i+1}",
            importance=0.3 + i * 0.15,  # Varying importance
            emotional_valence=-0.4 + i * 0.2,  # Varying valence
            life_period=f"phase_{i % 3 + 1}"  # Different phases
        )
    
    # Get statistics
    stats = system.get_memory_statistics()
    
    assert stats["total_memories"] == 5
    assert stats["memory_system_status"] == "active"
    assert stats["life_period_count"] == 3  # phase_1, phase_2, phase_3
    assert "importance_stats" in stats
    assert "emotional_stats" in stats
    assert "consolidation_stats" in stats
    assert "access_stats" in stats
    
    # Check importance statistics
    importance_stats = stats["importance_stats"]
    assert 0.3 <= importance_stats["mean"] <= 1.0
    assert importance_stats["min"] >= 0.3
    assert importance_stats["max"] <= 1.0
    
    # Check emotional statistics
    emotional_stats = stats["emotional_stats"]
    assert "positive_memories" in emotional_stats
    assert "negative_memories" in emotional_stats
    assert "neutral_memories" in emotional_stats
    
    print(f"✓ Statistics for {stats['total_memories']} memories")
    print(f"✓ {stats['life_period_count']} life periods tracked")
    print(f"✓ Mean importance: {importance_stats['mean']:.2f}")
    print(f"✓ Emotional distribution: {emotional_stats['positive_memories']} pos, {emotional_stats['negative_memories']} neg, {emotional_stats['neutral_memories']} neutral")
    print("✓ Memory statistics tests passed")

def test_memory_clearing():
    """Test memory clearing functionality"""
    print("Testing memory clearing...")
    
    system = EpisodicMemorySystem(
        chroma_persist_dir="test_data/chroma_episodic_clear",
        collection_name="test_clear",
        enable_json_backup=True,
        storage_path="test_data/episodic_clear"
    )
    
    # Store memories with different characteristics
    old_time = datetime.now() - timedelta(days=2)
    recent_time = datetime.now() - timedelta(minutes=30)
    
    # Old, low importance memory (should be cleared)
    old_memory_id = system.store_memory(
        summary="Old unimportant memory",
        detailed_content="This is an old memory with low importance",
        importance=0.2
    )
    system._memory_cache[old_memory_id].timestamp = old_time
    
    # Recent, important memory (should be kept)
    recent_memory_id = system.store_memory(
        summary="Recent important memory", 
        detailed_content="This is a recent memory with high importance",
        importance=0.8
    )
    system._memory_cache[recent_memory_id].timestamp = recent_time
    
    # Low importance memory (should be cleared by importance)
    low_imp_memory_id = system.store_memory(
        summary="Low importance memory",
        detailed_content="This memory has low importance",
        importance=0.1
    )
    
    initial_count = len(system._memory_cache)
    assert initial_count == 3
    
    # Clear old memories
    system.clear_memory(older_than=timedelta(days=1))
    
    # Should have removed the old memory
    assert old_memory_id not in system._memory_cache
    assert recent_memory_id in system._memory_cache
    assert low_imp_memory_id in system._memory_cache
    
    # Clear low importance memories
    system.clear_memory(importance_threshold=0.5)
    
    # Should have removed the low importance memory
    assert low_imp_memory_id not in system._memory_cache
    assert recent_memory_id in system._memory_cache
    
    final_count = len(system._memory_cache)
    print(f"✓ Cleared memories: {initial_count} -> {final_count}")
    print("✓ Memory clearing tests passed")

def run_comprehensive_test():
    """Run all episodic memory tests"""
    print("=" * 60)
    print("COMPREHENSIVE EPISODIC MEMORY SYSTEM TESTS")
    print("=" * 60)
    
    try:
        # Run all test functions
        test_episodic_context()
        print()
        
        test_episodic_memory()
        print()
        
        test_episodic_memory_system_initialization()
        print()
        
        test_memory_storage_and_retrieval()
        print()
        
        test_memory_search()
        print()
        
        test_related_memories()
        print()
        
        test_autobiographical_timeline()
        print()
        
        test_memory_consolidation()
        print()
        
        test_memory_statistics()
        print()
        
        test_memory_clearing()
        print()
        
        print("=" * 60)
        print("✅ ALL EPISODIC MEMORY TESTS PASSED!")
        print("=" * 60)
        print()
        print("The episodic memory system is working correctly with:")
        print("• Rich contextual metadata storage")
        print("• Cross-references to STM and LTM systems")
        print("• Temporal and semantic search capabilities")
        print("• Autobiographical organization")
        print("• Memory consolidation pipeline integration")
        print("• Comprehensive statistics and management")
        print("• ChromaDB and JSON backup storage")
        
        return True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)
