#!/usr/bin/env python3
"""
Test Episodic Memory Integration with Memory System
Validates the fixed integration between MemorySystem and EpisodicMemorySystem
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from datetime import datetime, timedelta
import tempfile
import shutil
from src.memory.memory_system import MemorySystem
from src.memory.episodic.episodic_memory import EpisodicContext


def test_episodic_memory_integration():
    """Test the corrected episodic memory integration"""
    print("🧠 Testing Episodic Memory Integration (Fixed)")
    print("=" * 60)
    
    # Setup temporary directory for testing
    test_dir = tempfile.mkdtemp(prefix="episodic_integration_test_")
    
    try:
        # Initialize integrated memory system
        print("\n1. Initializing Memory System with Episodic Memory...")
        memory_system = MemorySystem(
            chroma_persist_dir=os.path.join(test_dir, "chroma"),
            use_vector_stm=True,
            use_vector_ltm=True
        )
        
        # Verify episodic memory is initialized
        print(f"   Episodic Memory Available: {hasattr(memory_system, 'episodic') and memory_system.episodic is not None}")
        
        if not hasattr(memory_system, 'episodic') or memory_system.episodic is None:
            print("   ❌ ERROR: Episodic memory not initialized!")
            return
        
        # Test 1: Create episodic memory
        print("\n2. Creating Episodic Memory...")
        episode_id = memory_system.create_episodic_memory(
            summary="Python programming discussion",
            detailed_content="Had a detailed conversation about Python best practices, including proper error handling, code organization, and testing strategies. Discussed the importance of clean code and documentation.",
            participants=["User", "AI Assistant"],
            location="Development Environment",
            cognitive_load=0.6,
            importance=0.8,
            emotional_valence=0.3,
            life_period="learning_session"
        )
        
        print(f"   Created Episode ID: {episode_id}")
        print(f"   Episode Created: {episode_id is not None}")
        
        # Test 2: Search episodic memories
        print("\n3. Searching Episodic Memories...")
        results = memory_system.search_episodic_memories(
            query="Python programming best practices",
            max_results=5,
            min_similarity=0.3
        )
        
        print(f"   Search Results Found: {len(results)}")
        for i, memory in enumerate(results):
            print(f"   {i+1}. {memory.summary[:50]}...")
            print(f"       Importance: {memory.importance:.2f}")
            print(f"       Emotional Valence: {memory.emotional_valence:.2f}")
        
        # Test 3: Create another episodic memory with cross-references
        print("\n4. Creating Second Episode with Cross-References...")
        
        # Store some memories in STM first
        stm_id1 = "stm_concept_1"
        stm_id2 = "stm_concept_2"
        
        memory_system.store_memory(
            memory_id=stm_id1,
            content="Object-oriented programming principles",
            importance=0.7
        )
        
        memory_system.store_memory(
            memory_id=stm_id2,
            content="Unit testing frameworks",
            importance=0.6
        )
        
        # Create episodic memory with STM cross-references
        episode_id2 = memory_system.create_episodic_memory(
            summary="Advanced programming concepts review",
            detailed_content="Reviewed advanced programming concepts including OOP principles and testing frameworks. Connected theory with practical applications.",
            participants=["User", "AI Assistant"],
            location="Development Environment",
            stm_ids=[stm_id1, stm_id2],
            importance=0.7,
            emotional_valence=0.4,
            life_period="learning_session"
        )
        
        print(f"   Second Episode ID: {episode_id2}")
        print(f"   With STM Cross-References: {[stm_id1, stm_id2]}")
        
        # Test 4: Search for cross-referenced episodes
        print("\n5. Testing Cross-Reference Search...")
        cross_ref_results = memory_system.get_cross_referenced_episodes(
            memory_id=stm_id1,
            memory_system="stm"
        )
        
        print(f"   Cross-Referenced Episodes Found: {len(cross_ref_results)}")
        for memory in cross_ref_results:
            print(f"   - {memory.summary[:50]}...")
            print(f"     Associated STM IDs: {memory.associated_stm_ids}")
        
        # Test 5: Test date range search
        print("\n6. Testing Date Range Search...")
        now = datetime.now()
        one_hour_ago = now - timedelta(hours=1)
        one_hour_from_now = now + timedelta(hours=1)
        
        date_range_results = memory_system.search_episodic_memories(
            query="programming",
            max_results=10,
            date_range=(one_hour_ago, one_hour_from_now),
            min_similarity=0.2
        )
        
        print(f"   Results in Date Range: {len(date_range_results)}")
        
        # Test 6: Test life period filter
        print("\n7. Testing Life Period Filter...")
        life_period_results = memory_system.search_episodic_memories(
            query="learning",
            life_periods=["learning_session"],
            max_results=5,
            min_similarity=0.2
        )
        
        print(f"   Results for 'learning_session': {len(life_period_results)}")
        
        # Test 7: Get episodic memory statistics
        print("\n8. Getting Episodic Memory Statistics...")
        if hasattr(memory_system.episodic, 'get_memory_statistics'):
            stats = memory_system.episodic.get_memory_statistics()
            print(f"   Total Episodic Memories: {stats.get('total_memories', 0)}")
            print(f"   Life Periods: {stats.get('life_period_count', 0)}")
            print(f"   Memory System Status: {stats.get('memory_system_status', 'unknown')}")
            print(f"   ChromaDB Available: {stats.get('chromadb_available', False)}")
        
        # Test 8: Test memory system status
        print("\n9. Testing Memory System Status...")
        status = memory_system.get_status()
        print(f"   Memory System Active: {status.get('system_active', False)}")
        print(f"   STM Count: {status.get('stm', {}).get('item_count', 0)}")
        print(f"   LTM Count: {status.get('ltm', {}).get('total_memories', 0)}")
        
        print("\n✅ Episodic Memory Integration Test Completed Successfully!")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(test_dir)
            print(f"\n🧹 Cleaned up test directory: {test_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up test directory: {e}")


if __name__ == "__main__":
    success = test_episodic_memory_integration()
    sys.exit(0 if success else 1)
