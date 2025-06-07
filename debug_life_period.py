#!/usr/bin/env python3
"""Debug script to investigate life period filtering issues"""

import os
import tempfile
import shutil
from datetime import datetime, timedelta

# Set up path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from memory.memory_system import MemorySystem

def debug_life_period_search():
    """Debug the life period search functionality"""
    print("🔍 Debugging Life Period Search")
    print("=" * 50)
    
    # Create temporary directory for testing
    temp_dir = tempfile.mkdtemp(prefix="life_period_debug_")
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Initialize memory system with episodic memory
        print("\n1. Initializing Memory System...")
        memory_system = MemorySystem(
            ltm_storage_dir=temp_dir,
            auto_create_dirs=True,
            ltm_backup_enabled=False  # Disable backup for speed
        )
        
        # Initialize episodic memory system
        episodic_dir = os.path.join(temp_dir, "episodic")
        os.makedirs(episodic_dir, exist_ok=True)
        
        from memory.episodic.episodic_memory import EpisodicMemorySystem
        memory_system.episodic = EpisodicMemorySystem(
            chroma_persist_dir=os.path.join(episodic_dir, "chroma"),
            collection_name="debug_episodic",
            enable_json_backup=True,
            storage_path=episodic_dir
        )
        print("   ✓ Memory systems initialized")
        
        # Create a test episode with specific life period
        print("\n2. Creating Test Episode...")
        episode_id = memory_system.create_episodic_memory(
            summary="Test episode for debugging",
            detailed_content="This is a test episode created specifically to debug life period filtering functionality. It should be found when searching for the learning_session life period.",
            participants=["User", "AI"],
            location="Debug Environment",
            cognitive_load=0.5,
            importance=0.8,
            emotional_valence=0.2,
            life_period="learning_session"
        )
        print(f"   ✓ Created episode: {episode_id}")
        
        # Check episode data directly
        if hasattr(memory_system.episodic, 'retrieve_memory'):
            episode = memory_system.episodic.retrieve_memory(episode_id)
            if episode:
                print(f"   Episode summary: {episode.summary}")
                print(f"   Episode life_period: {episode.life_period}")
                print(f"   Episode importance: {episode.importance}")
            else:
                print("   ⚠️ Could not retrieve episode directly")
        
        # Wait for ChromaDB indexing
        import time
        time.sleep(2)
        
        # Test 1: Direct search method call
        print("\n3. Testing Direct EpisodicMemorySystem search...")
        if hasattr(memory_system.episodic, 'search_memories'):
            direct_results = memory_system.episodic.search_memories(
                query="debug",
                life_period="learning_session",
                min_relevance=0.1,
                limit=10
            )
            print(f"   Direct search results: {len(direct_results)}")
            for result in direct_results:
                print(f"   - {result.memory.summary} (life_period: {result.memory.life_period})")
        
        # Test 2: MemorySystem wrapper search
        print("\n4. Testing MemorySystem search_episodic_memories...")
        wrapper_results = memory_system.search_episodic_memories(
            query="debug",
            life_periods=["learning_session"],
            min_similarity=0.1,
            max_results=10
        )
        print(f"   Wrapper search results: {len(wrapper_results)}")
        for result in wrapper_results:
            if hasattr(result, 'summary'):
                print(f"   - {result.summary} (life_period: {result.life_period})")
            else:
                print(f"   - {result}")
        
        # Test 3: Search without life period filter
        print("\n5. Testing search without life period filter...")
        all_results = memory_system.search_episodic_memories(
            query="debug",
            min_similarity=0.1,
            max_results=10
        )
        print(f"   No filter results: {len(all_results)}")
        for result in all_results:
            if hasattr(result, 'summary'):
                print(f"   - {result.summary} (life_period: {result.life_period})")
            else:
                print(f"   - {result}")
        
        # Test 4: Check ChromaDB metadata
        print("\n6. Checking ChromaDB metadata...")
        if hasattr(memory_system.episodic, 'collection') and memory_system.episodic.collection:
            try:
                # Get all documents to check metadata
                all_docs = memory_system.episodic.collection.get()
                print(f"   Total ChromaDB documents: {len(all_docs['ids'])}")
                for i, doc_id in enumerate(all_docs['ids']):
                    metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                    print(f"   Doc {doc_id}: life_period = {metadata.get('life_period', 'MISSING')}")
            except Exception as e:
                print(f"   ChromaDB metadata check failed: {e}")
        
        # Test 5: Check cache
        print("\n7. Checking memory cache...")
        if hasattr(memory_system.episodic, '_memory_cache'):
            cache_size = len(memory_system.episodic._memory_cache)
            print(f"   Memory cache size: {cache_size}")
            for mem_id, memory in memory_system.episodic._memory_cache.items():
                print(f"   Cache {mem_id}: life_period = {memory.life_period}")
        
        print("\n✅ Debug completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Debug failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up
        try:
            shutil.rmtree(temp_dir)
            print(f"\n🧹 Cleaned up temp directory: {temp_dir}")
        except:
            print(f"\n⚠️ Could not clean up temp directory: {temp_dir}")

if __name__ == "__main__":
    debug_life_period_search()
