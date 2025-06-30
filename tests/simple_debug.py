#!/usr/bin/env python3
"""
Simple debug script for Life Period filtering
"""

print("🔍 Starting Life Period Debug")

try:
    from src.memory.memory_system import MemorySystem
    from src.memory.episodic.episodic_memory import EpisodicMemorySystem
    import tempfile
    import os
    print("✅ Import successful")
    
    # Create temporary directory
    temp_dir = tempfile.mkdtemp(prefix="debug_test_")
    
    # Create memory system
    memory_system = MemorySystem()
    
    # Initialize episodic memory system
    episodic_dir = os.path.join(temp_dir, "episodic")
    os.makedirs(episodic_dir, exist_ok=True)
    
    memory_system.episodic = EpisodicMemorySystem(
        chroma_persist_dir=os.path.join(episodic_dir, "chroma"),
        collection_name="debug_metadata",
        enable_json_backup=True,
        storage_path=episodic_dir
    )
    print("✅ Memory system initialized")
    
    # Create episode
    episode_id = memory_system.create_episodic_memory(
        summary="Test episode for life period debugging",
        detailed_content="Debug context with test episode content",
        life_period="debug_session"
    )
    print(f"✅ Episode created: {episode_id}")
    
    # Test search
    if memory_system.episodic.collection:
        search_result = memory_system.episodic.collection.query(
            query_texts=["test episode"],
            n_results=5,
            where={"life_period": "debug_session"}
        )
        print(f"✅ Search completed, found {len(search_result['ids'][0])} results")
    
    # Cleanup
    import shutil
    shutil.rmtree(temp_dir)
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    traceback.print_exc()
    traceback.print_exc()
    traceback.print_exc()
    traceback.print_exc()
