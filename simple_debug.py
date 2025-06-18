#!/usr/bin/env python3
"""
Simple debug script for Life Period filtering
"""

print("🔍 Starting Life Period Debug")

try:
    from src.memory.episodic.episodic_memory import EpisodicMemorySystem
    print("✅ Import successful")
    
    # Create episodic memory system
    episodic = EpisodicMemorySystem(
        chroma_persist_dir="./temp_debug_episodic",
        embedding_model="all-MiniLM-L6-v2"
    )
    print(f"✅ Episodic memory initialized: {episodic.is_available()}")
    
    # Create episode
    episode_id = episodic.create_episode(
        event_description="Test episode for life period debugging",
        context="Debug context",
        participants=["user"],
        location="test",
        emotional_state="neutral",
        significance=0.5,
        life_period="debug_session"
    )
    print(f"✅ Episode created: {episode_id}")
    
    # Test search
    results = episodic.search_memories(
        query="test episode",
        life_period="debug_session",
        max_results=5
    )
    print(f"✅ Search completed, found {len(results)} results")
    
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
