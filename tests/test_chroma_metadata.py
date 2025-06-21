#!/usr/bin/env python3
"""Simple test to check ChromaDB metadata storage"""

import os
import sys
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_chroma_metadata():
    temp_dir = tempfile.mkdtemp(prefix="chroma_test_")
    
    try:
        from memory.memory_system import MemorySystem
        from memory.episodic.episodic_memory import EpisodicMemorySystem
        
        # Initialize memory system
        memory_system = MemorySystem(
            ltm_storage_dir=temp_dir,
            auto_create_dirs=True,
            ltm_backup_enabled=False
        )
        
        # Initialize episodic memory system
        episodic_dir = os.path.join(temp_dir, "episodic")
        os.makedirs(episodic_dir, exist_ok=True)
        
        memory_system.episodic = EpisodicMemorySystem(
            chroma_persist_dir=os.path.join(episodic_dir, "chroma"),
            collection_name="test_metadata",
            enable_json_backup=True,
            storage_path=episodic_dir
        )
        
        print("Creating episode...")
        episode_id = memory_system.create_episodic_memory(
            summary="Test episode",
            detailed_content="This is a test.",
            life_period="learning_session"
        )
        print(f"Created: {episode_id}")
        
        # Check if ChromaDB has the metadata
        import time
        time.sleep(1)
        
        if memory_system.episodic.collection:
            print("\nChecking ChromaDB...")
            all_data = memory_system.episodic.collection.get()
            print(f"Number of documents: {len(all_data['ids'])}")
            
            for i, doc_id in enumerate(all_data['ids']):
                metadata = all_data['metadatas'][i]
                print(f"Doc {doc_id}:")
                print(f"  life_period: {metadata.get('life_period', 'MISSING')}")
                print(f"  summary: {metadata.get('summary', 'MISSING')}")
                
            # Test search with filter
            print("\nTesting search with life_period filter...")
            search_result = memory_system.episodic.collection.query(
                query_texts=["test"],
                n_results=10,
                where={"life_period": "learning_session"}
            )
            print(f"Search with filter found: {len(search_result['ids'][0])}")
            
            # Test search without filter
            print("\nTesting search without filter...")
            search_result2 = memory_system.episodic.collection.query(
                query_texts=["test"],
                n_results=10
            )
            print(f"Search without filter found: {len(search_result2['ids'][0])}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    test_chroma_metadata()
