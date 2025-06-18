#!/usr/bin/env python3
"""
Debug script for Life Period filtering in Episodic Memory
"""
import tempfile
import shutil
from src.memory.episodic.episodic_memory import EpisodicMemorySystem

def debug_life_period():
    print("🔍 Debugging Life Period Filtering")
    print("=" * 50)
    
    # Create temporary directory for test
    temp_dir = tempfile.mkdtemp(prefix="life_period_debug_")
    print(f"Using temp directory: {temp_dir}")
    
    try:
        # Initialize episodic memory system directly
        episodic = EpisodicMemorySystem(
            chroma_persist_dir=f"{temp_dir}/episodic_chroma",
            embedding_model="all-MiniLM-L6-v2"
        )
        
        print(f"Episodic memory initialized: {episodic.is_available()}")
        
        # Create episode with specific life period
        test_life_period = "test_learning_session"
        episode_data = {
            "event_description": "Debug test for life period filtering",
            "context": "Testing whether life period metadata is stored correctly",
            "participants": ["user", "system"],
            "location": "debug_environment",
            "emotional_state": "focused",
            "significance": 0.8,
            "life_period": test_life_period
        }
        
        print(f"\n1. Creating episode with life_period: '{test_life_period}'")
        episode_id = episodic.create_episode(**episode_data)
        print(f"   Episode ID: {episode_id}")
        
        # Wait for indexing
        import time
        time.sleep(2)
        
        # Check what's actually stored in ChromaDB
        print("\n2. Checking ChromaDB collection contents:")
        collection = episodic.collection
        if collection:
            # Get all documents
            all_docs = collection.get()
            print(f"   Total documents in collection: {len(all_docs['ids'])}")
            
            # Check each document's metadata
            for i, doc_id in enumerate(all_docs['ids']):
                metadata = all_docs['metadatas'][i] if all_docs['metadatas'] else {}
                print(f"   Document {i+1}: ID={doc_id}")
                print(f"      Metadata: {metadata}")
                if 'life_period' in metadata:
                    print(f"      ✅ life_period found: '{metadata['life_period']}'")
                else:
                    print("      ❌ life_period NOT found in metadata")
        
        # Test direct search in episodic system
        print("\n3. Testing direct search in EpisodicMemorySystem:")
        results = episodic.search_memories(
            query="debug test",
            life_period=test_life_period,
            max_results=10
        )
        print(f"   Direct search results for life_period '{test_life_period}': {len(results)}")
        for result in results:
            print(f"      - {result.get('event_description', 'No description')}")
            print(f"        Life Period: {result.get('life_period', 'Not found')}")
        
        # Test search without life period filter
        print("\n4. Testing search without life period filter:")
        all_results = episodic.search_memories(
            query="debug test",
            max_results=10
        )
        print(f"   Search results without filter: {len(all_results)}")
        for result in all_results:
            print(f"      - {result.get('event_description', 'No description')}")
            print(f"        Life Period: {result.get('life_period', 'Not found')}")
        
        # Test ChromaDB query directly with where clause
        print("\n5. Testing ChromaDB direct query with where clause:")
        try:
            direct_results = collection.query(
                query_texts=["debug test"],
                where={"life_period": test_life_period},
                n_results=10
            )
            print(f"   Direct ChromaDB query results: {len(direct_results['ids'][0])}")
            if direct_results['metadatas'][0]:
                for metadata in direct_results['metadatas'][0]:
                    print(f"      - Metadata: {metadata}")
        except Exception as e:
            print(f"   ❌ Direct ChromaDB query failed: {e}")
        
        # Test if the issue is in the query construction
        print("\n6. Testing various query constructions:")
        test_queries = [
            {"life_period": test_life_period},
            {"life_period": {"$eq": test_life_period}},
        ]
        
        for i, where_clause in enumerate(test_queries):
            try:
                query_results = collection.query(
                    query_texts=["debug"],
                    where=where_clause,
                    n_results=10
                )
                print(f"   Query {i+1} ({where_clause}): {len(query_results['ids'][0])} results")
            except Exception as e:
                print(f"   Query {i+1} failed: {e}")
    
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        try:
            shutil.rmtree(temp_dir)
            print(f"\n✅ Cleaned up temp directory: {temp_dir}")
        except Exception as e:
            print(f"⚠️ Could not clean up temp directory: {e}")

if __name__ == "__main__":
    debug_life_period()
