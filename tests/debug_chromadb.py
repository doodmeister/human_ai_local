#!/usr/bin/env python3
"""
Debug ChromaDB initialization issues
"""
import sys
import tempfile
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("Importing ChromaDB...")
try:
    import chromadb
    from chromadb.config import Settings
    print("✓ ChromaDB imported successfully")
except ImportError as e:
    print(f"❌ ChromaDB import failed: {e}")
    sys.exit(1)

print("Testing basic ChromaDB initialization...")
temp_dir = tempfile.mkdtemp()
print(f"Using temp dir: {temp_dir}")

try:
    # Test basic client creation
    print("Creating ChromaDB client...")
    
    settings = Settings(
        allow_reset=True,
        anonymized_telemetry=False,
        persist_directory=str(temp_dir)
    )
    
    client = chromadb.PersistentClient(
        path=str(temp_dir),
        settings=settings
    )
    print("✓ ChromaDB client created")
    
    # Test collection creation
    print("Creating collection...")
    collection = client.create_collection(
        name="debug_test",
        metadata={"description": "Debug test collection"}
    )
    print("✓ Collection created")
    
    # Test basic operations
    print("Testing basic operations...")
    collection.upsert(
        ids=["test1"],
        documents=["This is a test document"],
        metadatas=[{"type": "test"}]
    )
    print("✓ Document stored")
    
    result = collection.get(ids=["test1"])
    print(f"✓ Document retrieved: {result}")
    
    print("✓ All ChromaDB operations successful")
    
except Exception as e:
    print(f"❌ ChromaDB test failed: {e}")
    import traceback
    traceback.print_exc()
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)
    print("✓ Cleanup completed")
