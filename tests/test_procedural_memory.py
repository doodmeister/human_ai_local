from src.memory.procedural.procedural_memory import ProceduralMemory
from src.memory.stm import VectorShortTermMemory, STMConfiguration
from src.memory.ltm.vector_ltm import VectorLongTermMemory
import tempfile
import os
import pytest

# Use module-level temp dir to avoid Windows file handle issues with ChromaDB
_temp_dir = None

@pytest.fixture(scope="module")
def temp_dir():
    global _temp_dir
    import atexit
    import shutil
    _temp_dir = tempfile.mkdtemp()
    
    def cleanup():
        try:
            shutil.rmtree(_temp_dir, ignore_errors=True)
        except Exception:
            pass
    atexit.register(cleanup)
    return _temp_dir

def make_pm(temp_dir_path=None):
    if temp_dir_path is None:
        # Fallback for standalone use
        temp_dir_path = tempfile.mkdtemp()
    
    test_id = os.urandom(4).hex()  # Unique collection per test
    config = STMConfiguration(
        chroma_persist_dir=os.path.join(temp_dir_path, f"proc_stm_{test_id}"),
        collection_name=f"procedural_stm_{test_id}",
        capacity=10
    )
    stm = VectorShortTermMemory(config)
    # Give LTM a unique persist dir and collection too
    ltm = VectorLongTermMemory(
        chroma_persist_dir=os.path.join(temp_dir_path, f"proc_ltm_{test_id}"),
        collection_name=f"procedural_ltm_{test_id}"
    )
    pm = ProceduralMemory(stm=stm, ltm=ltm)
    return pm

def test_store_and_retrieve(temp_dir):
    pm = make_pm(temp_dir)
    proc_id = pm.store(
        description="How to make tea",
        steps=["Boil water", "Add tea leaves", "Steep", "Pour into cup"],
        tags=["kitchen", "beverage"],
        memory_type="stm"
    )
    proc = pm.retrieve(proc_id)
    assert proc is not None
    assert proc["description"] == "How to make tea"
    assert "Boil water" in proc["steps"]

def test_store_and_retrieve_ltm(temp_dir):
    pm = make_pm(temp_dir)
    proc_id = pm.store(
        description="How to make bread",
        steps=["Mix ingredients", "Knead dough", "Bake"],
        tags=["kitchen", "baking"],
        memory_type="ltm"
    )
    proc = pm.retrieve(proc_id)
    assert proc is not None
    assert proc["description"] == "How to make bread"
    assert "Knead dough" in proc["steps"]

def test_search(temp_dir):
    pm = make_pm(temp_dir)
    pm.store(description="How to tie shoes", steps=["Cross laces", "Pull tight"], tags=["shoes"], memory_type="stm")
    pm.store(description="How to make coffee", steps=["Boil water", "Add coffee grounds"], tags=["kitchen"], memory_type="ltm")
    results = pm.search("coffee")
    # Filter unique by id in case of duplicates
    unique = {r["id"]: r for r in results}.values()
    assert len(unique) == 1
    assert next(iter(unique))["description"] == "How to make coffee"

def test_use_strengthens_memory(temp_dir):
    pm = make_pm(temp_dir)
    proc_id = pm.store(description="How to jump rope", steps=["Hold handles", "Swing rope", "Jump"], tags=["exercise"], memory_type="stm")
    pm.use(proc_id)
    pm.use(proc_id)
    proc = pm.retrieve(proc_id)
    assert proc is not None
    assert proc["usage_count"] == 2
    assert proc["strength"] > 0.1

def test_delete_and_clear(temp_dir):
    pm = make_pm(temp_dir)
    proc_id = pm.store(description="How to whistle", steps=["Purse lips", "Blow air"], tags=["music"], memory_type="stm")
    assert pm.delete(proc_id)
    assert pm.retrieve(proc_id) is None
    pm.store(description="How to snap fingers", steps=["Position fingers", "Snap"], tags=["music"], memory_type="ltm")
    pm.clear()
    assert len(pm.all_procedures()) == 0

