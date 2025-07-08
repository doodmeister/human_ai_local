from src.memory.procedural.procedural_memory import ProceduralMemory
from src.memory.stm.vector_stm import VectorShortTermMemory, STMConfiguration
from src.memory.ltm.vector_ltm import VectorLongTermMemory
import tempfile
import os

def make_pm():
    with tempfile.TemporaryDirectory() as temp_dir:
        config = STMConfiguration(
            chroma_persist_dir=os.path.join(temp_dir, "proc_stm"),
            collection_name="procedural_test",
            capacity=10
        )
        stm = VectorShortTermMemory(config)
        ltm = VectorLongTermMemory()
        pm = ProceduralMemory(stm=stm, ltm=ltm)
        return pm

def test_store_and_retrieve():
    pm = make_pm()
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

def test_store_and_retrieve_ltm():
    pm = make_pm()
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

def test_search():
    pm = make_pm()
    pm.store(description="How to tie shoes", steps=["Cross laces", "Pull tight"], tags=["shoes"], memory_type="stm")
    pm.store(description="How to make coffee", steps=["Boil water", "Add coffee grounds"], tags=["kitchen"], memory_type="ltm")
    results = pm.search("coffee")
    # Filter unique by id in case of duplicates
    unique = {r["id"]: r for r in results}.values()
    assert len(unique) == 1
    assert next(iter(unique))["description"] == "How to make coffee"

def test_use_strengthens_memory():
    pm = make_pm()
    proc_id = pm.store(description="How to jump rope", steps=["Hold handles", "Swing rope", "Jump"], tags=["exercise"], memory_type="stm")
    pm.use(proc_id)
    pm.use(proc_id)
    proc = pm.retrieve(proc_id)
    assert proc is not None
    assert proc["usage_count"] == 2
    assert proc["strength"] > 0.1

def test_delete_and_clear():
    pm = make_pm()
    proc_id = pm.store(description="How to whistle", steps=["Purse lips", "Blow air"], tags=["music"], memory_type="stm")
    assert pm.delete(proc_id)
    assert pm.retrieve(proc_id) is None
    pm.store(description="How to snap fingers", steps=["Position fingers", "Snap"], tags=["music"], memory_type="ltm")
    pm.clear()
    assert len(pm.all_procedures()) == 0

