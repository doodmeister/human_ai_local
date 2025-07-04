import unittest
import os
import shutil
from memory.semantic.semantic_memory import SemanticMemorySystem

class TestSemanticMemorySystem(unittest.TestCase):

    def setUp(self):
        """Set up a temporary storage directory and file for testing."""
        self.test_dir = "temp_test_semantic_memory"
        os.makedirs(self.test_dir, exist_ok=True)
        self.chroma_dir = os.path.join(self.test_dir, "chroma_semantic")
        self.sms = SemanticMemorySystem(chroma_persist_dir=self.chroma_dir)
        self.sms.clear()  # Ensure clean state for each test

    def tearDown(self):
        """Clean up the temporary storage directory and file."""
        if hasattr(self, 'sms') and self.sms:
            self.sms.shutdown()
        import time
        time.sleep(0.5)  # Give ChromaDB time to release file handles
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_store_and_retrieve_fact(self):
        """Test storing a new fact and retrieving it by its ID."""
        fact_id = self.sms.store_fact("Paris", "is_capital_of", "France")
        self.assertIsNotNone(fact_id)
        import time
        retrieved_fact = None
        for _ in range(5):
            rf = self.sms.retrieve_fact(fact_id)
            if rf is not None:
                retrieved_fact = rf
                break
            time.sleep(0.2)
        self.assertIsNotNone(retrieved_fact, f"Fact with ID {fact_id} should exist but was not found")
        self.assertEqual(retrieved_fact["subject"], "paris")
        self.assertEqual(retrieved_fact["predicate"], "is_capital_of")
        self.assertEqual(retrieved_fact["object"], "France")

    def test_find_facts_by_subject(self):
        """Test finding facts that match a specific subject."""
        self.sms.store_fact("Paris", "is_capital_of", "France")
        self.sms.store_fact("Paris", "has_population", 2141000)
        self.sms.store_fact("Berlin", "is_capital_of", "Germany")

        # Always query with normalized subject (lowercase)
        paris_facts = self.sms.find_facts(subject="paris")
        self.assertEqual(len(paris_facts), 2)
        subjects = {fact["subject"] for fact in paris_facts}
        self.assertEqual(subjects, {"paris"})

    def test_find_facts_by_predicate(self):
        """Test finding facts that match a specific predicate."""
        self.sms.store_fact("Paris", "is_capital_of", "France")
        self.sms.store_fact("Berlin", "is_capital_of", "Germany")
        self.sms.store_fact("Jupiter", "is_planet", True)

        # Always query with normalized predicate (lowercase)
        capital_facts = self.sms.find_facts(predicate="is_capital_of")
        self.assertEqual(len(capital_facts), 2)
        predicates = {fact["predicate"] for fact in capital_facts}
        self.assertEqual(predicates, {"is_capital_of"})

    def test_find_facts_by_subject_and_predicate(self):
        """Test finding facts that match both a subject and a predicate."""
        self.sms.store_fact("Paris", "is_capital_of", "France")
        self.sms.store_fact("Paris", "has_population", 2141000)

        # Always query with normalized subject and predicate (lowercase)
        fact = self.sms.find_facts(subject="paris", predicate="is_capital_of")
        self.assertEqual(len(fact), 1)
        self.assertEqual(fact[0]["object"], "France")

    def test_delete_fact(self):
        """Test deleting a fact from the knowledge base."""
        fact_id = self.sms.store_fact("Pluto", "is_planet", False)
        self.assertIsNotNone(self.sms.retrieve_fact(fact_id))

        # Always query with normalized subject/predicate and string object
        deleted = self.sms.delete_fact("Pluto", "is_planet", False)
        self.assertTrue(deleted)
        facts = self.sms.find_facts(subject="pluto", predicate="is_planet", object_val="False")
        self.assertEqual(len(facts), 0)

    def test_persistence(self):
        """Test that the knowledge base is correctly saved to and loaded from ChromaDB."""
        fact_id = self.sms.store_fact("Earth", "orbits", "Sun")
        self.assertIsNotNone(fact_id)
        self.sms.shutdown()
        import time
        new_sms = SemanticMemorySystem(chroma_persist_dir=self.chroma_dir)
        retrieved_fact = None
        for _ in range(5):
            rf = new_sms.retrieve_fact(fact_id)
            if rf is not None:
                retrieved_fact = rf
                break
            time.sleep(0.2)
        self.assertIsNotNone(retrieved_fact, f"Fact with ID {fact_id} should exist but was not found")
        self.assertEqual(retrieved_fact["subject"], "earth")
        new_sms.shutdown()

if __name__ == '__main__':
    unittest.main()
