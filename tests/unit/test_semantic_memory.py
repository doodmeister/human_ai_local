import unittest
import os
import json
import shutil
from src.memory.semantic.semantic_memory import SemanticMemorySystem

class TestSemanticMemorySystem(unittest.TestCase):

    def setUp(self):
        """Set up a temporary storage directory and file for testing."""
        self.test_dir = "temp_test_semantic_memory"
        os.makedirs(self.test_dir, exist_ok=True)
        self.storage_path = os.path.join(self.test_dir, "semantic_kb.json")
        self.sms = SemanticMemorySystem(storage_path=self.storage_path)

    def tearDown(self):
        """Clean up the temporary storage directory and file."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_store_and_retrieve_fact(self):
        """Test storing a new fact and retrieving it by its ID."""
        fact_id = self.sms.store_fact("Paris", "is_capital_of", "France")
        self.assertIsNotNone(fact_id)
        
        retrieved_fact = self.sms.retrieve_fact(fact_id)
        self.assertIsNotNone(retrieved_fact, f"Fact with ID {fact_id} should exist but was not found")
        self.assertEqual(retrieved_fact["subject"], "paris")
        self.assertEqual(retrieved_fact["predicate"], "is_capital_of")
        self.assertEqual(retrieved_fact["object"], "France")

    def test_find_facts_by_subject(self):
        """Test finding facts that match a specific subject."""
        self.sms.store_fact("Paris", "is_capital_of", "France")
        self.sms.store_fact("Paris", "has_population", 2141000)
        self.sms.store_fact("Berlin", "is_capital_of", "Germany")

        paris_facts = self.sms.find_facts(subject="Paris")
        self.assertEqual(len(paris_facts), 2)
        subjects = {fact["subject"] for fact in paris_facts}
        self.assertEqual(subjects, {"paris"})

    def test_find_facts_by_predicate(self):
        """Test finding facts that match a specific predicate."""
        self.sms.store_fact("Paris", "is_capital_of", "France")
        self.sms.store_fact("Berlin", "is_capital_of", "Germany")
        self.sms.store_fact("Jupiter", "is_planet", True)

        capital_facts = self.sms.find_facts(predicate="is_capital_of")
        self.assertEqual(len(capital_facts), 2)
        predicates = {fact["predicate"] for fact in capital_facts}
        self.assertEqual(predicates, {"is_capital_of"})

    def test_find_facts_by_subject_and_predicate(self):
        """Test finding facts that match both a subject and a predicate."""
        self.sms.store_fact("Paris", "is_capital_of", "France")
        self.sms.store_fact("Paris", "has_population", 2141000)

        fact = self.sms.find_facts(subject="Paris", predicate="is_capital_of")
        self.assertEqual(len(fact), 1)
        self.assertEqual(fact[0]["object"], "France")

    def test_delete_fact(self):
        """Test deleting a fact from the knowledge base."""
        fact_id = self.sms.store_fact("Pluto", "is_planet", False)
        self.assertIsNotNone(self.sms.retrieve_fact(fact_id))

        deleted = self.sms.delete_fact(fact_id)
        self.assertTrue(deleted)
        self.assertIsNone(self.sms.retrieve_fact(fact_id))

    def test_persistence(self):
        """Test that the knowledge base is correctly saved to and loaded from the file."""
        fact_id = self.sms.store_fact("Earth", "orbits", "Sun")
        
        # Create a new instance to force loading from the file
        new_sms = SemanticMemorySystem(storage_path=self.storage_path)
        retrieved_fact = new_sms.retrieve_fact(fact_id)
        self.assertIsNotNone(retrieved_fact, f"Fact with ID {fact_id} should exist but was not found")
        self.assertEqual(retrieved_fact["subject"], "earth")

if __name__ == '__main__':
    unittest.main()
