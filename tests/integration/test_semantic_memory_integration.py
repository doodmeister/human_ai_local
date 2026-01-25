
import unittest
import os
import shutil
from src.orchestration.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig, MemoryConfig

class TestSemanticMemoryIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = "test_semantic_memory_integration_data"
        os.makedirs(self.test_dir, exist_ok=True)
        
        # Create a config with a dedicated path for semantic memory
        self.config = CognitiveConfig(
            memory=MemoryConfig(
                semantic_storage_path=os.path.join(self.test_dir, "semantic_memory.db")
            )
        )
        
        self.agent = CognitiveAgent(config=self.config)

    def tearDown(self):
        # Clean up the test directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def test_agent_fact_storage_retrieval_and_deletion(self):
        # 1. Store a fact through the agent
        subject = "test_subject"
        predicate = "test_predicate"
        object_ = "test_object"
        self.agent.store_fact(subject, predicate, object_)

        # 2. Retrieve the fact using the agent
        found_facts = self.agent.find_facts(subject=subject)
        self.assertEqual(len(found_facts), 1)
        self.assertEqual(found_facts[0], (subject, predicate, object_))

        # 3. Delete the fact using the agent
        deleted = self.agent.delete_fact(subject, predicate, object_)
        self.assertTrue(deleted)

        # 4. Verify the fact is gone
        found_facts_after_delete = self.agent.find_facts(subject=subject)
        self.assertEqual(len(found_facts_after_delete), 0)

if __name__ == '__main__':
    unittest.main()
