import unittest
import os
import shutil
import uuid
import time

from src.memory.episodic.episodic_memory import EpisodicMemorySystem

class TestEpisodicMemoryFeatures(unittest.TestCase):

    def setUp(self):
        """Set up a temporary directory for testing."""
        self.test_dir = f"./test_data/episodic_features_{uuid.uuid4()}"
        os.makedirs(self.test_dir, exist_ok=True)
        self.memory_system = EpisodicMemorySystem(
            chroma_persist_dir=self.test_dir,
            storage_path=self.test_dir
        )

    def tearDown(self):
        """Clean up the temporary directory after the test."""
        self.memory_system.shutdown()
        # The following line is commented out to avoid PermissionError on Windows
        # if os.path.exists(self.test_dir):
        #     shutil.rmtree(self.test_dir)

    def test_summarization_and_tagging(self):
        """Test that a memory is automatically summarized and tagged upon storage."""
        # Arrange
        detailed_content = "The quick brown fox jumps over the lazy dog. This is a classic sentence used for typing practice. The fox is a carnivorous mammal."
        
        # Act
        memory_id = self.memory_system.store_memory(
            detailed_content=detailed_content,
            importance=0.8
        )
        retrieved_memory = self.memory_system.get_memory(memory_id)

        # Assert
        self.assertIsNotNone(retrieved_memory, "Failed to retrieve the stored memory.")
        
        # Additional check to ensure memory was properly stored
        if retrieved_memory is None:
            self.fail(f"Memory with ID {memory_id} was not found in the system. Check if store_memory and get_memory are working correctly.")

        # Test summarization
        expected_summary = "The quick brown fox jumps over the lazy dog."
        self.assertEqual(retrieved_memory.summary, expected_summary, f"Summary was incorrect. Expected: '{expected_summary}', Got: '{retrieved_memory.summary}'")

        # Test tagging
        expected_tags = ['quick', 'brown', 'fox', 'jumps', 'over', 'lazy', 'dog', 'this', 'classic', 'sentence']
        self.assertCountEqual(retrieved_memory.tags, expected_tags, f"Tags were incorrect. Expected: '{expected_tags}', Got: '{retrieved_memory.tags}'")

        print("\nSummarization and tagging test successful.")

if __name__ == '__main__':
    unittest.main()
