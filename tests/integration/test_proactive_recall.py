import asyncio
import unittest
from unittest.mock import AsyncMock, patch
import uuid
import shutil
import os
import gc
import time

from src.core.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig

class TestProactiveRecall(unittest.TestCase):

    def setUp(self):
        """Set up a cognitive agent for testing."""
        config = CognitiveConfig()
        # Use a temporary directory for ChromaDB for this test
        self.test_dir = f"./test_data/chroma_proactive_recall_{uuid.uuid4()}"
        config.memory.chroma_persist_dir = self.test_dir
        config.memory.ltm_storage_path = self.test_dir  # Also use for json backups
        self.agent = CognitiveAgent(config=config)

    def tearDown(self):
        """Clean up the temporary directory after the test."""
        # Give ChromaDB time to release file handles
        if hasattr(self, 'agent') and self.agent:
            # Attempt to close/cleanup any ChromaDB connections
            try:
                if hasattr(self.agent, 'memory'):
                    if hasattr(self.agent.memory, 'ltm') and hasattr(self.agent.memory.ltm, '_chroma_client'):
                        del self.agent.memory.ltm._chroma_client
                    if hasattr(self.agent.memory, 'stm') and hasattr(self.agent.memory.stm, '_chroma_client'):
                        del self.agent.memory.stm._chroma_client
            except Exception:
                pass
            del self.agent
        
        gc.collect()
        time.sleep(0.1)  # Small delay for file handle release
        
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except PermissionError:
                # On Windows, ChromaDB files may still be locked
                # Mark for later cleanup or ignore
                pass

    @patch('src.core.cognitive_agent.CognitiveAgent._generate_response', new_callable=AsyncMock)
    def test_proactive_recall_retrieves_contextual_memory(self, mock_generate_response):
        """Test that the agent can proactively recall a memory based on conversation context."""
        # Arrange
        # 1. Manually add a specific memory that should be recalled later.
        initial_memory_content = "The user mentioned their favorite hobby is space exploration."
        self.agent.memory.store_memory(
            memory_id=str(uuid.uuid4()),
            content=initial_memory_content,
            importance=0.9
        )

        # 2. Define the conversation flow.
        conversation = [
            {"user_input": "I'm looking for a new activity.", "ai_response": "There are many activities to choose from. Do you prefer indoor or outdoor?"},
            {"user_input": "I enjoy things that make you think, especially about space.", "ai_response": "That's fascinating! Exploring the cosmos is a wonderful pursuit."},
            {"user_input": "What was that thing I told you I enjoyed?", "ai_response": "You mentioned you enjoy space exploration."}
        ]

        # Mock the AI's responses to control the conversation
        mock_generate_response.side_effect = [turn["ai_response"] for turn in conversation]

        # Act & Assert
        async def run_conversation():
            # We will capture the memory context retrieved during the final turn.
            final_memory_context = []
            call_count = 0

            # Patch the memory search to intercept the results of the last call
            original_search = self.agent.memory.search_memories
            
            def patched_search(*args, **kwargs):
                nonlocal final_memory_context, call_count
                call_count += 1
                # Call the original search method
                result = original_search(*args, **kwargs)
                # Capture the context on the last turn (3rd call)
                if call_count == 3:
                    final_memory_context = result
                return result

            with patch.object(self.agent.memory, 'search_memories', side_effect=patched_search):
                for turn in conversation:
                    await self.agent.process_input(turn["user_input"])

            # 3. Assert that the correct memory was recalled in the final step.
            self.assertTrue(final_memory_context, "Memory context was not captured or is empty.")

            def get_content(mem_obj):
                if hasattr(mem_obj, 'detailed_content'):  # Episodic
                    return mem_obj.detailed_content
                if hasattr(mem_obj, 'content'):  # STM/LTM
                    return mem_obj.content
                if isinstance(mem_obj, dict) and 'content' in mem_obj:  # LTM (dict)
                    return mem_obj['content']
                return ''

            self.assertTrue(any(initial_memory_content in get_content(mem[0]) for mem in final_memory_context),
                            f"The initial memory about space exploration was not found in the final retrieved context. Context: {final_memory_context}")
            print("\nProactive recall test successful. The agent correctly recalled the memory about space exploration.")

        # Run the async test
        asyncio.run(run_conversation())

if __name__ == '__main__':
    unittest.main()
