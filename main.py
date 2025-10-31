"""
LEGACY CLI Demo Entry Point (Retained for Reference)

This file provides a simple command-line interface for the Human-AI Cognition Framework.
It demonstrates basic agent initialization and conversation flow.

**STATUS: LEGACY / REFERENCE ONLY**
For production use, see:
- `start_george.py` - Streamlit UI interface
- `start_server.py` - FastAPI REST API server

This file is kept as:
1. A minimal example of direct CognitiveAgent usage
2. Reference for CLI-based interactions
3. Quick testing without UI dependencies

Usage:
    python main.py

Note: Requires all dependencies from requirements.txt.
This may become outdated if core APIs change.
"""
import asyncio
import sys
from pathlib import Path

# Add src to Python path for development
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core import CognitiveAgent, CognitiveConfig
from src.utils import setup_logging

async def main():
    """Main demonstration of the cognitive framework"""
    
    # Setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting Human-AI Cognition Framework demonstration")
    
    try:
        # Initialize cognitive agent
        config = CognitiveConfig.from_env()
        agent = CognitiveAgent(config)
        
        logger.info("Cognitive agent initialized successfully")
        logger.info(f"Session ID: {agent.session_id}")
        
        # Display cognitive status
        status = agent.get_cognitive_status()
        logger.info(f"Initial cognitive status: {status}")
        
        # Demonstrate basic conversation
        print("\n" + "="*60)
        print("🧠 HUMAN-AI COGNITION FRAMEWORK DEMO")
        print("="*60)
        print("Type 'quit' to exit, 'status' for cognitive state")
        print()
        
        while True:
            try:
                user_input = input("You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("Agent: Goodbye! Entering dream state for memory consolidation...")
                    await agent.enter_dream_state()
                    break
                
                if user_input.lower() == 'status':
                    status = agent.get_cognitive_status()
                    print(f"Agent Status: {status}")
                    continue
                
                if not user_input:
                    continue
                
                # Process input through cognitive architecture
                response = await agent.process_input(user_input)
                print(f"Agent: {response}")
                
                # Show cognitive state changes
                status = agent.get_cognitive_status()
                print(f"[Fatigue: {status['fatigue_level']:.3f}, Conversations: {status['conversation_length']}]")
                print()
                
            except KeyboardInterrupt:
                print("\nAgent: Interrupted. Shutting down gracefully...")
                break
            except Exception as e:
                logger.error(f"Error in conversation loop: {e}")
                print(f"Agent: Sorry, I encountered an error: {e}")
        
        # Shutdown
        await agent.shutdown()
        logger.info("Cognitive agent shutdown complete")
        
    except Exception as e:
        logger.error(f"Fatal error in main: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
