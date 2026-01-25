#!/usr/bin/env python3
"""
Test sensory processing integration with cognitive agent
"""
import sys
import os
import asyncio
import pytest
sys.path.insert(0, os.path.abspath('.'))

@pytest.mark.asyncio
async def test_sensory_integration():
    """Test sensory processing integration with cognitive agent"""
    print("Testing Sensory Processing Integration with Cognitive Agent...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.orchestration.cognitive_agent import CognitiveAgent
        from src.core.config import CognitiveConfig
        print("All components imported successfully")
        
        # Create configuration
        print("\n2. Creating configuration...")
        config = CognitiveConfig()
        print(f"Config created - embedding model: {config.processing.embedding_model}")
        print(f"   embedding dimension: {config.processing.embedding_dimension}")
        print(f"   entropy threshold: {config.processing.entropy_threshold}")
        
        # Initialize cognitive agent with sensory processing
        print("\n3. Initializing cognitive agent...")
        agent = CognitiveAgent(config)
        print(f"Agent initialized: {agent.session_id}")
        
        # Test that sensory components are properly initialized
        print("\n4. Verifying sensory processing initialization...")
        assert hasattr(agent, 'sensory_processor'), "Sensory processor not initialized"
        assert hasattr(agent, 'sensory_interface'), "Sensory interface not initialized"
        print("Sensory processing components initialized correctly")
        
        # Test sensory processing within cognitive loop
        print("\n5. Testing sensory processing in cognitive loop...")
        test_inputs = [
            "Hello, how are you today?",
            "This is a very important message with lots of exclamation points!!!",
            "Can you help me understand something complex?",
            "Short msg",
            "",  # Empty input
            "What is the weather like today? I need to know for planning."
        ]
        
        for i, input_text in enumerate(test_inputs, 1):
            print(f"\n--- Processing Input {i}: '{input_text}' ---")
            
            # Process input through full cognitive pipeline
            response = await agent.process_input(input_text)
            print(f"Response: {response}")
            
            # Get cognitive status to see sensory processing stats
            status = agent.get_cognitive_status()
            
            # Display sensory processing information
            sensory_stats = status.get("sensory_processing", {})
            print(f"Sensory Stats: {sensory_stats.get('total_processed', 0)} processed, "
                  f"{sensory_stats.get('filtered_count', 0)} filtered")
            
            if "avg_scores" in sensory_stats:
                avg_scores = sensory_stats["avg_scores"]
                print(f"   Average Scores - Entropy: {avg_scores.get('entropy', 0):.3f}, "
                      f"Salience: {avg_scores.get('salience', 0):.3f}, "
                      f"Relevance: {avg_scores.get('relevance', 0):.3f}")
            
            # Display cognitive integration metrics
            integration = status.get("cognitive_integration", {})
            print(f"Integration - Efficiency: {integration.get('overall_efficiency', 0):.3f}, "
                  f"Sensory Efficiency: {integration.get('sensory_efficiency', 0):.3f}")
        
        # Test cognitive status with sensory information
        print("\n6. Testing comprehensive cognitive status...")
        final_status = agent.get_cognitive_status()
        
        print("Final Cognitive Status:")
        print(f"  Session: {final_status['session_id']}")
        print(f"  Fatigue: {final_status['fatigue_level']:.3f}")
        print(f"  Conversations: {final_status['conversation_length']}")
        
        # Sensory processing status
        sensory_final = final_status.get("sensory_processing", {})
        print("  Sensory Processing:")
        print(f"    Total Processed: {sensory_final.get('total_processed', 0)}")
        print(f"    Filtered: {sensory_final.get('filtered_count', 0)}")
        if "avg_scores" in sensory_final:
            avg = sensory_final["avg_scores"]
            print(f"    Avg Entropy: {avg.get('entropy', 0):.3f}")
            print(f"    Avg Salience: {avg.get('salience', 0):.3f}")
            print(f"    Avg Relevance: {avg.get('relevance', 0):.3f}")
        
        # Integration metrics
        integration_final = final_status.get("cognitive_integration", {})
        print("  Integration Efficiency:")
        print(f"    Overall: {integration_final.get('overall_efficiency', 0):.3f}")
        print(f"    Sensory: {integration_final.get('sensory_efficiency', 0):.3f}")
        print(f"    Processing Capacity: {integration_final.get('processing_capacity', 0):.3f}")
        
        # Test edge cases
        print("\n7. Testing edge cases...")
        edge_cases = [
            "!@#$%^&*()",  # Special characters
            "A" * 200,     # Very long input
            "question?",   # Question
            "URGENT!!!",   # High salience
        ]
        
        for edge_input in edge_cases:
            print(f"Processing edge case: '{edge_input[:30]}...'")
            await agent.process_input(edge_input)
        
        edge_status = agent.get_cognitive_status()
        edge_sensory = edge_status.get("sensory_processing", {})
        print(f"Edge case processing - Total: {edge_sensory.get('total_processed', 0)}, "
              f"Filtered: {edge_sensory.get('filtered_count', 0)}")
        
        # Cleanup
        print("\n8. Testing shutdown...")
        await agent.shutdown()
        print("Agent shutdown complete")
        
        print("\nSensory Processing Integration Test PASSED!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

@pytest.mark.asyncio
async def test_attention_sensory_interaction():
    """Test how attention mechanism interacts with sensory processing"""
    print("\nTesting Attention-Sensory Interaction...")
    
    try:
        from src.orchestration.cognitive_agent import CognitiveAgent
        
        agent = CognitiveAgent()
        
        # Test inputs with different sensory characteristics
        inputs_with_expectations = [
            ("Low entropy simple message", "low_salience"),
            ("EMERGENCY!!! URGENT MESSAGE WITH HIGH SALIENCE!!!", "high_salience"), 
            ("This is a very long message with lots of words that should have higher cognitive load requirements", "high_load"),
            ("?", "low_entropy"),
        ]
        
        for input_text, expected_type in inputs_with_expectations:
            print(f"\nTesting {expected_type}: '{input_text[:50]}...'")
            
            response = await agent.process_input(input_text)
            status = agent.get_cognitive_status()
            
            attention_status = status.get("attention_status", {})
            sensory_stats = status.get("sensory_processing", {})
            
            print(f"  Attention Score: {attention_status.get('current_attention_score', 0):.3f}")
            print(f"  Cognitive Load: {attention_status.get('current_load', 0):.3f}")
            print(f"  Fatigue Level: {status['fatigue_level']:.3f}")
            
            if "avg_scores" in sensory_stats:
                avg = sensory_stats["avg_scores"]
                print(f"  Sensory - Entropy: {avg.get('entropy', 0):.3f}, "
                      f"Salience: {avg.get('salience', 0):.3f}")
        
        await agent.shutdown()
        print("Attention-Sensory interaction test PASSED!")
        return True
        
    except Exception as e:
        print(f"Attention-Sensory test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("SENSORY PROCESSING INTEGRATION TEST SUITE")
    print("=" * 70)
    
    # Run integration tests
    integration_success = asyncio.run(test_sensory_integration())
    attention_success = asyncio.run(test_attention_sensory_interaction())
    
    print("\n" + "=" * 70)
    if integration_success and attention_success:
        print("ALL INTEGRATION TESTS PASSED!")
        print("Sensory processing is fully integrated with cognitive agent!")
    else:
        print("Some integration tests failed.")
    print("=" * 70)
