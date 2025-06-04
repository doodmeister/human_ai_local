#!/usr/bin/env python3
"""
Comprehensive test for the Sensory Processing module
"""
import sys
import os
import numpy as np
import time

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath('.'))

def test_sensory_processing():
    """Test all sensory processing functionality"""
    print("Testing Sensory Processing Module...")
    
    try:
        # Test imports
        print("1. Testing imports...")
        from src.processing.sensory import (
            SensoryProcessor,
            SensoryInput,
            ProcessedSensoryData,
            SensoryInterface,
            SensoryInputBuilder,
            quick_text_input,
            quick_process_text
        )
        print("All sensory processing components imported successfully")
        
        # Test SensoryInput creation
        print("\n2. Testing SensoryInput creation...")
        input1 = SensoryInput(
            content="Hello, this is a test message!",
            modality="text",
            source="user"
        )
        print(f"SensoryInput created: {input1.content[:30]}...")
        assert input1.timestamp is not None
        assert input1.modality == "text"
        assert input1.source == "user"
        
        # Test SensoryProcessor
        print("\n3. Testing SensoryProcessor...")
        processor = SensoryProcessor()
        print("SensoryProcessor initialized")
        
        # Test single input processing
        processed = processor.process_input(input1)
        print(f"Input processed - Entropy: {processed.entropy_score:.3f}, "
              f"Salience: {processed.salience_score:.3f}, "
              f"Relevance: {processed.relevance_score:.3f}")
        
        assert isinstance(processed, ProcessedSensoryData)
        assert processed.embedding.shape == (384,)
        assert 0.0 <= processed.entropy_score <= 1.0
        assert 0.0 <= processed.salience_score <= 1.0
        assert 0.0 <= processed.relevance_score <= 1.0
        
        # Test batch processing
        print("\n4. Testing batch processing...")
        batch_inputs = [
            SensoryInput("First message", "text", source="user"),
            SensoryInput("Second message with more content!", "text", source="user"),
            SensoryInput("", "text", source="system"),  # Empty content
            SensoryInput("Question? This should have higher salience!", "text", source="user")
        ]
        
        batch_results = processor.process_batch(batch_inputs)
        print(f"Batch processed: {len(batch_results)} items")
        assert len(batch_results) == len(batch_inputs)
        
        # Check filtering
        filtered_count = sum(1 for r in batch_results if r.filtered)
        print(f"Filtered items: {filtered_count}")
        
        # Test processing statistics
        print("\n5. Testing processing statistics...")
        stats = processor.get_processing_stats()
        print(f"Statistics: {stats['total_processed']} processed, "
              f"{stats['filtered_count']} filtered")
        assert stats['total_processed'] > 0
        assert 'avg_scores' in stats
        
        # Test SensoryInterface
        print("\n6. Testing SensoryInterface...")
        interface = SensoryInterface(processor)
        print("SensoryInterface initialized")
        
        # Test user input processing
        user_result = interface.process_user_input("Hello from interface!")
        print("User input processed via interface")
        assert isinstance(user_result, ProcessedSensoryData)
        assert user_result.input_data.source == "user"
        
        # Test quick utilities
        print("\n7. Testing quick utilities...")
        quick_input = quick_text_input("Quick input test")
        quick_result = quick_process_text("Quick processing test")
        print("Quick utilities working")
        assert quick_input.modality == "text"
        assert isinstance(quick_result, ProcessedSensoryData)
        
        print("\nCore sensory processing tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("SENSORY PROCESSING MODULE TEST")
    print("=" * 60)
    
    success = test_sensory_processing()
    
    print("\n" + "=" * 60)
    if success:
        print("SENSORY PROCESSING TEST PASSED!")
    else:
        print("Test failed.")
    print("=" * 60)
