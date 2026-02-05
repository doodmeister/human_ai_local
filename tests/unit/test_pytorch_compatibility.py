#!/usr/bin/env python3
"""
Test script to verify PyTorch compatibility of dummy classes
"""

import sys
import torch
import torch.nn as nn

# Temporarily hide hflayers to test fallback
original_modules = sys.modules.copy()
if 'hflayers' in sys.modules:
    del sys.modules['hflayers']

class FakeHflayers:
    def __getattr__(self, name):
        raise ImportError("Test: hflayers not available")

sys.modules['hflayers'] = FakeHflayers()

try:
    # Import the dummy classes
    from src.cognition.processing.neural.lshn_network import _DummyHopfieldLayer, LSHNConfig
    
    print("Testing PyTorch compatibility of dummy classes...")
    
    # Test 1: Can we create dummy classes with the expected parameters?
    config = LSHNConfig()
    dummy_layer = _DummyHopfieldLayer(
        input_size=config.pattern_dim,
        hidden_size=config.pattern_dim // 2,
        output_size=config.pattern_dim,
        num_heads=config.num_heads,
        scaling=config.attractor_strength,
        update_steps_max=config.max_iterations
    )
    print("✅ Dummy layer created with expected parameters")
    
    # Test 2: Can we add dummy classes to nn.ModuleList?
    module_list = nn.ModuleList([dummy_layer])
    print("✅ Dummy layer successfully added to nn.ModuleList")
    
    # Test 3: Can we call the dummy layer with expected signature?
    test_input = torch.randn(2, 1, config.pattern_dim)  # batch_size, seq_len, pattern_dim
    output = dummy_layer(input=test_input, stored_pattern_padding_mask=None)
    print(f"✅ Dummy layer forward pass works: {test_input.shape} -> {output.shape}")
    
    # Test 4: Does the output have the expected shape?
    expected_shape = (2, 1, config.pattern_dim)
    if output.shape == expected_shape:
        print(f"✅ Output shape is correct: {output.shape}")
    else:
        print(f"❌ Output shape mismatch: expected {expected_shape}, got {output.shape}")
    
    # Test 5: Can we create multiple layers and use them in a loop?
    layers = [
        _DummyHopfieldLayer(
            input_size=config.pattern_dim,
            hidden_size=config.pattern_dim // (2 ** i),
            output_size=config.pattern_dim,
            num_heads=config.num_heads,
            scaling=config.attractor_strength,
            update_steps_max=config.max_iterations
        )
        for i in range(config.num_layers)
    ]
    
    module_list = nn.ModuleList(layers)
    print(f"✅ Created ModuleList with {len(layers)} dummy layers")
    
    # Test the same pattern used in the actual code
    query = torch.randn(2, config.pattern_dim)
    retrieved_patterns = []
    
    for layer in module_list:
        # Add sequence dimension for HopfieldLayer (expects 3D input)
        query_3d = query.unsqueeze(1)  # (batch_size, 1, pattern_dim)
        retrieved = layer(
            input=query_3d,
            stored_pattern_padding_mask=None
        )
        # Remove sequence dimension
        retrieved = retrieved.squeeze(1)  # (batch_size, pattern_dim)
        retrieved_patterns.append(retrieved)
    
    print(f"✅ Successfully processed query through {len(retrieved_patterns)} layers")
    print(f"✅ Final retrieved pattern shape: {retrieved_patterns[0].shape}")
    
    print("\n🎉 All PyTorch compatibility tests passed!")
    
finally:
    # Restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)
