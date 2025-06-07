#!/usr/bin/env python3
"""
Test script to verify dummy classes work correctly when hopfield-layers is not available
"""

import sys
import torch
import torch.nn as nn

# Temporarily hide hflayers to test fallback
original_modules = sys.modules.copy()
if 'hflayers' in sys.modules:
    del sys.modules['hflayers']

# Also add a fake hflayers that raises ImportError
class FakeHflayers:
    def __getattr__(self, name):
        raise ImportError("Test: hflayers not available")

sys.modules['hflayers'] = FakeHflayers()

try:
    # Now import our module - it should use dummy classes
    from src.processing.neural.lshn_network import LSHNNetwork, LSHNConfig, HOPFIELD_AVAILABLE
    
    print(f"HOPFIELD_AVAILABLE: {HOPFIELD_AVAILABLE}")
    
    if not HOPFIELD_AVAILABLE:
        print("✅ Successfully detected hopfield-layers as unavailable")
        
        # Test that we can create a config and network
        config = LSHNConfig()
        print("✅ Config created successfully")
        
        # Test that we can create the network (should use simplified memory)
        network = LSHNNetwork(config)
        print("✅ Network created successfully with fallback implementation")
        
        # Test that encoder works
        test_input = torch.randn(2, config.embedding_dim)
        encoded = network.encoder(test_input)
        print(f"✅ Encoder works: input shape {test_input.shape} -> output shape {encoded.shape}")
        
        # Test that associative memory is None (fallback mode)
        if network.associative_memory is None:
            print("✅ Associative memory correctly disabled in fallback mode")
        else:
            print("❌ Associative memory should be None in fallback mode")
            
        print("\n🎉 All dummy class tests passed!")
        
    else:
        print("❌ HOPFIELD_AVAILABLE should be False when hflayers import fails")
        
finally:
    # Restore original modules
    sys.modules.clear()
    sys.modules.update(original_modules)
