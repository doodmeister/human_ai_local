#!/usr/bin/env python3
"""
Test LSHN Neural Replay Integration

This test specifically verifies that LSHN is working during neural replay
by ensuring we meet the memory threshold requirements and verify both
DPAD and LSHN processing occur.
"""

import warnings
# Suppress scikit-learn deprecation warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

import asyncio
import sys
import os
import pytest
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.core.cognitive_agent import CognitiveAgent
from src.core.config import CognitiveConfig


@pytest.mark.asyncio
async def test_lshn_neural_replay_integration():
    """Test LSHN integration during neural replay with sufficient memories"""
    # Suppress scikit-learn deprecation warnings
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")
    
    print("🧠 TESTING LSHN NEURAL REPLAY INTEGRATION")
    print("=" * 60)
    
    # Create config
    config = CognitiveConfig()
    agent = CognitiveAgent(config)
    
    print(f"✅ Agent initialized: {agent.session_id}")
    
    # Create enough memories to trigger neural replay (threshold is 5)
    print("\n📝 STEP 1: Creating sufficient memories for neural replay...")
    memory_inputs = [
        "My name is Alice and I work as a data scientist at TechCorp.",
        "I specialize in machine learning and neural network architectures.",
        "Today I had a breakthrough with our new AI model using transformers.",
        "The weather is beautiful today, perfect for outdoor activities.",
        "I'm planning to visit my grandmother this weekend in Boston.",
        "Our team is developing an advanced cognitive AI system.",
        "I enjoy reading research papers about artificial intelligence.",
        "The coffee at the new downtown cafe is absolutely excellent.",
        "This evening I will attend a conference on deep learning.",
        "My favorite hobby is playing chess and studying game theory.",
    ]
    
    for i, text in enumerate(memory_inputs, 1):
        response = await agent.process_input(text)
        print(f"   Memory {i}: {text[:50]}...")
    
    # Check memory status
    status = agent.get_cognitive_status()
    memory_status = status["memory_status"]
    stm_count = memory_status['stm'].get('vector_db_count', 0)
    initial_ltm_count = memory_status['ltm'].get('total_memories', memory_status['ltm'].get('vector_db_count', 0))
    
    print("\n📊 Initial Memory Status:")
    print(f"   STM: {stm_count} items")
    print(f"   LTM: {initial_ltm_count} items")
    
    # Check neural integration is available
    if not agent.neural_integration:
        print("❌ Neural integration not available!")
        await agent.shutdown()
        return False
    
    print("✅ Neural integration available")
      # Test direct neural memory replay to verify LSHN processing
    print("\n🔬 STEP 2: Testing direct neural replay...")
    
    # Create test embeddings directly since MemoryItem doesn't have embeddings
    memory_embeddings = [torch.randn(384) for _ in range(8)]  # Create 8 test embeddings
    importance_scores = [0.5 + 0.3 * np.random.random() for _ in range(8)]  # Random importance scores
    
    print(f"   Created {len(memory_embeddings)} test embeddings for replay")
    print(f"   Importance scores: {[f'{score:.3f}' for score in importance_scores]}")
    
    # Perform direct neural replay
    neural_replay_results = await agent.neural_integration.neural_memory_replay(
        memory_embeddings,
        importance_scores,
        attention_context={'test_mode': True}
    )
    
    print("\n🔍 STEP 3: Analyzing neural replay results...")
    print(f"   Replay results keys: {list(neural_replay_results.keys())}")
    
    # Verify both DPAD and LSHN results are present
    if 'dpad_replay' in neural_replay_results:
        dpad_results = neural_replay_results['dpad_replay']
        print(f"   ✅ DPAD replay: {dpad_results.get('replayed_memories', 0)} memories")
        print(f"      Reconstruction quality: {dpad_results.get('reconstruction_quality', 0):.3f}")
        print(f"      Consolidation strength: {dpad_results.get('consolidation_strength', 0):.3f}")
    else:
        print("   ❌ DPAD replay results missing!")
    
    if 'lshn_consolidation' in neural_replay_results:
        lshn_results = neural_replay_results['lshn_consolidation']
        print("   ✅ LSHN consolidation:")
        print(f"      Associations created: {lshn_results.get('associations_created', 0)}")
        print(f"      Episodic patterns: {lshn_results.get('episodic_patterns', 0)}")
        print(f"      Consolidation strength: {lshn_results.get('consolidation_strength', 0):.3f}")
        print(f"      Memory associations: {len(lshn_results.get('memory_associations', []))}")
    else:
        print("   ❌ LSHN consolidation results missing!")
      # Test REM dream cycle which should trigger neural replay
    print("\n🌈 STEP 4: Testing REM dream cycle with neural replay...")
    
    dream_results = await agent.enter_dream_state('rem')
    
    print("   Dream cycle completed")
    if 'neural_replay' in dream_results:
        neural_replay = dream_results['neural_replay']
        print(f"   Neural replay in dream: {neural_replay}")
        print(f"      Memories replayed: {neural_replay.get('memories_replayed', 0)}")
        print(f"      Strength increased: {neural_replay.get('strength_increased', 0):.3f}")
        print(f"      Patterns reinforced: {neural_replay.get('patterns_reinforced', 0)}")
    
    # Final status check
    print("\n📈 STEP 5: Final analysis...")
    final_status = agent.get_cognitive_status()
    final_memory_status = final_status["memory_status"]
    final_stm_count = final_memory_status['stm'].get('vector_db_count', 0)
    final_ltm_count = final_memory_status['ltm'].get('total_memories', final_memory_status['ltm'].get('vector_db_count', 0))
    
    print(f"   Final STM: {final_stm_count} items (was {stm_count})")
    print(f"   Final LTM: {final_ltm_count} items (was {initial_ltm_count})")
    print(f"   Memory transfer: {initial_ltm_count} → {final_ltm_count} (+{final_ltm_count - initial_ltm_count})")
    
    # Verify LSHN integration was successful
    success_checks = [
        ('DPAD replay executed', 'dpad_replay' in neural_replay_results),
        ('LSHN consolidation executed', 'lshn_consolidation' in neural_replay_results),
        ('Sufficient memories processed', len(memory_embeddings) >= 5),
        ('Associations created', neural_replay_results.get('total_associations_created', 0) > 0),
        ('Neural integration functioning', agent.neural_integration is not None)
    ]
    
    print("\n✅ SUCCESS CRITERIA:")
    all_passed = True
    for check_name, check_result in success_checks:
        status_icon = "✅" if check_result else "❌"
        print(f"   {status_icon} {check_name}: {check_result}")
        if not check_result:
            all_passed = False
    
    await agent.shutdown()
    
    if all_passed:
        print("\n🎉 LSHN NEURAL REPLAY INTEGRATION TEST PASSED!")
        print("   Both DPAD and LSHN networks are functioning correctly")
        print("   Memory consolidation pipeline includes episodic memory processing")
        return True
    else:
        print("\n❌ LSHN NEURAL REPLAY INTEGRATION TEST FAILED!")
        return False


if __name__ == "__main__":
    result = asyncio.run(test_lshn_neural_replay_integration())
    exit(0 if result else 1)
