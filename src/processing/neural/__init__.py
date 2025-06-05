"""
Neural Processing Module

This module implements neural networks for biologically-inspired cognitive processing:
- DPAD (Dual-Path Attention Dynamics) neural network architecture
- LSHN (Latent Structured Hopfield Networks) for episodic memory

Key Components:
- DPADNetwork: Core neural network with dual-path processing
- LSHNNetwork: Hopfield-based episodic memory network
- NeuralIntegrationManager: Integration with cognitive architecture
"""

from .dpad_network import (
    DPADNetwork,
    DPADTrainer,
    DPADConfig,
    NonlinearityType,
    FlexibleNonlinearity,
    AttentionGate,
    BehaviorPredictionPath,
    ResidualPredictionPath
)

from .lshn_network import (
    LSHNNetwork,
    LSHNTrainer,
    LSHNConfig,
    EpisodicMemoryEncoder,
    HopfieldAssociativeMemory
)

from .neural_integration import (
    NeuralIntegrationManager,
    NeuralIntegrationConfig
)

__all__ = [
    'DPADNetwork',
    'DPADTrainer', 
    'DPADConfig',
    'NonlinearityType',
    'FlexibleNonlinearity',
    'AttentionGate',
    'BehaviorPredictionPath',
    'ResidualPredictionPath',
    'LSHNNetwork',
    'LSHNTrainer',
    'LSHNConfig', 
    'EpisodicMemoryEncoder',
    'HopfieldAssociativeMemory',
    'NeuralIntegrationManager',
    'NeuralIntegrationConfig'
]