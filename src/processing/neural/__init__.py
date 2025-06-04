"""
Neural Processing Module

This module implements the DPAD (Dual-Path Attention Dynamics) neural network
architecture for biologically-inspired cognitive processing.

Key Components:
- DPADNetwork: Core neural network with dual-path processing
- DPADTrainer: Training manager for the network
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
    'NeuralIntegrationManager',
    'NeuralIntegrationConfig'
]