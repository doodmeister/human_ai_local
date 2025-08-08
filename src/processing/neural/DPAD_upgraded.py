"""
DPAD (Dual-Path Attention Dynamics) Neural Network Implementation - Deprecated Shim

This module is deprecated and will be removed in future versions.
It re-exports the merged implementation from dpad_network.py for compatibility.
"""

import warnings
from .dpad_network import DPADNetwork, DPADConfig, NonlinearityType

__all__ = ["DPADNetwork", "DPADConfig", "NonlinearityType"]

warnings.warn(
    "DPAD_upgraded.py is deprecated. Use 'from .dpad_network import DPADNetwork' instead.",
    DeprecationWarning,
    stacklevel=2,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Any
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class NonlinearityType(Enum):
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"

@dataclass
class DPADConfig:
    latent_dim: int = 64
    embedding_dim: int = 384
    behavior_weight: float = 1.0
    residual_weight: float = 0.5
    salience_threshold: float = 0.8
    auto_nonlinearity: bool = True
    learning_rate: float = 0.001
    dropout_rate: float = 0.1
    hidden_layers: int = 2
    attention_heads: int = 8
    replay_strength: float = 0.1
    consolidation_rate: float = 0.05

class FlexibleNonlinearity(nn.Module):
    def __init__(self, nonlinearity_type: NonlinearityType = NonlinearityType.GELU):
        super().__init__()
        self.nonlinearity_type = nonlinearity_type
        self.performance_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.nonlinearity_type == NonlinearityType.RELU:
            return F.relu(x)
        elif self.nonlinearity_type == NonlinearityType.GELU:
            return F.gelu(x)
        elif self.nonlinearity_type == NonlinearityType.SWISH:
            return x * torch.sigmoid(x)
        elif self.nonlinearity_type == NonlinearityType.TANH:
            return torch.tanh(x)
        elif self.nonlinearity_type == NonlinearityType.SIGMOID:
            return torch.sigmoid(x)
        elif self.nonlinearity_type == NonlinearityType.LEAKY_RELU:
            return F.leaky_relu(x)
        elif self.nonlinearity_type == NonlinearityType.ELU:
            return F.elu(x)
        else:
            return F.gelu(x)
    
    def update_performance(self, loss: float):
        self.performance_history.append(loss)
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]

class AttentionGate(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5

        self.last_attention_weights = None
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        self.last_attention_weights = attn_weights.detach().cpu()
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        
        return self.out_proj(attn_output)

class BehaviorPredictionPath(nn.Module):
    def __init__(self, config: DPADConfig):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.embedding_dim, config.latent_dim)
        self.hidden_layers = nn.ModuleList()
        self.nonlinearities = nn.ModuleList()
        
        for _ in range(config.hidden_layers):
            self.hidden_layers.append(nn.Linear(config.latent_dim, config.latent_dim))
            self.nonlinearities.append(FlexibleNonlinearity(NonlinearityType.GELU))
        
        self.attention_gate = AttentionGate(config.latent_dim, config.attention_heads)
        self.behavior_head = nn.Linear(config.latent_dim, config.latent_dim)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.dropout(x)
        
        for layer, nonlinearity in zip(self.hidden_layers, self.nonlinearities):
            residual = x
            x = layer(x)
            x = nonlinearity(x)
            x = self.dropout(x)
            x = x + residual
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.attention_gate(x)
        
        if x.size(1) == 1:
            x = x.squeeze(1)
        
        return self.behavior_head(x)

class ResidualPredictionPath(nn.Module):
    def __init__(self, config: DPADConfig):
        super().__init__()
        self.config = config
        self.residual_proj = nn.Linear(config.embedding_dim, config.latent_dim)
        self.residual_norm = nn.LayerNorm(config.latent_dim)
        self.residual_head = nn.Linear(config.latent_dim, config.latent_dim)
        self.nonlinearity = FlexibleNonlinearity(NonlinearityType.RELU)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.residual_proj(x)
        x = self.residual_norm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        return self.residual_head(x)

class DPADNetwork(nn.Module):
    def __init__(self, config: DPADConfig):
        super().__init__()
        self.config = config
        self.behavior_path = BehaviorPredictionPath(config)
        self.residual_path = ResidualPredictionPath(config)
        self.path_weights = nn.Parameter(torch.tensor([config.behavior_weight, config.residual_weight]))
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_head = nn.Linear(config.latent_dim, config.embedding_dim)
        
        self.training_step = 0
        self.performance_history = []
        self.path_weights_history = []
        self.replay_history = []
        self.nonlinearity_history = []
        
        logger.info(f"DPAD Network initialized with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor, return_paths: bool = False):
        behavior_output = self.behavior_path(x)
        residual_output = self.residual_path(x)
        weights = F.softmax(self.path_weights, dim=0)
        self.path_weights_history.append(weights.detach().cpu().numpy().tolist())
        
        combined = (weights[0] * behavior_output + weights[1] * residual_output)
        combined = self.output_norm(combined)
        output = self.output_head(combined)
        
        if return_paths:
            return output, {
                'behavior_output': behavior_output,
                'residual_output': residual_output,
                'path_weights': weights,
                'combined_latent': combined
            }
        return output

    def get_network_stats(self) -> Dict[str, Any]:
        stats = {
            'training_step': self.training_step,
            'path_weights': self.path_weights.detach().cpu().numpy().tolist(),
            'path_weights_history': self.path_weights_history[-10:],
            'parameter_count': self._count_parameters(),
            'performance_trend': self._calculate_performance_trend(),
            'replay_history': self.replay_history[-10:],
            'nonlinearity_summary': self.get_nonlinearity_summary(),
        }
        if self.behavior_path.attention_gate.last_attention_weights is not None:
            attn = self.behavior_path.attention_gate.last_attention_weights
            stats['attention_per_head'] = attn.mean(dim=-1).numpy().tolist()
            sparsity = (attn < 0.01).float().mean().item()
            stats['attention_sparsity'] = sparsity
        return stats

    def get_nonlinearity_summary(self) -> Dict[str, Any]:
        return {
            'behavior_path': [nl.nonlinearity_type.value for nl in self.behavior_path.nonlinearities],
            'residual_path': self.residual_path.nonlinearity.nonlinearity_type.value
        }

    def set_nonlinearity(self, path: str, layer_index: int, nonlinearity_type: NonlinearityType):
        if path == 'behavior':
            self.behavior_path.nonlinearities[layer_index].nonlinearity_type = nonlinearity_type
        elif path == 'residual':
            self.residual_path.nonlinearity.nonlinearity_type = nonlinearity_type

    def consolidate_replay_results(self, replay_result: Dict[str, Any]):
        self.replay_history.append(replay_result)
        if replay_result.get('reconstruction_quality', 1.0) < 0.8:
            self.optimize_nonlinearity(replay_result.get('total_loss', 0.0))

    def reset_network(self):
        self.__init__(self.config)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _calculate_performance_trend(self) -> str:
        if len(self.performance_history) < 10:
            return "insufficient_data"
        recent = np.mean(self.performance_history[-5:])
        older = np.mean(self.performance_history[-10:-5])
        if recent < older * 0.95:
            return "improving"
        elif recent > older * 1.05:
            return "degrading"
        else:
            return "stable"
