"""
DPAD (Dual-Path Attention Dynamics) Neural Network Implementation

This module implements a biologically-inspired neural network architecture that supports:
- Dual-path processing with behavior and residual prediction
- Flexible nonlinearity selection and optimization
- Background training during dream cycles
- Integration with attention mechanisms
- Hippocampal-style memory replay
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class NonlinearityType(Enum):
    """Supported nonlinearity types for flexible architecture"""
    RELU = "relu"
    GELU = "gelu"
    SWISH = "swish"
    TANH = "tanh"
    SIGMOID = "sigmoid"
    LEAKY_RELU = "leaky_relu"
    ELU = "elu"

@dataclass
class DPADConfig:
    """Configuration for DPAD neural network"""
    latent_dim: int = 64
    embedding_dim: int = 384  # Matches sentence transformer
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
    """Adaptive nonlinearity that can switch between different activation functions"""
    
    def __init__(self, nonlinearity_type: NonlinearityType = NonlinearityType.GELU):
        super().__init__()
        self.nonlinearity_type = nonlinearity_type
        self.performance_history = []
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the current nonlinearity"""
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
            return F.gelu(x)  # Default fallback
    
    def update_performance(self, loss: float):
        """Track performance for auto-optimization"""
        self.performance_history.append(loss)
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]

class AttentionGate(nn.Module):
    """Attention-based gating mechanism for dual-path processing"""
    
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
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Multi-head attention forward pass"""
        batch_size, seq_len, _ = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores.masked_fill_(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.dim
        )
        
        return self.out_proj(attn_output)

class BehaviorPredictionPath(nn.Module):
    """Behavior prediction pathway of DPAD network"""
    
    def __init__(self, config: DPADConfig):
        super().__init__()
        self.config = config
        
        # Input projection
        self.input_proj = nn.Linear(config.embedding_dim, config.latent_dim)
        
        # Hidden layers with flexible nonlinearity
        self.hidden_layers = nn.ModuleList()
        self.nonlinearities = nn.ModuleList()
        
        for i in range(config.hidden_layers):
            self.hidden_layers.append(
                nn.Linear(config.latent_dim, config.latent_dim)
            )
            self.nonlinearities.append(
                FlexibleNonlinearity(NonlinearityType.GELU)
            )
        
        # Attention gate
        self.attention_gate = AttentionGate(config.latent_dim, config.attention_heads)
        
        # Output projection for behavior prediction
        self.behavior_head = nn.Linear(config.latent_dim, config.latent_dim)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for behavior prediction"""
        # Input projection
        x = self.input_proj(x)
        x = self.dropout(x)
        
        # Hidden processing
        for layer, nonlinearity in zip(self.hidden_layers, self.nonlinearities):
            residual = x
            x = layer(x)
            x = nonlinearity(x)
            x = self.dropout(x)
            x = x + residual  # Residual connection
        
        # Add sequence dimension if needed for attention
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, dim)
        
        # Attention processing
        x = self.attention_gate(x)
        
        # Remove sequence dimension if added
        if x.size(1) == 1:
            x = x.squeeze(1)
        
        # Behavior prediction
        behavior_pred = self.behavior_head(x)
        
        return behavior_pred

class ResidualPredictionPath(nn.Module):
    """Residual prediction pathway of DPAD network"""
    
    def __init__(self, config: DPADConfig):
        super().__init__()
        self.config = config
        
        # Simplified pathway for residual prediction
        self.residual_proj = nn.Linear(config.embedding_dim, config.latent_dim)
        self.residual_norm = nn.LayerNorm(config.latent_dim)
        self.residual_head = nn.Linear(config.latent_dim, config.latent_dim)
        
        # Nonlinearity
        self.nonlinearity = FlexibleNonlinearity(NonlinearityType.RELU)
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for residual prediction"""
        x = self.residual_proj(x)
        x = self.residual_norm(x)
        x = self.nonlinearity(x)
        x = self.dropout(x)
        x = self.residual_head(x)
        
        return x

class DPADNetwork(nn.Module):
    """
    Dual-Path Attention Dynamics Network
    
    Implements biologically-inspired dual-path processing with:
    - Behavior prediction pathway (complex processing)
    - Residual prediction pathway (simple baseline)
    - Attention-based gating
    - Flexible nonlinearity optimization
    """
    
    def __init__(self, config: DPADConfig):
        super().__init__()
        self.config = config
        
        # Dual paths
        self.behavior_path = BehaviorPredictionPath(config)
        self.residual_path = ResidualPredictionPath(config)
        
        # Combination weights
        self.path_weights = nn.Parameter(
            torch.tensor([config.behavior_weight, config.residual_weight])
        )
        
        # Output processing
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_head = nn.Linear(config.latent_dim, config.embedding_dim)
        
        # Training state
        self.training_step = 0
        self.performance_history = []
        
        logger.info(f"DPAD Network initialized with {self._count_parameters()} parameters")
    
    def forward(self, x: torch.Tensor, return_paths: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass through dual paths
        
        Args:
            x: Input embeddings (batch_size, embedding_dim)
            return_paths: Whether to return individual path outputs
        
        Returns:
            Combined output or (output, path_details)
        """
        # Process through both paths
        behavior_output = self.behavior_path(x)
        residual_output = self.residual_path(x)
        
        # Normalize weights
        weights = F.softmax(self.path_weights, dim=0)
        
        # Combine paths
        combined = (weights[0] * behavior_output + 
                   weights[1] * residual_output)
        
        # Final processing
        combined = self.output_norm(combined)
        output = self.output_head(combined)
        
        if return_paths:
            path_details = {
                'behavior_output': behavior_output,
                'residual_output': residual_output,
                'path_weights': weights,
                'combined_latent': combined
            }
            return output, path_details
        
        return output
    
    def compute_loss(
        self,
        input_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dual-path loss with attention weighting
        
        Args:
            input_embeddings: Input memory embeddings
            target_embeddings: Target reconstructed embeddings
            attention_scores: Optional attention weights for loss weighting
        
        Returns:
            Dictionary of loss components
        """
        # Forward pass with path details
        output, path_details = self.forward(input_embeddings, return_paths=True)
        
        # Reconstruction loss
        reconstruction_loss = F.mse_loss(output, target_embeddings)        # Path-specific losses
        # Project path outputs to embedding space for comparison
        behavior_target_emb = self.output_head(path_details['behavior_output'])
        residual_target_emb = self.output_head(path_details['residual_output'])
        
        behavior_loss = F.mse_loss(behavior_target_emb, target_embeddings)
        residual_loss = F.mse_loss(residual_target_emb, target_embeddings)
        
        # Attention-weighted loss if provided
        if attention_scores is not None:
            attention_weights = attention_scores.unsqueeze(-1)
            reconstruction_loss = (reconstruction_loss * attention_weights).mean()
        
        # Combined loss
        total_loss = (reconstruction_loss + 
                     self.config.behavior_weight * behavior_loss +
                     self.config.residual_weight * residual_loss)
        
        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'behavior_loss': behavior_loss,
            'residual_loss': residual_loss,
            'path_weights': path_details['path_weights']
        }
    
    def memory_replay(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
        replay_strength: float = None
    ) -> Dict[str, Any]:
        """
        Perform memory replay during dream cycles
        
        Args:
            memory_embeddings: List of memory embeddings to replay
            importance_scores: Importance scores for each memory
            replay_strength: Strength of replay (default from config)
        
        Returns:
            Replay results and statistics
        """
        if not memory_embeddings:
            return {
                'replayed_memories': 0,
                'total_loss': 0.0,
                'consolidation_strength': 0.0
            }
        
        if replay_strength is None:
            replay_strength = self.config.replay_strength
        
        # Stack embeddings
        embeddings = torch.stack(memory_embeddings)
        importance_weights = torch.tensor(importance_scores, dtype=torch.float32)
        
        # Forward pass for replay
        reconstructed = self.forward(embeddings)
        
        # Compute replay loss with importance weighting
        replay_loss = F.mse_loss(reconstructed, embeddings, reduction='none')
        replay_loss = (replay_loss * importance_weights.unsqueeze(-1)).mean()
        
        # Consolidation effect (strengthen important memories)
        consolidation_strength = (importance_weights * replay_strength).sum().item()
        
        return {
            'replayed_memories': len(memory_embeddings),
            'total_loss': replay_loss.item(),
            'consolidation_strength': consolidation_strength,
            'reconstruction_quality': 1.0 / (1.0 + replay_loss.item())
        }
    
    def optimize_nonlinearity(self, validation_loss: float):
        """
        Optimize nonlinearity selection based on performance
        
        Args:
            validation_loss: Current validation loss
        """
        if not self.config.auto_nonlinearity:
            return
        
        self.performance_history.append(validation_loss)
        
        # Only optimize after sufficient data
        if len(self.performance_history) < 20:
            return
        
        # Check if current performance is degrading
        recent_avg = np.mean(self.performance_history[-10:])
        older_avg = np.mean(self.performance_history[-20:-10])
        
        if recent_avg > older_avg * 1.1:  # 10% performance degradation
            # Try different nonlinearity
            current_types = [NonlinearityType.GELU, NonlinearityType.RELU, 
                           NonlinearityType.SWISH, NonlinearityType.ELU]
            
            # Cycle through different types
            for path in [self.behavior_path, self.residual_path]:
                for i, nonlinearity in enumerate(path.nonlinearities):
                    current_idx = current_types.index(nonlinearity.nonlinearity_type)
                    new_idx = (current_idx + 1) % len(current_types)
                    nonlinearity.nonlinearity_type = current_types[new_idx]
            
            logger.info("DPAD: Switched nonlinearity due to performance degradation")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get current training statistics"""
        return {
            'training_step': self.training_step,
            'path_weights': self.path_weights.detach().numpy().tolist(),
            'parameter_count': self._count_parameters(),
            'performance_trend': self._calculate_performance_trend()
        }
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def _calculate_performance_trend(self) -> str:
        """Calculate performance trend from history"""
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

class DPADTrainer:
    """Training manager for DPAD network"""
    
    def __init__(self, network: DPADNetwork, config: DPADConfig):
        self.network = network
        self.config = config
        self.optimizer = optim.Adam(network.parameters(), lr=config.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.training_history = []
        self.best_loss = float('inf')
        
    def train_step(
        self,
        input_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            input_embeddings: Input memory embeddings
            target_embeddings: Target embeddings for reconstruction
            attention_scores: Optional attention weights
        
        Returns:
            Loss components and metrics
        """
        self.network.train()
        self.optimizer.zero_grad()
        
        # Compute losses
        loss_dict = self.network.compute_loss(
            input_embeddings, target_embeddings, attention_scores
        )
        
        # Backward pass
        loss_dict['total_loss'].backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        
        # Optimizer step
        self.optimizer.step()
        
        # Update training step
        self.network.training_step += 1
          # Convert to float dict for logging
        float_losses = {}
        for k, v in loss_dict.items():
            if k == 'path_weights':
                float_losses[k] = v.detach().cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
            else:
                float_losses[k] = v.item() if isinstance(v, torch.Tensor) else v
        
        self.training_history.append(float_losses)
        
        return float_losses
    
    def background_training(
        self,
        memory_batch: List[torch.Tensor],
        importance_scores: List[float],
        steps: int = 5
    ) -> Dict[str, Any]:
        """
        Background training during dream cycles
        
        Args:
            memory_batch: Batch of memory embeddings
            importance_scores: Importance scores for memories
            steps: Number of training steps
        
        Returns:
            Training results
        """
        if not memory_batch:
            return {'trained': False, 'reason': 'empty_batch'}
        
        results = {
            'trained': True,
            'steps_completed': 0,
            'avg_loss': 0.0,
            'memories_processed': len(memory_batch)
        }
        
        total_loss = 0.0
        
        for step in range(steps):
            try:
                # Sample memories with importance weighting
                batch_embeddings = torch.stack(memory_batch)
                target_embeddings = batch_embeddings.clone()  # Autoencoder-style
                
                # Optional: Add slight noise for robustness
                noise_scale = 0.01
                noisy_input = batch_embeddings + torch.randn_like(batch_embeddings) * noise_scale
                
                # Training step
                step_losses = self.train_step(
                    noisy_input,
                    target_embeddings,
                    attention_scores=torch.tensor(importance_scores)
                )
                
                total_loss += step_losses['total_loss']
                results['steps_completed'] += 1
                
            except Exception as e:
                logger.error(f"Error in background training step {step}: {e}")
                break
        
        if results['steps_completed'] > 0:
            results['avg_loss'] = total_loss / results['steps_completed']
            
            # Update learning rate based on performance
            self.scheduler.step(results['avg_loss'])
            
            # Check for nonlinearity optimization
            self.network.optimize_nonlinearity(results['avg_loss'])
        
        return results
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint"""
        checkpoint = {
            'network_state': self.network.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'config': self.config,
            'training_step': self.network.training_step        }
        torch.save(checkpoint, path)
        logger.info(f"DPAD checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            self.network.load_state_dict(checkpoint['network_state'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.training_history = checkpoint['training_history']
            self.network.training_step = checkpoint['training_step']
            logger.info(f"DPAD checkpoint loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
            logger.info("Continuing with fresh network initialization")
