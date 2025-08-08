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
        self.performance_history: List[float] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the current nonlinearity"""
        nl = self.nonlinearity_type
        if nl == NonlinearityType.RELU:
            return F.relu(x)
        if nl == NonlinearityType.GELU:
            return F.gelu(x)
        if nl == NonlinearityType.SWISH:
            return x * torch.sigmoid(x)
        if nl == NonlinearityType.TANH:
            return torch.tanh(x)
        if nl == NonlinearityType.SIGMOID:
            return torch.sigmoid(x)
        if nl == NonlinearityType.LEAKY_RELU:
            return F.leaky_relu(x)
        if nl == NonlinearityType.ELU:
            return F.elu(x)
        return F.gelu(x)  # Default fallback

    def update_performance(self, loss: float):
        """Track performance for auto-optimization"""
        self.performance_history.append(loss)
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-50:]

# New: Multi-head attention gate for behavior path
class AttentionGate(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        # Ensure divisibility to avoid view errors; fallback to single-head if needed
        if dim % max(1, num_heads) != 0:
            logger.warning(
                "AttentionGate: latent_dim (%d) not divisible by attention_heads (%d); using single head",
                dim, num_heads
            )
            self.num_heads = 1
        self.head_dim = dim // self.num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(0.1)
        self.scale = self.head_dim ** -0.5
        self.last_attention_weights: Optional[torch.Tensor] = None  # stored on CPU

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)  # [B,1,D]
        b, t, _ = x.shape
        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        self.last_attention_weights = attn.detach().cpu()
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, t, self.dim)
        out = self.out_proj(out)
        if out.size(1) == 1:
            out = out.squeeze(1)  # [B,D]
        return out

# New: Behavior prediction path with residual MLP blocks + attention
class BehaviorPredictionPath(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.input_proj = nn.Linear(config.embedding_dim, config.latent_dim)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(config.latent_dim, config.latent_dim) for _ in range(config.hidden_layers)]
        )
        self.nonlinearities = nn.ModuleList(
            [FlexibleNonlinearity(NonlinearityType.GELU) for _ in range(config.hidden_layers)]
        )
        self.attention_gate = AttentionGate(config.latent_dim, config.attention_heads)
        self.behavior_head = nn.Linear(config.latent_dim, config.latent_dim)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = self.dropout(x)
        for layer, nl in zip(self.hidden_layers, self.nonlinearities):
            residual = x
            x = layer(x)
            x = nl(x)
            x = self.dropout(x)
            x = x + residual
        x = self.attention_gate(x)
        return self.behavior_head(x)

# New: Residual prediction path (simple projection + norm + activation)
class ResidualPredictionPath(nn.Module):
    def __init__(self, config):
        super().__init__()
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
        # Paths and mixers
        self.behavior_path = BehaviorPredictionPath(config)
        self.residual_path = ResidualPredictionPath(config)
        self.path_weights = nn.Parameter(torch.tensor([config.behavior_weight, config.residual_weight], dtype=torch.float32))
        self.output_norm = nn.LayerNorm(config.latent_dim)
        self.output_head = nn.Linear(config.latent_dim, config.embedding_dim)

        # Optimizer for replay/dream-state updates
        self.optimizer = optim.Adam(self.parameters(), lr=config.learning_rate)

        # Histories/telemetry
        self.training_step = 0
        self.performance_history: List[float] = []
        self.path_weights_history: List[Any] = []
        self.replay_history: List[Dict[str, Any]] = []
        self.nonlinearity_history: List[Any] = []

        logger.info(f"DPAD Network initialized with {self._count_parameters()} parameters")

    def forward(self, x: torch.Tensor, return_paths: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Compute path outputs
        behavior_output = self.behavior_path(x)
        residual_output = self.residual_path(x)
        # Softmax mix
        weights = F.softmax(self.path_weights, dim=0)
        self.path_weights_history.append(weights.detach().cpu().numpy().tolist())
        combined = weights[0] * behavior_output + weights[1] * residual_output
        combined = self.output_norm(combined)
        output = self.output_head(combined)
        if return_paths:
            return output, {
                'behavior_output': behavior_output,
                'residual_output': residual_output,
                'path_weights': weights.detach(),
                'combined_latent': combined.detach()
            }
        return output
    
    def _attention_weighted_mse(self, pred: torch.Tensor, target: torch.Tensor, attention_scores: Optional[torch.Tensor]) -> Tuple[torch.Tensor, bool]:
        if attention_scores is None:
            return F.mse_loss(pred, target), False
        # Unreduced elementwise squared error
        err = (pred - target) ** 2
        if err.dim() == 2:
            # [B, D] -> per-sample mean
            per_sample = err.mean(dim=-1)  # [B]
            attn = attention_scores
            if attn.dim() > 1:
                attn = attn.view(attn.size(0), -1).mean(dim=-1)
            attn = attn.to(per_sample.dtype).to(per_sample.device)
            loss = (per_sample * attn).mean()
            return loss, True
        elif err.dim() == 3:
            # [B, T, D] -> [B, T]
            per_token = err.mean(dim=-1)
            attn = attention_scores
            if attn.dim() == 2:
                attn = attn
            elif attn.dim() == 1:
                attn = attn.unsqueeze(-1)  # [B,1]
            attn = attn.to(per_token.dtype).to(per_token.device)
            loss = (per_token * attn).mean()
            return loss, True
        else:
            return err.mean(), False

    def compute_loss(
        self,
        input_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute dual-path loss with optional attention weighting.
        Returns total_loss, reconstruction_loss, behavior_loss, residual_loss, path_weights.
        """
        output, path_details = self.forward(input_embeddings, return_paths=True)

        # Reconstruction loss (fixed to support attention weighting)
        reconstruction_loss, _ = self._attention_weighted_mse(output, target_embeddings, attention_scores)

        # Path-specific losses (project to embedding space then compare)
        behavior_target_emb = self.output_head(path_details['behavior_output'])
        residual_target_emb = self.output_head(path_details['residual_output'])
        behavior_loss = F.mse_loss(behavior_target_emb, target_embeddings)
        residual_loss = F.mse_loss(residual_target_emb, target_embeddings)

        total_loss = (
            reconstruction_loss
            + self.config.behavior_weight * behavior_loss
            + self.config.residual_weight * residual_loss
        )

        return {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss,
            'behavior_loss': behavior_loss,
            'residual_loss': residual_loss,
            'path_weights': path_details['path_weights']  # This is correct if path_details is a dict
        }

    def memory_replay(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
        replay_strength: Optional[float] = None
    ) -> Dict[str, Any]:
        """Perform memory replay during dream cycles with importance-weighted updates."""
        self.train()
        strength = replay_strength if replay_strength is not None else self.config.replay_strength
        device = next(self.parameters()).device

        total_loss_val = 0.0
        num_items = 0
        with torch.enable_grad():
            for emb, imp in zip(memory_embeddings, importance_scores):
                if emb is None:
                    continue
                m = emb.to(device)
                # Add noise inversely proportional to importance
                noise = torch.randn_like(m) * (0.1 * (1.0 - float(min(1.0, max(0.0, imp)))))
                noisy = m + noise

                losses = self.compute_loss(noisy, m)
                loss = losses['total_loss'] * (strength * max(0.05, float(imp)))

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                self.optimizer.step()

                total_loss_val += float(losses['total_loss'].detach().cpu().item())
                num_items += 1

        avg_loss = total_loss_val / max(1, num_items)
        self.training_step += 1
        self.performance_history.append(avg_loss)
        result = {
            'items_replayed': num_items,
            'avg_loss': avg_loss,
            'total_loss': total_loss_val,
            'reconstruction_quality': float(np.exp(-avg_loss)),  # heuristic
        }
        self.replay_history.append(result)
        return result

    def get_network_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {
            'training_step': self.training_step,
            'path_weights': self.path_weights.detach().cpu().numpy().tolist(),
            'path_weights_history': self.path_weights_history[-10:],
            'parameter_count': self._count_parameters(),
            'performance_trend': self._calculate_performance_trend(),
            'replay_history': self.replay_history[-10:],
            'nonlinearity_summary': self.get_nonlinearity_summary(),
        }
        attn = self.behavior_path.attention_gate.last_attention_weights
        if attn is not None:
            stats['attention_per_head'] = attn.mean(dim=-1).numpy().tolist()  # [B,H,T]
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

    def optimize_nonlinearity(self, loss_signal: float):
        """Heuristic placeholder to record loss trends for potential nonlinearity tuning."""
        if loss_signal is None:
            return
        for nl in self.behavior_path.nonlinearities:
            nl.update_performance(float(loss_signal))
        self.residual_path.nonlinearity.update_performance(float(loss_signal))

    def reset_network(self):
        self.__init__(self.config)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _calculate_performance_trend(self) -> str:
        if len(self.performance_history) < 10:
            return "insufficient_data"
        recent = float(np.mean(self.performance_history[-5:]))
        older = float(np.mean(self.performance_history[-10:-5]))
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
            self.optimizer, mode="min", patience=10, factor=0.5
        )
        self.training_history: List[Dict[str, float]] = []
        self.best_loss = float("inf")
        
    def train_step(
        self,
        input_embeddings: torch.Tensor,
        target_embeddings: torch.Tensor,
        attention_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        """Single training step"""
        self.network.train()
        self.optimizer.zero_grad()
        loss_dict = self.network.compute_loss(
            input_embeddings, target_embeddings, attention_scores
        )
        loss_dict["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.network.training_step += 1

        # Convert to float dict for logging
        float_losses: Dict[str, Any] = {}
        for k, v in loss_dict.items():
            if k == "path_weights":
                float_losses[k] = v.detach().cpu().numpy().tolist() if isinstance(v, torch.Tensor) else v
            else:
                float_losses[k] = v.item() if isinstance(v, torch.Tensor) else float(v)
        self.training_history.append(float_losses)
        return float_losses
    
    def background_training(
        self,
        memory_batch: List[torch.Tensor],
        importance_scores: List[float],
        steps: int = 5,
    ) -> Dict[str, Any]:
        """Background training during dream cycles"""
        if not memory_batch:
            return {"trained": False, "reason": "empty_batch"}
        
        results = {
            "trained": True,
            "steps_completed": 0,
            "avg_loss": 0.0,
            "memories_processed": len(memory_batch),
        }
        total_loss = 0.0

        # Prepare batch tensors on the correct device
        device = next(self.network.parameters()).device
        batch_embeddings = torch.stack([m.to(device) for m in memory_batch])
        target_embeddings = batch_embeddings.clone()
        attn_scores = torch.tensor(
            importance_scores, dtype=batch_embeddings.dtype, device=device
        )

        for step in range(steps):
            try:
                noise_scale = 0.01
                noisy_input = batch_embeddings + torch.randn_like(batch_embeddings) * noise_scale
                step_losses = self.train_step(
                    noisy_input,
                    target_embeddings,
                    attention_scores=attn_scores,
                )
                total_loss += float(step_losses["total_loss"])
                results["steps_completed"] += 1
            except Exception as e:
                logger.error(f"Error in background training step {step}: {e}")
                break
        
        if results["steps_completed"] > 0:
            results["avg_loss"] = total_loss / results["steps_completed"]
            self.scheduler.step(results["avg_loss"])
            self.network.optimize_nonlinearity(results["avg_loss"])
        
        return results
    
    def save_checkpoint(self, path: str) -> None:
        """Save training checkpoint"""
        checkpoint = {
            "network_state": self.network.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "training_history": self.training_history,
            "config": self.config,
            "training_step": self.network.training_step,
        }
        torch.save(checkpoint, path)
        logger.info(f"DPAD checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """Load training checkpoint"""
        try:
            checkpoint = torch.load(path, map_location="cpu")
            self.network.load_state_dict(checkpoint["network_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            self.training_history = checkpoint.get("training_history", [])
            self.network.training_step = checkpoint.get("training_step", 0)
            logger.info(f"DPAD checkpoint loaded from {path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {path}: {e}")
            logger.info("Continuing with fresh network initialization")
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
            logger.info("Continuing with fresh network initialization")
