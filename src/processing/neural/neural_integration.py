"""
Neural Integration Manager

This module manages the integration of DPAD neural networks with the cognitive architecture:
- Integration with attention mechanisms
- Memory-guided neural training
- Dream cycle neural replay
- Performance monitoring and optimization
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple
import torch
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from threading import Lock

from .dpad_network import DPADNetwork, DPADTrainer, DPADConfig
from ...core.config import CognitiveConfig

logger = logging.getLogger(__name__)

@dataclass
class NeuralIntegrationConfig:
    """Configuration for neural integration"""
    enable_neural_replay: bool = True
    enable_background_training: bool = True
    replay_memory_threshold: int = 5  # Minimum memories for replay
    training_batch_size: int = 16
    dream_training_steps: int = 10
    checkpoint_interval: int = 100  # Steps between checkpoints
    performance_monitoring: bool = True
    adaptive_learning_rate: bool = True
    consolidation_strength_threshold: float = 0.3

class NeuralIntegrationManager:
    """
    Manages integration between DPAD neural networks and cognitive architecture
    
    Handles:
    - Neural network initialization and configuration
    - Integration with attention mechanisms
    - Memory-guided training during dream cycles
    - Performance monitoring and optimization
    - Checkpoint management
    """
    
    def __init__(
        self,
        cognitive_config: CognitiveConfig,
        neural_config: Optional[NeuralIntegrationConfig] = None,
        model_save_path: Optional[str] = None
    ):
        self.cognitive_config = cognitive_config
        self.neural_config = neural_config or NeuralIntegrationConfig()
        self.model_save_path = model_save_path or "./models/dpad"
        
        # Initialize DPAD configuration from cognitive config
        self.dpad_config = self._create_dpad_config()
        
        # Initialize network and trainer
        self.network = DPADNetwork(self.dpad_config)
        self.trainer = DPADTrainer(self.network, self.dpad_config)
        
        # State management
        self.is_training = False
        self.training_lock = Lock()
        self.performance_metrics = {
            'total_training_steps': 0,
            'total_replays': 0,
            'avg_reconstruction_quality': 0.0,
            'consolidation_events': 0,
            'last_checkpoint': None
        }
        
        # Ensure model directory exists
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoint if available
        self._load_latest_checkpoint()
        
        logger.info("Neural Integration Manager initialized")
    
    def _create_dpad_config(self) -> DPADConfig:
        """Create DPAD configuration from cognitive config"""
        return DPADConfig(
            latent_dim=getattr(self.cognitive_config.processing, 'dpad_latent_dim', 64),
            embedding_dim=self.cognitive_config.processing.embedding_dimension,
            behavior_weight=getattr(self.cognitive_config.processing, 'dpad_behavior_weight', 1.0),
            residual_weight=getattr(self.cognitive_config.processing, 'dpad_residual_weight', 0.5),
            salience_threshold=getattr(self.cognitive_config.attention, 'salience_threshold', 0.8),
            auto_nonlinearity=getattr(self.cognitive_config.processing, 'dpad_auto_nonlinearity', True),
            learning_rate=getattr(self.cognitive_config.processing, 'dpad_learning_rate', 0.001),
            dropout_rate=getattr(self.cognitive_config.processing, 'dpad_dropout_rate', 0.1),
            replay_strength=getattr(self.cognitive_config.processing, 'dpad_replay_strength', 0.1),
            consolidation_rate=getattr(self.cognitive_config.processing, 'dpad_consolidation_rate', 0.05)
        )
    
    async def process_attention_update(
        self,
        embeddings: torch.Tensor,
        attention_scores: torch.Tensor,
        salience_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Process attention updates through the neural network
        
        Args:
            embeddings: Input embeddings from attention mechanism
            attention_scores: Attention allocation scores
            salience_scores: Optional salience scores
        
        Returns:
            Processing results and neural predictions
        """
        try:
            self.network.eval()
            
            with torch.no_grad():
                # Forward pass through network
                output, path_details = self.network.forward(embeddings, return_paths=True)
                
                # Calculate attention-guided predictions
                attention_weighted_output = output * attention_scores.unsqueeze(-1)
                
                # Compute novelty detection (difference from expected)
                novelty_scores = torch.norm(output - embeddings, dim=-1)
                
                results = {
                    'neural_output': output,
                    'attention_weighted_output': attention_weighted_output,
                    'novelty_scores': novelty_scores,
                    'path_weights': path_details['path_weights'],
                    'processing_quality': self._calculate_processing_quality(embeddings, output)
                }
                
                return results
                
        except Exception as e:
            logger.error(f"Error in attention processing: {e}")
            return {'error': str(e)}
    
    async def neural_memory_replay(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
        attention_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform neural replay of memories during dream cycles
        
        Args:
            memory_embeddings: Memory embeddings to replay
            importance_scores: Importance scores for each memory
            attention_context: Optional attention context
        
        Returns:
            Replay results and consolidation effects
        """
        if not self.neural_config.enable_neural_replay:
            return {'status': 'disabled', 'replayed_memories': 0}
        
        if len(memory_embeddings) < self.neural_config.replay_memory_threshold:
            return {
                'status': 'insufficient_memories',
                'replayed_memories': len(memory_embeddings),
                'threshold': self.neural_config.replay_memory_threshold
            }
        
        try:
            # Perform memory replay
            replay_results = self.network.memory_replay(
                memory_embeddings,
                importance_scores,
                replay_strength=self.dpad_config.replay_strength
            )
            
            # Update performance metrics
            self.performance_metrics['total_replays'] += 1
            self.performance_metrics['avg_reconstruction_quality'] = (
                self.performance_metrics['avg_reconstruction_quality'] * 0.9 +
                replay_results['reconstruction_quality'] * 0.1
            )
            
            # Check for consolidation threshold
            if replay_results['consolidation_strength'] > self.neural_config.consolidation_strength_threshold:
                self.performance_metrics['consolidation_events'] += 1
                
                # Trigger background training if enabled
                if self.neural_config.enable_background_training:
                    training_results = await self._background_training_step(
                        memory_embeddings, importance_scores
                    )
                    replay_results.update(training_results)
            
            return replay_results
            
        except Exception as e:
            logger.error(f"Error in neural memory replay: {e}")
            return {'error': str(e), 'replayed_memories': 0}
    
    async def _background_training_step(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Perform background training step during dream cycles
        
        Args:
            memory_embeddings: Memory embeddings for training
            importance_scores: Importance scores for weighting
        
        Returns:
            Training step results
        """
        if not self.neural_config.enable_background_training:
            return {'background_training': 'disabled'}
        
        # Acquire training lock to prevent concurrent training
        if not self.training_lock.acquire(blocking=False):
            return {'background_training': 'busy'}
        
        try:
            self.is_training = True
            
            # Perform background training
            training_results = self.trainer.background_training(
                memory_embeddings,
                importance_scores,
                steps=self.neural_config.dream_training_steps
            )
            
            # Update metrics
            if training_results['trained']:
                self.performance_metrics['total_training_steps'] += training_results['steps_completed']
                
                # Check for checkpoint
                if (self.performance_metrics['total_training_steps'] % 
                    self.neural_config.checkpoint_interval == 0):
                    await self._save_checkpoint()
            
            return {'background_training': training_results}
            
        except Exception as e:
            logger.error(f"Error in background training: {e}")
            return {'background_training': {'error': str(e)}}
        
        finally:
            self.is_training = False
            self.training_lock.release()
    
    def calculate_attention_enhancement(
        self,
        base_attention_scores: torch.Tensor,
        neural_predictions: torch.Tensor,
        novelty_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate attention enhancement based on neural network predictions
        
        Args:
            base_attention_scores: Base attention scores from attention mechanism
            neural_predictions: Neural network predictions
            novelty_scores: Novelty detection scores
        
        Returns:
            Enhanced attention scores
        """
        try:
            # Normalize novelty scores
            novelty_normalized = torch.sigmoid(novelty_scores - novelty_scores.mean())
            
            # Calculate prediction confidence
            prediction_confidence = 1.0 / (1.0 + torch.norm(neural_predictions, dim=-1))
            
            # Enhance attention based on novelty and uncertainty
            novelty_boost = novelty_normalized * 0.2  # Up to 20% boost for novel items
            uncertainty_boost = (1.0 - prediction_confidence) * 0.1  # Up to 10% boost for uncertain items
            
            enhanced_scores = base_attention_scores + novelty_boost + uncertainty_boost
            
            # Normalize to maintain total attention budget
            enhanced_scores = enhanced_scores / enhanced_scores.sum() * base_attention_scores.sum()
            
            return enhanced_scores
            
        except Exception as e:
            logger.error(f"Error in attention enhancement: {e}")
            return base_attention_scores  # Return original scores on error
    
    def get_neural_status(self) -> Dict[str, Any]:
        """Get current neural integration status"""
        network_stats = self.network.get_training_stats()
        
        return {
            'network_status': {
                'parameters': network_stats['parameter_count'],
                'training_step': network_stats['training_step'],
                'path_weights': network_stats['path_weights'],
                'performance_trend': network_stats['performance_trend']
            },
            'integration_status': {
                'is_training': self.is_training,
                'neural_replay_enabled': self.neural_config.enable_neural_replay,
                'background_training_enabled': self.neural_config.enable_background_training
            },
            'performance_metrics': self.performance_metrics.copy(),
            'config': {
                'dpad_config': asdict(self.dpad_config),
                'neural_config': asdict(self.neural_config)
            }
        }
    
    def _calculate_processing_quality(
        self,
        input_embeddings: torch.Tensor,
        output_embeddings: torch.Tensor
    ) -> float:
        """Calculate processing quality metric"""
        try:
            # Calculate cosine similarity between input and output
            cosine_sim = F.cosine_similarity(input_embeddings, output_embeddings, dim=-1)
            return cosine_sim.mean().item()
        except:
            return 0.0
    
    async def _save_checkpoint(self):
        """Save neural network checkpoint"""
        try:
            timestamp = int(time.time())
            checkpoint_path = f"{self.model_save_path}/dpad_checkpoint_{timestamp}.pt"
            
            self.trainer.save_checkpoint(checkpoint_path)
            
            # Save integration state
            integration_state = {
                'performance_metrics': self.performance_metrics,
                'dpad_config': asdict(self.dpad_config),
                'neural_config': asdict(self.neural_config),
                'timestamp': timestamp
            }
            
            state_path = f"{self.model_save_path}/integration_state_{timestamp}.json"
            with open(state_path, 'w') as f:
                json.dump(integration_state, f, indent=2)
            
            self.performance_metrics['last_checkpoint'] = checkpoint_path
            logger.info(f"Neural checkpoint saved: {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _load_latest_checkpoint(self):
        """Load the latest checkpoint if available"""
        try:
            model_dir = Path(self.model_save_path)
            if not model_dir.exists():
                return
            
            # Find latest checkpoint
            checkpoints = list(model_dir.glob("dpad_checkpoint_*.pt"))
            if not checkpoints:
                return
            
            latest_checkpoint = max(checkpoints, key=lambda p: p.stat().st_mtime)
            
            # Load checkpoint
            self.trainer.load_checkpoint(str(latest_checkpoint))
            
            # Load integration state if available
            timestamp = latest_checkpoint.stem.split('_')[-1]
            state_path = model_dir / f"integration_state_{timestamp}.json"
            
            if state_path.exists():
                with open(state_path, 'r') as f:
                    state = json.load(f)
                    self.performance_metrics.update(state['performance_metrics'])
            
            logger.info(f"Loaded neural checkpoint: {latest_checkpoint}")
            
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    
    async def optimize_performance(self):
        """Optimize neural network performance based on recent metrics"""
        try:
            # Check performance trends
            if len(self.trainer.training_history) < 10:
                return
            
            recent_losses = [h['total_loss'] for h in self.trainer.training_history[-10:]]
            avg_recent_loss = np.mean(recent_losses)
            
            # Trigger nonlinearity optimization
            self.network.optimize_nonlinearity(avg_recent_loss)
            
            # Adaptive learning rate adjustment
            if self.neural_config.adaptive_learning_rate:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                
                # Check if loss is stagnating
                if len(recent_losses) >= 5:
                    loss_variance = np.var(recent_losses[-5:])
                    if loss_variance < 1e-6:  # Very low variance indicates stagnation
                        new_lr = current_lr * 0.9  # Reduce learning rate
                        for param_group in self.trainer.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.info(f"Reduced learning rate to {new_lr}")
            
        except Exception as e:
            logger.error(f"Error in performance optimization: {e}")
    
    def shutdown(self):
        """Shutdown neural integration manager"""
        logger.info("Shutting down Neural Integration Manager...")
        
        # Save final checkpoint
        asyncio.create_task(self._save_checkpoint())
        
        # Wait for any ongoing training to complete
        if self.training_lock.locked():
            logger.info("Waiting for training to complete...")
            time.sleep(1.0)
        
        logger.info("Neural Integration Manager shutdown complete")

# Import torch.nn.functional for cosine similarity
import torch.nn.functional as F
