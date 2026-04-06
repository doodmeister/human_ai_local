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
from typing import Any, Callable, Dict, List, Optional
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, asdict
from pathlib import Path
import json
import time
from threading import Lock

from .dpad_network import DPADNetwork, DPADTrainer, DPADConfig
from .lshn_network import LSHNNetwork, LSHNTrainer, LSHNConfig
from src.core.config import CognitiveConfig
from src.optimization import PerformanceConfig, create_performance_optimizer

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
        self._missing_config_fields: set[str] = set()
        
        # Initialize DPAD configuration from cognitive config
        self.dpad_config = self._create_dpad_config()
        self.lshn_config = self._create_lshn_config()
        self._log_missing_config_fields()

        # Initialize network and trainer
        self.network = DPADNetwork(self.dpad_config)
        self.trainer = DPADTrainer(self.network, self.dpad_config)

        # Initialize LSHN network and trainer
        self.lshn_network = LSHNNetwork(self.lshn_config)
        self.lshn_trainer = LSHNTrainer(self.lshn_network, self.lshn_config)
        
        # Initialize performance optimization
        perf_config = PerformanceConfig(
            enable_batch_optimization=True,
            enable_memory_pooling=True,
            enable_mixed_precision=True,
            enable_parallel_processing=True,
            max_batch_size=neural_config.training_batch_size if neural_config else 16
        )
        self.performance_optimizer = create_performance_optimizer(perf_config)
        
        # State management
        self.is_training = False
        self.training_lock = Lock()
        self.performance_metrics = self._default_performance_metrics()
        
        # Ensure model directory exists
        Path(self.model_save_path).mkdir(parents=True, exist_ok=True)
        
        # Load existing checkpoint if available
        self._load_latest_checkpoint()
        
        logger.info("Neural Integration Manager initialized")

    def _ensure_missing_config_fields(self) -> None:
        """Initialize the missing-config tracker for direct __new__ test paths."""
        if not hasattr(self, '_missing_config_fields'):
            self._missing_config_fields = set()

    def _get_config_value(self, section_name: str, attr_name: str, default: Any) -> Any:
        """Fetch a config value and record when a default is being used."""
        section = getattr(self.cognitive_config, section_name, None)
        if section is None or not hasattr(section, attr_name):
            self._missing_config_fields.add(f"{section_name}.{attr_name}")
            return default
        return getattr(section, attr_name)

    def _log_missing_config_fields(self) -> None:
        """Summarize missing config attributes instead of silently falling back."""
        if not self._missing_config_fields:
            return
        logger.warning(
            "Using default neural config values for missing cognitive config fields: %s",
            ", ".join(sorted(self._missing_config_fields)),
        )

    def _default_performance_metrics(self) -> Dict[str, Any]:
        """Create the canonical performance metric payload."""
        return {
            'total_training_steps': 0,
            'total_replays': 0,
            'avg_reconstruction_quality': 0.0,
            'reconstruction_quality_samples': 0,
            'consolidation_events': 0,
            'last_checkpoint': None,
        }

    def _sanitize_performance_metrics(self, metrics: Any) -> Dict[str, Any]:
        """Merge checkpoint metrics against the current schema."""
        sanitized = self._default_performance_metrics()
        if not isinstance(metrics, dict):
            return sanitized

        int_keys = (
            'total_training_steps',
            'total_replays',
            'reconstruction_quality_samples',
            'consolidation_events',
        )
        for key in int_keys:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                sanitized[key] = int(value)

        avg_quality = metrics.get('avg_reconstruction_quality')
        if isinstance(avg_quality, (int, float)):
            sanitized['avg_reconstruction_quality'] = float(avg_quality)

        last_checkpoint = metrics.get('last_checkpoint')
        if isinstance(last_checkpoint, str) or last_checkpoint is None:
            sanitized['last_checkpoint'] = last_checkpoint

        if (
            sanitized['reconstruction_quality_samples'] == 0
            and sanitized['avg_reconstruction_quality'] > 0.0
        ):
            sanitized['reconstruction_quality_samples'] = 1

        return sanitized
    
    def _create_dpad_config(self) -> DPADConfig:
        """Create DPAD configuration from cognitive config"""
        self._ensure_missing_config_fields()
        return DPADConfig(
            latent_dim=self._get_config_value('processing', 'dpad_latent_dim', 64),
            embedding_dim=self._get_config_value('processing', 'embedding_dimension', 384),
            behavior_weight=self._get_config_value('processing', 'dpad_behavior_weight', 1.0),
            residual_weight=self._get_config_value('processing', 'dpad_residual_weight', 0.5),
            salience_threshold=self._get_config_value('attention', 'salience_threshold', 0.8),
            auto_nonlinearity=self._get_config_value('processing', 'dpad_auto_nonlinearity', True),
            learning_rate=self._get_config_value('processing', 'dpad_learning_rate', 0.001),
            dropout_rate=self._get_config_value('processing', 'dpad_dropout_rate', 0.1),
            replay_strength=self._get_config_value('processing', 'dpad_replay_strength', 0.1),
            consolidation_rate=self._get_config_value('processing', 'dpad_consolidation_rate', 0.05)
        )
    
    def _create_lshn_config(self) -> LSHNConfig:
        """Create LSHN configuration from cognitive config"""
        self._ensure_missing_config_fields()
        return LSHNConfig(
            embedding_dim=self._get_config_value('processing', 'embedding_dimension', 384),
            pattern_dim=self._get_config_value('processing', 'lshn_pattern_dim', 512),
            hidden_dim=self._get_config_value('processing', 'lshn_hidden_dim', 256),
            memory_capacity=self._get_config_value('processing', 'lshn_memory_capacity', 4096),
            attractor_strength=self._get_config_value('processing', 'lshn_attractor_strength', 0.8),
            convergence_threshold=self._get_config_value('processing', 'lshn_convergence_threshold', 0.95),
            max_iterations=self._get_config_value('processing', 'lshn_max_iterations', 50),
            episodic_trace_decay=self._get_config_value('processing', 'lshn_episodic_trace_decay', 0.05),
            association_threshold=self._get_config_value('processing', 'lshn_association_threshold', 0.3),
            consolidation_strength=self._get_config_value('processing', 'lshn_consolidation_strength', 0.3),
            learning_rate=self._get_config_value('processing', 'lshn_learning_rate', 0.001),
            dropout_rate=self._get_config_value('processing', 'lshn_dropout_rate', 0.1),
            temperature=self._get_config_value('processing', 'lshn_temperature', 1.0)
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
        was_training = self.network.training
        try:
            self.network.eval()
            
            with torch.no_grad():
                # Forward pass through network
                forward_result = self.network.forward(embeddings, return_paths=True)
                if not isinstance(forward_result, tuple) or len(forward_result) != 2:
                    raise TypeError("Expected DPAD forward pass to return output and path details")

                output, path_details = forward_result
                if not isinstance(path_details, dict):
                    raise TypeError("Expected path details to be a dict")
                
                # Calculate attention-guided predictions
                attention_weighted_output = output * attention_scores.unsqueeze(-1)
                
                # Compute novelty detection (difference from expected)
                novelty_scores = torch.norm(output - embeddings, dim=-1)
                
                results = {
                    'status': 'completed',
                    'neural_output': output,
                    'attention_weighted_output': attention_weighted_output,
                    'novelty_scores': novelty_scores,
                    'path_weights': path_details.get('path_weights'),
                    'processing_quality': self._calculate_processing_quality(embeddings, output)
                }
                
                return results
                
        except Exception as e:
            logger.error(f"Error in attention processing: {e}")
            return {'status': 'error', 'error': str(e)}
        finally:
            if was_training:
                self.network.train()

    def _replay_ready_result(self, memory_embeddings: List[torch.Tensor]) -> Optional[Dict[str, Any]]:
        """Validate replay preconditions."""
        if not self.neural_config.enable_neural_replay:
            return {'status': 'disabled', 'replayed_memories': 0}

        if len(memory_embeddings) < self.neural_config.replay_memory_threshold:
            return {
                'status': 'insufficient_memories',
                'replayed_memories': len(memory_embeddings),
                'threshold': self.neural_config.replay_memory_threshold,
            }

        return None

    def _ensure_batch_tensor(self, batch_embeddings: Any) -> torch.Tensor:
        """Normalize batch inputs to a tensor."""
        if isinstance(batch_embeddings, list):
            return torch.stack(batch_embeddings)
        return batch_embeddings

    def _extract_forward_output(self, batch_result: Any) -> torch.Tensor:
        """Get the tensor output from optimized forward-pass results."""
        if isinstance(batch_result, tuple) and batch_result:
            return batch_result[0]
        if isinstance(batch_result, torch.Tensor):
            return batch_result
        raise TypeError(f"Unsupported replay batch result type: {type(batch_result)!r}")

    def _calculate_replay_quality(
        self,
        input_batches: List[torch.Tensor],
        batch_results: List[Any],
    ) -> float:
        """Calculate replay quality from actual DPAD inputs and outputs."""
        total_quality = 0.0
        total_items = 0

        for input_batch, batch_result in zip(input_batches, batch_results):
            output_batch = self._extract_forward_output(batch_result)
            batch_size = input_batch.size(0) if input_batch.dim() > 1 else 1
            total_quality += self._calculate_processing_quality(input_batch, output_batch) * batch_size
            total_items += batch_size

        return total_quality / total_items if total_items else 0.0

    def _update_reconstruction_quality_metrics(self, quality: float) -> None:
        """Track reconstruction quality without cold-start bias."""
        samples = int(self.performance_metrics.get('reconstruction_quality_samples', 0))
        if samples == 0:
            avg_quality = quality
        else:
            current_avg = float(self.performance_metrics.get('avg_reconstruction_quality', 0.0))
            avg_quality = ((current_avg * samples) + quality) / (samples + 1)

        self.performance_metrics['avg_reconstruction_quality'] = avg_quality
        self.performance_metrics['reconstruction_quality_samples'] = samples + 1

    def _training_response(self, status: str, training_mode: str, **data: Any) -> Dict[str, Any]:
        """Build a consistent background-training response payload."""
        payload = {
            'status': status,
            'trained': data.pop('trained', status == 'completed'),
            'training_mode': training_mode,
        }
        payload.update(data)
        return {'background_training': payload}

    async def _finalize_training_results(self, training_results: Dict[str, Any]) -> None:
        """Update training metrics and checkpoint when needed."""
        if not training_results.get('trained'):
            return

        steps_completed = int(training_results.get('steps_completed', 0))
        self.performance_metrics['total_training_steps'] += steps_completed

        if (
            steps_completed > 0
            and self.performance_metrics['total_training_steps'] % self.neural_config.checkpoint_interval == 0
        ):
            await self._save_checkpoint()

    def _build_optimized_training_results(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
    ) -> Dict[str, Any]:
        """Run optimized background training batches."""
        training_batches = self.performance_optimizer.batch_processor.create_optimized_batches(
            memory_embeddings,
            importance_scores,
        )

        total_loss = 0.0
        steps_completed = 0
        training_start_time = time.time()

        for batch_embeddings, batch_importance in training_batches:
            try:
                batch_tensor = self._ensure_batch_tensor(batch_embeddings)
                if isinstance(batch_importance, list):
                    importance_tensor = torch.tensor(
                        batch_importance,
                        dtype=batch_tensor.dtype,
                        device=batch_tensor.device,
                    )
                else:
                    importance_tensor = batch_importance

                training_result = self.performance_optimizer.optimize_training_step(
                    model=self.network,
                    optimizer=self.trainer.optimizer,
                    loss_fn=lambda outputs, targets: self.network.compute_loss(
                        outputs,
                        targets,
                        importance_tensor,
                    )['total_loss'],
                    inputs=batch_tensor,
                    targets=batch_tensor,
                )

                total_loss += float(training_result['loss'])
                steps_completed += 1

            except Exception as e:
                logger.error(f"Error in optimized training step: {e}")
                break

        training_time = time.time() - training_start_time

        return {
            'status': 'completed' if steps_completed > 0 else 'no_progress',
            'trained': steps_completed > 0,
            'training_mode': 'optimized',
            'steps_completed': steps_completed,
            'avg_loss': total_loss / max(1, steps_completed),
            'training_time': training_time,
            'memories_processed': len(memory_embeddings),
            'optimization_stats': self.performance_optimizer.get_optimization_stats(),
        }

    async def _run_training_step(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
        training_fn: Callable[[List[torch.Tensor], List[float]], Dict[str, Any]],
        training_mode: str,
    ) -> Dict[str, Any]:
        """Run a single background training path with shared locking and metrics."""
        if not self.neural_config.enable_background_training:
            return self._training_response('disabled', training_mode, trained=False)

        if not self.training_lock.acquire(blocking=False):
            return self._training_response('busy', training_mode, trained=False)

        try:
            self.is_training = True
            training_results = training_fn(memory_embeddings, importance_scores)
            await self._finalize_training_results(training_results)
            return {'background_training': training_results}
        except Exception as e:
            logger.error(f"Error in {training_mode} background training: {e}")
            return self._training_response(
                'error',
                training_mode,
                trained=False,
                error=str(e),
            )
        finally:
            self.is_training = False
            self.training_lock.release()

    async def _replay_core(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
        *,
        return_paths: bool,
        include_optimization_metrics: bool,
        training_step: Callable[[List[torch.Tensor], List[float]], Any],
    ) -> Dict[str, Any]:
        """Shared replay implementation for standard and optimized replay paths."""
        replay_guard = self._replay_ready_result(memory_embeddings)
        if replay_guard is not None:
            return replay_guard

        try:
            dpad_optimization = self.performance_optimizer.optimize_neural_forward_pass(
                self.network,
                memory_embeddings,
                importance_scores,
                return_paths=return_paths,
            )

            reconstruction_quality = self._calculate_replay_quality(
                dpad_optimization.get('input_batches', []),
                dpad_optimization['results'],
            )
            consolidation_strength = float(np.mean(importance_scores)) if importance_scores else 0.5

            dpad_results = {
                'replayed_memories': len(memory_embeddings),
                'reconstruction_quality': reconstruction_quality,
                'consolidation_strength': consolidation_strength,
                'processing_time': dpad_optimization['processing_time'],
                'memory_usage': dpad_optimization['memory_usage'],
                'optimized_batch_size': dpad_optimization['optimized_batch_size'],
                'num_batches': dpad_optimization['num_batches'],
            }

            lshn_start_time = time.time()
            lshn_results = dict(
                self.lshn_network.consolidate_memories(
                    memory_embeddings,
                    importance_scores,
                    association_threshold=self.lshn_config.association_threshold,
                )
            )
            lshn_processing_time = time.time() - lshn_start_time
            lshn_memory_usage = self.performance_optimizer.memory_monitor.get_memory_usage().get(
                'gpu_percent',
                0.0,
            )
            lshn_results.update({
                'processing_time': lshn_processing_time,
                'memory_usage': lshn_memory_usage,
                'optimized_batch_size': getattr(
                    self.performance_optimizer.batch_processor,
                    'current_batch_size',
                    len(memory_embeddings),
                ),
            })

            combined_results = {
                'status': 'completed',
                'replayed_memories': len(memory_embeddings),
                'dpad_replay': dpad_results,
                'lshn_consolidation': lshn_results,
                'total_replayed_memories': len(memory_embeddings),
                'total_associations_created': lshn_results.get('associations_created', 0),
                'reconstruction_quality': reconstruction_quality,
                'consolidation_strength': max(
                    consolidation_strength,
                    lshn_results.get('consolidation_strength', 0.0),
                ),
                'episodic_patterns': lshn_results.get(
                    'episodic_patterns',
                    lshn_results.get('episodic_patterns_formed', 0),
                ),
                'memory_associations': lshn_results.get(
                    'memory_associations',
                    lshn_results.get('memory_stats', {}),
                ),
            }

            if include_optimization_metrics:
                combined_results['optimization_metrics'] = {
                    'dpad_processing_time': dpad_optimization['processing_time'],
                    'dpad_memory_usage': dpad_optimization['memory_usage'],
                    'dpad_batch_size': dpad_optimization['optimized_batch_size'],
                    'dpad_num_batches': dpad_optimization['num_batches'],
                    'lshn_processing_time': lshn_processing_time,
                    'lshn_memory_usage': lshn_memory_usage,
                    'lshn_batch_size': lshn_results['optimized_batch_size'],
                    'lshn_num_batches': 1,
                }

            self.performance_metrics['total_replays'] += 1
            self._update_reconstruction_quality_metrics(reconstruction_quality)

            if combined_results['consolidation_strength'] > self.neural_config.consolidation_strength_threshold:
                self.performance_metrics['consolidation_events'] += 1
                if self.neural_config.enable_background_training:
                    training_results = await training_step(memory_embeddings, importance_scores)
                    combined_results.update(training_results)

            return combined_results

        except Exception as e:
            logger.error(f"Error in neural memory replay: {e}")
            return {'status': 'error', 'error': str(e), 'replayed_memories': 0}
    
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
        return await self._replay_core(
            memory_embeddings,
            importance_scores,
            return_paths=True,
            include_optimization_metrics=False,
            training_step=self._background_training_step,
        )
    
    async def optimized_neural_memory_replay(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
        attention_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Perform optimized neural replay of memories using performance optimization
        
        Args:
            memory_embeddings: Memory embeddings to replay
            importance_scores: Importance scores for each memory
            attention_context: Optional attention context
        
        Returns:
            Replay results with optimization metrics
        """
        return await self._replay_core(
            memory_embeddings,
            importance_scores,
            return_paths=False,
            include_optimization_metrics=True,
            training_step=self._optimized_background_training_step,
        )
    
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
        def run_standard_training(
            embeddings: List[torch.Tensor],
            scores: List[float],
        ) -> Dict[str, Any]:
            training_results = self.trainer.background_training(
                embeddings,
                scores,
                steps=self.neural_config.dream_training_steps,
            )
            training_results['status'] = 'completed' if training_results.get('trained') else 'no_progress'
            training_results['training_mode'] = 'standard'
            return training_results

        return await self._run_training_step(
            memory_embeddings,
            importance_scores,
            training_fn=run_standard_training,
            training_mode='standard',
        )
    
    async def _optimized_background_training_step(
        self,
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float]
    ) -> Dict[str, Any]:
        """
        Perform optimized background training step using performance optimization
        
        Args:
            memory_embeddings: Memory embeddings for training
            importance_scores: Importance scores for weighting
        
        Returns:
            Optimized training step results
        """
        return await self._run_training_step(
            memory_embeddings,
            importance_scores,
            training_fn=self._build_optimized_training_results,
            training_mode='optimized',
        )

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
        network_stats = self.network.get_network_stats()
        
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
        except Exception:
            return 0.0
    
    async def _save_checkpoint(self):
        await asyncio.to_thread(self._save_checkpoint_sync)

    def _save_checkpoint_sync(self) -> None:
        """Save neural network checkpoint"""
        try:
            timestamp = int(time.time())
            model_dir = Path(self.model_save_path)
            checkpoint_path = model_dir / f"dpad_checkpoint_{timestamp}.pt"
            
            self.trainer.save_checkpoint(str(checkpoint_path))
            
            # Save integration state
            integration_state = {
                'performance_metrics': self.performance_metrics,
                'dpad_config': asdict(self.dpad_config),
                'neural_config': asdict(self.neural_config),
                'timestamp': timestamp
            }
            
            state_path = model_dir / f"integration_state_{timestamp}.json"
            with state_path.open('w', encoding='utf-8') as f:
                json.dump(integration_state, f, indent=2)
            
            self.performance_metrics['last_checkpoint'] = str(checkpoint_path)
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
            self.performance_metrics['last_checkpoint'] = str(latest_checkpoint)
            
            if state_path.exists():
                with state_path.open('r', encoding='utf-8') as f:
                    state = json.load(f)
                    self.performance_metrics = self._sanitize_performance_metrics(
                        state.get('performance_metrics')
                    )
                    self.performance_metrics['last_checkpoint'] = str(latest_checkpoint)
            
            logger.info(f"Loaded neural checkpoint: {latest_checkpoint}")
            
        except Exception as e:
            logger.warning(f"Could not load checkpoint: {e}")
    
    async def optimize_performance(self):
        """Optimize neural network performance based on recent metrics"""
        try:
            # Check performance trends
            if len(self.trainer.training_history) < 10:
                return
            
            recent_losses = [float(h['total_loss']) for h in self.trainer.training_history[-10:]]
            avg_recent_loss = float(np.mean(recent_losses))
            earlier_window = recent_losses[:5]
            recent_window = recent_losses[-5:]
            
            # Trigger nonlinearity optimization
            self.network.optimize_nonlinearity(avg_recent_loss)
            
            # Adaptive learning rate adjustment
            if self.neural_config.adaptive_learning_rate:
                current_lr = self.trainer.optimizer.param_groups[0]['lr']
                base_lr = float(self.dpad_config.learning_rate)
                
                # Check if loss is stagnating
                if len(recent_window) >= 5:
                    loss_variance = float(np.var(recent_window))
                    if loss_variance < 1e-6:  # Very low variance indicates stagnation
                        new_lr = max(base_lr * 0.1, current_lr * 0.9)
                        for param_group in self.trainer.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        logger.info(f"Reduced learning rate to {new_lr}")
                    else:
                        avg_earlier_loss = float(np.mean(earlier_window))
                        if avg_recent_loss < avg_earlier_loss * 0.95 and current_lr < base_lr:
                            new_lr = min(base_lr, current_lr * 1.05)
                            for param_group in self.trainer.optimizer.param_groups:
                                param_group['lr'] = new_lr
                            logger.info(f"Increased learning rate to {new_lr}")
            
        except Exception as e:
            logger.error(f"Error in performance optimization: {e}")
    
    def shutdown(self):
        """Shutdown neural integration manager"""
        logger.info("Shutting down Neural Integration Manager...")

        # Wait for any ongoing training to complete.
        if self.training_lock.locked():
            logger.info("Waiting for training to complete...")
            self.training_lock.acquire()
            self.training_lock.release()

        self._save_checkpoint_sync()
        optimizer_shutdown = getattr(self.performance_optimizer, 'shutdown', None)
        if callable(optimizer_shutdown):
            optimizer_shutdown()
        
        logger.info("Neural Integration Manager shutdown complete")
