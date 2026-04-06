"""
LSHN (Latent Structured Hopfield Networks) Implementation - Patched for Dashboard Integration

This version extends the original LSHN with:
- Safe fallback retrieval when Hopfield layers are unavailable
- Dashboard-friendly state summaries (frontier, salience/recency)
- Metacognitive metrics for human-likeness tracking
- Time-based decay and reweighting of stored patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, cast
from dataclasses import dataclass
import logging

# Initialize logger
logger = logging.getLogger(__name__)

# Dummy classes for fallback
class _DummyHopfield(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

class _DummyHopfieldLayer(nn.Module):
    def __init__(self, input_size=None, hidden_size=None, output_size=None, 
                 num_heads=None, scaling=None, update_steps_max=None, **kwargs):
        super().__init__()
        self.output_size = output_size or input_size or 512
        self.linear = nn.Linear(input_size or 512, self.output_size)
    def forward(self, input=None, stored_pattern_padding_mask=None, **kwargs):
        if input is None:
            raise ValueError("Input is required for DummyHopfieldLayer")
        return self.linear(input)

class _DummyHopfieldPooling(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x

try:
    from hflayers import Hopfield, HopfieldLayer, HopfieldPooling
    HOPFIELD_AVAILABLE = True
    logger.debug("Successfully imported hopfield-layers")
except Exception as e:
    HOPFIELD_AVAILABLE = False
    error_type = type(e).__name__
    logger.debug(f"Failed to import hopfield-layers ({error_type}): {e}")
    logging.warning(f"hopfield-layers not available ({error_type}). LSHN functionality will be limited.")
    Hopfield = _DummyHopfield
    HopfieldLayer = _DummyHopfieldLayer
    HopfieldPooling = _DummyHopfieldPooling

@dataclass
class LSHNConfig:
    embedding_dim: int = 384
    pattern_dim: int = 512
    hidden_dim: int = 256
    memory_capacity: int = 4096
    attractor_strength: float = 0.8
    convergence_threshold: float = 0.95
    max_iterations: int = 50
    episodic_trace_decay: float = 0.05
    association_threshold: float = 0.7
    consolidation_strength: float = 0.3
    learning_rate: float = 0.001
    dropout_rate: float = 0.1
    temperature: float = 1.0
    num_heads: int = 8
    num_layers: int = 2
    use_layernorm: bool = True

class EpisodicMemoryEncoder(nn.Module):
    def __init__(self, config: LSHNConfig):
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.pattern_projection = nn.Linear(config.hidden_dim, config.pattern_dim)
        self.temporal_encoding = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout_rate,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(config.pattern_dim) if config.use_layernorm else None
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, embeddings: torch.Tensor, temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        hidden = self.input_projection(embeddings)
        hidden = F.relu(hidden)
        if temporal_context is not None:
            temporal_hidden = self.input_projection(temporal_context)
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1)
            combined = torch.cat([hidden, temporal_hidden], dim=1)
            encoded = self.temporal_encoding(combined)
            hidden = encoded[:, 0, :]
        patterns = self.pattern_projection(hidden)
        patterns = self.dropout(patterns)
        if self.layer_norm:
            patterns = self.layer_norm(patterns)
        return patterns

class HopfieldAssociativeMemory(nn.Module):
    def __init__(self, config: LSHNConfig):
        super().__init__()
        self.config = config
        if config.memory_capacity <= 0:
            raise ValueError("memory_capacity must be positive")
        if not HOPFIELD_AVAILABLE:
            logger.warning("Hopfield layers not available - using simplified retrieval")
        self.hopfield_layers = nn.ModuleList([
            HopfieldLayer(
                input_size=config.pattern_dim,
                hidden_size=config.pattern_dim // (2 ** i),
                output_size=config.pattern_dim,
                num_heads=config.num_heads,
                scaling=config.attractor_strength,
                update_steps_max=config.max_iterations
            )
            for i in range(config.num_layers)
        ])
        self.consolidation_net = nn.Sequential(
            nn.Linear(config.pattern_dim * config.num_layers, config.pattern_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.pattern_dim, config.pattern_dim)
        )
        self.register_buffer('memory_patterns', torch.zeros(config.memory_capacity, config.pattern_dim))
        self.register_buffer('pattern_weights', torch.zeros(config.memory_capacity))
        self.register_buffer('access_counts', torch.zeros(config.memory_capacity))
        self.register_buffer('last_access_steps', torch.zeros(config.memory_capacity, dtype=torch.long))
        self.register_buffer('active_count', torch.zeros((), dtype=torch.long))
        self.register_buffer('write_index', torch.zeros((), dtype=torch.long))
        self.register_buffer('memory_step', torch.zeros((), dtype=torch.long))

    def _tensor_buffer(self, name: str) -> torch.Tensor:
        return cast(torch.Tensor, self._buffers[name])

    def _memory_patterns(self) -> torch.Tensor:
        return self._tensor_buffer('memory_patterns')

    def _pattern_weights(self) -> torch.Tensor:
        return self._tensor_buffer('pattern_weights')

    def _access_counts(self) -> torch.Tensor:
        return self._tensor_buffer('access_counts')

    def _last_access_steps(self) -> torch.Tensor:
        return self._tensor_buffer('last_access_steps')

    def _active_count(self) -> torch.Tensor:
        return self._tensor_buffer('active_count')

    def _write_index(self) -> torch.Tensor:
        return self._tensor_buffer('write_index')

    def _memory_step(self) -> torch.Tensor:
        return self._tensor_buffer('memory_step')

    def _active_indices(self) -> torch.Tensor:
        active_count_tensor = self._active_count()
        memory_patterns = self._memory_patterns()
        active_count = int(active_count_tensor.item())
        device = memory_patterns.device
        if active_count == 0:
            return torch.zeros(0, dtype=torch.long, device=device)
        if active_count < self.config.memory_capacity:
            return torch.arange(active_count, device=device)
        start = int(self._write_index().item())
        return (torch.arange(active_count, device=device) + start) % self.config.memory_capacity

    def _active_view(self, tensor: torch.Tensor) -> torch.Tensor:
        indices = self._active_indices()
        if indices.numel() == 0:
            return tensor[:0]
        return tensor.index_select(0, indices)

    def _advance_step(self) -> int:
        memory_step = self._memory_step()
        next_step = int(memory_step.item()) + 1
        memory_step.fill_(next_step)
        return next_step

    def _aggregate_patterns(self, patterns: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scaled_scores = scores / max(self.config.temperature, 1e-6)
        weights = torch.softmax(scaled_scores, dim=-1)
        return torch.sum(patterns * weights.unsqueeze(-1), dim=1)

    def _retrieve_with_hopfield(self, query: torch.Tensor, candidate_patterns: torch.Tensor) -> torch.Tensor:
        if not HOPFIELD_AVAILABLE:
            return self._aggregate_patterns(candidate_patterns, torch.ones_like(candidate_patterns[..., 0]))

        query_states = query.unsqueeze(1)
        retrieved_patterns = []
        for layer in self.hopfield_layers:
            hopfield = cast(Any, layer).hopfield
            hopfield_output = hopfield(
                input=(candidate_patterns, query_states, candidate_patterns)
            )
            retrieved_patterns.append(hopfield_output.squeeze(1))

        if len(retrieved_patterns) > 1:
            consolidated_input = torch.cat(retrieved_patterns, dim=-1)
            return self.consolidation_net(consolidated_input)
        return retrieved_patterns[0]

    def reset(self):
        self._memory_patterns().zero_()
        self._pattern_weights().zero_()
        self._access_counts().zero_()
        self._last_access_steps().zero_()
        self._active_count().zero_()
        self._write_index().zero_()
        self._memory_step().zero_()
        
    def store_pattern(self, pattern: torch.Tensor, weight: float | torch.Tensor = 1.0):
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)

        memory_patterns = self._memory_patterns()
        pattern_weights = self._pattern_weights()
        access_counts = self._access_counts()
        last_access_steps = self._last_access_steps()
        active_count = self._active_count()
        write_index = self._write_index()

        pattern = pattern.to(device=memory_patterns.device, dtype=memory_patterns.dtype)
        weight_tensor = torch.as_tensor(weight, device=pattern_weights.device, dtype=pattern_weights.dtype)
        if weight_tensor.dim() == 0:
            weight_tensor = weight_tensor.repeat(pattern.size(0))
        else:
            weight_tensor = weight_tensor.reshape(-1)
        if weight_tensor.numel() != pattern.size(0):
            raise ValueError("weight must be a scalar or provide one value per pattern")

        for row, row_weight in zip(pattern, weight_tensor):
            slot = int(write_index.item())
            current_step = self._advance_step()
            memory_patterns[slot].copy_(row)
            pattern_weights[slot] = row_weight
            access_counts[slot] = 0
            last_access_steps[slot] = current_step
            if int(active_count.item()) < self.config.memory_capacity:
                active_count.add_(1)
            write_index.fill_((slot + 1) % self.config.memory_capacity)
        
    def retrieve_pattern(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        active_indices = self._active_indices()
        if active_indices.numel() == 0:
            return torch.zeros_like(query), torch.zeros(query.size(0), 1, device=query.device, dtype=query.dtype)

        memory_patterns = self._memory_patterns()
        access_counts = self._access_counts()
        last_access_steps = self._last_access_steps()

        active_patterns = memory_patterns.index_select(0, active_indices)
        similarities = F.cosine_similarity(
            query.unsqueeze(1),
            active_patterns.unsqueeze(0),
            dim=-1
        )
        top_scores, top_indices = torch.topk(similarities, min(top_k, similarities.size(-1)), dim=-1)
        flat_top_indices = top_indices.reshape(-1)
        candidate_patterns = active_patterns.index_select(0, flat_top_indices).view(query.size(0), -1, self.config.pattern_dim)
        if HOPFIELD_AVAILABLE:
            final_retrieved = self._retrieve_with_hopfield(query, candidate_patterns)
        else:
            final_retrieved = self._aggregate_patterns(candidate_patterns, top_scores)

        accessed_buffer_indices = active_indices.index_select(0, flat_top_indices)
        access_updates = torch.ones_like(accessed_buffer_indices, dtype=access_counts.dtype)
        access_counts.index_add_(0, accessed_buffer_indices, access_updates)
        current_step = self._advance_step()
        unique_indices = torch.unique(accessed_buffer_indices)
        last_access_steps.index_fill_(0, unique_indices, current_step)
        return final_retrieved, top_scores
    
    def get_memory_stats(self) -> Dict[str, Any]:
        active_patterns = self._active_view(self._memory_patterns())
        active_access_counts = self._active_view(self._access_counts())
        return {
            'total_patterns': int(self._active_count().item()),
            'memory_capacity': self.config.memory_capacity,
            'pattern_dimension': active_patterns.size(-1) if active_patterns.size(0) > 0 else 0,
            'avg_access_count': active_access_counts.mean().item() if active_access_counts.size(0) > 0 else 0,
            'memory_utilization': (active_access_counts > 0).float().mean().item() if active_access_counts.size(0) > 0 else 0
        }

class LSHNNetwork(nn.Module):
    def __init__(self, config: LSHNConfig):
        super().__init__()
        self.config = config
        self.encoder = EpisodicMemoryEncoder(config)
        self.associative_memory = HopfieldAssociativeMemory(config)
        self.trace_formation = nn.Sequential(
            nn.Linear(config.pattern_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.pattern_dim),
            nn.Tanh()
        )
        self.consolidation_scorer = nn.Sequential(
            nn.Linear(config.pattern_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        self.stats = {
            'patterns_encoded': 0,
            'patterns_stored': 0,
            'retrievals_performed': 0,
            'consolidations_triggered': 0,
            'patterns_associated': 0,
            'association_matches': 0
        }

    def _encode_embeddings(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        episodic_patterns = self.encoder(embeddings)
        episodic_traces = self.trace_formation(episodic_patterns)
        return episodic_patterns, episodic_traces

    def _store_patterns(self, episodic_patterns: torch.Tensor, importance_scores: Optional[torch.Tensor]) -> int:
        if importance_scores is None:
            selected_patterns = episodic_patterns
            selected_weights = torch.ones(
                episodic_patterns.size(0),
                device=episodic_patterns.device,
                dtype=episodic_patterns.dtype,
            )
        else:
            weights = importance_scores.to(device=episodic_patterns.device, dtype=episodic_patterns.dtype)
            selected_mask = weights > self.config.consolidation_strength
            if not torch.any(selected_mask):
                return 0
            selected_patterns = episodic_patterns[selected_mask]
            selected_weights = weights[selected_mask]

        self.associative_memory.store_pattern(selected_patterns, selected_weights)
        stored_count = int(selected_patterns.size(0))
        self.stats['patterns_stored'] += stored_count
        return stored_count

    def _build_results(
        self,
        episodic_patterns: torch.Tensor,
        episodic_traces: torch.Tensor,
        *,
        store_patterns: bool,
        retrieve_similar: bool,
        importance_scores: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        batch_size = episodic_patterns.size(0)
        self.stats['patterns_encoded'] += batch_size

        results = {
            'episodic_patterns': episodic_patterns,
            'episodic_traces': episodic_traces,
            'retrieved_patterns': None,
            'similarity_scores': None,
            'consolidation_scores': None
        }
        if retrieve_similar:
            retrieved_patterns, similarity_scores = self.associative_memory.retrieve_pattern(episodic_patterns)
            results['retrieved_patterns'] = retrieved_patterns
            results['similarity_scores'] = similarity_scores
            self.stats['retrievals_performed'] += batch_size
            pattern_pairs = torch.cat([episodic_patterns, retrieved_patterns], dim=-1)
            results['consolidation_scores'] = self.consolidation_scorer(pattern_pairs)
        if store_patterns:
            self._store_patterns(episodic_patterns, importance_scores)
        return results
    
    def forward(self, embeddings: torch.Tensor, store_patterns=True, retrieve_similar=True, importance_scores=None) -> Dict[str, torch.Tensor]:
        episodic_patterns, episodic_traces = self._encode_embeddings(embeddings)
        return self._build_results(
            episodic_patterns,
            episodic_traces,
            store_patterns=store_patterns,
            retrieve_similar=retrieve_similar,
            importance_scores=importance_scores,
        )
    
    def decay_and_reweight(self, decay_factor=None):
        decay = decay_factor or self.config.episodic_trace_decay
        active_indices = self.associative_memory._active_indices()
        if active_indices.numel() > 0:
            pattern_weights = self.associative_memory._pattern_weights()
            decayed_weights = pattern_weights.index_select(0, active_indices) * (1.0 - decay)
            pattern_weights.index_copy_(0, active_indices, decayed_weights)
    
    def get_frontier_patterns(self, top_k=10):
        active_patterns = self.associative_memory._active_view(self.associative_memory._memory_patterns())
        active_access_counts = self.associative_memory._active_view(self.associative_memory._access_counts())
        if active_patterns.size(0) > 0:
            access_order = torch.argsort(active_access_counts, descending=True)
            frontier = access_order[:top_k]
            return active_patterns[frontier].detach().cpu().numpy().tolist()
        return []
    
    def get_salience_recency(self):
        active_weights = self.associative_memory._active_view(self.associative_memory._pattern_weights())
        active_access_counts = self.associative_memory._active_view(self.associative_memory._access_counts())
        active_last_access = self.associative_memory._active_view(self.associative_memory._last_access_steps())
        if active_weights.size(0) > 0:
            current_step = max(1, int(self.associative_memory._memory_step().item()))
            salience = active_weights.detach().cpu().numpy().tolist()
            access_frequency = active_access_counts.detach().cpu().numpy().tolist()
            recency = (active_last_access.float() / current_step).detach().cpu().numpy().tolist()
            return [
                {"salience": s, "recency": r, "access_frequency": f}
                for s, r, f in zip(salience, recency, access_frequency)
            ]
        return []
    
    def get_metacognitive_metrics(self):
        return {
            "memory_fidelity": float(self.stats['patterns_stored'] / max(1, self.stats['patterns_encoded'])),
            "attentional_adaptation": float(self.stats['retrievals_performed']) / max(1, self.stats['patterns_encoded']),
            "consolidation_precision": float(self.stats['patterns_associated']) / max(1, self.stats['patterns_stored'])
        }
    
    def get_dashboard_state(self):
        return {
            "stats": self.get_network_stats(),
            "frontier_patterns": self.get_frontier_patterns(),
            "salience_recency": self.get_salience_recency(),
            "metacognition": self.get_metacognitive_metrics()
        }
    
    def get_network_stats(self):
        stats = self.stats.copy()
        stats.update(self.associative_memory.get_memory_stats())
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        stats.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hopfield_available': HOPFIELD_AVAILABLE
        })
        return stats
    
    def consolidate_memories(self, memory_embeddings: List[torch.Tensor], importance_scores: List[float], association_threshold=None) -> Dict[str, Any]:
        if not memory_embeddings:
            return {'consolidated_memories': 0, 'associations_created': 0, 'consolidation_strength': 0.0}
        self.decay_and_reweight()
        threshold = association_threshold or self.config.association_threshold
        embeddings = torch.stack(memory_embeddings)
        importance_weights = torch.tensor(importance_scores, dtype=embeddings.dtype, device=embeddings.device)
        episodic_patterns, episodic_traces = self._encode_embeddings(embeddings)
        self._store_patterns(episodic_patterns, importance_weights)
        results = self._build_results(
            episodic_patterns,
            episodic_traces,
            store_patterns=False,
            retrieve_similar=True,
            importance_scores=importance_weights,
        )
        associations_created = 0
        patterns_associated = 0
        if results['similarity_scores'] is not None:
            high_similarity = results['similarity_scores'] > threshold
            associations_created = int(high_similarity.sum().item())
            patterns_associated = int(high_similarity.any(dim=-1).sum().item())
        consolidation_strength = 0.0
        if results['consolidation_scores'] is not None:
            consolidation_strength = (results['consolidation_scores'] * importance_weights.unsqueeze(-1)).sum().item()
        self.stats['consolidations_triggered'] += 1
        self.stats['association_matches'] += associations_created
        self.stats['patterns_associated'] += patterns_associated
        return {
            'consolidated_memories': len(memory_embeddings),
            'associations_created': associations_created,
            'consolidation_strength': consolidation_strength,
            'episodic_patterns_formed': results['episodic_patterns'].size(0),
            'memory_stats': self.associative_memory.get_memory_stats()
        }
    
    def reset_memory(self):
        self.associative_memory.reset()
        self.stats = {
            'patterns_encoded': 0,
            'patterns_stored': 0,
            'retrievals_performed': 0,
            'consolidations_triggered': 0,
            'patterns_associated': 0,
            'association_matches': 0,
        }

class LSHNTrainer:
    def __init__(self, network: LSHNNetwork, config: LSHNConfig):
        self.network = network
        self.config = config
        self.optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        self.training_history: List[Dict[str, float]] = []
        
    def train_step(self, embeddings: torch.Tensor, target_patterns: Optional[torch.Tensor] = None, importance_scores: Optional[torch.Tensor] = None) -> Dict[str, float]:
        self.optimizer.zero_grad()
        results = self.network.forward(
            embeddings,
            store_patterns=False,
            retrieve_similar=True,
            importance_scores=importance_scores
        )
        total_loss = 0.0
        metrics = {}
        if target_patterns is not None:
            recon_loss = F.mse_loss(results['episodic_patterns'], target_patterns)
            total_loss += recon_loss
            metrics['reconstruction_loss'] = recon_loss.item()
        consistency_loss = F.mse_loss(results['episodic_patterns'], results['episodic_traces'])
        total_loss += consistency_loss * 0.1
        metrics['consistency_loss'] = consistency_loss.item()
        if importance_scores is not None:
            importance_weights = importance_scores.unsqueeze(-1)
            weighted_loss = (F.mse_loss(results['episodic_patterns'], results['episodic_traces'], reduction='none') * importance_weights).mean()
            total_loss += weighted_loss * 0.1
            metrics['importance_weighted_loss'] = weighted_loss.item()
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
        metrics['total_loss'] = total_loss.item()
        self.training_history.append(metrics.copy())
        return metrics
