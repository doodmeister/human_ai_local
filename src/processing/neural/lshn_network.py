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
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

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
    print("LSHN DEBUG: Successfully imported hopfield-layers")
except Exception as e:
    HOPFIELD_AVAILABLE = False
    error_type = type(e).__name__
    print(f"LSHN DEBUG: Failed to import hopfield-layers ({error_type}): {e}")
    logging.warning(f"hopfield-layers not available ({error_type}). LSHN functionality will be limited.")
    Hopfield = _DummyHopfield
    HopfieldLayer = _DummyHopfieldLayer
    HopfieldPooling = _DummyHopfieldPooling

logger = logging.getLogger(__name__)

@dataclass
class LSHNConfig:
    embedding_dim: int = 384
    pattern_dim: int = 512
    hidden_dim: int = 256
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
        self.register_buffer('memory_patterns', torch.zeros(0, config.pattern_dim))
        self.register_buffer('pattern_weights', torch.zeros(0))
        self.register_buffer('access_counts', torch.zeros(0))
        
    def store_pattern(self, pattern: torch.Tensor, weight: float = 1.0):
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
        self.memory_patterns = torch.cat([self.memory_patterns, pattern], dim=0)
        self.pattern_weights = torch.cat([self.pattern_weights, torch.tensor([weight])])
        self.access_counts = torch.cat([self.access_counts, torch.zeros(1)])
        
    def retrieve_pattern(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_patterns.size(0) == 0:
            return torch.zeros_like(query), torch.zeros(query.size(0), 1)
        if not HOPFIELD_AVAILABLE:
            sims = F.cosine_similarity(
                query.unsqueeze(1),
                self.memory_patterns.unsqueeze(0),
                dim=-1
            )
            top_scores, _ = torch.topk(sims, min(top_k, sims.size(-1)), dim=-1)
            return query, top_scores
        retrieved_patterns = []
        for layer in self.hopfield_layers:
            query_3d = query.unsqueeze(1)
            retrieved = layer(input=query_3d)
            retrieved_patterns.append(retrieved.squeeze(1))
        if len(retrieved_patterns) > 1:
            consolidated_input = torch.cat(retrieved_patterns, dim=-1)
            final_retrieved = self.consolidation_net(consolidated_input)
        else:
            final_retrieved = retrieved_patterns[0]
        similarities = F.cosine_similarity(
            query.unsqueeze(1),
            self.memory_patterns.unsqueeze(0),
            dim=-1
        )
        top_scores, top_indices = torch.topk(similarities, min(top_k, similarities.size(-1)), dim=-1)
        for idx in top_indices.flatten():
            self.access_counts[idx] += 1
        return final_retrieved, top_scores
    
    def get_memory_stats(self) -> Dict[str, Any]:
        return {
            'total_patterns': self.memory_patterns.size(0),
            'pattern_dimension': self.memory_patterns.size(-1) if self.memory_patterns.size(0) > 0 else 0,
            'avg_access_count': self.access_counts.mean().item() if self.access_counts.size(0) > 0 else 0,
            'memory_utilization': (self.access_counts > 0).float().mean().item() if self.access_counts.size(0) > 0 else 0
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
            'consolidations_triggered': 0
        }
    
    def forward(self, embeddings: torch.Tensor, store_patterns=True, retrieve_similar=True, importance_scores=None) -> Dict[str, torch.Tensor]:
        batch_size = embeddings.size(0)
        episodic_patterns = self.encoder(embeddings)
        self.stats['patterns_encoded'] += batch_size
        episodic_traces = self.trace_formation(episodic_patterns)
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
            for i in range(batch_size):
                weight = importance_scores[i].item() if importance_scores is not None else 1.0
                if weight > self.config.consolidation_strength:
                    self.associative_memory.store_pattern(episodic_patterns[i], weight)
                    self.stats['patterns_stored'] += 1
        return results
    
    def decay_and_reweight(self, decay_factor=None):
        decay = decay_factor or self.config.episodic_trace_decay
        if self.associative_memory.pattern_weights.size(0) > 0:
            self.associative_memory.pattern_weights *= (1.0 - decay)
    
    def get_frontier_patterns(self, top_k=10):
        if self.associative_memory.memory_patterns.size(0) > 0:
            access_order = torch.argsort(self.associative_memory.access_counts, descending=True)
            frontier = access_order[:top_k]
            return self.associative_memory.memory_patterns[frontier].detach().cpu().numpy().tolist()
        return []
    
    def get_salience_recency(self):
        if self.associative_memory.memory_patterns.size(0) > 0:
            salience = self.associative_memory.pattern_weights.detach().cpu().numpy().tolist()
            recency = self.associative_memory.access_counts.detach().cpu().numpy().tolist()
            return [{"salience": s, "recency": r} for s, r in zip(salience, recency)]
        return []
    
    def get_metacognitive_metrics(self):
        return {
            "memory_fidelity": 1.0 - float(self.stats['patterns_stored'] / max(1, self.stats['patterns_encoded'])),
            "attentional_adaptation": float(self.stats['retrievals_performed']) / max(1, self.stats['patterns_encoded']),
            "consolidation_precision": float(self.stats['consolidations_triggered']) / max(1, self.stats['patterns_stored'] or 1)
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
        importance_weights = torch.tensor(importance_scores, dtype=torch.float32)
        self.forward(embeddings, store_patterns=True, retrieve_similar=False, importance_scores=importance_weights)
        results = self.forward(embeddings, store_patterns=False, retrieve_similar=True, importance_scores=importance_weights)
        associations_created = 0
        if results['similarity_scores'] is not None:
            high_similarity = (results['similarity_scores'] > threshold).sum().item()
            associations_created = high_similarity
        consolidation_strength = 0.0
        if results['consolidation_scores'] is not None:
            consolidation_strength = (results['consolidation_scores'] * importance_weights.unsqueeze(-1)).sum().item()
        self.stats['consolidations_triggered'] += 1
        return {
            'consolidated_memories': len(memory_embeddings),
            'associations_created': associations_created,
            'consolidation_strength': consolidation_strength,
            'episodic_patterns_formed': results['episodic_patterns'].size(0),
            'memory_stats': self.associative_memory.get_memory_stats()
        }
    
    def reset_memory(self):
        self.associative_memory.memory_patterns = torch.zeros(0, self.config.pattern_dim)
        self.associative_memory.pattern_weights = torch.zeros(0)
        self.associative_memory.access_counts = torch.zeros(0)
        self.stats = {'patterns_encoded': 0, 'patterns_stored': 0, 'retrievals_performed': 0, 'consolidations_triggered': 0}

class LSHNTrainer:
    def __init__(self, network: LSHNNetwork, config: LSHNConfig):
        self.network = network
        self.config = config
        self.optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
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
        return metrics
