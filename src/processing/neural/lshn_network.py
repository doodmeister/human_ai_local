"""
LSHN (Latent Structured Hopfield Networks) Implementation

This module implements biologically-inspired Hopfield networks for episodic memory:
- Hippocampal-style associative memory formation
- Content-addressable memory retrieval
- Continuous attractor dynamics
- Episodic trace consolidation
- Semantic association enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

# Define dummy classes for fallback
class _DummyHopfield: pass
class _DummyHopfieldLayer: pass  
class _DummyHopfieldPooling: pass

try:
    from hflayers import Hopfield, HopfieldLayer, HopfieldPooling
    HOPFIELD_AVAILABLE = True
    print("LSHN DEBUG: Successfully imported hopfield-layers")
except Exception as e:
    HOPFIELD_AVAILABLE = False
    error_type = type(e).__name__
    print(f"LSHN DEBUG: Failed to import hopfield-layers ({error_type}): {e}")
    logging.warning(f"hopfield-layers not available ({error_type}). LSHN functionality will be limited.")
    # Use dummy classes for fallback
    Hopfield = _DummyHopfield
    HopfieldLayer = _DummyHopfieldLayer
    HopfieldPooling = _DummyHopfieldPooling

logger = logging.getLogger(__name__)

@dataclass
class LSHNConfig:
    """Configuration for LSHN neural network"""
    # Core dimensions
    embedding_dim: int = 384  # Matches sentence transformer
    pattern_dim: int = 512    # Stored episodic patterns dimension
    hidden_dim: int = 256     # Hidden layer dimension
    
    # Hopfield parameters
    attractor_strength: float = 0.8      # Hopfield attractor dynamics strength
    convergence_threshold: float = 0.95  # Convergence criteria for memory retrieval
    max_iterations: int = 50             # Maximum iterations for attractor convergence
    
    # Memory parameters
    episodic_trace_decay: float = 0.05   # Decay rate for episodic memory traces
    association_threshold: float = 0.7   # Threshold for creating associations
    consolidation_strength: float = 0.3  # Strength of memory consolidation
    
    # Training parameters
    learning_rate: float = 0.001
    dropout_rate: float = 0.1
    temperature: float = 1.0             # Temperature for softmax attention
    
    # Architecture parameters
    num_heads: int = 8                   # Multi-head attention
    num_layers: int = 2                  # Number of Hopfield layers
    use_layernorm: bool = True           # Use layer normalization
    
class EpisodicMemoryEncoder(nn.Module):
    """
    Encoder for converting experiences into episodic memory patterns
    """
    
    def __init__(self, config: LSHNConfig):
        super().__init__()
        self.config = config
        
        # Embedding transformation layers
        self.input_projection = nn.Linear(config.embedding_dim, config.hidden_dim)
        self.pattern_projection = nn.Linear(config.hidden_dim, config.pattern_dim)
        
        # Temporal context encoding
        self.temporal_encoding = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout_rate,
            batch_first=True
        )
        
        # Normalization and activation
        self.layer_norm = nn.LayerNorm(config.pattern_dim) if config.use_layernorm else None
        self.dropout = nn.Dropout(config.dropout_rate)
        
    def forward(self, embeddings: torch.Tensor, temporal_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode embeddings into episodic memory patterns
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            temporal_context: Optional temporal context [batch_size, seq_len, embedding_dim]
        
        Returns:
            Episodic memory patterns [batch_size, pattern_dim]
        """
        # Project to hidden dimension
        hidden = self.input_projection(embeddings)
        hidden = F.relu(hidden)
        
        # Apply temporal context if available
        if temporal_context is not None:
            temporal_hidden = self.input_projection(temporal_context)
            # Add sequence dimension for transformer
            if hidden.dim() == 2:
                hidden = hidden.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            combined = torch.cat([hidden, temporal_hidden], dim=1)
            encoded = self.temporal_encoding(combined)
            hidden = encoded[:, 0, :]  # Take first (target) position
        
        # Project to pattern dimension
        patterns = self.pattern_projection(hidden)
        patterns = self.dropout(patterns)
        
        if self.layer_norm:
            patterns = self.layer_norm(patterns)
            
        return patterns

class HopfieldAssociativeMemory(nn.Module):
    """
    Hopfield-based associative memory for episodic pattern storage and retrieval
    """
    
    def __init__(self, config: LSHNConfig):
        super().__init__()
        self.config = config
        
        if not HOPFIELD_AVAILABLE:
            raise ImportError("hopfield-layers package required for HopfieldAssociativeMemory")
          # Multiple Hopfield layers for hierarchical memory
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
        
        # Memory consolidation network
        self.consolidation_net = nn.Sequential(
            nn.Linear(config.pattern_dim * config.num_layers, config.pattern_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.pattern_dim, config.pattern_dim)
        )
        
        # Stored memory patterns (will be populated during training)
        self.register_buffer('memory_patterns', torch.zeros(0, config.pattern_dim))
        self.register_buffer('pattern_weights', torch.zeros(0))
        self.register_buffer('access_counts', torch.zeros(0))
        
    def store_pattern(self, pattern: torch.Tensor, weight: float = 1.0):
        """
        Store a new episodic pattern in associative memory
        
        Args:
            pattern: Episodic pattern to store [pattern_dim]
            weight: Importance weight for the pattern
        """
        if pattern.dim() == 1:
            pattern = pattern.unsqueeze(0)
        
        # Add to memory
        self.memory_patterns = torch.cat([self.memory_patterns, pattern], dim=0)
        self.pattern_weights = torch.cat([self.pattern_weights, torch.tensor([weight])])
        self.access_counts = torch.cat([self.access_counts, torch.zeros(1)])
        
    def retrieve_pattern(self, query: torch.Tensor, top_k: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve similar patterns using Hopfield dynamics
        
        Args:
            query: Query pattern [batch_size, pattern_dim]
            top_k: Number of top patterns to retrieve
        
        Returns:
            Retrieved patterns and similarity scores
        """
        if self.memory_patterns.size(0) == 0:
            # No stored patterns
            return torch.zeros_like(query), torch.zeros(query.size(0), 1)
        
        batch_size = query.size(0)
        retrieved_patterns = []
        similarity_scores = []        # Process through Hopfield layers
        for layer in self.hopfield_layers:
            # Add sequence dimension for HopfieldLayer (expects 3D input)
            query_3d = query.unsqueeze(1)  # (batch_size, 1, pattern_dim)
            retrieved = layer(
                input=query_3d,
                stored_pattern_padding_mask=None
            )
            # Remove sequence dimension
            retrieved = retrieved.squeeze(1)  # (batch_size, pattern_dim)
            retrieved_patterns.append(retrieved)
        
        # Consolidate multi-layer retrievals
        if len(retrieved_patterns) > 1:
            consolidated_input = torch.cat(retrieved_patterns, dim=-1)
            final_retrieved = self.consolidation_net(consolidated_input)
        else:
            final_retrieved = retrieved_patterns[0]
        
        # Calculate similarity scores
        similarities = F.cosine_similarity(
            query.unsqueeze(1), 
            self.memory_patterns.unsqueeze(0), 
            dim=-1
        )
        
        # Get top-k most similar patterns
        top_scores, top_indices = torch.topk(similarities, min(top_k, similarities.size(-1)), dim=-1)
        
        # Update access counts
        for idx in top_indices.flatten():
            self.access_counts[idx] += 1
        
        return final_retrieved, top_scores
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get statistics about stored memory patterns"""
        return {
            'total_patterns': self.memory_patterns.size(0),
            'pattern_dimension': self.memory_patterns.size(-1) if self.memory_patterns.size(0) > 0 else 0,
            'avg_access_count': self.access_counts.mean().item() if self.access_counts.size(0) > 0 else 0,
            'memory_utilization': (self.access_counts > 0).float().mean().item() if self.access_counts.size(0) > 0 else 0
        }

class LSHNNetwork(nn.Module):
    """
    Main LSHN (Latent Structured Hopfield Networks) implementation
    
    Combines episodic memory encoding with Hopfield-based associative retrieval
    for biologically-inspired memory formation and consolidation.
    """
    
    def __init__(self, config: LSHNConfig):
        super().__init__()
        self.config = config
        
        # Core components
        self.encoder = EpisodicMemoryEncoder(config)
        
        if HOPFIELD_AVAILABLE:
            self.associative_memory = HopfieldAssociativeMemory(config)
        else:
            self.associative_memory = None
            logger.warning("Hopfield layers not available - using simplified memory")
        
        # Episodic trace formation
        self.trace_formation = nn.Sequential(
            nn.Linear(config.pattern_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim, config.pattern_dim),
            nn.Tanh()  # Bound episodic traces
        )
        
        # Memory consolidation scoring
        self.consolidation_scorer = nn.Sequential(
            nn.Linear(config.pattern_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Statistics tracking
        self.stats = {
            'patterns_encoded': 0,
            'patterns_stored': 0,
            'retrievals_performed': 0,
            'consolidations_triggered': 0
        }
    
    def forward(
        self, 
        embeddings: torch.Tensor, 
        store_patterns: bool = True,
        retrieve_similar: bool = True,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through LSHN network
        
        Args:
            embeddings: Input embeddings [batch_size, embedding_dim]
            store_patterns: Whether to store new patterns in memory
            retrieve_similar: Whether to retrieve similar patterns
            importance_scores: Optional importance scores for weighting
        
        Returns:
            Dictionary containing encoded patterns, retrieved memories, etc.
        """
        batch_size = embeddings.size(0)
        
        # Encode into episodic patterns
        episodic_patterns = self.encoder(embeddings)
        self.stats['patterns_encoded'] += batch_size
        
        # Form episodic traces
        episodic_traces = self.trace_formation(episodic_patterns)
        
        results = {
            'episodic_patterns': episodic_patterns,
            'episodic_traces': episodic_traces,
            'retrieved_patterns': None,
            'similarity_scores': None,
            'consolidation_scores': None
        }
        if self.associative_memory and retrieve_similar:
            # Retrieve similar patterns
            retrieved_patterns, similarity_scores = self.associative_memory.retrieve_pattern(episodic_patterns)
            results['retrieved_patterns'] = retrieved_patterns
            results['similarity_scores'] = similarity_scores
            self.stats['retrievals_performed'] += batch_size
            
            # Calculate consolidation scores
            pattern_pairs = torch.cat([episodic_patterns, retrieved_patterns], dim=-1)
            consolidation_scores = self.consolidation_scorer(pattern_pairs)
            results['consolidation_scores'] = consolidation_scores
        
        if self.associative_memory and store_patterns:
            # Store new patterns in associative memory
            patterns_stored_count = 0
            print(f"LSHN DEBUG: About to store {batch_size} patterns with consolidation_strength={self.config.consolidation_strength}")
            for i in range(batch_size):
                weight = importance_scores[i].item() if importance_scores is not None else 1.0
                print(f"LSHN DEBUG: Pattern {i}: weight={weight:.3f}, consolidation_strength={self.config.consolidation_strength}")
                if weight > self.config.consolidation_strength:  # Only store important patterns
                    self.associative_memory.store_pattern(episodic_patterns[i], weight)
                    self.stats['patterns_stored'] += 1
                    patterns_stored_count += 1
                    print(f"LSHN DEBUG: Stored pattern {i} with weight {weight:.3f}")
                else:
                    print(f"LSHN DEBUG: Rejected pattern {i} (weight {weight:.3f} <= threshold {self.config.consolidation_strength})")
            print(f"LSHN DEBUG: Total patterns stored in this batch: {patterns_stored_count}")
            print(f"LSHN DEBUG: Total memory patterns now: {self.associative_memory.memory_patterns.size(0) if self.associative_memory else 0}")
        
        return results
    
    def consolidate_memories(
        self, 
        memory_embeddings: List[torch.Tensor],
        importance_scores: List[float],
        association_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Consolidate memories during dream cycles using LSHN
        
        Args:
            memory_embeddings: List of memory embeddings to consolidate
            importance_scores: Importance scores for each memory
            association_threshold: Threshold for creating associations
            
        Returns:
            Consolidation results and statistics
        """
        if not memory_embeddings:
            return {
                'consolidated_memories': 0,
                'associations_created': 0,
                'consolidation_strength': 0.0
            }
        
        threshold = association_threshold or self.config.association_threshold
          # Stack embeddings for batch processing
        embeddings = torch.stack(memory_embeddings)
        importance_weights = torch.tensor(importance_scores, dtype=torch.float32)
        
        # First pass: Store patterns in memory
        print("LSHN DEBUG: First pass - storing patterns")
        store_results = self.forward(
            embeddings,
            store_patterns=True,
            retrieve_similar=False,  # Don't retrieve on first pass
            importance_scores=importance_weights
        )
          # Second pass: Retrieve similarities with stored patterns
        print("LSHN DEBUG: Second pass - retrieving similarities")
        results = self.forward(
            embeddings,
            store_patterns=False,  # Don't store again
            retrieve_similar=True,  # Now retrieve similarities
            importance_scores=importance_weights
        )
        
        # Count associations based on similarity scores
        associations_created = 0
        if results['similarity_scores'] is not None:
            print(f"LSHN DEBUG: similarity_scores shape: {results['similarity_scores'].shape}")
            print(f"LSHN DEBUG: similarity_scores range: [{results['similarity_scores'].min():.3f}, {results['similarity_scores'].max():.3f}]")
            print(f"LSHN DEBUG: association_threshold: {threshold}")
            high_similarity = (results['similarity_scores'] > threshold).sum().item()
            print(f"LSHN DEBUG: scores above threshold: {high_similarity}")
            associations_created = high_similarity
        else:
            print("LSHN DEBUG: similarity_scores is None")
        
        # Calculate consolidation strength
        consolidation_strength = 0.0
        if results['consolidation_scores'] is not None:
            consolidation_strength = (results['consolidation_scores'] * importance_weights.unsqueeze(-1)).sum().item()
        
        self.stats['consolidations_triggered'] += 1
        
        return {
            'consolidated_memories': len(memory_embeddings),
            'associations_created': associations_created,
            'consolidation_strength': consolidation_strength,
            'episodic_patterns_formed': results['episodic_patterns'].size(0),
            'memory_stats': self.associative_memory.get_memory_stats() if self.associative_memory else {}
        }
    
    def get_network_stats(self) -> Dict[str, Any]:
        """Get comprehensive network statistics"""
        stats = self.stats.copy()
        
        if self.associative_memory:
            stats.update(self.associative_memory.get_memory_stats())
        
        # Add parameter count
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        stats.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'hopfield_available': HOPFIELD_AVAILABLE
        })
        
        return stats
    
    def reset_memory(self):
        """Reset stored memory patterns"""
        if self.associative_memory:
            self.associative_memory.memory_patterns = torch.zeros(0, self.config.pattern_dim)
            self.associative_memory.pattern_weights = torch.zeros(0)
            self.associative_memory.access_counts = torch.zeros(0)
        
        # Reset statistics
        self.stats = {
            'patterns_encoded': 0,
            'patterns_stored': 0,
            'retrievals_performed': 0,
            'consolidations_triggered': 0
        }

class LSHNTrainer:
    """
    Training manager for LSHN networks
    """
    
    def __init__(self, network: LSHNNetwork, config: LSHNConfig):
        self.network = network
        self.config = config
        self.optimizer = torch.optim.Adam(network.parameters(), lr=config.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=10)
        
    def train_step(
        self, 
        embeddings: torch.Tensor, 
        target_patterns: Optional[torch.Tensor] = None,
        importance_scores: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Single training step for LSHN
        
        Args:
            embeddings: Input embeddings
            target_patterns: Optional target patterns for supervised learning
            importance_scores: Optional importance scores
            
        Returns:
            Training metrics
        """
        self.optimizer.zero_grad()
        
        # Forward pass
        results = self.network.forward(
            embeddings,
            store_patterns=False,  # Don't store during training
            retrieve_similar=True,
            importance_scores=importance_scores
        )
        
        # Calculate loss
        total_loss = 0.0
        metrics = {}
        
        # Reconstruction loss if targets provided
        if target_patterns is not None:
            recon_loss = F.mse_loss(results['episodic_patterns'], target_patterns)
            total_loss += recon_loss
            metrics['reconstruction_loss'] = recon_loss.item()
        
        # Consistency loss between patterns and traces
        consistency_loss = F.mse_loss(results['episodic_patterns'], results['episodic_traces'])
        total_loss += consistency_loss * 0.1
        metrics['consistency_loss'] = consistency_loss.item()
        
        # Importance-weighted loss
        if importance_scores is not None:
            importance_weights = importance_scores.unsqueeze(-1)
            weighted_loss = (F.mse_loss(results['episodic_patterns'], results['episodic_traces'], reduction='none') * importance_weights).mean()
            total_loss += weighted_loss * 0.1
            metrics['importance_weighted_loss'] = weighted_loss.item()
        
        # Backward pass
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1.0)
            self.optimizer.step()
        
        metrics['total_loss'] = total_loss.item()
        return metrics
