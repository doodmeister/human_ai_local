"""
Dream State Processor - Automated Memory Consolidation Pipeline

This module implements a sophisticated dream-state processing system that mimics
human sleep cycles for memory consolidation, neural replay, and cognitive maintenance.

Features:
- Scheduled consolidation cycles
- Meta-cognitive memory analysis
- Cluster-based memory organization
- DPAD neural network replay during dreams
- Adaptive consolidation timing
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import torch

# Try to import HDBSCAN from hdbscan library, fall back if not available
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
    HDBSCAN = hdbscan.HDBSCAN
except ImportError:
    HDBSCAN_AVAILABLE = False
    HDBSCAN = None

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix
import schedule
import time
from threading import Thread

from src.core.cognitive_tick import CognitiveStep, CognitiveTick

from src.learning.learning_law import clamp01, utility_score

logger = logging.getLogger(__name__)

@dataclass
class DreamCycle:
    """Configuration for dream processing cycles"""
    cycle_type: str  # 'light', 'deep', 'rem'
    duration_minutes: int
    consolidation_threshold: float
    replay_intensity: float
    association_strength: float

@dataclass
class ConsolidationCandidate:
    """Memory candidate for consolidation"""
    memory_id: str
    content: Any
    importance_score: float
    access_frequency: int
    emotional_salience: float
    temporal_relevance: float
    cluster_id: Optional[int] = None

class DreamProcessor:
    """
    Advanced dream-state processor for memory consolidation
    
    Implements multiple types of sleep cycles with different consolidation strategies:
    - Light Sleep: Basic STM decay and maintenance
    - Deep Sleep: Intensive memory consolidation and clustering    - REM Sleep: Creative associations and neural replay
    """
    
    def __init__(
        self,
        memory_system,
        meta_cognition_engine=None,
        enable_scheduling: bool = True,
        consolidation_threshold: float = 0.6,
        cluster_min_size: int = 2,
        neural_integration_manager=None
    ):
        """
        Initialize dream processor
        
        Args:
            memory_system: Reference to the memory system
            meta_cognition_engine: Reference to meta-cognition engine
            enable_scheduling: Enable automatic scheduling
            consolidation_threshold: Minimum score for consolidation
            cluster_min_size: Minimum cluster size for grouping
            neural_integration_manager: Neural integration manager for DPAD replay
        """
        self.memory_system = memory_system
        self.meta_cognition = meta_cognition_engine
        self.neural_integration = neural_integration_manager
        self.enable_scheduling = enable_scheduling
        self.consolidation_threshold = consolidation_threshold
        self.cluster_min_size = cluster_min_size
        
        # Dream cycle configurations
        self.dream_cycles = {
            'light': DreamCycle('light', 5, 0.4, 0.2, 0.3),
            'deep': DreamCycle('deep', 15, 0.7, 0.6, 0.5),
            'rem': DreamCycle('rem', 10, 0.5, 0.8, 0.7)
        }
        
        # Processing state
        self.last_dream_cycle = None
        self.consolidation_history = []
        self.is_dreaming = False
        self.dream_statistics = {
            'total_cycles': 0,
            'memories_consolidated': 0,
            'associations_created': 0,
            'clusters_formed': 0
        }
        
        # Scheduling
        self.scheduler_thread = None
        if enable_scheduling:
            self._setup_dream_schedule()
        
        logger.info("Dream processor initialized")
    
    def _setup_dream_schedule(self):
        """Setup automatic dream cycle scheduling"""
        # Light sleep every 2 hours during active period
        schedule.every(2).hours.do(self._scheduled_light_sleep)
        
        # Deep sleep every 8 hours
        schedule.every(8).hours.do(self._scheduled_deep_sleep)
        
        # REM sleep every 6 hours
        schedule.every(6).hours.do(self._scheduled_rem_sleep)
        
        # Start scheduler thread
        self.scheduler_thread = Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("Dream cycle scheduling enabled")
    
    def _run_scheduler(self):
        """Run the dream cycle scheduler"""
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute

    def _launch_dream_cycle(self, cycle_type: str) -> None:
        """Start a dream cycle, handling both async and non-async contexts."""
        if self.is_dreaming:
            logger.debug("Dream cycle already in progress; skipping scheduled start")
            return

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(self.enter_dream_cycle(cycle_type))
            return

        def _run_cycle() -> None:
            try:
                asyncio.run(self.enter_dream_cycle(cycle_type))
            except Exception:  # pragma: no cover - logged for visibility
                logger.exception(f"Dream cycle {cycle_type} failed")

        thread = Thread(target=_run_cycle, daemon=True)
        thread.start()
    
    def _scheduled_light_sleep(self):
        """Scheduled light sleep cycle"""
        self._launch_dream_cycle('light')
    
    def _scheduled_deep_sleep(self):
        """Scheduled deep sleep cycle"""
        self._launch_dream_cycle('deep')
    
    def _scheduled_rem_sleep(self):
        """Scheduled REM sleep cycle"""
        self._launch_dream_cycle('rem')
    
    async def enter_dream_cycle(self, cycle_type: str = 'deep') -> Dict[str, Any]:
        """
        Enter a specific dream cycle for memory consolidation
        
        Args:
            cycle_type: Type of dream cycle ('light', 'deep', 'rem')
        
        Returns:
            Dream cycle results
        """
        tick = CognitiveTick(owner="dream_processor", kind=f"dream_cycle:{cycle_type}")
        tick.state["cycle_type"] = cycle_type

        # Perceive
        tick.assert_step(CognitiveStep.PERCEIVE)
        if self.is_dreaming:
            logger.warning("Already in dream state, skipping cycle")
            return {"error": "already_dreaming"}

        cycle_config = self.dream_cycles.get(cycle_type, self.dream_cycles['deep'])
        tick.state["cycle_duration_minutes"] = cycle_config.duration_minutes
        tick.advance(CognitiveStep.PERCEIVE)

        # Update STM (enter dream-state / reserve resources)
        tick.assert_step(CognitiveStep.UPDATE_STM)
        self.is_dreaming = True
        logger.info(f"Entering {cycle_type} dream cycle for {cycle_config.duration_minutes} minutes")
        tick.advance(CognitiveStep.UPDATE_STM)

        try:
            results = await self._process_dream_cycle(cycle_config, tick=tick)

            # Consolidate (persist dream-cycle state + stats)
            tick.assert_step(CognitiveStep.CONSOLIDATE)
            self.last_dream_cycle = datetime.now()
            self.dream_statistics['total_cycles'] += 1
            tick.finish()

            return results

        finally:
            self.is_dreaming = False
    
    async def _process_dream_cycle(self, cycle_config: DreamCycle, tick: CognitiveTick | None = None) -> Dict[str, Any]:
        """
        Process a complete dream cycle
        
        Args:
            cycle_config: Dream cycle configuration
        
        Returns:
            Processing results
        """
        start_time = datetime.now()
        results = {
            "cycle_type": cycle_config.cycle_type,
            "start_time": start_time,
            "duration_minutes": cycle_config.duration_minutes
        }

        if tick is not None:
            tick.assert_step(CognitiveStep.RETRIEVE)
        
        # Retrieve
        # Phase 1: Identify consolidation candidates
        candidates = await self._identify_consolidation_candidates(cycle_config)
        results["candidates_identified"] = len(candidates)
        
        # Phase 2: Cluster related memories
        if len(candidates) >= self.cluster_min_size:
            clusters = await self._cluster_memories(candidates)
            results["clusters_formed"] = len(clusters)
            self.dream_statistics['clusters_formed'] += len(clusters)
        else:
            clusters = []
            results["clusters_formed"] = 0

        if tick is not None:
            tick.state["candidates_identified"] = results["candidates_identified"]
            tick.state["clusters_formed"] = results["clusters_formed"]
            tick.advance(CognitiveStep.RETRIEVE)
        
        # Decide
        if tick is not None:
            tick.assert_step(CognitiveStep.DECIDE)

        # Phase 3: Meta-cognitive evaluation
        if self.meta_cognition:
            evaluated_candidates = await self._meta_cognitive_evaluation(candidates)
        else:
            evaluated_candidates = candidates

        if tick is not None:
            tick.mark_decided({
                "evaluated_candidates": len(evaluated_candidates),
                "has_meta_cognition": bool(self.meta_cognition),
            })
            tick.advance(CognitiveStep.DECIDE)

        # Act
        if tick is not None:
            tick.assert_step(CognitiveStep.ACT)

        # Phase 4: Create associations (before consolidation, while memories are still in STM)
        associations = await self._create_dream_associations(candidates, clusters, cycle_config)
        results["associations_created"] = associations
        self.dream_statistics['associations_created'] += associations
        
        # Phase 5: Consolidate memories (after associations are created)
        consolidated = await self._consolidate_memories(evaluated_candidates, cycle_config)
        results["memories_consolidated"] = consolidated
        self.dream_statistics['memories_consolidated'] += consolidated
        
        # Phase 6: Neural replay (if applicable)
        if cycle_config.cycle_type == 'rem':
            replay_results = await self._neural_replay(candidates, cycle_config)
            results["neural_replay"] = replay_results
        
        # Phase 7: Cleanup and maintenance
        cleanup_results = await self._dream_cleanup(cycle_config)
        results["cleanup"] = cleanup_results

        if tick is not None:
            tick.advance(CognitiveStep.ACT)
        
        # Reflect
        if tick is not None:
            tick.assert_step(CognitiveStep.REFLECT)

        end_time = datetime.now()
        results["actual_duration"] = (end_time - start_time).total_seconds() / 60
        results["end_time"] = end_time
        
        # Store in consolidation history
        self.consolidation_history.append(results)

        if tick is not None:
            tick.state["dream_results"] = {
                "associations_created": associations,
                "memories_consolidated": consolidated,
                "cycle_type": cycle_config.cycle_type,
            }
            tick.advance(CognitiveStep.REFLECT)
        
        logger.info(f"Dream cycle completed: {results}")
        return results
    
    async def _identify_consolidation_candidates(
        self,
        cycle_config: DreamCycle
    ) -> List[ConsolidationCandidate]:
        """
        Identify memories that are candidates for consolidation
        
        Args:
            cycle_config: Dream cycle configuration
        
        Returns:
            List of consolidation candidates
        """
        candidates = []
        current_time = datetime.now()
        
        # Get all STM items
        stm_items = self.memory_system.stm.get_all_memories()
        
        for item in stm_items:
            # Calculate various scores
            importance_score = item.importance
            access_frequency = item.access_count
            
            # Temporal relevance (newer memories get higher scores)
            age_hours = (current_time - item.encoding_time).total_seconds() / 3600
            temporal_relevance = max(0.1, 1.0 - (age_hours / 24))  # Decay over 24 hours
            
            # Emotional salience
            emotional_salience = abs(item.emotional_valence)
            
            # Combined score
            combined_score = (
                importance_score * 0.4 +
                min(access_frequency / 10, 1.0) * 0.3 +
                temporal_relevance * 0.2 +
                emotional_salience * 0.1
            )

            # Phase 7: utility-based learning law.
            # Candidate selection is a utility thresholding step; cost is treated as ~0 during dream retrieval.
            utility = utility_score(benefit=combined_score, cost=0.0)
            
            # Check if meets threshold for this cycle type
            if utility >= cycle_config.consolidation_threshold:
                candidate = ConsolidationCandidate(
                    memory_id=item.id,
                    content=item.content,
                    importance_score=clamp01(utility),
                    access_frequency=access_frequency,
                    emotional_salience=emotional_salience,
                    temporal_relevance=temporal_relevance
                )
                candidates.append(candidate)
        
        # Sort by importance
        candidates.sort(key=lambda x: x.importance_score, reverse=True)
        
        logger.debug(f"Identified {len(candidates)} consolidation candidates")
        return candidates
    
    async def _cluster_memories(
        self,
        candidates: List[ConsolidationCandidate]
    ) -> List[List[ConsolidationCandidate]]:
        """
        Cluster related memories for coherent consolidation
        
        Args:        candidates: List of consolidation candidates
        
        Returns:
            List of memory clusters
        """
        if len(candidates) < self.cluster_min_size:
            # If not enough candidates for clustering, create simple pairs
            if len(candidates) >= 2:
                clusters = []
                for i in range(0, len(candidates), 2):
                    if i + 1 < len(candidates):
                        clusters.append([candidates[i], candidates[i + 1]])
                    else:
                        clusters.append([candidates[i]])
                return clusters
            return []
        
        try:
            # Extract text content for clustering
            texts = []
            for candidate in candidates:
                if isinstance(candidate.content, dict):
                    # Extract text from structured content
                    text_parts = []
                    for value in candidate.content.values():
                        if isinstance(value, str):
                            text_parts.append(value)
                        elif isinstance(value, dict):
                            text_parts.extend([str(v) for v in value.values()])
                    texts.append(" ".join(text_parts))
                else:
                    texts.append(str(candidate.content))            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
            X = vectorizer.fit_transform(texts)
            
            # Convert sparse matrix to dense array  
            # Convert sparse matrix to dense array
            if isinstance(X, csr_matrix):
                X_dense = X.toarray()  # type: ignore
            else:
                X_dense = X
              # Apply HDBSCAN clustering if available
            if HDBSCAN_AVAILABLE and HDBSCAN is not None:
                try:
                    clusterer = HDBSCAN(min_cluster_size=self.cluster_min_size, metric='cosine')
                    cluster_labels = clusterer.fit_predict(X_dense)
                except ValueError as e:
                    if "Unrecognized metric" in str(e):
                        # Fallback to euclidean metric (always supported)
                        logger.debug("Cosine metric not supported, falling back to euclidean")
                        clusterer = HDBSCAN(min_cluster_size=self.cluster_min_size, metric='euclidean')
                        cluster_labels = clusterer.fit_predict(X_dense)
                    else:
                        raise e
            else:
                # fallback: all in one cluster (or implement another strategy)
                cluster_labels = [0] * len(candidates)

            # Group candidates by cluster
            clusters = {}
            for i, label in enumerate(cluster_labels):
                if label != -1:  # -1 is noise in HDBSCAN
                    candidates[i].cluster_id = label
                    if label not in clusters:
                        clusters[label] = []
                    clusters[label].append(candidates[i])
            
            cluster_list = list(clusters.values())
            logger.debug(f"Formed {len(cluster_list)} memory clusters")
            return cluster_list
        
        except Exception as e:
            logger.error(f"Error in memory clustering: {e}")
            return []
    
    async def _meta_cognitive_evaluation(
        self,
        candidates: List[ConsolidationCandidate]
    ) -> List[ConsolidationCandidate]:
        """
        Use meta-cognition to evaluate memory importance
        
        Args:
            candidates: List of consolidation candidates
        
        Returns:
            Re-evaluated candidates
        """
        if not self.meta_cognition:
            return candidates
        
        try:
            # Prepare context for meta-cognitive analysis
            memory_summaries = []
            for candidate in candidates:
                summary = {
                    "id": candidate.memory_id,
                    "content_preview": str(candidate.content)[:200],
                    "importance": candidate.importance_score,
                    "access_count": candidate.access_frequency,
                    "emotional_salience": candidate.emotional_salience
                }
                memory_summaries.append(summary)
            
            # Get meta-cognitive assessment
            assessment = await self.meta_cognition.evaluate_memories_for_consolidation(
                memory_summaries
            )
            
            # Update candidate scores based on assessment
            if assessment and 'memory_evaluations' in assessment:
                for i, candidate in enumerate(candidates):
                    if i < len(assessment['memory_evaluations']):
                        eval_data = assessment['memory_evaluations'][i]
                        # Adjust importance based on meta-cognitive insight
                        meta_importance = eval_data.get('consolidation_priority', candidate.importance_score)
                        candidate.importance_score = (candidate.importance_score + meta_importance) / 2
            
            logger.debug("Meta-cognitive evaluation completed")
            return candidates
        
        except Exception as e:
            logger.error(f"Error in meta-cognitive evaluation: {e}")
            return candidates
    
    async def _consolidate_memories(
        self,
        candidates: List[ConsolidationCandidate],
        cycle_config: DreamCycle
    ) -> int:
        """
        Consolidate selected memories to LTM
        
        Args:
            candidates: Evaluated consolidation candidates
            cycle_config: Dream cycle configuration
        
        Returns:
            Number of memories consolidated
        """
        consolidated_count = 0
        
        for candidate in candidates:
            try:
                # Get the actual memory item from STM
                stm_item = self.memory_system.stm.retrieve(candidate.memory_id)
                if not stm_item:
                    continue
                
                # Store in LTM with enhanced metadata
                success = self.memory_system.ltm.store(
                    memory_id=candidate.memory_id,
                    content=candidate.content,
                    memory_type="consolidated_episodic",
                    importance=candidate.importance_score,
                    emotional_valence=stm_item.emotional_valence,
                    tags=["dream_consolidated", cycle_config.cycle_type],
                    associations=stm_item.associations
                )
                
                if success:
                    # Remove from STM
                    self.memory_system.stm.remove_item(candidate.memory_id)
                    consolidated_count += 1
                    logger.debug(f"Consolidated memory {candidate.memory_id}")
            
            except Exception as e:
                logger.error(f"Error consolidating memory {candidate.memory_id}: {e}")
        
        logger.info(f"Consolidated {consolidated_count} memories during {cycle_config.cycle_type} cycle")
        return consolidated_count
    
    async def _create_dream_associations(
        self,
        candidates: List[ConsolidationCandidate],
        clusters: List[List[ConsolidationCandidate]],
        cycle_config: DreamCycle
    ) -> int:
        """
        Create associative links between memories during dream processing
        
        Args:
            candidates: All consolidation candidates
            clusters: Memory clusters
            cycle_config: Dream cycle configuration
        
        Returns:
            Number of associations created        """
        associations_created = 0
        
        logger.debug(f"Creating associations for {len(candidates)} candidates and {len(clusters)} clusters")
        
        try:            # Create intra-cluster associations
            logger.debug(f"Processing {len(clusters)} clusters for intra-cluster associations")
            for cluster in clusters:
                for i, candidate1 in enumerate(cluster):
                    for candidate2 in cluster[i+1:]:
                        # Add mutual associations
                        stm_item1 = self.memory_system.stm.retrieve(candidate1.memory_id)
                        stm_item2 = self.memory_system.stm.retrieve(candidate2.memory_id)
                        
                        if stm_item1 and stm_item2:
                            if candidate2.memory_id not in stm_item1.associations:
                                stm_item1.associations.append(candidate2.memory_id)
                                associations_created += 1
                            
                            if candidate1.memory_id not in stm_item2.associations:
                                stm_item2.associations.append(candidate1.memory_id)
                                associations_created += 1            # Create temporal associations (enhanced for all cycle types)
            logger.debug(f"Processing temporal associations for {cycle_config.cycle_type} cycle with {len(candidates)} candidates")
            
            # Adjust association criteria based on cycle type
            if cycle_config.cycle_type == 'rem':
                time_window = 7200  # 2 hours for REM (creative associations)
                emotional_threshold = 0.3
                importance_threshold = 0.3
            elif cycle_config.cycle_type == 'light':
                time_window = 3600  # 1 hour for light sleep (recent memories)
                emotional_threshold = 0.4  
                importance_threshold = 0.2
            else:  # deep sleep
                time_window = 1800  # 30 minutes for deep sleep (very recent)
                emotional_threshold = 0.5
                importance_threshold = 0.4
            
            for i, candidate1 in enumerate(candidates):
                for candidate2 in candidates[i+1:]:
                    # Check temporal proximity
                    stm_item1 = self.memory_system.stm.retrieve(candidate1.memory_id)
                    stm_item2 = self.memory_system.stm.retrieve(candidate2.memory_id)
                    
                    if stm_item1 and stm_item2:
                        time_diff = abs((stm_item1.encoding_time - stm_item2.encoding_time).total_seconds())
                        
                        # Enhanced association criteria with cycle-specific thresholds
                        should_associate = False
                        association_reasons = []
                        
                        # Criteria 1: Temporal proximity (cycle-specific window)
                        if time_diff < time_window:
                            should_associate = True
                            association_reasons.append(f"temporal_{time_diff}s")
                        
                        # Criteria 2: High emotional salience 
                        if (candidate1.emotional_salience >= emotional_threshold and 
                            candidate2.emotional_salience >= emotional_threshold):
                            should_associate = True
                            association_reasons.append("emotional")
                        
                        # Criteria 3: Similar importance scores
                        importance_diff = abs(candidate1.importance_score - candidate2.importance_score)
                        if importance_diff < importance_threshold:
                            should_associate = True
                            association_reasons.append("importance_similarity")
                        
                        # Criteria 4: Both have high importance
                        if (candidate1.importance_score > 0.6 and 
                            candidate2.importance_score > 0.6):
                            should_associate = True
                            association_reasons.append("high_importance")
                        
                        if should_associate:
                            logger.debug(f"Creating association between {candidate1.memory_id} and {candidate2.memory_id}: {', '.join(association_reasons)}")
                            
                            if candidate2.memory_id not in stm_item1.associations:
                                stm_item1.associations.append(candidate2.memory_id)
                                associations_created += 1
                            
                            if candidate1.memory_id not in stm_item2.associations:
                                stm_item2.associations.append(candidate1.memory_id)
                                associations_created += 1
        
        except Exception as e:
            logger.error(f"Error creating dream associations: {e}")
        
        logger.debug(f"Created {associations_created} associations during dream processing")
        return associations_created
    
    async def _neural_replay(
        self,
        candidates: List[ConsolidationCandidate],
        cycle_config: DreamCycle
    ) -> Dict[str, Any]:
        """
        Simulate neural replay during REM sleep
        
        Args:
            candidates: Consolidation candidates
            cycle_config: Dream cycle configuration
          Returns:
            Replay results
        """
        replay_results = {
            "memories_replayed": 0,
            "strength_increased": 0.0,
            "patterns_reinforced": 0
        }
        
        try:
            # Phase 7: utility-based learning law.
            # Replay is applied when reinforcement utility is sufficiently positive.
            # Benefit: candidate utility; Cost: replay intensity (compute/energy proxy).
            replay_cost = clamp01(cycle_config.replay_intensity)
            for candidate in candidates:
                u = utility_score(benefit=candidate.importance_score, cost=0.35 * replay_cost)
                if u < 0.35:
                    continue
                stm_item = self.memory_system.stm.retrieve(candidate.memory_id)
                if stm_item:
                    # Strengthen memory through replay
                    strength_boost = float(cycle_config.replay_intensity) * 0.1 * clamp01(u)
                    stm_item.importance = min(1.0, stm_item.importance + strength_boost)
                    
                    replay_results["memories_replayed"] += 1
                    replay_results["strength_increased"] += strength_boost
              # Integrate with DPAD neural network for actual replay
            await self._dpad_neural_replay(candidates, cycle_config, replay_results)
            
            logger.debug(f"Neural replay completed: {replay_results}")
        
        except Exception as e:
            logger.error(f"Error in neural replay: {e}")
        
        return replay_results
    
    async def _dpad_neural_replay(
        self,
        candidates: List[ConsolidationCandidate],
        cycle_config: DreamCycle,
        replay_results: Dict[str, Any]
    ) -> None:
        """
        Perform DPAD neural network replay during dream cycles
        
        Args:
            candidates: Consolidation candidates
            cycle_config: Dream cycle configuration
            replay_results: Results dict to update
        """
        if not self.neural_integration:
            logger.debug("Neural integration not available for DPAD replay")
            return
        
        try:
            # Extract memory embeddings and importance scores
            memory_embeddings = []
            importance_scores = []
            
            for candidate in candidates:
                stm_item = self.memory_system.stm.retrieve(candidate.memory_id)
                if stm_item and hasattr(stm_item, 'embedding') and stm_item.embedding is not None:
                    # Convert embedding to torch tensor
                    if isinstance(stm_item.embedding, np.ndarray):
                        embedding_tensor = torch.from_numpy(stm_item.embedding).float()
                    else:
                        # Assume it's already a tensor or convertible
                        embedding_tensor = torch.tensor(stm_item.embedding, dtype=torch.float32)
                    
                    memory_embeddings.append(embedding_tensor)
                    importance_scores.append(candidate.importance_score)
            
            if not memory_embeddings:
                logger.debug("No embeddings available for DPAD neural replay")
                return
            
            # Perform neural replay through DPAD network
            neural_replay_results = await self.neural_integration.neural_memory_replay(
                memory_embeddings,
                importance_scores,
                attention_context={
                    'cycle_type': cycle_config.cycle_type,
                    'replay_intensity': cycle_config.replay_intensity
                }
            )
            
            # Update replay results with neural network information
            if 'error' not in neural_replay_results:
                replay_results.update({
                    'neural_replay_enabled': True,
                    'neural_consolidation_strength': neural_replay_results.get('consolidation_strength', 0.0),
                    'neural_reconstruction_quality': neural_replay_results.get('reconstruction_quality', 0.0),
                    'neural_replayed_memories': neural_replay_results.get('replayed_memories', 0)
                })
                
                # Apply consolidation effects back to memories.
                # Phase 7: importance strengthening is utility-gated (benefit from consolidation vs replay cost).
                consolidation_strength = float(neural_replay_results.get('consolidation_strength', 0.0) or 0.0)
                replay_cost = clamp01(getattr(cycle_config, "replay_intensity", 0.5))
                for candidate in candidates:
                    stm_item = self.memory_system.stm.retrieve(candidate.memory_id)
                    if not stm_item:
                        continue

                    benefit = clamp01(clamp01(consolidation_strength) * clamp01(candidate.importance_score))
                    u = utility_score(benefit=benefit, cost=clamp01(0.35 * replay_cost))
                    if u <= 0.0:
                        continue

                    neural_boost = clamp01(consolidation_strength) * 0.1 * clamp01(u)
                    if neural_boost <= 0.0:
                        continue

                    stm_item.importance = min(1.0, stm_item.importance + neural_boost)
                    replay_results["strength_increased"] += neural_boost
                
                logger.debug(f"DPAD neural replay: {neural_replay_results.get('replayed_memories', 0)} memories, "
                           f"quality: {neural_replay_results.get('reconstruction_quality', 0.0):.3f}")
            else:
                logger.warning(f"DPAD neural replay error: {neural_replay_results['error']}")
                replay_results['neural_replay_error'] = neural_replay_results['error']
                
        except Exception as e:
            logger.error(f"Error in DPAD neural replay: {e}")
            replay_results['neural_replay_error'] = str(e)
    
    async def _dream_cleanup(self, cycle_config: DreamCycle) -> Dict[str, Any]:
        """
        Perform memory cleanup and maintenance during dream cycle
        
        Args:
            cycle_config: Dream cycle configuration
        
        Returns:
            Cleanup results
        """
        cleanup_results = {
            "weak_memories_removed": 0,
            "duplicate_associations_cleaned": 0,
            "decay_applied": False
        }
        
        try:
            # Remove very weak memories
            weak_threshold = 0.1
            items_to_remove = []
            
            all_memories = self.memory_system.stm.get_all_memories()
            for item in all_memories:
                if item.importance < weak_threshold and item.access_count == 0:
                    items_to_remove.append(item.id)
            
            for item_id in items_to_remove:
                self.memory_system.stm.remove_item(item_id)
                cleanup_results["weak_memories_removed"] += 1
            
            # Clean duplicate associations
            for item in all_memories:
                original_count = len(item.associations)
                item.associations = list(set(item.associations))  # Remove duplicates
                cleaned_count = original_count - len(item.associations)
                cleanup_results["duplicate_associations_cleaned"] += cleaned_count
            
            # Apply gentle decay during light sleep
            if cycle_config.cycle_type == 'light':
                forgotten_ids = self.memory_system.stm.decay_memories()
                cleanup_results["decay_applied"] = True
                cleanup_results["items_decayed"] = len(forgotten_ids)
            
            logger.debug(f"Dream cleanup completed: {cleanup_results}")
        
        except Exception as e:
            logger.error(f"Error in dream cleanup: {e}")
        
        return cleanup_results
    
    def get_dream_statistics(self) -> Dict[str, Any]:
        """Get comprehensive dream processing statistics"""
        return {
            "statistics": self.dream_statistics.copy(),
            "last_dream_cycle": self.last_dream_cycle,
            "consolidation_history_length": len(self.consolidation_history),
            "is_currently_dreaming": self.is_dreaming,
            "scheduling_enabled": self.enable_scheduling,
            "recent_cycles": self.consolidation_history[-5:] if self.consolidation_history else []
        }
    
    def force_dream_cycle(self, cycle_type: str = 'deep') -> None:
        """Force an immediate dream cycle (for testing/manual triggering)"""
        if self.is_dreaming:
            logger.warning("Cannot force dream cycle - already dreaming")
            return
        self._launch_dream_cycle(cycle_type)
    
    def shutdown(self):
        """Shutdown the dream processor"""
        if self.scheduler_thread and self.scheduler_thread.is_alive():
            # Note: In a production system, you'd want a more graceful shutdown
            logger.info("Dream processor shutting down")
        
        logger.info("Dream processor shutdown complete")
