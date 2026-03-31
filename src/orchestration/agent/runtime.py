from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

from ...cognition.attention.attention_mechanism import AttentionMechanism
from ...cognition.processing.sensory import SensoryInterface, SensoryProcessor
from ...core.config import CognitiveConfig
from ...memory import MemorySystem, MemorySystemConfig


logger = logging.getLogger(__name__)

DreamProcessor = None
PerformanceOptimizer = None


def _lazy_import_dream():
    global DreamProcessor
    if DreamProcessor is None:
        from ...cognition.processing.dream import DreamProcessor as _dream_processor

        DreamProcessor = _dream_processor
    return DreamProcessor


def _lazy_import_optimizer():
    global PerformanceOptimizer
    if PerformanceOptimizer is None:
        from ...optimization.performance_optimizer import PerformanceOptimizer as _performance_optimizer

        PerformanceOptimizer = _performance_optimizer
    return PerformanceOptimizer


@dataclass
class CognitiveAgentRuntime:
    memory: MemorySystem
    attention: AttentionMechanism
    sensory_processor: SensoryProcessor
    sensory_interface: SensoryInterface
    neural_integration: Any
    dream_processor: Any
    performance_optimizer: Any


class CognitiveAgentRuntimeBuilder:
    def __init__(self, config: CognitiveConfig) -> None:
        self._config = config

    def build(self) -> CognitiveAgentRuntime:
        fast_init = os.getenv("FAST_AGENT_INIT") == "1"

        memory_config = MemorySystemConfig(
            stm_capacity=self._config.memory.stm_capacity,
            stm_decay_threshold=self._config.memory.stm_decay_threshold,
            ltm_storage_path=self._config.memory.ltm_storage_path,
            use_vector_ltm=self._config.memory.use_vector_ltm,
            use_vector_stm=self._config.memory.use_vector_stm,
            chroma_persist_dir=self._config.memory.chroma_persist_dir,
            stm_collection_name=self._config.memory.stm_collection_name,
            ltm_collection_name=self._config.memory.ltm_collection_name,
            embedding_model=self._config.processing.embedding_model,
            semantic_storage_path=self._config.memory.semantic_storage_path,
        )
        if fast_init:
            memory_config.use_vector_stm = False
            memory_config.use_vector_ltm = False

        memory = MemorySystem(memory_config)
        attention = AttentionMechanism(self._config.attention)
        sensory_processor = SensoryProcessor()
        sensory_interface = SensoryInterface(sensory_processor)

        if fast_init:
            class _NoopDreamProcessor:
                def __init__(self) -> None:
                    self.enable_scheduling = False

            return CognitiveAgentRuntime(
                memory=memory,
                attention=attention,
                sensory_processor=sensory_processor,
                sensory_interface=sensory_interface,
                neural_integration=None,
                dream_processor=_NoopDreamProcessor(),
                performance_optimizer=None,
            )

        neural_integration = None
        try:
            from ...cognition.processing.neural import NeuralIntegrationManager

            neural_integration = NeuralIntegrationManager(
                cognitive_config=self._config,
                model_save_path="./data/models/dpad",
            )
            logger.info("Neural integration (DPAD) initialized")
        except ImportError as exc:
            logger.info("Neural integration disabled: %s", exc)

        dream_processor = _lazy_import_dream()(
            memory_system=memory,
            enable_scheduling=True,
            consolidation_threshold=0.6,
            neural_integration_manager=neural_integration,
        )

        performance_optimizer = None
        if self._config.performance.enabled:
            performance_optimizer = _lazy_import_optimizer()(
                config=self._config.performance,
            )
            logger.info("Performance optimizer initialized")

        return CognitiveAgentRuntime(
            memory=memory,
            attention=attention,
            sensory_processor=sensory_processor,
            sensory_interface=sensory_interface,
            neural_integration=neural_integration,
            dream_processor=dream_processor,
            performance_optimizer=performance_optimizer,
        )