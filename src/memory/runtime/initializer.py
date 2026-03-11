from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

from src.memory.episodic import EpisodicMemorySystem
from src.memory.ltm import VectorLongTermMemory
from src.memory.procedural.procedural_memory import ProceduralMemory
from src.memory.stm import STMConfiguration, VectorShortTermMemory


logger = logging.getLogger(__name__)


@dataclass
class MemorySubsystemBundle:
    stm: VectorShortTermMemory
    ltm: VectorLongTermMemory
    episodic: EpisodicMemorySystem
    semantic: Any
    prospective: Any
    procedural: ProceduralMemory


class MemorySubsystemInitializer:
    def __init__(self, config: Any) -> None:
        self._config = config

    def initialize(self) -> MemorySubsystemBundle:
        stm_config = STMConfiguration(
            chroma_persist_dir=self._config.chroma_persist_dir,
            embedding_model=self._config.embedding_model,
            capacity=self._config.stm_capacity,
            enable_gpu=getattr(self._config, "enable_gpu", True),
            lazy_embeddings=self._config.lazy_embeddings,
        )
        stm = VectorShortTermMemory(stm_config)

        if not self._config.use_vector_ltm:
            raise RuntimeError("LTM initialization failed: VectorLTM must be used")

        ltm = VectorLongTermMemory(
            chroma_persist_dir=self._config.chroma_persist_dir,
            embedding_model=self._config.embedding_model,
            lazy_embeddings=self._config.lazy_embeddings,
        )

        episodic = EpisodicMemorySystem(
            chroma_persist_dir=self._config.chroma_persist_dir,
            embedding_model=self._config.embedding_model,
            lazy_embeddings=self._config.lazy_embeddings,
        )

        semantic = self._initialize_semantic()
        prospective = self._initialize_prospective()
        procedural = ProceduralMemory(stm=stm, ltm=ltm)

        return MemorySubsystemBundle(
            stm=stm,
            ltm=ltm,
            episodic=episodic,
            semantic=semantic,
            prospective=prospective,
            procedural=procedural,
        )

    def _initialize_semantic(self) -> Optional[Any]:
        if os.environ.get("DISABLE_SEMANTIC_MEMORY") == "1":
            logger.info("Semantic memory disabled via DISABLE_SEMANTIC_MEMORY")
            return None

        try:
            from src.memory.semantic.semantic_memory import SemanticMemorySystem

            return SemanticMemorySystem(
                chroma_persist_dir=self._config.chroma_persist_dir or "data/memory_stores/chroma_semantic",
                embedding_model=self._config.embedding_model,
                lazy_embeddings=self._config.lazy_embeddings,
            )
        except Exception as exc:
            logger.warning("Semantic memory unavailable; continuing without it: %s", exc)
            return None

    def _initialize_prospective(self) -> Any:
        try:
            if getattr(self._config, "use_vector_prospective", False):
                from src.memory.prospective.prospective_memory import ProspectiveMemoryVectorStore

                prospective = ProspectiveMemoryVectorStore(
                    chroma_persist_dir=self._config.chroma_persist_dir,
                    embedding_model=self._config.embedding_model,
                )
                logger.info("Prospective memory backend: vector store (ChromaDB)")
                return prospective

            from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory

            prospective = get_inmemory_prospective_memory()
            logger.info("Prospective memory backend: in-memory singleton")
            return prospective
        except Exception as exc:
            logger.warning("Prospective memory init fallback to in-memory due to error: %s", exc)
            from src.memory.prospective.prospective_memory import get_inmemory_prospective_memory

            prospective = get_inmemory_prospective_memory()
            logger.info("Prospective memory backend: in-memory singleton (fallback)")
            return prospective