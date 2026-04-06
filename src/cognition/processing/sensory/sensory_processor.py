"""Minimal sensory processing module used by the cognitive runtime."""

from collections import Counter, deque
from dataclasses import dataclass, field
import hashlib
import math
import re
import time
from typing import Any, Deque, Dict, List, Optional

import numpy as np


_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "but",
    "by",
    "for",
    "from",
    "if",
    "in",
    "into",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "with",
}

_SALIENT_TERMS = {
    "alert",
    "asap",
    "critical",
    "deadline",
    "emergency",
    "error",
    "help",
    "important",
    "need",
    "now",
    "remember",
    "urgent",
    "warning",
}

@dataclass
class SensoryInput:
    """Represents a sensory input with metadata"""
    content: str
    modality: str = 'text'
    timestamp: Optional[float] = None
    source: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

@dataclass
class ProcessedSensoryData:
    """Represents processed sensory data with embeddings and scores"""
    input_data: SensoryInput
    embedding: np.ndarray
    entropy_score: float
    salience_score: float
    relevance_score: float
    filtered: bool = False
    processing_metadata: Dict[str, Any] = field(default_factory=dict)

class SensoryProcessor:
    """Minimal sensory processor with deterministic fallback embeddings."""
    
    def __init__(
        self,
        max_history_size: int = 512,
        embedding_dimension: int = 384,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_model_embeddings: bool = False,
        embedding_manager: Optional[Any] = None,
    ):
        self.embedding_dimension = max(1, int(embedding_dimension))
        self.embedding_model = embedding_model
        self.use_model_embeddings = use_model_embeddings or embedding_manager is not None
        self._embedding_manager = embedding_manager
        self.processing_history: Deque[ProcessedSensoryData] = deque(
            maxlen=max(1, int(max_history_size))
        )
        self._total_processed = 0
        self._filtered_count = 0

    def _get_embedding_manager(self) -> Optional[Any]:
        if self._embedding_manager is not None:
            return self._embedding_manager
        return None

    def _generate_deterministic_embedding(self, content: str) -> np.ndarray:
        vector = np.zeros(self.embedding_dimension, dtype=np.float32)
        normalized = content.lower().strip()
        if not normalized:
            return vector

        tokens = re.findall(r"\b\w+\b|[^\w\s]", normalized)
        if not tokens:
            tokens = [normalized]

        for index, token in enumerate(tokens):
            self._accumulate_feature(vector, f"tok:{token}", 1.0)
            if index + 1 < len(tokens):
                next_token = tokens[index + 1]
                self._accumulate_feature(vector, f"bi:{token}|{next_token}", 0.5)

        for index in range(max(0, len(normalized) - 2)):
            trigram = normalized[index:index + 3]
            self._accumulate_feature(vector, f"tri:{trigram}", 0.15)

        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return vector

    def _accumulate_feature(self, vector: np.ndarray, feature: str, weight: float) -> None:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % self.embedding_dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += np.float32(weight * sign)

    def _generate_embedding(self, content: str) -> tuple[np.ndarray, str]:
        manager = self._get_embedding_manager()
        if manager is not None:
            try:
                embedding = manager.encode(content)
            except Exception:
                embedding = None

            if embedding is not None:
                array = np.asarray(embedding, dtype=np.float32)
                norm = float(np.linalg.norm(array))
                if norm > 0.0:
                    array = array / norm
                return array, "model"

        return self._generate_deterministic_embedding(content), "deterministic_hash"

    def _estimate_attention_salience_and_valence(self, content: str) -> tuple[float, float]:
        if not content:
            return 0.0, 0.0

        try:
            from src.cognition.attention.attention_manager import get_attention_manager

            return get_attention_manager().estimate_salience_and_valence(content)
        except Exception:
            words = re.findall(r"\b\w+\b", content.lower())
            length_factor = min(1.0, len(words) / 25.0)
            punctuation_boost = min(0.25, 0.07 * (content.count("!") + content.count("?")))
            uppercase_ratio = sum(1 for char in content if char.isupper()) / max(1, len(content))
            emphasis_boost = min(0.25, uppercase_ratio * 0.6)
            salience = min(1.0, 0.25 + (length_factor * 0.4) + punctuation_boost + emphasis_boost)
            return salience, 0.0

    def _calculate_entropy_score(self, content: str, words: List[str]) -> float:
        if not content:
            return 0.0

        signal_units = words or list(content)
        counts = Counter(signal_units)
        length = len(signal_units)
        entropy = 0.0
        for count in counts.values():
            probability = count / length
            entropy -= probability * math.log2(probability)

        max_entropy = math.log2(max(2, length))
        normalized_entropy = entropy / max_entropy if max_entropy > 0.0 else 0.0
        length_damping = 1.0 - math.exp(-length / 12.0)
        return min(1.0, normalized_entropy * length_damping)

    def _calculate_salience_score(
        self,
        content: str,
        words: List[str],
        entropy_score: float,
        relevance_score: float,
    ) -> float:
        base_salience, _valence = self._estimate_attention_salience_and_valence(content)
        salient_term_hits = sum(1 for word in words if word in _SALIENT_TERMS)
        term_boost = min(0.18, salient_term_hits * 0.06)

        score = (
            (base_salience * 0.7)
            + (entropy_score * 0.15)
            + (relevance_score * 0.15)
            + term_boost
        )
        return min(1.0, score)

    def _calculate_relevance_score(self, words: List[str]) -> float:
        if not words:
            return 0.0

        content_words = [word for word in words if word not in _STOPWORDS]
        analysis_words = content_words or words
        word_count = len(words)
        analysis_count = len(analysis_words)
        lexical_diversity = len(set(analysis_words)) / analysis_count
        length_signal = 1.0 - math.exp(-analysis_count / 18.0)
        density = analysis_count / word_count
        structure_signal = min(1.0, word_count / 8.0)
        repetition_penalty = 0.5 + (0.5 * lexical_diversity)
        score = (
            (0.45 * length_signal)
            + (0.35 * lexical_diversity)
            + (0.20 * density)
        ) * structure_signal * repetition_penalty
        return min(1.0, score)

    def _classify_filter(
        self,
        content: str,
        words: List[str],
        entropy_score: float,
        relevance_score: float,
    ) -> Optional[str]:
        if not content:
            return "empty"
        if not any(char.isalnum() for char in content):
            return "punctuation_only"
        if len(content) >= 8 and len(set(content.lower())) <= 2:
            return "repeated_characters"
        if len(words) >= 4 and len(set(words)) == 1:
            return "repeated_single_token"
        if len(words) <= 1 and len(content) >= 8 and entropy_score < 0.08:
            return "low_information_single_token"
        if relevance_score < 0.05 and entropy_score < 0.05:
            return "low_information"
        return None
        
    def process_input(self, sensory_input: SensoryInput) -> ProcessedSensoryData:
        """Process a sensory input"""
        content = str(sensory_input.content) if sensory_input.content else ""
        stripped_content = content.strip()
        words = re.findall(r"\b\w+\b", stripped_content.lower())
        
        entropy_score = self._calculate_entropy_score(stripped_content, words)
        relevance_score = self._calculate_relevance_score(words)
        salience_score = self._calculate_salience_score(
            stripped_content,
            words,
            entropy_score,
            relevance_score,
        )
        
        embedding, embedding_source = self._generate_embedding(stripped_content)
        
        filter_reason = self._classify_filter(
            stripped_content,
            words,
            entropy_score,
            relevance_score,
        )
        filtered = filter_reason is not None
        if filtered:
            entropy_score = min(entropy_score, 0.05)
            salience_score = min(salience_score, 0.15)
            relevance_score = min(relevance_score, 0.1)
        
        processed = ProcessedSensoryData(
            input_data=sensory_input,
            embedding=embedding,
            entropy_score=float(entropy_score),
            salience_score=float(salience_score),
            relevance_score=float(relevance_score),
            filtered=filtered,
            processing_metadata={
                "embedding_source": embedding_source,
                "word_count": len(words),
                "history_capacity": self.processing_history.maxlen,
                "filter_reason": filter_reason,
            },
        )
        
        self.processing_history.append(processed)
        self._total_processed += 1
        if filtered:
            self._filtered_count += 1
        return processed
    
    def process_batch(self, inputs: List[SensoryInput]) -> List[ProcessedSensoryData]:
        """Process a batch of inputs"""
        return [self.process_input(inp) for inp in inputs]
    
    def get_processing_stats(self) -> Dict:
        """Get processing statistics"""
        if not self.processing_history:
            return {
                'total_processed': self._total_processed,
                'filtered_count': self._filtered_count,
                'history_size': 0,
                'history_capacity': self.processing_history.maxlen,
            }

        history = list(self.processing_history)
        
        return {
            'total_processed': self._total_processed,
            'filtered_count': self._filtered_count,
            'history_size': len(history),
            'history_capacity': self.processing_history.maxlen,
            'avg_scores': {
                'entropy': float(np.mean([p.entropy_score for p in history])),
                'salience': float(np.mean([p.salience_score for p in history])),
                'relevance': float(np.mean([p.relevance_score for p in history]))
            }
        }
