"""
Search Strategies for Episodic Memory

Implements robust fallback search mechanisms when vector database is unavailable.
Provides tiered search quality: semantic (best) → TF-IDF/BM25 (good) → word overlap (basic)
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from collections import Counter
import math
import re

logger = logging.getLogger(__name__)

# Optional imports for advanced search
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available. Advanced search fallbacks will be limited.")


@dataclass
class SearchResult:
    """Generic search result with relevance score"""
    doc_id: str
    relevance: float
    match_type: str
    metadata: Dict[str, Any]


class SearchStrategy:
    """Base class for search strategies"""
    
    def search(self, query: str, documents: Dict[str, str], limit: int = 10, 
               min_relevance: float = 0.3) -> List[SearchResult]:
        """
        Search documents for query
        
        Args:
            query: Search query string
            documents: Dict mapping doc_id -> document text
            limit: Maximum results to return
            min_relevance: Minimum relevance threshold
            
        Returns:
            List of SearchResult objects sorted by relevance
        """
        raise NotImplementedError


class TfidfSearchStrategy(SearchStrategy):
    """
    TF-IDF based search using scikit-learn
    
    Provides better relevance ranking than simple substring matching by:
    - Accounting for term frequency and document frequency
    - Downweighting common words
    - Computing proper similarity scores
    
    Uses caching to avoid recomputing vectorizer for every search.
    """
    
    def __init__(self, max_features: int = 1000, ngram_range: Tuple[int, int] = (1, 2)):
        """
        Initialize TF-IDF search strategy
        
        Args:
            max_features: Maximum number of features (vocabulary size)
            ngram_range: Range of n-grams to consider (1-2 = unigrams and bigrams)
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.vectorizer: Optional[Any] = None
        self.doc_matrix: Optional[Any] = None
        self.doc_ids: List[str] = []
        self._corpus_hash: Optional[int] = None
        
        if not SKLEARN_AVAILABLE:
            logger.warning("TfidfSearchStrategy initialized but sklearn not available")
    
    def _compute_corpus_hash(self, documents: Dict[str, str]) -> int:
        """Compute hash of document corpus for cache invalidation"""
        doc_str = ''.join(f"{k}:{v}" for k, v in sorted(documents.items()))
        return hash(doc_str)
    
    def _update_vectorizer(self, documents: Dict[str, str]) -> None:
        """Update TF-IDF vectorizer with current document corpus"""
        if not SKLEARN_AVAILABLE:
            return
        
        corpus_hash = self._compute_corpus_hash(documents)
        
        # Skip update if corpus hasn't changed
        if corpus_hash == self._corpus_hash and self.vectorizer is not None:
            return
        
        try:
            self.doc_ids = list(documents.keys())
            corpus = [documents[doc_id] for doc_id in self.doc_ids]
            
            # Create and fit vectorizer
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                ngram_range=self.ngram_range,
                stop_words='english',
                lowercase=True,
                strip_accents='unicode'
            )
            
            self.doc_matrix = self.vectorizer.fit_transform(corpus)
            self._corpus_hash = corpus_hash
            
            logger.debug(f"TF-IDF vectorizer updated with {len(corpus)} documents, "
                        f"vocabulary size: {len(self.vectorizer.vocabulary_)}")
        except Exception as e:
            logger.error(f"Failed to update TF-IDF vectorizer: {e}")
            self.vectorizer = None
    
    def search(self, query: str, documents: Dict[str, str], limit: int = 10,
               min_relevance: float = 0.3) -> List[SearchResult]:
        """
        Search using TF-IDF similarity
        
        Returns documents ranked by cosine similarity of TF-IDF vectors
        """
        if not SKLEARN_AVAILABLE or not documents:
            return []
        
        try:
            # Update vectorizer if corpus changed
            self._update_vectorizer(documents)
            
            if self.vectorizer is None or self.doc_matrix is None:
                logger.warning("TF-IDF vectorizer not available, skipping search")
                return []
            
            # Transform query
            query_vec = self.vectorizer.transform([query])
            
            # Compute similarities
            similarities = cosine_similarity(query_vec, self.doc_matrix)[0]
            
            # Create results
            results = []
            for idx, similarity in enumerate(similarities):
                if similarity >= min_relevance:
                    results.append(SearchResult(
                        doc_id=self.doc_ids[idx],
                        relevance=float(similarity),
                        match_type="tfidf",
                        metadata={
                            "similarity": float(similarity),
                            "method": "cosine_tfidf"
                        }
                    ))
            
            # Sort by relevance and limit
            results.sort(key=lambda x: x.relevance, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"TF-IDF search failed: {e}")
            return []


class BM25SearchStrategy(SearchStrategy):
    """
    BM25 ranking function - state-of-the-art bag-of-words retrieval
    
    BM25 improves on TF-IDF by:
    - Saturating term frequency (prevents over-weighting repeated terms)
    - Length normalization (accounts for document length)
    - Tunable parameters (k1, b) for different collections
    
    Generally performs better than TF-IDF for information retrieval tasks.
    """
    
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        Initialize BM25 search
        
        Args:
            k1: Term frequency saturation parameter (typically 1.2-2.0)
            b: Length normalization parameter (0=no normalization, 1=full)
        """
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.doc_freqs: Dict[str, int] = {}  # Term -> document frequency
        self.doc_lengths: Dict[str, int] = {}  # Doc ID -> length
        self.avg_doc_length: float = 0.0
        self.num_docs: int = 0
        self._corpus_hash: Optional[int] = None
        self.tokenized_docs: Dict[str, List[str]] = {}
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric"""
        text = text.lower()
        tokens = re.findall(r'\b\w+\b', text)
        # Filter out very short tokens and common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                     'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'be', 
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 
                     'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 
                     'these', 'those', 'it', 'its', 'i', 'you', 'he', 'she', 'we', 'they'}
        return [t for t in tokens if len(t) > 2 and t not in stop_words]
    
    def _compute_corpus_hash(self, documents: Dict[str, str]) -> int:
        """Compute hash of document corpus for cache invalidation"""
        doc_str = ''.join(f"{k}:{v}" for k, v in sorted(documents.items()))
        return hash(doc_str)
    
    def _index_documents(self, documents: Dict[str, str]) -> None:
        """Build BM25 index from document corpus"""
        corpus_hash = self._compute_corpus_hash(documents)
        
        # Skip if corpus hasn't changed
        if corpus_hash == self._corpus_hash and self.doc_ids:
            return
        
        try:
            self.doc_ids = list(documents.keys())
            self.num_docs = len(self.doc_ids)
            self.doc_freqs = {}
            self.doc_lengths = {}
            self.tokenized_docs = {}
            
            # Tokenize all documents and compute statistics
            for doc_id in self.doc_ids:
                tokens = self._tokenize(documents[doc_id])
                self.tokenized_docs[doc_id] = tokens
                self.doc_lengths[doc_id] = len(tokens)
                
                # Update document frequencies
                unique_tokens = set(tokens)
                for token in unique_tokens:
                    self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1
            
            # Compute average document length
            if self.doc_lengths:
                self.avg_doc_length = sum(self.doc_lengths.values()) / len(self.doc_lengths)
            else:
                self.avg_doc_length = 0.0
            
            self._corpus_hash = corpus_hash
            logger.debug(f"BM25 index built with {self.num_docs} documents, "
                        f"avg length: {self.avg_doc_length:.1f}, "
                        f"vocabulary: {len(self.doc_freqs)} terms")
        
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {e}")
    
    def _bm25_score(self, query_tokens: List[str], doc_id: str) -> float:
        """Compute BM25 score for a document given query tokens"""
        if doc_id not in self.tokenized_docs:
            return 0.0
        
        doc_tokens = self.tokenized_docs[doc_id]
        doc_length = self.doc_lengths[doc_id]
        
        # Term frequencies in document
        tf = Counter(doc_tokens)
        
        score = 0.0
        for term in query_tokens:
            if term not in tf:
                continue
            
            # Term frequency in document
            term_freq = tf[term]
            
            # Inverse document frequency
            df = self.doc_freqs.get(term, 0)
            if df == 0:
                continue
            idf = math.log((self.num_docs - df + 0.5) / (df + 0.5) + 1.0)
            
            # BM25 formula
            numerator = term_freq * (self.k1 + 1)
            denominator = term_freq + self.k1 * (
                1 - self.b + self.b * (doc_length / self.avg_doc_length)
            )
            
            score += idf * (numerator / denominator)
        
        return score
    
    def search(self, query: str, documents: Dict[str, str], limit: int = 10,
               min_relevance: float = 0.3) -> List[SearchResult]:
        """
        Search using BM25 ranking
        
        Returns documents ranked by BM25 score
        """
        if not documents:
            return []
        
        try:
            # Build/update index
            self._index_documents(documents)
            
            if not self.doc_ids:
                return []
            
            # Tokenize query
            query_tokens = self._tokenize(query)
            if not query_tokens:
                return []
            
            # Score all documents
            scores = []
            for doc_id in self.doc_ids:
                score = self._bm25_score(query_tokens, doc_id)
                if score > 0:
                    scores.append((doc_id, score))
            
            if not scores:
                return []
            
            # Normalize scores to 0-1 range for relevance
            max_score = max(s[1] for s in scores)
            if max_score > 0:
                normalized_scores = [(doc_id, score / max_score) 
                                    for doc_id, score in scores]
            else:
                normalized_scores = scores
            
            # Filter by min relevance and create results
            results = []
            for doc_id, relevance in normalized_scores:
                if relevance >= min_relevance:
                    results.append(SearchResult(
                        doc_id=doc_id,
                        relevance=relevance,
                        match_type="bm25",
                        metadata={
                            "bm25_score": relevance,
                            "method": "bm25",
                            "k1": self.k1,
                            "b": self.b
                        }
                    ))
            
            # Sort and limit
            results.sort(key=lambda x: x.relevance, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"BM25 search failed: {e}")
            return []


class EnhancedWordOverlapStrategy(SearchStrategy):
    """
    Enhanced word overlap with stemming and better scoring
    
    Improvements over naive word overlap:
    - Simple stemming (removing common suffixes)
    - Boost for exact phrase matches
    - Consideration of term rarity
    - Better score normalization
    """
    
    def __init__(self):
        self.doc_ids: List[str] = []
        self.term_doc_freq: Dict[str, int] = {}  # Term -> number of docs containing it
        self._corpus_hash: Optional[int] = None
    
    def _simple_stem(self, word: str) -> str:
        """Apply simple suffix stripping"""
        word = word.lower()
        suffixes = ['ing', 'ed', 'es', 's', 'ly', 'er', 'est']
        for suffix in suffixes:
            if word.endswith(suffix) and len(word) > len(suffix) + 2:
                return word[:-len(suffix)]
        return word
    
    def _tokenize_and_stem(self, text: str) -> List[str]:
        """Tokenize and apply simple stemming"""
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [self._simple_stem(t) for t in tokens if len(t) > 2]
    
    def _compute_corpus_hash(self, documents: Dict[str, str]) -> int:
        """Compute hash of document corpus"""
        doc_str = ''.join(f"{k}:{v}" for k, v in sorted(documents.items()))
        return hash(doc_str)
    
    def _build_term_frequencies(self, documents: Dict[str, str]) -> None:
        """Build term document frequency map"""
        corpus_hash = self._compute_corpus_hash(documents)
        
        if corpus_hash == self._corpus_hash and self.term_doc_freq:
            return
        
        self.doc_ids = list(documents.keys())
        self.term_doc_freq = {}
        
        for doc_id in self.doc_ids:
            tokens = set(self._tokenize_and_stem(documents[doc_id]))
            for token in tokens:
                self.term_doc_freq[token] = self.term_doc_freq.get(token, 0) + 1
        
        self._corpus_hash = corpus_hash
    
    def search(self, query: str, documents: Dict[str, str], limit: int = 10,
               min_relevance: float = 0.3) -> List[SearchResult]:
        """
        Search using enhanced word overlap
        
        Scores based on:
        - Stemmed word overlap
        - Exact phrase matches (bonus)
        - Term rarity (IDF-like weighting)
        """
        if not documents:
            return []
        
        try:
            self._build_term_frequencies(documents)
            
            query_lower = query.lower()
            query_tokens = self._tokenize_and_stem(query)
            
            if not query_tokens:
                return []
            
            query_set = set(query_tokens)
            results = []
            
            for doc_id, doc_text in documents.items():
                doc_lower = doc_text.lower()
                doc_tokens = self._tokenize_and_stem(doc_text)
                doc_set = set(doc_tokens)
                
                # Compute overlap
                overlap = query_set & doc_set
                if not overlap:
                    continue
                
                # Base score from overlap ratio
                overlap_ratio = len(overlap) / len(query_set)
                
                # Weight by term rarity (less common terms = higher weight)
                idf_weight = 0.0
                num_docs = len(documents)
                for term in overlap:
                    df = self.term_doc_freq.get(term, 1)
                    idf_weight += math.log(num_docs / df) if df > 0 else 0
                
                # Normalize IDF weight
                if overlap:
                    idf_weight /= len(overlap)
                
                # Bonus for exact phrase match
                phrase_bonus = 0.2 if query_lower in doc_lower else 0.0
                
                # Combined relevance score
                relevance = min(1.0, 0.3 * overlap_ratio + 0.2 * idf_weight + phrase_bonus)
                
                if relevance >= min_relevance:
                    results.append(SearchResult(
                        doc_id=doc_id,
                        relevance=relevance,
                        match_type="word_overlap_enhanced",
                        metadata={
                            "overlap_terms": len(overlap),
                            "overlap_ratio": overlap_ratio,
                            "phrase_match": phrase_bonus > 0
                        }
                    ))
            
            # Sort and limit
            results.sort(key=lambda x: x.relevance, reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Enhanced word overlap search failed: {e}")
            return []


class TieredSearchStrategy:
    """
    Tiered fallback search combining multiple strategies
    
    Tries strategies in order of quality:
    1. Semantic (ChromaDB vector search) - handled by caller
    2. BM25 - best lexical retrieval
    3. TF-IDF - good lexical retrieval with sklearn
    4. Enhanced word overlap - basic but robust
    
    Each tier is tried if previous tiers return insufficient results.
    """
    
    def __init__(self):
        self.bm25 = BM25SearchStrategy()
        self.tfidf = TfidfSearchStrategy() if SKLEARN_AVAILABLE else None
        self.word_overlap = EnhancedWordOverlapStrategy()
        
        logger.info(f"TieredSearchStrategy initialized with: "
                   f"BM25=available, TF-IDF={'available' if SKLEARN_AVAILABLE else 'unavailable'}, "
                   f"EnhancedWordOverlap=available")
    
    def search(self, query: str, documents: Dict[str, str], limit: int = 10,
               min_relevance: float = 0.3, min_results: int = 3) -> List[SearchResult]:
        """
        Search using tiered fallback strategy
        
        Args:
            query: Search query
            documents: Documents to search
            limit: Maximum results
            min_relevance: Minimum relevance threshold
            min_results: Minimum results needed before trying next tier
            
        Returns:
            Best results from available strategies
        """
        if not documents:
            return []
        
        # Try BM25 first (best lexical search)
        results = self.bm25.search(query, documents, limit, min_relevance)
        if len(results) >= min_results:
            logger.debug(f"BM25 search returned {len(results)} results")
            return results
        
        # Try TF-IDF if BM25 insufficient and sklearn available
        if self.tfidf is not None:
            results = self.tfidf.search(query, documents, limit, min_relevance)
            if len(results) >= min_results:
                logger.debug(f"TF-IDF search returned {len(results)} results")
                return results
        
        # Fall back to enhanced word overlap
        results = self.word_overlap.search(query, documents, limit, min_relevance)
        logger.debug(f"Enhanced word overlap search returned {len(results)} results")
        return results
