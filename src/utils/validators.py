"""
Input validation utilities for the cognitive agent system

This module provides comprehensive input validation and sanitization
to ensure data integrity and security throughout the cognitive pipeline.
"""

import re
import html
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Security constants
MAX_INPUT_LENGTH = 10000
MAX_CONTENT_LENGTH = 50000
MIN_INPUT_LENGTH = 1
ALLOWED_SPECIAL_CHARS = r"[a-zA-Z0-9\s\.\,\!\?\-\'\"\(\)\[\]\{\}\:\;\@\#\$\%\&\*\+\=\|\\\/_~`]"


class ValidationError(Exception):
    """Raised when input validation fails"""
    pass


class InputValidator:
    """Production-grade input validation and sanitization"""
    
    @staticmethod
    def validate_text_input(text: Any) -> str:
        """
        Validate and sanitize text input
        
        Args:
            text: Raw text input to validate
            
        Returns:
            Sanitized text string
            
        Raises:
            ValidationError: If input is invalid or unsafe
        """
        if text is None:
            raise ValidationError("Input cannot be None")
        
        # Convert to string if not already
        if not isinstance(text, str):
            text = str(text)
        
        # Length validation
        if len(text) < MIN_INPUT_LENGTH:
            raise ValidationError(f"Input too short (minimum {MIN_INPUT_LENGTH} characters)")
        
        if len(text) > MAX_INPUT_LENGTH:
            raise ValidationError(f"Input too long (maximum {MAX_INPUT_LENGTH} characters)")
        
        # Basic sanitization
        text = text.strip()
        
        # HTML entity encoding for security
        text = html.escape(text, quote=False)
        
        # Remove potentially dangerous characters
        text = re.sub(r'[^\w\s\.\,\!\?\-\'\"\(\)\[\]\{\}\:\;\@\#\$\%\&\*\+\=\|\\\/_~`]', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text
    
    @staticmethod
    def validate_memory_content(content: Any) -> str:
        """
        Validate content for memory storage
        
        Args:
            content: Memory content to validate
            
        Returns:
            Validated content string
            
        Raises:
            ValidationError: If content is invalid
        """
        if content is None:
            raise ValidationError("Memory content cannot be None")
        
        if not isinstance(content, str):
            content = str(content)
        
        if len(content) > MAX_CONTENT_LENGTH:
            raise ValidationError(f"Memory content too long (maximum {MAX_CONTENT_LENGTH} characters)")
        
        # Basic sanitization
        content = content.strip()
        content = html.escape(content, quote=False)
        
        return content
    
    @staticmethod
    def validate_importance_score(score: Any) -> float:
        """
        Validate importance score
        
        Args:
            score: Importance score to validate
            
        Returns:
            Validated score as float
            
        Raises:
            ValidationError: If score is invalid
        """
        try:
            score = float(score)
        except (TypeError, ValueError):
            raise ValidationError("Importance score must be a number")
        
        if not 0.0 <= score <= 1.0:
            raise ValidationError("Importance score must be between 0.0 and 1.0")
        
        return score
    
    @staticmethod
    def validate_attention_params(
        salience: Optional[float] = None,
        novelty: Optional[float] = None,
        priority: Optional[float] = None,
        effort: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Validate attention mechanism parameters
        
        Args:
            salience: Salience score (0.0-1.0)
            novelty: Novelty score (0.0-1.0)
            priority: Priority score (0.0-1.0)
            effort: Effort required (0.0-1.0)
            
        Returns:
            Dictionary of validated parameters
            
        Raises:
            ValidationError: If any parameter is invalid
        """
        result = {}
        
        for name, value in [
            ("salience", salience),
            ("novelty", novelty),
            ("priority", priority),
            ("effort", effort)
        ]:
            if value is not None:
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise ValidationError(f"{name} must be a number")
                
                if not 0.0 <= value <= 1.0:
                    raise ValidationError(f"{name} must be between 0.0 and 1.0")
                
                result[name] = value
        
        return result
    
    @staticmethod
    def validate_session_id(session_id: Any) -> str:
        """
        Validate session identifier
        
        Args:
            session_id: Session ID to validate
            
        Returns:
            Validated session ID
            
        Raises:
            ValidationError: If session ID is invalid
        """
        if not session_id:
            raise ValidationError("Session ID cannot be empty")
        
        session_id = str(session_id)
        
        # Check format and length
        if not re.match(r'^[a-zA-Z0-9_\-]{8,64}$', session_id):
            raise ValidationError("Invalid session ID format")
        
        return session_id
    
    @staticmethod
    def validate_duration(duration: Any) -> float:
        """
        Validate duration parameter
        
        Args:
            duration: Duration in minutes
            
        Returns:
            Validated duration
            
        Raises:
            ValidationError: If duration is invalid
        """
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            raise ValidationError("Duration must be a number")
        
        if duration <= 0:
            raise ValidationError("Duration must be positive")
        
        if duration > 1440:  # 24 hours
            raise ValidationError("Duration cannot exceed 24 hours")
        
        return duration
