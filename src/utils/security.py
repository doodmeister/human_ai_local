"""
Security utilities for the cognitive agent system

This module provides security features including rate limiting,
credential management, and security monitoring to protect
the cognitive agent from various threats.
"""

import hashlib
import secrets
import logging
import re
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Dict, Set

logger = logging.getLogger(__name__)

# Security constants
MAX_REQUESTS_PER_MINUTE = 60
MAX_REQUESTS_PER_HOUR = 1000
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION = 300  # 5 minutes
TOKEN_EXPIRY_HOURS = 24


class SecurityError(Exception):
    """Base security exception"""
    pass


class RateLimitExceeded(SecurityError):
    """Raised when rate limit is exceeded"""
    pass


class AccessDenied(SecurityError):
    """Raised when access is denied"""
    pass


class SecurityManager:
    """Production-grade security manager for cognitive agent"""
    
    def __init__(self):
        """Initialize security manager with default settings"""
        self._request_counts: Dict[str, deque] = defaultdict(deque)
        self._failed_attempts: Dict[str, int] = defaultdict(int)
        self._locked_out: Dict[str, datetime] = {}
        self._active_sessions: Set[str] = set()
        self._session_tokens: Dict[str, datetime] = {}
        
        logger.info("SecurityManager initialized")
    
    def check_rate_limit(self, client_id: str) -> bool:
        """
        Check if client has exceeded rate limits
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if within limits, False otherwise
            
        Raises:
            RateLimitExceeded: If rate limit is exceeded
        """
        now = datetime.now()
        
        # Clean old entries
        minute_ago = now - timedelta(minutes=1)
        hour_ago = now - timedelta(hours=1)
        
        # Remove old requests
        while (self._request_counts[client_id] and 
               self._request_counts[client_id][0] < hour_ago):
            self._request_counts[client_id].popleft()
        
        # Check per-minute limit
        recent_requests = sum(
            1 for req_time in self._request_counts[client_id]
            if req_time > minute_ago
        )
        
        if recent_requests >= MAX_REQUESTS_PER_MINUTE:
            logger.warning(f"Rate limit exceeded for client {client_id}: {recent_requests}/min")
            raise RateLimitExceeded(f"Too many requests per minute: {recent_requests}")
        
        # Check per-hour limit
        if len(self._request_counts[client_id]) >= MAX_REQUESTS_PER_HOUR:
            logger.warning(f"Hourly rate limit exceeded for client {client_id}")
            raise RateLimitExceeded(f"Too many requests per hour: {len(self._request_counts[client_id])}")
        
        # Record this request
        self._request_counts[client_id].append(now)
        return True
    
    def check_lockout(self, client_id: str) -> bool:
        """
        Check if client is locked out due to failed attempts
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            True if not locked out, False if locked out
            
        Raises:
            AccessDenied: If client is locked out
        """
        if client_id in self._locked_out:
            lockout_time = self._locked_out[client_id]
            if datetime.now() - lockout_time < timedelta(seconds=LOCKOUT_DURATION):
                remaining = LOCKOUT_DURATION - (datetime.now() - lockout_time).seconds
                logger.warning(f"Client {client_id} is locked out for {remaining} more seconds")
                raise AccessDenied(f"Client locked out for {remaining} seconds")
            else:
                # Lockout expired, remove it
                del self._locked_out[client_id]
                self._failed_attempts[client_id] = 0
        
        return True
    
    def record_failed_attempt(self, client_id: str) -> None:
        """
        Record a failed authentication/validation attempt
        
        Args:
            client_id: Unique client identifier
        """
        self._failed_attempts[client_id] += 1
        
        if self._failed_attempts[client_id] >= MAX_FAILED_ATTEMPTS:
            self._locked_out[client_id] = datetime.now()
            logger.warning(f"Client {client_id} locked out after {MAX_FAILED_ATTEMPTS} failed attempts")
    
    def record_successful_attempt(self, client_id: str) -> None:
        """
        Record a successful authentication/validation attempt
        
        Args:
            client_id: Unique client identifier
        """
        if client_id in self._failed_attempts:
            del self._failed_attempts[client_id]
        if client_id in self._locked_out:
            del self._locked_out[client_id]
    
    def generate_session_token(self, client_id: str) -> str:
        """
        Generate a secure session token
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            Secure session token
        """
        token = secrets.token_urlsafe(32)
        self._session_tokens[token] = datetime.now()
        self._active_sessions.add(client_id)
        
        logger.info(f"Generated session token for client {client_id}")
        return token
    
    def validate_session_token(self, token: str) -> bool:
        """
        Validate a session token
        
        Args:
            token: Session token to validate
            
        Returns:
            True if valid, False otherwise
        """
        if token not in self._session_tokens:
            return False
        
        # Check if token has expired
        token_time = self._session_tokens[token]
        if datetime.now() - token_time > timedelta(hours=TOKEN_EXPIRY_HOURS):
            del self._session_tokens[token]
            return False
        
        return True
    
    def revoke_session_token(self, token: str) -> None:
        """
        Revoke a session token
        
        Args:
            token: Session token to revoke
        """
        if token in self._session_tokens:
            del self._session_tokens[token]
            logger.info("Revoked session token")
    
    def sanitize_input(self, input_text: str) -> str:
        """
        Sanitize input text for security
        
        Args:
            input_text: Text to sanitize
            
        Returns:
            Sanitized text
        """
        # Remove potential script injections
        dangerous_patterns = [
            r'<script.*?</script>',
            r'javascript:',
            r'onload=',
            r'onerror=',
            r'eval\(',
            r'exec\(',
        ]
        
        sanitized = input_text
        for pattern in dangerous_patterns:
            sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
        
        return sanitized
    
    def hash_sensitive_data(self, data: str) -> str:
        """
        Hash sensitive data using secure algorithm
        
        Args:
            data: Data to hash
            
        Returns:
            Hashed data
        """
        salt = secrets.token_bytes(32)
        hashed = hashlib.pbkdf2_hmac('sha256', data.encode(), salt, 100000)
        return salt.hex() + hashed.hex()
    
    def verify_hash(self, data: str, hashed: str) -> bool:
        """
        Verify hashed data
        
        Args:
            data: Original data
            hashed: Hashed data to verify against
            
        Returns:
            True if match, False otherwise
        """
        try:
            salt = bytes.fromhex(hashed[:64])
            stored_hash = bytes.fromhex(hashed[64:])
            new_hash = hashlib.pbkdf2_hmac('sha256', data.encode(), salt, 100000)
            return new_hash == stored_hash
        except (ValueError, TypeError):
            return False
    
    def get_security_metrics(self) -> Dict[str, int]:
        """
        Get security metrics for monitoring
        
        Returns:
            Dictionary of security metrics
        """
        return {
            "active_sessions": len(self._active_sessions),
            "locked_out_clients": len(self._locked_out),
            "total_clients_tracked": len(self._request_counts),
            "total_session_tokens": len(self._session_tokens)
        }
    
    def cleanup_expired_data(self) -> None:
        """Clean up expired security data"""
        now = datetime.now()
        
        # Clean expired tokens
        expired_tokens = [
            token for token, timestamp in self._session_tokens.items()
            if now - timestamp > timedelta(hours=TOKEN_EXPIRY_HOURS)
        ]
        for token in expired_tokens:
            del self._session_tokens[token]
        
        # Clean expired lockouts
        expired_lockouts = [
            client_id for client_id, timestamp in self._locked_out.items()
            if now - timestamp > timedelta(seconds=LOCKOUT_DURATION)
        ]
        for client_id in expired_lockouts:
            del self._locked_out[client_id]
            if client_id in self._failed_attempts:
                del self._failed_attempts[client_id]
        
        logger.debug(f"Cleaned up {len(expired_tokens)} expired tokens and {len(expired_lockouts)} expired lockouts")
