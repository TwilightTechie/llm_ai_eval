"""
Rate limiting utilities for API calls.
Supports token bucket and sliding window strategies.
"""

import time
import threading
from typing import Optional
from dataclasses import dataclass
from collections import deque

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class RateLimitConfig:
    """Configuration for rate limiting."""
    enabled: bool = True
    strategy: str = "token_bucket"  # token_bucket or sliding_window
    
    # Token bucket settings
    bucket_size: int = 100
    refill_rate: float = 10.0  # tokens per second
    
    # Sliding window settings
    window_size_seconds: int = 60
    max_requests_per_window: int = 60
    
    # Provider-specific settings
    requests_per_minute: int = 60
    tokens_per_minute: int = 150000
    max_retries: int = 3
    retry_delay_seconds: float = 1.0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter.
    Allows bursts up to bucket_size, then limits to refill_rate.
    """
    
    def __init__(self, bucket_size: int = 100, refill_rate: float = 10.0):
        """
        Initialize token bucket.
        
        Args:
            bucket_size: Maximum tokens in bucket
            refill_rate: Tokens added per second
        """
        self.bucket_size = bucket_size
        self.refill_rate = refill_rate
        self.tokens = bucket_size
        self.last_refill = time.monotonic()
        self._lock = threading.Lock()
        
        logger.debug(
            f"TokenBucket initialized: size={bucket_size}, rate={refill_rate}/s"
        )
    
    def _refill(self):
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_refill
        tokens_to_add = elapsed * self.refill_rate
        
        self.tokens = min(self.bucket_size, self.tokens + tokens_to_add)
        self.last_refill = now
    
    def acquire(self, tokens: int = 1, blocking: bool = True, 
                timeout: Optional[float] = None) -> bool:
        """
        Acquire tokens from the bucket.
        
        Args:
            tokens: Number of tokens to acquire
            blocking: Wait for tokens if not available
            timeout: Maximum time to wait (None = infinite)
        
        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._refill()
                
                if self.tokens >= tokens:
                    self.tokens -= tokens
                    logger.debug(f"Acquired {tokens} tokens, {self.tokens:.1f} remaining")
                    return True
                
                if not blocking:
                    return False
                
                # Calculate wait time
                tokens_needed = tokens - self.tokens
                wait_time = tokens_needed / self.refill_rate
            
            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    logger.warning(f"Rate limit timeout after {elapsed:.2f}s")
                    return False
            
            logger.debug(f"Waiting {wait_time:.2f}s for tokens")
            time.sleep(min(wait_time, 0.1))  # Sleep in small increments
    
    def get_available_tokens(self) -> float:
        """Get current number of available tokens."""
        with self._lock:
            self._refill()
            return self.tokens


class SlidingWindowRateLimiter:
    """
    Sliding window rate limiter.
    Limits requests to max_requests within window_size seconds.
    """
    
    def __init__(self, window_size_seconds: int = 60, 
                 max_requests: int = 60):
        """
        Initialize sliding window.
        
        Args:
            window_size_seconds: Size of the sliding window
            max_requests: Maximum requests allowed in window
        """
        self.window_size = window_size_seconds
        self.max_requests = max_requests
        self.requests = deque()
        self._lock = threading.Lock()
        
        logger.debug(
            f"SlidingWindow initialized: window={window_size_seconds}s, "
            f"max={max_requests}"
        )
    
    def _cleanup_old_requests(self):
        """Remove requests outside the current window."""
        cutoff = time.monotonic() - self.window_size
        while self.requests and self.requests[0] < cutoff:
            self.requests.popleft()
    
    def acquire(self, blocking: bool = True, 
                timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make a request.
        
        Args:
            blocking: Wait if limit reached
            timeout: Maximum time to wait
        
        Returns:
            True if request allowed, False if timeout
        """
        start_time = time.monotonic()
        
        while True:
            with self._lock:
                self._cleanup_old_requests()
                
                if len(self.requests) < self.max_requests:
                    self.requests.append(time.monotonic())
                    logger.debug(
                        f"Request allowed, {len(self.requests)}/{self.max_requests} "
                        f"in window"
                    )
                    return True
                
                if not blocking:
                    return False
                
                # Calculate wait time until oldest request expires
                oldest = self.requests[0]
                wait_time = oldest + self.window_size - time.monotonic()
            
            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start_time
                if elapsed + wait_time > timeout:
                    logger.warning(f"Rate limit timeout after {elapsed:.2f}s")
                    return False
            
            logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
            time.sleep(min(max(wait_time, 0), 0.1))
    
    def get_remaining_requests(self) -> int:
        """Get number of remaining requests in current window."""
        with self._lock:
            self._cleanup_old_requests()
            return self.max_requests - len(self.requests)


class CompositeRateLimiter:
    """
    Combines multiple rate limiters for comprehensive limiting.
    Useful for APIs with both request and token limits.
    """
    
    def __init__(self, config: RateLimitConfig):
        """
        Initialize composite rate limiter from config.
        
        Args:
            config: Rate limit configuration
        """
        self.config = config
        self.enabled = config.enabled
        
        if not self.enabled:
            logger.info("Rate limiting disabled")
            return
        
        # Request rate limiter
        if config.strategy == "token_bucket":
            # Convert requests per minute to tokens per second
            refill_rate = config.requests_per_minute / 60.0
            self.request_limiter = TokenBucketRateLimiter(
                bucket_size=config.bucket_size,
                refill_rate=refill_rate
            )
        else:
            self.request_limiter = SlidingWindowRateLimiter(
                window_size_seconds=config.window_size_seconds,
                max_requests=config.max_requests_per_window
            )
        
        # Token rate limiter (for LLM APIs)
        if config.tokens_per_minute > 0:
            token_refill_rate = config.tokens_per_minute / 60.0
            self.token_limiter = TokenBucketRateLimiter(
                bucket_size=config.tokens_per_minute,
                refill_rate=token_refill_rate
            )
        else:
            self.token_limiter = None
        
        logger.info(
            f"Rate limiter initialized: strategy={config.strategy}, "
            f"rpm={config.requests_per_minute}, tpm={config.tokens_per_minute}"
        )
    
    def acquire_request(self, blocking: bool = True, 
                        timeout: Optional[float] = None) -> bool:
        """Acquire permission for a request."""
        if not self.enabled:
            return True
        return self.request_limiter.acquire(blocking=blocking, timeout=timeout)
    
    def acquire_tokens(self, tokens: int, blocking: bool = True,
                       timeout: Optional[float] = None) -> bool:
        """Acquire tokens for an API call."""
        if not self.enabled or self.token_limiter is None:
            return True
        return self.token_limiter.acquire(tokens, blocking=blocking, timeout=timeout)
    
    def acquire(self, estimated_tokens: int = 0, blocking: bool = True,
                timeout: Optional[float] = None) -> bool:
        """
        Acquire both request and token permissions.
        
        Args:
            estimated_tokens: Estimated tokens for the request
            blocking: Wait if limits reached
            timeout: Maximum wait time
        
        Returns:
            True if both acquired, False otherwise
        """
        if not self.enabled:
            return True
        
        # First acquire request permission
        if not self.acquire_request(blocking=blocking, timeout=timeout):
            return False
        
        # Then acquire token permission
        if estimated_tokens > 0:
            if not self.acquire_tokens(estimated_tokens, blocking=blocking, 
                                       timeout=timeout):
                return False
        
        return True


def create_rate_limiter(config: dict) -> CompositeRateLimiter:
    """
    Create a rate limiter from configuration dictionary.
    
    Args:
        config: Configuration dict (from evaluation.yaml rate_limiter section)
    
    Returns:
        Configured CompositeRateLimiter
    """
    rate_config = RateLimitConfig(
        enabled=config.get('enabled', True),
        strategy=config.get('strategy', 'token_bucket'),
        bucket_size=config.get('bucket_size', 100),
        refill_rate=config.get('refill_rate', 10.0),
        window_size_seconds=config.get('window_size_seconds', 60),
        max_requests_per_window=config.get('max_requests_per_window', 60),
        requests_per_minute=config.get('requests_per_minute', 60),
        tokens_per_minute=config.get('tokens_per_minute', 150000),
        max_retries=config.get('max_retries', 3),
        retry_delay_seconds=config.get('retry_delay_seconds', 1.0),
    )
    
    return CompositeRateLimiter(rate_config)


