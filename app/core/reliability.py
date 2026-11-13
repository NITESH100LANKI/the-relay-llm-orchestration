"""
Reliability Engineering and Traffic Shaping Mechanisms for The Relay:
- Token Bucket Rate Limiter
- Concurrency Semaphore & Admission Control
- Request Deadlines & Timeout Cancellation
- Exponential Backoff with Full Jitter
- Circuit Breaker
- In-memory Metrics and Outcome Tracking

Loop-Aware Primitives:
Guarantees Semaphore and Locks are always bound to the currently running asyncio loop.
"""

import asyncio
import time
import random
from typing import Dict, Any, Optional

class TokenBucket:
    def __init__(self, rate_per_second: float, capacity: float):
        self.rate = rate_per_second
        self.capacity = capacity
        self.tokens = capacity
        self.last_update = time.monotonic()
        self._lock: Optional[asyncio.Lock] = None
        self._loop = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    async def acquire(self, tokens: float = 1.0) -> bool:
        async with self._get_lock():
            now = time.monotonic()
            elapsed = now - self.last_update
            self.last_update = now
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, recovery_timeout_sec: float = 10.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout_sec
        self.state = "closed"  # "closed", "open", "half_open"
        self.consecutive_failures = 0
        self.last_failure_time = 0.0
        self._lock: Optional[asyncio.Lock] = None
        self._loop = None

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._loop is not loop:
            self._lock = asyncio.Lock()
            self._loop = loop
        return self._lock

    async def allow_request(self) -> bool:
        async with self._get_lock():
            if self.state == "closed":
                return True
            now = time.monotonic()
            if self.state == "open":
                if now - self.last_failure_time > self.recovery_timeout:
                    self.state = "half_open"
                    return True
                return False
            return True

    async def record_success(self):
        async with self._get_lock():
            self.consecutive_failures = 0
            self.state = "closed"

    async def record_failure(self):
        async with self._get_lock():
            self.consecutive_failures += 1
            self.last_failure_time = time.monotonic()
            if self.consecutive_failures >= self.failure_threshold:
                self.state = "open"

def compute_backoff(attempt: int, base_delay: float = 0.5, max_delay: float = 5.0) -> float:
    """Exponential backoff with full jitter to prevent thundering herd."""
    delay = min(max_delay, base_delay * (2 ** attempt))
    return random.uniform(0.1, delay)

class ReliabilityManager:
    def __init__(self, max_concurrency: int = 5, rpm_limit: int = 30):
        self.max_concurrency = max_concurrency
        self.rpm_limit = rpm_limit
        self._semaphore: Optional[asyncio.Semaphore] = None
        self._sem_loop = None
        self._lock: Optional[asyncio.Lock] = None
        self._lock_loop = None
        
        self.current_queue_length = 0
        self.peak_queue_length = 0
        self.rate_limiter = TokenBucket(rate_per_second=rpm_limit / 60.0, capacity=max(5.0, rpm_limit / 10.0))
        self.circuit_breaker = CircuitBreaker(failure_threshold=4, recovery_timeout_sec=5.0)
        
        # Metrics aggregates
        self.total_requests = 0
        self.successful_requests = 0
        self.rate_limited_requests = 0
        self.timed_out_requests = 0
        self.error_requests = 0
        self.total_retries = 0

    def get_semaphore(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._semaphore is None or self._sem_loop is not loop:
            self._semaphore = asyncio.Semaphore(self.max_concurrency)
            self._sem_loop = loop
        return self._semaphore

    def _get_lock(self) -> asyncio.Lock:
        loop = asyncio.get_running_loop()
        if self._lock is None or self._lock_loop is not loop:
            self._lock = asyncio.Lock()
            self._lock_loop = loop
        return self._lock

    async def track_queue_enter(self):
        async with self._get_lock():
            self.current_queue_length += 1
            if self.current_queue_length > self.peak_queue_length:
                self.peak_queue_length = self.current_queue_length

    async def track_queue_leave(self):
        async with self._get_lock():
            self.current_queue_length = max(0, self.current_queue_length - 1)

    async def record_outcome(self, status: str, retries: int = 0):
        async with self._get_lock():
            self.total_requests += 1
            self.total_retries += retries
            if status == "ok":
                self.successful_requests += 1
                await self.circuit_breaker.record_success()
            elif status == "rate_limited":
                self.rate_limited_requests += 1
                await self.circuit_breaker.record_failure()
            elif status == "timeout":
                self.timed_out_requests += 1
                await self.circuit_breaker.record_failure()
            else:
                self.error_requests += 1
                await self.circuit_breaker.record_failure()

    def get_metrics(self) -> Dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "timed_out_requests": self.timed_out_requests,
            "error_requests": self.error_requests,
            "total_retries": self.total_retries,
            "current_queue_length": self.current_queue_length,
            "peak_queue_length": self.peak_queue_length,
            "circuit_breaker_state": self.circuit_breaker.state
        }

reliability_mgr = ReliabilityManager()
