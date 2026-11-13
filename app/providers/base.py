"""
Abstract Base Provider interface for LLM inference (Real Groq and Stand-in Simulator).
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Dict, Any, Optional
from pydantic import BaseModel, Field

class ProviderResponse(BaseModel):
    content: str
    tokens_in: int = 0
    tokens_out: int = 0
    first_token_ms: Optional[float] = None
    total_ms: float = 0.0
    status: str = "ok"  # "ok", "rate_limited", "timeout", "error"
    error_message: Optional[str] = None
    rate_limit_headers: Dict[str, str] = Field(default_factory=dict)
    retry_after_sec: Optional[float] = None
    source: str = "stand-in"

class LLMProvider(ABC):
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> ProviderResponse:
        """Non-streaming generation."""
        pass

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Streaming generation. Yields dict chunks:
        {"delta": str, "is_first": bool, "done": bool, "response": Optional[ProviderResponse]}
        """
        pass

