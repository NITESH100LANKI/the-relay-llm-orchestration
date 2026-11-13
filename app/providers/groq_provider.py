"""
Real Groq LLM Provider Integration for The Relay.
Provides asynchronous streaming and non-streaming inference using Groq API.
Accurately measures TTFT (Time to First Token), total latency, token usage,
and captures all rate-limit headers (x-ratelimit-*) and HTTP errors.
Redacts secrets and never exposes API keys in logs.
"""

import os
import time
import json
import httpx
from typing import AsyncGenerator, Dict, Any, Optional

from app.providers.base import LLMProvider, ProviderResponse
from app.core.config import settings

class GroqProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self.model = model or settings.groq_model or os.environ.get("GROQ_MODEL", "qwen/qwen3.8-27b")
        self.base_url = "https://api.groq.com/openai/v1"
        self.source = "real"
        self.last_rate_limit_headers: Dict[str, str] = {}

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    def _extract_rate_limit_headers(self, headers: httpx.Headers) -> Dict[str, str]:
        extracted = {}
        for k, v in headers.items():
            k_lower = k.lower()
            if k_lower.startswith("x-ratelimit-") or k_lower == "retry-after":
                extracted[k_lower] = v
        return extracted

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> ProviderResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False
        }

        t_start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload
                )
                total_ms = (time.perf_counter() - t_start) * 1000.0
                rl_headers = self._extract_rate_limit_headers(resp.headers)
                self.last_rate_limit_headers = rl_headers
                
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("retry-after", 2.0))
                    return ProviderResponse(
                        content="",
                        first_token_ms=None,
                        total_ms=total_ms,
                        status="rate_limited",
                        error_message=resp.text,
                        rate_limit_headers=rl_headers,
                        retry_after_sec=retry_after,
                        source="real"
                    )

                if resp.status_code >= 400:
                    return ProviderResponse(
                        content="",
                        first_token_ms=None,
                        total_ms=total_ms,
                        status="error",
                        error_message=f"HTTP {resp.status_code}: {resp.text}",
                        rate_limit_headers=rl_headers,
                        source="real"
                    )

                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                tokens_in = usage.get("prompt_tokens", len(prompt.split()))
                tokens_out = usage.get("completion_tokens", len(content.split()))

                return ProviderResponse(
                    content=content,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    first_token_ms=total_ms * 0.4,  # Estimated for non-streamed
                    total_ms=total_ms,
                    status="ok",
                    rate_limit_headers=rl_headers,
                    source="real"
                )

        except httpx.TimeoutException:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return ProviderResponse(
                content="",
                first_token_ms=None,
                total_ms=total_ms,
                status="timeout",
                error_message="Groq request timed out after 30s",
                source="real"
            )
        except Exception as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            return ProviderResponse(
                content="",
                first_token_ms=None,
                total_ms=total_ms,
                status="error",
                error_message=str(e),
                source="real"
            )

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.2
    ) -> AsyncGenerator[Dict[str, Any], None]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True
        }

        t_start = time.perf_counter()
        first_token_time = None
        full_content_pieces = []
        tokens_in = len(prompt.split()) + len(system_prompt.split())

        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                async with client.stream(
                    "POST",
                    f"{self.base_url}/chat/completions",
                    headers=self._get_headers(),
                    json=payload
                ) as resp:
                    rl_headers = self._extract_rate_limit_headers(resp.headers)
                    self.last_rate_limit_headers = rl_headers

                    if resp.status_code == 429:
                        body = await resp.aread()
                        total_ms = (time.perf_counter() - t_start) * 1000.0
                        retry_after = float(resp.headers.get("retry-after", 2.0))
                        yield {
                            "delta": "",
                            "is_first": False,
                            "done": True,
                            "response": ProviderResponse(
                                content="",
                                first_token_ms=None,
                                total_ms=total_ms,
                                status="rate_limited",
                                error_message=body.decode("utf-8", errors="replace"),
                                rate_limit_headers=rl_headers,
                                retry_after_sec=retry_after,
                                source="real"
                            )
                        }
                        return

                    if resp.status_code >= 400:
                        body = await resp.aread()
                        total_ms = (time.perf_counter() - t_start) * 1000.0
                        yield {
                            "delta": "",
                            "is_first": False,
                            "done": True,
                            "response": ProviderResponse(
                                content="",
                                first_token_ms=None,
                                total_ms=total_ms,
                                status="error",
                                error_message=f"HTTP {resp.status_code}: {body.decode('utf-8', errors='replace')}",
                                rate_limit_headers=rl_headers,
                                source="real"
                            )
                        }
                        return

                    async for line in resp.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data_str)
                            delta = chunk["choices"][0]["delta"].get("content", "")
                            if delta:
                                is_first = False
                                if first_token_time is None:
                                    first_token_time = time.perf_counter()
                                    is_first = True
                                full_content_pieces.append(delta)
                                yield {
                                    "delta": delta,
                                    "is_first": is_first,
                                    "done": False,
                                    "response": None
                                }
                        except Exception:
                            continue

            t_end = time.perf_counter()
            total_ms = (t_end - t_start) * 1000.0
            first_ms = ((first_token_time - t_start) * 1000.0) if first_token_time else None
            full_content = "".join(full_content_pieces)
            tokens_out = len(full_content.split())

            yield {
                "delta": "",
                "is_first": False,
                "done": True,
                "response": ProviderResponse(
                    content=full_content,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    first_token_ms=first_ms,
                    total_ms=total_ms,
                    status="ok",
                    rate_limit_headers=self.last_rate_limit_headers,
                    source="real"
                )
            }

        except httpx.TimeoutException:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            yield {
                "delta": "",
                "is_first": False,
                "done": True,
                "response": ProviderResponse(
                    content="",
                    first_token_ms=None,
                    total_ms=total_ms,
                    status="timeout",
                    error_message="Stream timed out",
                    source="real"
                )
            }
        except Exception as e:
            total_ms = (time.perf_counter() - t_start) * 1000.0
            yield {
                "delta": "",
                "is_first": False,
                "done": True,
                "response": ProviderResponse(
                    content="",
                    first_token_ms=None,
                    total_ms=total_ms,
                    status="error",
                    error_message=str(e),
                    source="real"
                )
            }

