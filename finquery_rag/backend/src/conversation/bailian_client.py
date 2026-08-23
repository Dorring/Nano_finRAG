"""Alibaba Cloud Bailian (DashScope) Qwen3.6-Flash Client Wrapper.

Provides an OpenAI-compatible HTTP client for the Conversation Context Layer with:
- Environment variable configuration
- Thinking mode explicitly disabled (enable_thinking=False)
- Bounded retries with exponential backoff and jitter on 429/timeout
- Strict output token caps
- Mock/Offline support for test isolation
"""

from __future__ import annotations

import json
import os
import random
import time
import urllib.error
import urllib.request
from typing import Any


class BailianClient:
    """Client for Alibaba Cloud Bailian Qwen3.6-Flash model."""

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_ms: float | None = None,
        max_retries: int | None = None,
        max_output_tokens: int | None = None,
        enable_thinking: bool | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("BAILIAN_API_KEY", "")
        self.base_url = (base_url or os.environ.get("BAILIAN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")).rstrip("/")
        self.model = model or os.environ.get("BAILIAN_CONTEXT_MODEL", "qwen3.6-flash")
        
        # Configuration
        timeout_val = timeout_ms or float(os.environ.get("CONTEXT_RESOLUTION_TIMEOUT_MS", "3000"))
        self.timeout_sec = max(0.5, timeout_val / 1000.0)
        self.max_retries = max_retries if max_retries is not None else int(os.environ.get("CONTEXT_MAX_RETRIES", "2"))
        self.max_output_tokens = max_output_tokens or int(os.environ.get("CONTEXT_RESOLVER_MAX_OUTPUT_TOKENS", "512"))
        
        if enable_thinking is not None:
            self.enable_thinking = enable_thinking
        else:
            thinking_env = os.environ.get("BAILIAN_CONTEXT_THINKING", "false").lower()
            self.enable_thinking = thinking_env in ("1", "true", "yes")

    def chat_completion(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        response_format: dict[str, str] | None = None,
    ) -> str | None:
        """Invokes chat completion with bounded retries and JSON decoding."""
        if not self.api_key:
            # Offline/unconfigured environment: return None to allow resolver deterministic fallback
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": self.max_output_tokens,
        }
        if response_format:
            payload["response_format"] = response_format
            
        if not self.enable_thinking:
            payload["enable_thinking"] = False

        data_bytes = json.dumps(payload).encode("utf-8")
        
        for attempt in range(self.max_retries + 1):
            try:
                req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
                with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    choices = resp_data.get("choices", [])
                    if choices and "message" in choices[0]:
                        return choices[0]["message"].get("content", "")
                    return None
            except urllib.error.HTTPError as e:
                if e.code == 429 or e.code >= 500:
                    if attempt < self.max_retries:
                        # Exponential backoff with jitter
                        backoff = (0.2 * (2 ** attempt)) + (random.uniform(0.01, 0.1))
                        time.sleep(backoff)
                        continue
                return None
            except Exception:
                if attempt < self.max_retries:
                    time.sleep(0.1 + random.uniform(0.01, 0.05))
                    continue
                return None
        return None
