"""HTTP client for OpenAI-compatible chat completions."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx


class LLMAuthError(RuntimeError):
    pass


class LLMRateLimitError(RuntimeError):
    pass


class LLMServerError(RuntimeError):
    pass


class LLMTimeoutError(RuntimeError):
    pass


class LLMEmptyResponseError(RuntimeError):
    pass


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass(frozen=True)
class ChatCompletionResult:
    content: str
    latency_ms: int


def _classify_http_error(status_code: int, body: str) -> RuntimeError:
    if status_code in (401, 403):
        return LLMAuthError(f"LLM authentication failed ({status_code})")
    if status_code == 429:
        return LLMRateLimitError(f"LLM rate limited ({status_code})")
    if status_code >= 500:
        return LLMServerError(f"LLM server error ({status_code}): {body[:200]}")
    return RuntimeError(f"LLM request failed ({status_code}): {body[:200]}")


class OpenAIChatClient:
    def __init__(
        self,
        *,
        api_url: str,
        api_key: str,
        model: str,
        temperature: float,
        max_tokens: int,
        timeout_sec: int,
        retry_max: int = 2,
        retry_backoff_sec: float = 2.0,
        deployment: str | None = None,
        api_version: str | None = None,
        provider: str = "openai_compatible",
    ) -> None:
        self._api_url = api_url.rstrip("/")
        self._api_key = api_key
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_sec = timeout_sec
        self._retry_max = retry_max
        self._retry_backoff_sec = retry_backoff_sec
        self._deployment = deployment
        self._api_version = api_version
        self._provider = provider

    def _build_url(self) -> str:
        if self._provider == "azure":
            assert self._deployment and self._api_version
            return (
                f"{self._api_url}/openai/deployments/{self._deployment}"
                f"/chat/completions?api-version={self._api_version}"
            )
        return f"{self._api_url}/chat/completions"

    def _build_payload(self, messages: list[ChatMessage]) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        if self._provider != "azure":
            payload["model"] = self._model
        return payload

    def complete(self, messages: list[ChatMessage]) -> ChatCompletionResult:
        url = self._build_url()
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = self._build_payload(messages)
        last_exc: Exception | None = None

        for attempt in range(self._retry_max + 1):
            start = time.perf_counter()
            try:
                with httpx.Client(timeout=self._timeout_sec) as client:
                    resp = client.post(url, headers=headers, json=payload)
                latency_ms = int((time.perf_counter() - start) * 1000)

                if resp.status_code >= 400:
                    exc = _classify_http_error(resp.status_code, resp.text)
                    if isinstance(exc, (LLMRateLimitError, LLMServerError)):
                        last_exc = exc
                        if attempt < self._retry_max:
                            time.sleep(self._retry_backoff_sec * (2**attempt))
                            continue
                    raise exc

                data = resp.json()
                choices = data.get("choices") or []
                if not choices:
                    raise LLMEmptyResponseError("LLM returned no choices")
                content = (choices[0].get("message") or {}).get("content", "")
                if not str(content).strip():
                    raise LLMEmptyResponseError("LLM returned empty content")
                return ChatCompletionResult(content=str(content).strip(), latency_ms=latency_ms)

            except httpx.TimeoutException as exc:
                last_exc = LLMTimeoutError("LLM request timed out")
                if attempt < self._retry_max:
                    time.sleep(self._retry_backoff_sec * (2**attempt))
                    continue
                raise last_exc from exc

        if last_exc:
            raise last_exc
        raise RuntimeError("LLM request failed")
