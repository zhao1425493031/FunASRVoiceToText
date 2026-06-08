"""OpenAIChatClient HTTP error handling tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from voicetotext.llm.client import (
    ChatMessage,
    LLMAuthError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    OpenAIChatClient,
)

from tests.conftest import mock_llm_http_body


def _client() -> OpenAIChatClient:
    return OpenAIChatClient(
        api_url="https://mock-llm.test/v1",
        api_key="sk-test",
        model="qwen-test",
        temperature=0.3,
        max_tokens=1024,
        timeout_sec=10,
        retry_max=0,
        retry_backoff_sec=0.01,
    )


def test_client_success_200() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_llm_http_body()
    with patch("httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        result = _client().complete([ChatMessage(role="user", content="hi")])
    assert "测试会议" in result.content
    assert result.latency_ms >= 0


def test_client_401_raises_auth_error() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "unauthorized"
    with patch("httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        with pytest.raises(LLMAuthError):
            _client().complete([ChatMessage(role="user", content="hi")])


def test_client_429_raises_rate_limit() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 429
    mock_resp.text = "rate limited"
    with patch("httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        with pytest.raises(LLMRateLimitError):
            _client().complete([ChatMessage(role="user", content="hi")])


def test_client_500_raises_server_error() -> None:
    mock_resp = MagicMock()
    mock_resp.status_code = 502
    mock_resp.text = "bad gateway"
    with patch("httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.return_value = mock_resp
        with pytest.raises(LLMServerError):
            _client().complete([ChatMessage(role="user", content="hi")])


def test_client_timeout_raises_timeout_error() -> None:
    with patch("httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.post.side_effect = httpx.TimeoutException(
            "timeout"
        )
        with pytest.raises(LLMTimeoutError):
            _client().complete([ChatMessage(role="user", content="hi")])
