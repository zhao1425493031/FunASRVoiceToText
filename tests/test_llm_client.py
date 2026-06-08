"""LLM HTTP client tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx
import pytest

from tests.conftest_llm import mock_chat_response, sample_summary_json
from voicetotext.llm.client import (
    ChatMessage,
    LLMAuthError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
    OpenAIChatClient,
)


def _client() -> OpenAIChatClient:
    return OpenAIChatClient(
        api_url="https://example.test/v1",
        api_key="sk-testkey1234567890abcdef",
        model="qwen-plus",
        temperature=0.3,
        max_tokens=1024,
        timeout_sec=10,
        retry_max=2,
        retry_backoff_sec=0,
    )


def test_complete_success() -> None:
    client = _client()
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = mock_chat_response(sample_summary_json())

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        result = client.complete([ChatMessage("system", "s"), ChatMessage("user", "u")])
        assert "案件進捗確認" in result.content
        assert result.latency_ms >= 0


def test_complete_auth_error() -> None:
    client = _client()
    mock_resp = MagicMock()
    mock_resp.status_code = 401
    mock_resp.text = "unauthorized"

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        with pytest.raises(LLMAuthError):
            client.complete([ChatMessage("user", "u")])


def test_complete_rate_limit_retries() -> None:
    client = _client()
    bad = MagicMock(status_code=429, text="rate")
    good = MagicMock(status_code=200)
    good.json.return_value = mock_chat_response(sample_summary_json())

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = [bad, good]
        mock_client_cls.return_value = mock_client

        result = client.complete([ChatMessage("user", "u")])
        assert result.content


def test_complete_timeout() -> None:
    client = _client()
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.side_effect = httpx.TimeoutException("timeout")
        mock_client_cls.return_value = mock_client

        with pytest.raises(LLMTimeoutError):
            client.complete([ChatMessage("user", "u")])


def test_complete_server_error() -> None:
    client = _client()
    mock_resp = MagicMock(status_code=500, text="boom")
    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__.return_value = mock_client
        mock_client.post.return_value = mock_resp
        mock_client_cls.return_value = mock_client

        with pytest.raises(LLMServerError):
            client.complete([ChatMessage("user", "u")])
