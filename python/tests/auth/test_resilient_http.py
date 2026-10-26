"""Tests for archi.auth.http: resilient HTTP session and TimeoutHTTPAdapter."""
from __future__ import annotations

import http.cookiejar
import io
from unittest.mock import MagicMock, patch

import pytest
import requests
from urllib3.exceptions import MaxRetryError
from urllib3.response import HTTPResponse
from urllib3.util import Retry

from archi.auth.http import (
    DEFAULT_BACKOFF_FACTOR,
    DEFAULT_RETRIES,
    DEFAULT_STATUS_FORCELIST,
    DEFAULT_TIMEOUT,
    USER_AGENT,
    TimeoutHTTPAdapter,
    create_resilient_session,
)


def test_default_session_configuration():
    session = create_resilient_session()
    assert isinstance(session, requests.Session)
    assert session.headers["User-Agent"] == USER_AGENT

    adapter_https = session.adapters["https://"]
    adapter_http = session.adapters["http://"]
    assert isinstance(adapter_https, TimeoutHTTPAdapter)
    assert isinstance(adapter_http, TimeoutHTTPAdapter)
    assert adapter_https.default_timeout == DEFAULT_TIMEOUT

    retries = adapter_https.max_retries
    assert isinstance(retries, Retry)
    assert retries.total == DEFAULT_RETRIES
    assert retries.backoff_factor == DEFAULT_BACKOFF_FACTOR
    assert set(retries.status_forcelist) == set(DEFAULT_STATUS_FORCELIST)
    assert retries.respect_retry_after_header is True


def test_session_custom_options():
    custom_headers = {"Authorization": "Bearer test-token", "X-Custom": "val"}
    jar = http.cookiejar.CookieJar()
    session = create_resilient_session(
        retries=2,
        backoff_factor=0.1,
        status_forcelist=(429, 503),
        timeout=15.0,
        headers=custom_headers,
        cookies=jar,
    )
    assert session.headers["Authorization"] == "Bearer test-token"
    assert session.headers["X-Custom"] == "val"
    assert session.cookies is jar

    adapter = session.adapters["https://"]
    assert adapter.default_timeout == 15.0
    assert adapter.max_retries.total == 2
    assert set(adapter.max_retries.status_forcelist) == {429, 503}


def test_timeout_adapter_applies_default_timeout():
    adapter = TimeoutHTTPAdapter(timeout=42.0)
    request = requests.Request("GET", "https://example.cern.ch/api").prepare()

    with patch.object(requests.adapters.HTTPAdapter, "send") as mock_send:
        mock_send.return_value = requests.Response()
        adapter.send(request)
        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs.get("timeout") == 42.0


def test_timeout_adapter_respects_explicit_timeout():
    adapter = TimeoutHTTPAdapter(timeout=42.0)
    request = requests.Request("GET", "https://example.cern.ch/api").prepare()

    with patch.object(requests.adapters.HTTPAdapter, "send") as mock_send:
        mock_send.return_value = requests.Response()
        adapter.send(request, timeout=10.0)
        mock_send.assert_called_once()
        _, kwargs = mock_send.call_args
        assert kwargs.get("timeout") == 10.0


def test_session_retries_on_429_then_succeeds():
    session = create_resilient_session(retries=2, backoff_factor=0.01)

    call_count = {"n": 0}

    def fake_urlopen(conn, *args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return HTTPResponse(
                body=io.BytesIO(b"rate limited"),
                status=429,
                headers={"Retry-After": "0"},
                preload_content=False,
            )
        return HTTPResponse(
            body=io.BytesIO(b"{\"status\": \"ok\"}"),
            status=200,
            preload_content=False,
        )

    with patch("urllib3.connectionpool.HTTPConnectionPool._make_request", side_effect=fake_urlopen):
        resp = session.get("http://test.cern.ch/api")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}
        assert call_count["n"] == 2


def test_session_retries_exhausted_on_500():
    session = create_resilient_session(retries=2, backoff_factor=0.01)

    call_count = {"n": 0}

    def fake_urlopen(conn, *args, **kwargs):
        call_count["n"] += 1
        return HTTPResponse(
            body=io.BytesIO(b"server error"),
            status=500,
            preload_content=False,
        )

    with patch("urllib3.connectionpool.HTTPConnectionPool._make_request", side_effect=fake_urlopen):
        resp = session.get("http://test.cern.ch/error")
        # With raise_on_status=False in Retry, final response with 500 is returned after retries
        assert resp.status_code == 500
        # 1 initial + 2 retries = 3 attempts total
        assert call_count["n"] == 3
