"""Resilient HTTP session adapter with automatic exponential backoff, jitter,
and HTTP 429 Too Many Requests (Retry-After) handling.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Mapping, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

logger = logging.getLogger("archi.http")

DEFAULT_STATUS_FORCELIST: tuple[int, ...] = (429, 500, 502, 503, 504)
DEFAULT_RETRIES: int = 5
DEFAULT_BACKOFF_FACTOR: float = 1.0
DEFAULT_TIMEOUT: float = 30.0
USER_AGENT: str = "archi-kb/3.0 (+https://github.com/archi-physics/archi)"


class TimeoutHTTPAdapter(HTTPAdapter):
    """HTTPAdapter that enforces a default timeout if none was passed."""

    def __init__(self, timeout: float = DEFAULT_TIMEOUT, **kwargs: Any) -> None:
        self.default_timeout = timeout
        super().__init__(**kwargs)

    def send(self, request: requests.PreparedRequest, **kwargs: Any) -> requests.Response:
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.default_timeout
        return super().send(request, **kwargs)


def create_resilient_session(
    *,
    retries: int = DEFAULT_RETRIES,
    backoff_factor: float = DEFAULT_BACKOFF_FACTOR,
    status_forcelist: Iterable[int] = DEFAULT_STATUS_FORCELIST,
    timeout: float = DEFAULT_TIMEOUT,
    headers: Optional[Mapping[str, str]] = None,
    cookies: Any = None,
) -> requests.Session:
    """Create a requests.Session configured with automatic retries, backoff, and default timeouts."""
    session = requests.Session()

    retry_strategy = Retry(
        total=retries,
        read=retries,
        connect=retries,
        backoff_factor=backoff_factor,
        status_forcelist=tuple(status_forcelist),
        raise_on_status=False,
        respect_retry_after_header=True,
    )

    adapter = TimeoutHTTPAdapter(timeout=timeout, max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    session.headers["User-Agent"] = USER_AGENT
    if headers:
        session.headers.update(headers)

    if cookies is not None:
        session.cookies = cookies

    return session
