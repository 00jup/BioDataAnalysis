"""공통 HTTP 래퍼 (tenacity retry + requests-cache)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import requests
import requests_cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

CACHE_DIR = Path.home() / ".cache" / "marketed_drugs"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_session: requests_cache.CachedSession | None = None


def get_session() -> requests_cache.CachedSession:
    """공유 캐시 세션을 반환 (싱글톤)."""
    global _session
    if _session is None:
        _session = requests_cache.CachedSession(
            cache_name=str(CACHE_DIR / "http_cache"),
            backend="sqlite",
            expire_after=7 * 24 * 60 * 60,
            allowable_methods=["GET"],
            allowable_codes=[200, 404],
        )
        _session.headers.update({"User-Agent": "marketed-drugs-collector/1.0"})
    return _session


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, requests.HTTPError)),
    reraise=True,
)
def fetch_json(url: str, params: dict[str, Any] | None = None, timeout: int = 30) -> Any:
    """JSON GET 요청 + 재시도. requests-cache로 자동 캐싱."""
    session = get_session()
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, requests.HTTPError)),
    reraise=True,
)
def fetch_text(url: str, params: dict[str, Any] | None = None, timeout: int = 60) -> str:
    """텍스트 GET 요청 + 재시도."""
    session = get_session()
    response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.text


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=2, min=4, max=60),
    retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout, requests.HTTPError)),
    reraise=True,
)
def fetch_bytes(url: str, params: dict[str, Any] | None = None, timeout: int = 120) -> bytes:
    """바이너리 GET 요청 + 재시도. 큰 dump 다운로드용."""
    session = get_session()
    response = session.get(url, params=params, timeout=timeout, stream=True)
    response.raise_for_status()
    return response.content
