"""Quality-first browser traffic optimizer for the V4 Arkose worker.

Only immutable, public Arkose client assets are eligible for direct download
and replay. Challenge/session requests always remain on the browser route.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import logging
import os
import re
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import parse_qsl, urlsplit, urlunsplit

import requests


LOG = logging.getLogger("ruyipage_http_v11")
MIB = 1024 * 1024
CACHE_VERSION = 1

_SESSION_QUERY_NAMES = {
    "blob",
    "challenge",
    "data",
    "gametoken",
    "license_blob",
    "requestedid",
    "requestid",
    "rid",
    "session",
    "sessionid",
    "sessiontoken",
    "signature",
    "st",
    "token",
}
_SESSION_PATH_PARTS = (
    "/fc/ca/",
    "/fc/gt2/",
    "/fc/gc/",
    "/fc/a/",
    "/rtig/",
)
_STATIC_EXTENSIONS = (
    ".css",
    ".gif",
    ".ico",
    ".jpeg",
    ".jpg",
    ".js",
    ".json",
    ".map",
    ".otf",
    ".png",
    ".svg",
    ".ttf",
    ".wasm",
    ".webp",
    ".woff",
    ".woff2",
)
_TEXT_CONTENT_TYPES = (
    "application/javascript",
    "application/json",
    "application/wasm",
    "application/x-javascript",
    "font/",
    "image/",
    "text/css",
    "text/javascript",
)
_REPLAY_HEADER_NAMES = {
    "access-control-allow-credentials",
    "access-control-allow-headers",
    "access-control-allow-methods",
    "access-control-allow-origin",
    "cache-control",
    "content-language",
    "content-type",
    "cross-origin-resource-policy",
    "etag",
    "expires",
    "last-modified",
    "timing-allow-origin",
    "vary",
    "x-content-type-options",
}
_REQUEST_HEADER_NAMES = {
    "accept",
    "accept-language",
    "origin",
    "referer",
    "user-agent",
}
_ALLOWED_VARY_HEADERS = {"accept-encoding", "origin"}
_CHALLENGE_IMAGE_QUERY_NAMES = {
    "challenge",
    "expires",
    "gametoken",
    "sessiontoken",
    "signature",
}


def _lower_headers(headers: Optional[Mapping[str, Any]]) -> dict[str, str]:
    return {
        str(name).strip().lower(): str(value).strip()
        for name, value in dict(headers or {}).items()
        if str(name).strip()
    }


def _arkose_host(host: str) -> bool:
    normalized = str(host or "").strip(".").lower()
    return normalized == "arkoselabs.com" or normalized.endswith(".arkoselabs.com")


def query_names(url: str) -> set[str]:
    return {
        str(name).strip().lower()
        for name, _value in parse_qsl(urlsplit(str(url or "")).query, keep_blank_values=True)
    }


def _normalized_query_names(url: str) -> set[str]:
    return {
        re.sub(r"[^a-z0-9]", "", name.lower())
        for name in query_names(url)
    }


def is_session_bound_url(url: str) -> bool:
    parsed = urlsplit(str(url or ""))
    path = parsed.path.lower()
    if any(part in path for part in _SESSION_PATH_PARTS):
        return True
    return bool(_normalized_query_names(url) & _SESSION_QUERY_NAMES)


def is_public_static_candidate(url: str, method: str = "GET") -> bool:
    if str(method or "").upper() != "GET":
        return False
    parsed = urlsplit(str(url or ""))
    if parsed.scheme.lower() != "https" or not _arkose_host(parsed.hostname or ""):
        return False
    if parsed.username or parsed.password or parsed.fragment or is_session_bound_url(url):
        return False

    path = parsed.path.lower()
    if path.startswith(("/cdn/fc/", "/fc/assets/")):
        return path.endswith(_STATIC_EXTENSIONS)
    return bool(re.fullmatch(r"/v2/[0-9a-f-]{20,}/api\.js", path, re.I))


def is_direct_challenge_image_candidate(url: str, method: str = "GET") -> bool:
    if str(method or "").upper() != "GET":
        return False
    parsed = urlsplit(str(url or ""))
    if parsed.scheme.lower() != "https" or not _arkose_host(parsed.hostname or ""):
        return False
    if parsed.username or parsed.password or parsed.fragment:
        return False
    if parsed.path.lower() != "/rtig/image":
        return False
    return _CHALLENGE_IMAGE_QUERY_NAMES <= _normalized_query_names(url)


def is_complete_jpeg(body: bytes) -> bool:
    data = bytes(body or b"")
    return (
        len(data) >= 1024
        and data.startswith(b"\xff\xd8")
        and data.endswith(b"\xff\xd9")
    )


def public_cache_lifetime(headers: Mapping[str, Any]) -> int:
    normalized = _lower_headers(headers)
    if "set-cookie" in normalized:
        return 0
    cache_control = normalized.get("cache-control", "").lower()
    if any(value in cache_control for value in ("private", "no-store", "no-cache")):
        return 0
    vary = {
        item.strip().lower()
        for item in normalized.get("vary", "").split(",")
        if item.strip()
    }
    if vary - _ALLOWED_VARY_HEADERS:
        return 0

    def directive_age(name: str) -> Optional[int]:
        match = re.search(
            rf"(?:^|,)\s*{re.escape(name)}\s*=\s*\"?(\d+)",
            cache_control,
        )
        return int(match.group(1)) if match else None

    shared_age = directive_age("s-maxage")
    browser_age = directive_age("max-age")
    max_age = shared_age if shared_age is not None else (browser_age or 0)
    if "immutable" in cache_control and max_age <= 0:
        max_age = 7 * 24 * 60 * 60
    if max_age <= 0:
        return 0
    return min(max_age, 30 * 24 * 60 * 60)


def _allowed_content_type(content_type: str, url: str) -> bool:
    value = str(content_type or "").split(";", 1)[0].strip().lower()
    if any(value == prefix or value.startswith(prefix) for prefix in _TEXT_CONTENT_TYPES):
        return True
    return urlsplit(url).path.lower().endswith(_STATIC_EXTENSIONS)


def sanitized_url(url: str) -> str:
    parsed = urlsplit(str(url or ""))
    names = sorted(query_names(url))
    query = "&".join(f"{name}=<redacted>" for name in names)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, query, ""))


def resource_type(url: str, content_type: str = "") -> str:
    content = str(content_type or "").split(";", 1)[0].lower()
    path = urlsplit(str(url or "")).path.lower()
    if "javascript" in content or path.endswith((".js", ".mjs")):
        return "script"
    if content == "text/css" or path.endswith(".css"):
        return "stylesheet"
    if content.startswith("font/") or path.endswith((".woff", ".woff2", ".ttf", ".otf")):
        return "font"
    if content.startswith("image/") or path.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg")):
        return "image"
    if content == "application/wasm" or path.endswith(".wasm"):
        return "wasm"
    if content in {"application/json", "text/json"} or path.endswith(".json"):
        return "json"
    if content.startswith("text/html"):
        return "document"
    return "other"


def _decode_typed_bytes(value: Any) -> bytes:
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    if not isinstance(value, dict):
        return str(value).encode("utf-8", "replace")
    for key in ("bytes", "base64"):
        if key not in value:
            continue
        decoded = _decode_typed_bytes(value.get(key))
        if decoded:
            return decoded
    typ = value.get("type")
    raw = value.get("value")
    if typ == "base64" and raw is not None:
        with contextlib.suppress(Exception):
            return base64.b64decode(str(raw))
        return b""
    if raw is not None:
        return str(raw).encode("utf-8")
    return b""


@dataclass(frozen=True)
class CachedResponse:
    body: bytes
    headers: dict[str, str]
    status_code: int = 200
    reason_phrase: str = "OK"
    expires_at: float = 0.0
    vary_request_headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FetchOutcome:
    response: Optional[CachedResponse]
    reason: str
    elapsed_seconds: float


def is_hard_direct_failure(reason: str) -> bool:
    normalized = str(reason or "").strip().lower()
    status_match = re.fullmatch(r"status-(\d+)", normalized)
    if status_match:
        status = int(status_match.group(1))
        return status in {401, 403, 407, 408, 425, 429} or status >= 500
    return normalized.startswith(
        (
            "connectionerror:",
            "connecttimeout:",
            "contentdecodingerror:",
            "proxyerror:",
            "readtimeout:",
            "retryerror:",
            "sslerror:",
        )
    )


class StaticResponseCache:
    def __init__(self, root: Path, max_entry_bytes: int = 8 * MIB):
        self.root = Path(root).expanduser().resolve()
        self.max_entry_bytes = max(1, int(max_entry_bytes))
        self.root.mkdir(parents=True, exist_ok=True)
        self._locks: dict[str, threading.Lock] = {}
        self._locks_guard = threading.Lock()

    @staticmethod
    def key(url: str) -> str:
        return hashlib.sha256(str(url).encode("utf-8")).hexdigest()

    def lock_for(self, url: str) -> threading.Lock:
        key = self.key(url)
        with self._locks_guard:
            return self._locks.setdefault(key, threading.Lock())

    def _paths(self, url: str) -> tuple[Path, Path]:
        key = self.key(url)
        return self.root / f"{key}.bin", self.root / f"{key}.json"

    def get(
        self,
        url: str,
        request_headers: Optional[Mapping[str, Any]] = None,
    ) -> Optional[CachedResponse]:
        body_path, metadata_path = self._paths(url)
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if int(metadata.get("version") or 0) != CACHE_VERSION:
                return None
            if metadata.get("urlSha256") != self.key(url):
                return None
            expires_at = float(metadata.get("expiresAt") or 0.0)
            if expires_at <= time.time():
                return None
            body = body_path.read_bytes()
            if not body or len(body) > self.max_entry_bytes:
                return None
            if len(body) != int(metadata.get("bytes") or -1):
                return None
            if hashlib.sha256(body).hexdigest() != metadata.get("sha256"):
                return None
            vary_request_headers = {
                str(k).lower(): str(v)
                for k, v in dict(metadata.get("varyRequestHeaders") or {}).items()
            }
            current_headers = _lower_headers(request_headers)
            if any(
                current_headers.get(name, "") != expected
                for name, expected in vary_request_headers.items()
            ):
                return None
            return CachedResponse(
                body=body,
                headers={str(k): str(v) for k, v in dict(metadata.get("headers") or {}).items()},
                status_code=int(metadata.get("statusCode") or 200),
                reason_phrase=str(metadata.get("reasonPhrase") or "OK"),
                expires_at=expires_at,
                vary_request_headers=vary_request_headers,
            )
        except (OSError, ValueError, TypeError):
            return None

    def put(self, url: str, response: CachedResponse) -> bool:
        if (
            not response.body
            or len(response.body) > self.max_entry_bytes
            or response.expires_at <= time.time()
        ):
            return False
        body_path, metadata_path = self._paths(url)
        nonce = f"{os.getpid()}.{threading.get_ident()}.{time.time_ns()}"
        body_tmp = body_path.with_name(body_path.name + f".{nonce}.tmp")
        metadata_tmp = metadata_path.with_name(metadata_path.name + f".{nonce}.tmp")
        metadata = {
            "version": CACHE_VERSION,
            "urlSha256": self.key(url),
            "statusCode": response.status_code,
            "reasonPhrase": response.reason_phrase,
            "headers": response.headers,
            "bytes": len(response.body),
            "sha256": hashlib.sha256(response.body).hexdigest(),
            "storedAt": time.time(),
            "expiresAt": response.expires_at,
            "varyRequestHeaders": response.vary_request_headers,
        }
        try:
            body_tmp.write_bytes(response.body)
            metadata_tmp.write_text(
                json.dumps(metadata, ensure_ascii=True, separators=(",", ":")),
                encoding="utf-8",
            )
            os.replace(body_tmp, body_path)
            os.replace(metadata_tmp, metadata_path)
            return True
        except OSError:
            return False
        finally:
            with contextlib.suppress(OSError):
                body_tmp.unlink()
            with contextlib.suppress(OSError):
                metadata_tmp.unlink()


class DirectPublicStaticFetcher:
    """Fetch public static assets without inheriting any process proxy."""

    def __init__(self, timeout: float = 8.0, max_entry_bytes: int = 8 * MIB):
        self.timeout = max(0.5, float(timeout))
        self.max_entry_bytes = max(1, int(max_entry_bytes))
        self.session = requests.Session()
        self.session.trust_env = False

    def close(self) -> None:
        self.session.close()

    def fetch(self, url: str, request_headers: Mapping[str, Any]) -> FetchOutcome:
        return self._fetch(url, request_headers, ephemeral_image=False)

    def fetch_ephemeral_image(
        self,
        url: str,
        request_headers: Mapping[str, Any],
    ) -> FetchOutcome:
        return self._fetch(url, request_headers, ephemeral_image=True)

    def _fetch(
        self,
        url: str,
        request_headers: Mapping[str, Any],
        *,
        ephemeral_image: bool,
    ) -> FetchOutcome:
        started = time.perf_counter()
        response = None
        headers = _lower_headers(request_headers)
        outgoing = {
            name: value
            for name, value in headers.items()
            if name in _REQUEST_HEADER_NAMES and value
        }
        outgoing["accept-encoding"] = "gzip, deflate"
        try:
            response = self.session.get(
                url,
                headers=outgoing,
                allow_redirects=False,
                stream=True,
                timeout=(min(3.0, self.timeout), self.timeout),
            )
            normalized = _lower_headers(response.headers)
            if response.status_code != 200:
                return FetchOutcome(None, f"status-{response.status_code}", time.perf_counter() - started)
            if not _allowed_content_type(normalized.get("content-type", ""), url):
                return FetchOutcome(None, "unsupported-content-type", time.perf_counter() - started)
            if "set-cookie" in normalized:
                return FetchOutcome(None, "response-sets-cookie", time.perf_counter() - started)
            vary_names = {
                item.strip().lower()
                for item in normalized.get("vary", "").split(",")
                if item.strip()
            }
            if not ephemeral_image and vary_names - _ALLOWED_VARY_HEADERS:
                return FetchOutcome(None, "unsupported-vary", time.perf_counter() - started)

            lifetime = 0 if ephemeral_image else public_cache_lifetime(normalized)
            replay_only = ephemeral_image or lifetime <= 0
            if replay_only and not ephemeral_image:
                cache_control = normalized.get("cache-control", "").lower()
                if any(
                    directive in cache_control
                    for directive in ("private", "no-store", "no-cache")
                ):
                    return FetchOutcome(
                        None,
                        "response-forbids-storage",
                        time.perf_counter() - started,
                    )
                if not (normalized.get("etag") or normalized.get("last-modified")):
                    return FetchOutcome(
                        None,
                        "response-has-no-validator",
                        time.perf_counter() - started,
                    )

            chunks: list[bytes] = []
            total = 0
            for chunk in response.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > self.max_entry_bytes:
                    return FetchOutcome(None, "response-too-large", time.perf_counter() - started)
                chunks.append(chunk)
            body = b"".join(chunks)
            if not body:
                return FetchOutcome(None, "empty-response", time.perf_counter() - started)
            if ephemeral_image:
                content_type = normalized.get("content-type", "").split(";", 1)[0].lower()
                if content_type != "image/jpeg":
                    return FetchOutcome(
                        None,
                        "challenge-image-not-jpeg",
                        time.perf_counter() - started,
                    )
                if not is_complete_jpeg(body):
                    return FetchOutcome(
                        None,
                        "challenge-image-incomplete",
                        time.perf_counter() - started,
                    )
                if not normalized.get("content-encoding"):
                    expected_length = 0
                    with contextlib.suppress(TypeError, ValueError):
                        expected_length = int(normalized.get("content-length") or 0)
                    if expected_length > 0 and expected_length != len(body):
                        return FetchOutcome(
                            None,
                            "challenge-image-length-mismatch",
                            time.perf_counter() - started,
                        )

            replay_headers = {
                name: value
                for name, value in normalized.items()
                if name in _REPLAY_HEADER_NAMES
            }
            replay_headers.pop("content-encoding", None)
            replay_headers["content-length"] = str(len(body))
            cached = CachedResponse(
                body=body,
                headers=replay_headers,
                status_code=200,
                reason_phrase=str(response.reason or "OK"),
                expires_at=(time.time() + lifetime) if not replay_only else 0.0,
                vary_request_headers={
                    name: outgoing.get(name, "")
                    for name in vary_names
                    if name != "accept-encoding"
                },
            )
            return FetchOutcome(
                cached,
                (
                    "ok-ephemeral-image"
                    if ephemeral_image
                    else ("ok-replay-only" if replay_only else "ok")
                ),
                time.perf_counter() - started,
            )
        except requests.RequestException as exc:
            return FetchOutcome(
                None,
                f"{type(exc).__name__}: {exc}",
                time.perf_counter() - started,
            )
        finally:
            with contextlib.suppress(Exception):
                if response is not None:
                    response.close()


class BrowserResourceOptimizer:
    def __init__(
        self,
        page: Any,
        cache_dir: Path,
        *,
        proxy_enabled: bool,
        direct_public_static: bool = True,
        direct_challenge_images: bool = True,
        block_nonessential: bool = True,
        fetch_timeout: float = 8.0,
        max_entry_bytes: int = 8 * MIB,
        direct_failure_limit: int = 2,
        should_block=None,
        fetcher: Optional[Any] = None,
    ):
        self.page = page
        self.cache = StaticResponseCache(cache_dir, max_entry_bytes=max_entry_bytes)
        self.proxy_enabled = bool(proxy_enabled)
        self.direct_public_static = bool(direct_public_static)
        self.direct_challenge_images = bool(direct_challenge_images)
        self.block_nonessential = bool(block_nonessential)
        self.direct_failure_limit = max(1, int(direct_failure_limit))
        self.should_block = should_block or (lambda _url: False)
        self.fetcher = fetcher or DirectPublicStaticFetcher(fetch_timeout, max_entry_bytes)
        self._owns_fetcher = fetcher is None
        self._started_at = time.time()
        self._lock = threading.RLock()
        self._handlers_done = threading.Condition(self._lock)
        self._active_handlers = 0
        self._started = False
        self._stopped = False
        self._local_request_ids: set[str] = set()
        self._counts: Counter[str] = Counter()
        self._bytes: Counter[str] = Counter()
        self._failure_reasons: Counter[str] = Counter()
        self._resources: list[dict[str, Any]] = []
        self._direct_fallbacks: list[dict[str, Any]] = []
        self._direct_failures = 0
        self._direct_circuit_open = False
        self._challenge_image_circuit_open = False

    def start(self) -> None:
        self.page.intercept.start(
            self._handle,
            phases=["beforeRequestSent", "responseStarted"],
            collect_response=True,
        )
        self._started = True

    def _handle(self, request: Any) -> None:
        with self._lock:
            self._active_handlers += 1
        try:
            if bool(getattr(request, "is_response_phase", False)):
                self._handle_response(request)
            else:
                self._handle_request(request)
        finally:
            with self._lock:
                self._active_handlers -= 1
                self._handlers_done.notify_all()

    def _handle_request(self, request: Any) -> None:
        url = str(getattr(request, "url", "") or "")
        method = str(getattr(request, "method", "GET") or "GET").upper()
        request_id = str(getattr(request, "request_id", "") or "")
        with self._lock:
            self._counts["requests"] += 1

        if self.block_nonessential and self.should_block(url):
            request.fail()
            with self._lock:
                self._counts["blockedRequests"] += 1
            return

        if is_direct_challenge_image_candidate(url, method):
            self._handle_challenge_image_request(
                request,
                url=url,
                request_id=request_id,
            )
            return

        if is_session_bound_url(url):
            request.continue_request()
            with self._lock:
                self._counts["sessionBoundRequests"] += 1
                self._counts["proxyPassThroughRequests"] += int(self.proxy_enabled)
            return

        if not is_public_static_candidate(url, method):
            request.continue_request()
            with self._lock:
                self._counts["proxyPassThroughRequests"] += int(self.proxy_enabled)
            return

        with self._lock:
            self._counts["publicStaticCandidates"] += 1
        key_lock = self.cache.lock_for(url)
        with key_lock:
            request_headers = getattr(request, "headers", {}) or {}
            cached = self.cache.get(url, request_headers)
            route = "cache"
            outcome: Optional[FetchOutcome] = None
            direct_allowed = self.direct_public_static and not self._direct_circuit_open
            if cached is None and direct_allowed:
                outcome = self.fetcher.fetch(url, request_headers)
                cached = outcome.response
                route = "direct-public-static"
                if cached is not None and cached.expires_at > time.time():
                    self.cache.put(url, cached)

            if cached is not None:
                try:
                    request.mock(
                        cached.body,
                        status_code=cached.status_code,
                        headers=cached.headers,
                        reason_phrase=cached.reason_phrase,
                    )
                except Exception as exc:
                    # RuyiPage marks a request handled before sending provideResponse.
                    # Restore the flag so a late/stale intercepted request can fall back.
                    with contextlib.suppress(Exception):
                        request._handled = False
                        request.continue_request()
                    with self._lock:
                        self._counts["mockFailures"] += 1
                        self._failure_reasons[f"mock-{type(exc).__name__}: {exc}"[:240]] += 1
                        self._counts["proxyPassThroughRequests"] += int(self.proxy_enabled)
                    return
                with self._lock:
                    if request_id:
                        self._local_request_ids.add(request_id)
                    if route == "cache":
                        self._counts["cacheHits"] += 1
                        self._bytes["cacheHitBytes"] += len(cached.body)
                    else:
                        self._counts["directStaticFetches"] += 1
                        if outcome and outcome.reason == "ok-replay-only":
                            self._counts["directStaticReplayOnly"] += 1
                            self._bytes["directStaticReplayOnlyBytes"] += len(
                                cached.body
                            )
                        self._bytes["directStaticBytes"] += len(cached.body)
                        self._bytes["directStaticFetchMillis"] += int(
                            1000 * float(outcome.elapsed_seconds if outcome else 0.0)
                        )
                    self._bytes["estimatedProxyBytesAvoided"] += len(cached.body)
                    self._record_locked(
                        url=url,
                        route=route,
                        status=cached.status_code,
                        content_type=cached.headers.get("content-type", ""),
                        wire_bytes=len(cached.body),
                        decoded_bytes=len(cached.body),
                    )
                return

            request.continue_request()
            with self._lock:
                self._counts["directStaticFallbacks"] += int(direct_allowed)
                self._counts["proxyPassThroughRequests"] += int(self.proxy_enabled)
                if outcome is not None:
                    self._failure_reasons[outcome.reason[:240]] += 1
                    hard_failure = is_hard_direct_failure(outcome.reason)
                    self._counts[
                        "directStaticHardFailures"
                        if hard_failure
                        else "directStaticPolicyFallbacks"
                    ] += 1
                    self._direct_fallbacks.append(
                        {
                            "url": sanitized_url(url),
                            "reason": outcome.reason[:240],
                            "hardFailure": hard_failure,
                        }
                    )
                    if hard_failure:
                        self._direct_failures += 1
                        if self._direct_failures >= self.direct_failure_limit:
                            self._direct_circuit_open = True
                            self._counts["directStaticCircuitOpened"] = 1
                elif self._direct_circuit_open:
                    self._counts["directStaticCircuitBypasses"] += 1

    def _handle_challenge_image_request(
        self,
        request: Any,
        *,
        url: str,
        request_id: str,
    ) -> None:
        with self._lock:
            self._counts["sessionBoundRequests"] += 1
            self._counts["challengeImageCandidates"] += 1
            circuit_open = self._challenge_image_circuit_open

        direct_allowed = (
            self.proxy_enabled
            and self.direct_challenge_images
            and not circuit_open
        )
        outcome: Optional[FetchOutcome] = None
        response: Optional[CachedResponse] = None
        if direct_allowed:
            fetch = getattr(self.fetcher, "fetch_ephemeral_image", None)
            if callable(fetch):
                outcome = fetch(url, getattr(request, "headers", {}) or {})
                response = outcome.response
            else:
                outcome = FetchOutcome(None, "ephemeral-fetch-unavailable", 0.0)

        if response is not None:
            try:
                request.mock(
                    response.body,
                    status_code=response.status_code,
                    headers=response.headers,
                    reason_phrase=response.reason_phrase,
                )
            except Exception as exc:
                with contextlib.suppress(Exception):
                    request._handled = False
                    request.continue_request()
                with self._lock:
                    self._counts["challengeImageMockFailures"] += 1
                    self._counts["directChallengeImageFallbacks"] += 1
                    self._counts["proxyPassThroughRequests"] += int(self.proxy_enabled)
                    reason = f"mock-{type(exc).__name__}: {exc}"[:240]
                    self._failure_reasons[f"challenge-image:{reason}"[:240]] += 1
                    self._direct_fallbacks.append(
                        {
                            "url": sanitized_url(url),
                            "reason": reason,
                            "hardFailure": True,
                            "category": "challenge-image",
                        }
                    )
                    self._challenge_image_circuit_open = True
                    self._counts["directChallengeImageCircuitOpened"] = 1
                return

            with self._lock:
                if request_id:
                    self._local_request_ids.add(request_id)
                self._counts["directChallengeImageFetches"] += 1
                self._bytes["directChallengeImageBytes"] += len(response.body)
                self._bytes["directChallengeImageFetchMillis"] += int(
                    1000 * float(outcome.elapsed_seconds if outcome else 0.0)
                )
                self._bytes["estimatedProxyBytesAvoided"] += len(response.body)
                self._record_locked(
                    url=url,
                    route="direct-challenge-image",
                    status=response.status_code,
                    content_type=response.headers.get("content-type", ""),
                    wire_bytes=len(response.body),
                    decoded_bytes=len(response.body),
                )
            return

        request.continue_request()
        with self._lock:
            self._counts["proxyPassThroughRequests"] += int(self.proxy_enabled)
            if direct_allowed:
                self._counts["directChallengeImageFallbacks"] += 1
                reason = outcome.reason if outcome is not None else "unknown"
                self._failure_reasons[f"challenge-image:{reason}"[:240]] += 1
                self._direct_fallbacks.append(
                    {
                        "url": sanitized_url(url),
                        "reason": reason[:240],
                        "hardFailure": True,
                        "category": "challenge-image",
                    }
                )
                self._challenge_image_circuit_open = True
                self._counts["directChallengeImageCircuitOpened"] = 1
            elif circuit_open:
                self._counts["directChallengeImageCircuitBypasses"] += 1

    def _handle_response(self, request: Any) -> None:
        url = str(getattr(request, "url", "") or "")
        request_id = str(getattr(request, "request_id", "") or "")
        status = int(getattr(request, "response_status", 0) or 0)
        headers = _lower_headers(getattr(request, "response_headers", {}) or {})

        with self._lock:
            if request_id and request_id in self._local_request_ids:
                self._local_request_ids.discard(request_id)
                # provideResponse already completed this mocked request. Mark the
                # synthetic response handled so RuyiPage does not auto-continue it.
                request._handled = True
                return
        request.continue_response()

        content_length = 0
        with contextlib.suppress(ValueError, TypeError):
            content_length = max(0, int(headers.get("content-length", "0")))
        body = b""
        cacheable_proxy_static = (
            status == 200
            and is_public_static_candidate(url, str(getattr(request, "method", "GET") or "GET"))
            and public_cache_lifetime(headers) > 0
            and _allowed_content_type(headers.get("content-type", ""), url)
        )
        if cacheable_proxy_static or content_length <= 0:
            body = self._collect_response_bytes(request, timeout=4.0)
        measured_bytes = content_length or len(body)
        route = "proxy" if self.proxy_enabled else "browser-direct"

        if cacheable_proxy_static and body:
            replay_headers = {
                name: value for name, value in headers.items() if name in _REPLAY_HEADER_NAMES
            }
            replay_headers.pop("content-encoding", None)
            replay_headers["content-length"] = str(len(body))
            self.cache.put(
                url,
                CachedResponse(
                    body=body,
                    headers=replay_headers,
                    status_code=status,
                    reason_phrase="OK",
                    expires_at=time.time() + public_cache_lifetime(headers),
                    vary_request_headers={
                        name: _lower_headers(getattr(request, "headers", {}) or {}).get(
                            name, ""
                        )
                        for name in {
                            item.strip().lower()
                            for item in headers.get("vary", "").split(",")
                            if item.strip() and item.strip().lower() != "accept-encoding"
                        }
                    },
                ),
            )
            with self._lock:
                self._counts["proxyStaticCacheStores"] += 1
                self._bytes["proxyStaticCacheStoreBytes"] += len(body)

        with self._lock:
            self._counts["responses"] += 1
            if route == "proxy":
                self._bytes["profiledProxyResponseBytes"] += measured_bytes
            self._record_locked(
                url=url,
                route=route,
                status=status,
                content_type=headers.get("content-type", ""),
                wire_bytes=measured_bytes,
                decoded_bytes=len(body),
            )

    @staticmethod
    def _collect_response_bytes(request: Any, timeout: float) -> bytes:
        collector = getattr(request, "_response_collector", None)
        request_id = str(getattr(request, "request_id", "") or "")
        if collector is None or not request_id:
            return b""
        deadline = time.monotonic() + max(0.1, float(timeout))
        data = b""
        while time.monotonic() < deadline:
            try:
                item = collector.get(request_id, data_type="response")
                raw = getattr(item, "raw", None)
                data = _decode_typed_bytes(raw)
                if data:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        with contextlib.suppress(Exception):
            collector.disown(request_id, data_type="response")
        return data

    def _record_locked(
        self,
        *,
        url: str,
        route: str,
        status: int,
        content_type: str,
        wire_bytes: int,
        decoded_bytes: int,
    ) -> None:
        parsed = urlsplit(url)
        category = "session-bound" if is_session_bound_url(url) else (
            "public-static" if is_public_static_candidate(url) else "other"
        )
        self._resources.append(
            {
                "url": sanitized_url(url),
                "host": (parsed.hostname or "").lower(),
                "route": route,
                "category": category,
                "resourceType": resource_type(url, content_type),
                "status": int(status),
                "contentType": str(content_type or "").split(";", 1)[0],
                "wireBodyBytesEstimate": max(0, int(wire_bytes)),
                "decodedBodyBytes": max(0, int(decoded_bytes)),
            }
        )

    def stop(self) -> dict[str, Any]:
        if self._stopped:
            return self.report()
        self._stopped = True
        if self._started:
            with contextlib.suppress(Exception):
                self.page.intercept.stop()
        deadline = time.monotonic() + 5.0
        with self._lock:
            while self._active_handlers and time.monotonic() < deadline:
                self._handlers_done.wait(timeout=min(0.1, deadline - time.monotonic()))
        if self._owns_fetcher:
            with contextlib.suppress(Exception):
                self.fetcher.close()
        return self.report()

    def report(self) -> dict[str, Any]:
        with self._lock:
            resources = [dict(item) for item in self._resources]
            counts = dict(self._counts)
            byte_counts = dict(self._bytes)
            failures = dict(self._failure_reasons)
            direct_fallbacks = [dict(item) for item in self._direct_fallbacks]

        by_host: dict[str, dict[str, int]] = defaultdict(lambda: {"requests": 0, "bytes": 0})
        by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"requests": 0, "bytes": 0})
        by_route: dict[str, dict[str, int]] = defaultdict(lambda: {"requests": 0, "bytes": 0})
        for item in resources:
            size = int(item.get("wireBodyBytesEstimate") or 0)
            for bucket, key in (
                (by_host, str(item.get("host") or "unknown")),
                (by_type, str(item.get("resourceType") or "other")),
                (by_route, str(item.get("route") or "unknown")),
            ):
                bucket[key]["requests"] += 1
                bucket[key]["bytes"] += size

        top = sorted(
            resources,
            key=lambda item: int(item.get("wireBodyBytesEstimate") or 0),
            reverse=True,
        )[:30]
        return {
            "enabled": True,
            "mode": "bandwidth-first-challenge-images-and-public-static-direct",
            "proxyEnabled": self.proxy_enabled,
            "directPublicStaticEnabled": self.direct_public_static,
            "directChallengeImagesEnabled": self.direct_challenge_images,
            "blockNonessentialEnabled": self.block_nonessential,
            "cacheDir": str(self.cache.root),
            "durationSeconds": round(max(0.0, time.time() - self._started_at), 3),
            "counts": counts,
            "bytes": {
                **byte_counts,
                "directStaticMiB": round(byte_counts.get("directStaticBytes", 0) / MIB, 4),
                "directChallengeImageMiB": round(
                    byte_counts.get("directChallengeImageBytes", 0) / MIB,
                    4,
                ),
                "cacheHitMiB": round(byte_counts.get("cacheHitBytes", 0) / MIB, 4),
                "estimatedProxyMiBAvoided": round(
                    byte_counts.get("estimatedProxyBytesAvoided", 0) / MIB, 4
                ),
            },
            "directFetchFailures": failures,
            "directFallbacks": direct_fallbacks[:30],
            "byHost": dict(sorted(by_host.items())),
            "byResourceType": dict(sorted(by_type.items())),
            "byRoute": dict(sorted(by_route.items())),
            "topResponses": top,
            "measurement": (
                "content-length when available, otherwise decoded BiDi response bytes; "
                "TLS/TCP framing is excluded and query values are redacted"
            ),
        }
