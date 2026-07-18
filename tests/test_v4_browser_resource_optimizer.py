import base64
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from v4_browser_resource_optimizer import (
    BrowserResourceOptimizer,
    CachedResponse,
    DirectPublicStaticFetcher,
    FetchOutcome,
    StaticResponseCache,
    is_public_static_candidate,
    is_session_bound_url,
    public_cache_lifetime,
)


PUBLIC_API = (
    "https://blizzard-api.arkoselabs.com/v2/"
    "E8A75615-1CBA-5DFF-8032-D16BCF234E10/api.js"
)
PUBLIC_ASSET = "https://blizzard-api.arkoselabs.com/cdn/fc/assets/app.js?v=123"


class FakeRequest:
    def __init__(
        self,
        url,
        *,
        request_id="request-1",
        method="GET",
        headers=None,
        response_status=0,
        response_headers=None,
        collector=None,
    ):
        self.url = url
        self.request_id = request_id
        self.method = method
        self.headers = headers or {"User-Agent": "Firefox/Test"}
        self.response_status = response_status
        self.response_headers = response_headers or {}
        self.is_response_phase = bool(response_status)
        self._response_collector = collector
        self.action = None
        self.mocked = None

    def continue_request(self):
        self.action = "continue-request"

    def continue_response(self):
        self.action = "continue-response"

    def fail(self):
        self.action = "fail"

    def mock(self, body, **kwargs):
        self.action = "mock"
        self.mocked = (body, kwargs)


class FakeFetcher:
    def __init__(self, response):
        self.response = response
        self.calls = []

    def fetch(self, url, headers):
        self.calls.append((url, dict(headers)))
        return FetchOutcome(self.response, "ok", 0.01)


def optimizer(tmp_path, fetcher, **kwargs):
    return BrowserResourceOptimizer(
        SimpleNamespace(),
        tmp_path,
        proxy_enabled=True,
        fetcher=fetcher,
        **kwargs,
    )


def test_only_public_arkose_assets_are_direct_candidates():
    assert is_public_static_candidate(PUBLIC_API)
    assert is_public_static_candidate(PUBLIC_ASSET)
    assert not is_public_static_candidate(
        "https://blizzard-api.arkoselabs.com/cdn/fc/assets/style-manager"
    )
    assert not is_public_static_candidate(
        "https://blizzard-api.arkoselabs.com/cdn/fc/assets/opaque-session-like-path"
    )
    assert not is_public_static_candidate(
        "https://blizzard-api.arkoselabs.com/rtig/image?challenge=0&sessionToken=secret"
    )
    assert not is_public_static_candidate(
        "https://blizzard-api.arkoselabs.com/cdn/fc/assets/app.js?session_token=secret"
    )
    assert not is_public_static_candidate(
        "https://blizzard-api.arkoselabs.com/fc/gt2/public_key"
    )
    assert not is_public_static_candidate(
        "https://account.battle.net/static/main.js"
    )


@pytest.mark.parametrize(
    ("headers", "cacheable"),
    [
        ({"cache-control": "public, max-age=3600"}, True),
        ({"cache-control": "public, max-age=31536000, immutable"}, True),
        ({"cache-control": "private, max-age=3600"}, False),
        ({"cache-control": "public, max-age=3600", "set-cookie": "x=1"}, False),
        ({"cache-control": "public, max-age=3600", "vary": "Origin"}, True),
        (
            {
                "cache-control": "public, max-age=0, s-maxage=31536000",
                "vary": "Accept-Encoding, Origin",
            },
            True,
        ),
        ({"cache-control": "public, max-age=3600", "vary": "Cookie"}, False),
        ({"cache-control": "max-age=0"}, False),
    ],
)
def test_public_cache_policy_is_strict(headers, cacheable):
    assert (public_cache_lifetime(headers) > 0) is cacheable


def test_direct_public_response_is_mocked_then_reused_from_disk_cache(tmp_path):
    payload = CachedResponse(
        body=b"console.log('cached');",
        headers={
            "content-type": "application/javascript",
            "cache-control": "public, max-age=3600",
            "content-length": "22",
        },
        expires_at=9999999999,
    )
    fetcher = FakeFetcher(payload)
    app = optimizer(tmp_path, fetcher)

    first = FakeRequest(PUBLIC_API, request_id="first")
    app._handle_request(first)
    second = FakeRequest(PUBLIC_API, request_id="second")
    app._handle_request(second)

    assert first.action == "mock"
    assert second.action == "mock"
    assert len(fetcher.calls) == 1
    report = app.report()
    assert report["counts"]["directStaticFetches"] == 1
    assert report["counts"]["cacheHits"] == 1
    assert report["bytes"]["estimatedProxyBytesAvoided"] == 44


def test_disk_cache_does_not_cross_origin_variants(tmp_path):
    cache = StaticResponseCache(tmp_path)
    payload = CachedResponse(
        body=b"origin-specific",
        headers={
            "content-type": "application/javascript",
            "cache-control": "public, max-age=3600",
            "vary": "Origin",
        },
        expires_at=9999999999,
        vary_request_headers={"origin": "https://account.battle.net"},
    )
    assert cache.put(PUBLIC_API, payload)

    assert cache.get(
        PUBLIC_API, {"Origin": "https://account.battle.net"}
    ) is not None
    assert cache.get(PUBLIC_API, {"Origin": "https://other.example"}) is None


def test_session_bound_image_always_uses_browser_proxy(tmp_path):
    fetcher = FakeFetcher(None)
    app = optimizer(tmp_path, fetcher)
    url = (
        "https://blizzard-api.arkoselabs.com/rtig/image?"
        "challenge=0&gameToken=game-secret&sessionToken=session-secret&signature=sig"
    )

    request = FakeRequest(url)
    app._handle_request(request)

    assert is_session_bound_url(url)
    assert request.action == "continue-request"
    assert fetcher.calls == []
    assert app.report()["counts"]["sessionBoundRequests"] == 1


def test_direct_failure_circuit_prevents_repeated_runner_timeouts(tmp_path):
    class FailingFetcher:
        def __init__(self):
            self.calls = 0

        def fetch(self, _url, _headers):
            self.calls += 1
            return FetchOutcome(None, "ConnectTimeout: timed out", 3.0)

    fetcher = FailingFetcher()
    app = optimizer(tmp_path, fetcher, direct_failure_limit=2)
    requests = [
        FakeRequest(f"https://blizzard-api.arkoselabs.com/cdn/fc/assets/{index}.js")
        for index in range(4)
    ]

    for request in requests:
        app._handle_request(request)

    assert fetcher.calls == 2
    assert all(request.action == "continue-request" for request in requests)
    counts = app.report()["counts"]
    assert counts["directStaticCircuitOpened"] == 1
    assert counts["directStaticCircuitBypasses"] == 2


def test_cache_policy_fallbacks_do_not_open_transport_circuit(tmp_path):
    class PolicyFallbackFetcher:
        def __init__(self):
            self.calls = 0

        def fetch(self, _url, _headers):
            self.calls += 1
            return FetchOutcome(None, "response-not-public-cacheable", 0.01)

    fetcher = PolicyFallbackFetcher()
    app = optimizer(tmp_path, fetcher, direct_failure_limit=2)
    requests = [
        FakeRequest(f"https://blizzard-api.arkoselabs.com/cdn/fc/assets/{index}.js")
        for index in range(4)
    ]

    for request in requests:
        app._handle_request(request)

    assert fetcher.calls == 4
    counts = app.report()["counts"]
    assert counts["directStaticPolicyFallbacks"] == 4
    assert counts.get("directStaticCircuitOpened", 0) == 0


def test_response_report_redacts_all_query_values(tmp_path):
    fetcher = FakeFetcher(None)
    app = optimizer(tmp_path, fetcher)
    url = (
        "https://blizzard-api.arkoselabs.com/rtig/image?"
        "challenge=0&sessionToken=do-not-store&signature=also-secret"
    )
    response = FakeRequest(
        url,
        response_status=200,
        response_headers={"content-type": "image/jpeg", "content-length": "4321"},
    )

    app._handle_response(response)

    top = app.report()["topResponses"][0]
    assert response.action == "continue-response"
    assert top["route"] == "proxy"
    assert top["category"] == "session-bound"
    assert top["wireBodyBytesEstimate"] == 4321
    assert "do-not-store" not in top["url"]
    assert "also-secret" not in top["url"]
    assert "sessiontoken=<redacted>" in top["url"]


def test_mocked_response_is_not_continued_a_second_time(tmp_path):
    app = optimizer(tmp_path, FakeFetcher(None))
    app._local_request_ids.add("mocked-request")
    response = FakeRequest(
        PUBLIC_API,
        request_id="mocked-request",
        response_status=200,
        response_headers={"content-type": "application/javascript"},
    )

    app._handle_response(response)

    assert response.action is None
    assert response._handled is True
    assert "mocked-request" not in app._local_request_ids
    assert app.report()["topResponses"] == []


def test_collector_binary_body_is_used_when_content_length_is_missing(tmp_path):
    body = b"binary-body"

    class Collector:
        def get(self, _request_id, data_type):
            assert data_type == "response"
            return SimpleNamespace(
                raw={
                    "bytes": None,
                    "base64": {
                        "type": "base64",
                        "value": base64.b64encode(body).decode("ascii"),
                    }
                }
            )

        def disown(self, _request_id, data_type):
            assert data_type == "response"

    app = optimizer(tmp_path, FakeFetcher(None))
    response = FakeRequest(
        "https://example.test/no-length",
        response_status=200,
        response_headers={"content-type": "application/octet-stream"},
        collector=Collector(),
    )

    app._handle_response(response)

    assert app.report()["topResponses"][0]["decodedBodyBytes"] == len(body)


def test_direct_fetcher_ignores_environment_proxy(tmp_path, monkeypatch):
    body = b"window.fixture = true;"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Cache-Control", "public, max-age=3600")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, _format, *_args):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:1")
    monkeypatch.setenv("NO_PROXY", "")
    fetcher = DirectPublicStaticFetcher(timeout=2)
    try:
        outcome = fetcher.fetch(
            f"http://127.0.0.1:{server.server_port}/asset.js",
            {"User-Agent": "Firefox/Test"},
        )
    finally:
        fetcher.close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    assert fetcher.session.trust_env is False
    assert outcome.reason == "ok"
    assert outcome.response is not None
    assert outcome.response.body == body
