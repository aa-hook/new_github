# -*- coding: utf-8 -*-
"""Battle.net persistent HTTP registration with RuyiPage + local Route V11.

The registration form is advanced by one persistent curl_cffi session.  A
short-lived RuyiPage Firefox process is started only after the server returns
the Arkose blob.  The browser renders the challenge, the existing V3 solver
captures each challenge strip and asks the local V11 service for the answer,
then the resulting token is submitted by the original HTTP session.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import logging
import os
import sys
import time
import traceback
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Optional
from urllib.parse import quote, unquote, urlsplit, urlunsplit

from battle_protocol_flow_v4 import BattleProtocolClient, PersistentFlowState
from proxy_traffic_meter import ProxyTrafficMeter
from v4_browser_resource_optimizer import BrowserResourceOptimizer, MIB


def _load_v3_solver_modules():
    """Load V3 helpers without installing or launching CloakBrowser.

    ``register.py`` has a legacy top-level CloakBrowser import even though the
    RuyiPage V3 solver never calls it.  A temporary import shim keeps the V4
    dependency set small; the shim is removed immediately after imports.
    """

    shimmed = False
    if importlib.util.find_spec("cloakbrowser") is None:
        module = types.ModuleType("cloakbrowser")

        def unavailable_launch(*_args, **_kwargs):
            raise RuntimeError("CloakBrowser is not part of the V4 runtime")

        module.launch = unavailable_launch
        sys.modules["cloakbrowser"] = module
        shimmed = True
    try:
        import register_ruyipage_v3 as solver
        from register import REGISTER_URL, generate_identity
        from ruyipage_manual_register import manual_same_browser_register_ruyipage as browser
    finally:
        if shimmed:
            sys.modules.pop("cloakbrowser", None)
    return solver, browser, REGISTER_URL, generate_identity


v3, base, REGISTER_URL, generate_identity = _load_v3_solver_modules()

LOG = logging.getLogger("ruyipage_http_v11")
PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ruyipage_http_v11_register" / "runs"
DEFAULT_SITE_KEY = "E8A75615-1CBA-5DFF-8032-D16BCF234E10"
DEFAULT_SURL = "blizzard-api.arkoselabs.com"
REGISTRATION_COUNTRY = "GBR"


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        with contextlib.suppress(Exception):
            stream.reconfigure(encoding="utf-8", errors="replace")


def run_id() -> str:
    return "run_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def setup_logging(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    formatter = logging.Formatter("%(asctime)s [RUYI-V4] %(message)s", "%H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)
    for name in ("urllib3", "PIL"):
        logging.getLogger(name).setLevel(logging.WARNING)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


@dataclass(frozen=True)
class ProxySettings:
    url: Optional[str]
    display: str
    scheme: Optional[str] = None
    host: Optional[str] = None
    port: Optional[int] = None
    has_auth: bool = False

    @property
    def enabled(self) -> bool:
        return bool(self.url)

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "display": self.display,
            "scheme": self.scheme,
            "host": self.host,
            "port": self.port,
            "hasAuth": self.has_auth,
        }


def _host_for_url(host: str) -> str:
    return f"[{host}]" if ":" in host and not host.startswith("[") else host


def parse_proxy(value: Optional[str]) -> ProxySettings:
    """Normalize blank, host:port, host:port:user:password, or a proxy URL."""

    raw = str(value or "").strip()
    if not raw:
        return ProxySettings(None, "direct")

    if "://" not in raw:
        parts = raw.split(":", 3)
        if len(parts) not in (2, 4):
            raise ValueError(
                "proxy must be blank, host:port, host:port:user:password, or a URL"
            )
        host, port_text = parts[0].strip(), parts[1].strip()
        username = parts[2] if len(parts) == 4 else None
        password = parts[3] if len(parts) == 4 else None
        scheme = "http"
    else:
        parsed = urlsplit(raw)
        scheme = parsed.scheme.lower()
        host = parsed.hostname or ""
        port_text = str(parsed.port or "")
        username = unquote(parsed.username) if parsed.username is not None else None
        password = unquote(parsed.password) if parsed.password is not None else None
        if parsed.path not in ("", "/") or parsed.query or parsed.fragment:
            raise ValueError("proxy URL must not contain a path, query, or fragment")

    if scheme not in {"http", "https", "socks5", "socks5h"}:
        raise ValueError(f"unsupported proxy scheme: {scheme}")
    if not host:
        raise ValueError("proxy host is empty")
    try:
        port = int(port_text)
    except (TypeError, ValueError) as exc:
        raise ValueError("proxy port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise ValueError(f"proxy port is out of range: {port}")
    if (username is None) != (password is None):
        raise ValueError("proxy username and password must be provided together")

    userinfo = ""
    has_auth = username is not None
    if has_auth:
        userinfo = f"{quote(str(username), safe='')}:{quote(str(password), safe='')}@"
    normalized = urlunsplit(
        (scheme, f"{userinfo}{_host_for_url(host)}:{port}", "", "", "")
    )
    return ProxySettings(
        normalized,
        f"{scheme}://{_host_for_url(host)}:{port}",
        scheme,
        host,
        port,
        has_auth,
    )


_BLOCKED_HOST_PARTS = (
    "google-analytics.com",
    "googletagmanager.com",
    "doubleclick.net",
    "newrelic.com",
    "nr-data.net",
    "hotjar.com",
    "fullstory.com",
    "sentry.io",
)
_BLOCKED_EXTENSIONS = (
    ".woff",
    ".woff2",
    ".ttf",
    ".otf",
    ".mp4",
    ".webm",
    ".mp3",
    ".wav",
)
_FIREFOX_BACKGROUND_DIRECT_HOSTS = (
    "ciscobinary.openh264.org",
    "content-signature-2.cdn.mozilla.net",
    "aus5.mozilla.org",
)
_FIREFOX_LOW_TRAFFIC_PREFS: dict[str, Any] = {
    "app.normandy.api_url": "",
    "app.normandy.enabled": False,
    "app.shield.optoutstudies.enabled": False,
    "app.update.auto": False,
    "app.update.disabledForTesting": True,
    "browser.discovery.enabled": False,
    "browser.newtabpage.activity-stream.feeds.snippets": False,
    "browser.safebrowsing.downloads.enabled": False,
    "browser.safebrowsing.malware.enabled": False,
    "browser.safebrowsing.phishing.enabled": False,
    "browser.search.update": False,
    "browser.startup.homepage": "about:blank",
    "browser.startup.page": 0,
    "datareporting.healthreport.service.enabled": False,
    "datareporting.healthreport.uploadEnabled": False,
    "extensions.blocklist.enabled": False,
    "extensions.getAddons.cache.enabled": False,
    "extensions.systemAddon.update.enabled": False,
    "extensions.update.enabled": False,
    "media.gmp-gmpopenh264.autoupdate": False,
    "media.gmp-gmpopenh264.enabled": False,
    "media.gmp-manager.updateEnabled": False,
    "media.gmp-manager.url": "",
    "media.gmp-widevinecdm.autoupdate": False,
    "media.gmp-widevinecdm.enabled": False,
    "network.captive-portal-service.enabled": False,
    "network.connectivity-service.enabled": False,
    "network.dns.disablePrefetch": True,
    "network.http.speculative-parallel-limit": 0,
    "network.predictor.enabled": False,
    "network.prefetch-next": False,
    "network.proxy.no_proxies_on": ",".join(_FIREFOX_BACKGROUND_DIRECT_HOSTS),
    "security.remote_settings.crlite_filters.enabled": False,
    "security.remote_settings.intermediates.enabled": False,
    "services.settings.server": "data:,",
    "toolkit.telemetry.archive.enabled": False,
    "toolkit.telemetry.enabled": False,
    "toolkit.telemetry.server": "data:,",
    "toolkit.telemetry.unified": False,
}


def should_block_resource(url: str) -> bool:
    parsed = urlsplit(str(url or ""))
    host = (parsed.hostname or "").lower()
    path = parsed.path.lower()
    if host.endswith("arkoselabs.com"):
        return False
    if any(part in host for part in _BLOCKED_HOST_PARTS):
        return True
    return path.endswith(_BLOCKED_EXTENSIONS)


def install_low_traffic_filter(page: Any) -> bool:
    def handler(request: Any) -> None:
        url = str(getattr(request, "url", "") or "")
        if should_block_resource(url):
            request.fail()
        else:
            request.continue_request()

    try:
        page.intercept.start_requests(handler)
        LOG.info("Low-traffic browser filter enabled (fonts/media/analytics blocked)")
        return True
    except Exception as exc:
        LOG.warning("Low-traffic filter unavailable: %s: %s", type(exc).__name__, exc)
        return False


def launch_ruyi_browser(
    args: argparse.Namespace,
    proxy: ProxySettings,
    runtime_proxy_url: Optional[str] = None,
):
    """Use the V3 RuyiPage launch settings without logging proxy credentials."""

    LOG.info(
        "Launching RuyiPage Firefox: headless=%s proxy=%s",
        args.headless,
        proxy.display,
    )
    launch_proxy = runtime_proxy_url if runtime_proxy_url is not None else proxy.url
    ruyi = base.ruyipage
    options_type = getattr(ruyi, "FirefoxOptions", None)
    page_type = getattr(ruyi, "FirefoxPage", None)
    if callable(options_type) and callable(page_type):
        options = options_type()
        options.quick_start(
            headless=bool(args.headless),
            proxy=launch_proxy,
            window_size=(1920, 1080),
            timeout_page_load=60,
            timeout_script=60,
            close_on_exit=True,
            failure_snapshot=True,
            snapshot_dir=str(Path(args.output_dir) / "_ruyi_failure_snapshots"),
        )
        for key, value in _FIREFOX_LOW_TRAFFIC_PREFS.items():
            options.set_pref(key, value)
        resolver = getattr(ruyi, "resolve_firefox_path", None)
        browser_path = resolver(None) if callable(resolver) else None
        if browser_path:
            options.set_browser_path(browser_path)
        page = page_type(options)
        LOG.info(
            "Firefox startup background network disabled: preferences=%s",
            len(_FIREFOX_LOW_TRAFFIC_PREFS),
        )
    else:
        LOG.warning(
            "RuyiPage options API unavailable; using launch() without startup preferences"
        )
        page = ruyi.launch(
            headless=bool(args.headless),
            proxy=launch_proxy,
            window_size=(1920, 1080),
            timeout_page_load=60,
            timeout_script=60,
            close_on_exit=True,
            failure_snapshot=True,
            snapshot_dir=str(Path(args.output_dir) / "_ruyi_failure_snapshots"),
        )
    with contextlib.suppress(Exception):
        page.set_bypass_csp(True)
    return page


def run_v4_solver_tab(
    page: Any,
    image_catcher: Any,
    args: argparse.Namespace,
    out: Path,
) -> dict[str, Any]:
    try:
        return v3.auto_solve_solver_tab(page, image_catcher, args, out)
    except v3.UnsupportedCaptchaQuestion as exc:
        details = dict(getattr(exc, "details", {}) or {})
        if (
            details.get("questionMatched") is True
            and details.get("imageSize") is None
        ):
            raise RuntimeError(
                "transient Arkose challenge image response was not decodable; retry required"
            ) from exc
        raise


def redact_proxy_text(value: Any, proxy: ProxySettings, raw_proxy: str = "") -> str:
    text = str(value or "")
    replacements = {
        candidate
        for candidate in (str(raw_proxy or "").strip(), str(proxy.url or ""))
        if candidate
    }
    for candidate in sorted(replacements, key=len, reverse=True):
        text = text.replace(candidate, proxy.display)
    return text


def write_proxy_traffic_report(out: Path, report: Mapping[str, Any]) -> None:
    clean_report = dict(report)
    write_json(out / "proxy_traffic.json", clean_report)
    summary_path = out / "summary.json"
    if summary_path.is_file():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            summary = {}
        if isinstance(summary, dict):
            summary["proxyTraffic"] = clean_report
            write_json(summary_path, summary)


def capture_proxy_traffic_snapshot(
    meter: Optional[ProxyTrafficMeter],
) -> dict[str, Any]:
    if meter is None:
        return {}
    snapshot = getattr(meter, "snapshot", None)
    if not callable(snapshot):
        return {}
    value = snapshot()
    return dict(value) if isinstance(value, Mapping) else {}


def _proxy_traffic_delta(
    start: Mapping[str, Any],
    end: Mapping[str, Any],
) -> dict[str, Any]:
    upload = max(0, int(end.get("uploadBytes") or 0) - int(start.get("uploadBytes") or 0))
    download = max(
        0,
        int(end.get("downloadBytes") or 0) - int(start.get("downloadBytes") or 0),
    )
    total = upload + download
    duration = max(
        0.0,
        float(end.get("durationSeconds") or 0.0)
        - float(start.get("durationSeconds") or 0.0),
    )
    return {
        "uploadBytes": upload,
        "downloadBytes": download,
        "totalBytes": total,
        "uploadMiB": round(upload / MIB, 4),
        "downloadMiB": round(download / MIB, 4),
        "totalMiB": round(total / MIB, 4),
        "connections": max(
            0,
            int(end.get("connections") or 0) - int(start.get("connections") or 0),
        ),
        "failures": max(
            0,
            int(end.get("failures") or 0) - int(start.get("failures") or 0),
        ),
        "durationSeconds": round(duration, 3),
    }


def build_proxy_traffic_phase_report(
    snapshots: Mapping[str, Mapping[str, Any]],
    final_report: Mapping[str, Any],
) -> dict[str, Any]:
    boundaries = {
        name: dict(value)
        for name, value in snapshots.items()
        if isinstance(value, Mapping) and value
    }
    if final_report:
        boundaries["final"] = dict(final_report)

    phase_specs = (
        ("protocolToCaptcha", "start", "captchaGate"),
        ("arkoseSolver", "captchaGate", "tokenReady"),
        ("captchaSubmit", "tokenReady", "final"),
    )
    phases: dict[str, dict[str, Any]] = {}
    for name, start_name, end_name in phase_specs:
        start = boundaries.get(start_name)
        end = boundaries.get(end_name)
        if start and end:
            phases[name] = {
                "startBoundary": start_name,
                "endBoundary": end_name,
                **_proxy_traffic_delta(start, end),
            }

    start = boundaries.get("start") or {}
    final = boundaries.get("final") or {}
    measured = _proxy_traffic_delta(start, final) if start and final else {}
    accounted = sum(int(item.get("totalBytes") or 0) for item in phases.values())
    measured_total = int(measured.get("totalBytes") or 0)
    unaccounted = max(0, measured_total - accounted)
    return {
        "enabled": bool(final_report.get("enabled")),
        "boundaries": boundaries,
        "phases": phases,
        "measured": measured,
        "accountedBytes": accounted,
        "accountedMiB": round(accounted / MIB, 4),
        "unaccountedBytes": unaccounted,
        "unaccountedMiB": round(unaccounted / MIB, 4),
        "complete": all(name in phases for name, *_unused in phase_specs),
    }


def write_proxy_traffic_phase_report(out: Path, report: Mapping[str, Any]) -> None:
    clean_report = dict(report)
    write_json(out / "proxy_traffic_phases.json", clean_report)
    summary_path = out / "summary.json"
    if summary_path.is_file():
        summary = read_json_object(summary_path)
        summary["proxyTrafficPhases"] = clean_report
        write_json(summary_path, summary)


def log_proxy_traffic_phases(report: Mapping[str, Any]) -> None:
    for name in ("protocolToCaptcha", "arkoseSolver", "captchaSubmit"):
        phase = dict((report.get("phases") or {}).get(name) or {})
        if not phase:
            continue
        LOG.info(
            "Proxy traffic phase %s: upload=%.4f MiB download=%.4f MiB "
            "total=%.4f MiB bytes=%s connections=%s failures=%s",
            name,
            float(phase.get("uploadMiB") or 0.0),
            float(phase.get("downloadMiB") or 0.0),
            float(phase.get("totalMiB") or 0.0),
            int(phase.get("totalBytes") or 0),
            int(phase.get("connections") or 0),
            int(phase.get("failures") or 0),
        )
    if int(report.get("unaccountedBytes") or 0):
        LOG.info(
            "Proxy traffic unaccounted: %.4f MiB bytes=%s",
            float(report.get("unaccountedMiB") or 0.0),
            int(report.get("unaccountedBytes") or 0),
        )


def log_proxy_traffic_targets(report: Mapping[str, Any]) -> None:
    for item in list(report.get("targets") or [])[:10]:
        LOG.info(
            "Proxy traffic target %s: upload=%.4f MiB download=%.4f MiB "
            "total=%.4f MiB bytes=%s connections=%s failures=%s",
            item.get("target"),
            float(item.get("uploadMiB") or 0.0),
            float(item.get("downloadMiB") or 0.0),
            float(item.get("totalMiB") or 0.0),
            int(item.get("totalBytes") or 0),
            int(item.get("connections") or 0),
            int(item.get("failures") or 0),
        )


def read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return dict(value) if isinstance(value, dict) else {}


def write_browser_traffic_report(out: Path, report: Mapping[str, Any]) -> None:
    clean_report = dict(report)
    write_json(out / "browser_traffic.json", clean_report)
    summary_path = out / "summary.json"
    if summary_path.is_file():
        summary = read_json_object(summary_path)
        summary["browserTraffic"] = clean_report
        write_json(summary_path, summary)


def stop_browser_optimizer(
    optimizer: BrowserResourceOptimizer,
    out: Path,
) -> dict[str, Any]:
    try:
        report = optimizer.stop()
    except Exception as exc:
        report = {
            "enabled": True,
            "error": f"{type(exc).__name__}: {exc}",
        }
        LOG.warning("Browser traffic optimizer stop failed: %s", report["error"])
    write_browser_traffic_report(out, report)
    counts = dict(report.get("counts") or {})
    byte_counts = dict(report.get("bytes") or {})
    LOG.info(
        "Browser traffic optimizer: directStatic=%.4f MiB directImages=%.4f MiB "
        "cacheHit=%.4f MiB estimatedProxyAvoided=%.4f MiB candidates=%s "
        "imageFallbacks=%s staticFallbacks=%s blocked=%s",
        float(byte_counts.get("directStaticMiB") or 0.0),
        float(byte_counts.get("directChallengeImageMiB") or 0.0),
        float(byte_counts.get("cacheHitMiB") or 0.0),
        float(byte_counts.get("estimatedProxyMiBAvoided") or 0.0),
        int(counts.get("publicStaticCandidates") or 0),
        int(counts.get("directChallengeImageFallbacks") or 0),
        int(counts.get("directStaticFallbacks") or 0),
        int(counts.get("blockedRequests") or 0),
    )
    failures = dict(report.get("directFetchFailures") or {})
    if failures:
        LOG.info(
            "Direct fetch fallback reasons: %s",
            sorted(failures.items(), key=lambda item: item[1], reverse=True)[:5],
        )
    for fallback in list(report.get("directFallbacks") or [])[:10]:
        LOG.info(
            "Direct fetch fallback: category=%s hard=%s reason=%s url=%s",
            fallback.get("category") or "public-static",
            bool(fallback.get("hardFailure")),
            fallback.get("reason"),
            fallback.get("url"),
        )
    proxy_responses = [
        item
        for item in list(report.get("topResponses") or [])
        if item.get("route") == "proxy"
    ][:10]
    for item in proxy_responses:
        LOG.info(
            "Proxy response: bytes=%s type=%s category=%s url=%s",
            int(item.get("wireBodyBytesEstimate") or 0),
            item.get("resourceType"),
            item.get("category"),
            item.get("url"),
        )
    return report


def public_arkose_context(value: Mapping[str, Any]) -> dict[str, Any]:
    blob = str(value.get("blob") or "")
    token = str(value.get("token") or "")
    return {
        "source": value.get("source"),
        "siteKey": value.get("siteKey"),
        "surl": value.get("surl"),
        "websiteURL": value.get("websiteURL"),
        "blobLength": len(blob),
        "blobSha256": hashlib.sha256(blob.encode()).hexdigest() if blob else None,
        "tokenLength": len(token),
        "tokenSha256": hashlib.sha256(token.encode()).hexdigest() if token else None,
    }


def configured_identity(args: argparse.Namespace) -> dict[str, str]:
    identity = dict(generate_identity())
    if args.email:
        identity["email"] = args.email.strip()
    if args.password:
        identity["password"] = args.password
    if args.battle_tag:
        identity["battle_tag"] = args.battle_tag.strip()
    identity["full_name"] = " ".join(
        part for part in (identity.get("first_name"), identity.get("last_name")) if part
    )
    identity.setdefault("phone_number", "")
    return {key: str(value) for key, value in identity.items()}


def configure_v3_clicks(args: argparse.Namespace) -> None:
    v3.CLICK_STYLE = args.click_style
    v3.HUMAN_MOVE_MIN_MS = max(100, int(args.human_move_min_ms))
    v3.HUMAN_MOVE_MAX_MS = max(
        v3.HUMAN_MOVE_MIN_MS, int(args.human_move_max_ms)
    )


def wait_rank_v11_service(base_url: str, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + max(0.1, float(timeout))
    last_error: Optional[Exception] = None
    while time.monotonic() < deadline:
        try:
            return v3.ensure_rank_v11_service(
                base_url, min(2.0, max(0.1, deadline - time.monotonic()))
            )
        except Exception as exc:
            last_error = exc
            time.sleep(0.2)
    raise TimeoutError(
        f"local V11 service was not ready within {timeout}s: {last_error}"
    )


def replace_document_low_traffic(page: Any, website_url: str, html: str) -> dict[str, Any]:
    expected_origin = base.origin_from_url(website_url)
    before = {}
    with contextlib.suppress(Exception):
        before = page.run_js(
            "return {url: location.href, origin: location.origin};", timeout=5
        ) or {}
    navigation_error = ""
    if before.get("origin") != expected_origin:
        bootstrap_url = expected_origin + "/robots.txt"
        try:
            page.get(bootstrap_url, wait="interactive", timeout=25)
        except Exception as exc:
            navigation_error = f"{type(exc).__name__}: {exc}"
        with contextlib.suppress(Exception):
            page.stop_loading()
        before = page.run_js(
            "return {url: location.href, origin: location.origin};", timeout=8
        ) or {}
    if before.get("origin") != expected_origin:
        raise RuntimeError(
            f"solver origin mismatch: expected={expected_origin}, actual={before}"
        )
    page.run_js(
        """function(html) {
          window.stop();
          document.open();
          document.write(html);
          document.close();
        }""",
        html,
        timeout=15,
    )
    return {
        "expectedOrigin": expected_origin,
        "beforeReplace": before,
        "navigationError": navigation_error,
    }


def recover_blob_with_ruyi(
    page: Any,
    client: BattleProtocolClient,
    args: argparse.Namespace,
    out: Path,
) -> dict[str, Any]:
    """Fallback for deployments where the captcha HTML omits the blob."""

    cookies = client.playwright_cookies()
    if cookies:
        page.set_cookies(cookies)
    base.write_json(
        out / "solver" / "cookie_import_summary.json",
        {"count": len(cookies), "names": sorted({item["name"] for item in cookies})},
    )
    catcher = base.RuyiArkoseCatcher(page)
    catcher.start()
    try:
        page.get(args.entry_url, wait="interactive", timeout=45)
        with contextlib.suppress(Exception):
            page.stop_loading()
        blob = catcher.wait_for_blob(timeout=min(12.0, args.blob_timeout))
        if not blob:
            base.click_arkose_verify(page, timeout=12)
            blob = catcher.wait_for_blob(timeout=args.blob_timeout)
        blob = blob or catcher.captured_blob
        if not blob:
            raise RuntimeError("RuyiPage fallback did not recover an Arkose blob")
        detected = base.detect_arkose_context(page, catcher)
        return {
            "blob": blob,
            "siteKey": detected.get("siteKey") or DEFAULT_SITE_KEY,
            "surl": detected.get("surl") or DEFAULT_SURL,
            "websiteURL": args.entry_url,
            "source": "ruyipage-cookie-import-bidi",
        }
    finally:
        with contextlib.suppress(Exception):
            catcher.stop()


def solve_arkose_with_ruyi(
    client: BattleProtocolClient,
    context: Mapping[str, Any],
    args: argparse.Namespace,
    proxy: ProxySettings,
    out: Path,
    runtime_proxy_url: Optional[str] = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    page = None
    image_catcher = None
    optimizer = None
    current = dict(context)
    try:
        page = launch_ruyi_browser(args, proxy, runtime_proxy_url)
        if not current.get("blob"):
            LOG.info("HTTP response had no blob; using the RuyiPage cookie-import fallback")
            current = recover_blob_with_ruyi(page, client, args, out)

        optimizer = BrowserResourceOptimizer(
            page,
            Path(args.static_cache_dir),
            proxy_enabled=proxy.enabled,
            direct_public_static=(
                proxy.enabled and not bool(args.no_direct_public_static)
            ),
            direct_challenge_images=(
                proxy.enabled and bool(args.direct_challenge_images)
            ),
            block_nonessential=not bool(args.no_resource_blocking),
            fetch_timeout=args.static_fetch_timeout,
            max_entry_bytes=max(1, int(args.static_cache_max_entry_mib * MIB)),
            should_block=should_block_resource,
        )
        optimizer.start()
        LOG.info(
            "Browser traffic optimizer enabled: publicStaticDirect=%s "
            "challengeImageDirect=%s cache=%s sessionControl=proxy",
            proxy.enabled and not bool(args.no_direct_public_static),
            proxy.enabled and bool(args.direct_challenge_images),
            Path(args.static_cache_dir).expanduser().resolve(),
        )

        blob = str(current.get("blob") or "")
        if not blob:
            raise RuntimeError("Arkose context has no blob")
        current["siteKey"] = str(current.get("siteKey") or DEFAULT_SITE_KEY)
        current["surl"] = str(current.get("surl") or DEFAULT_SURL)
        current["websiteURL"] = str(current.get("websiteURL") or args.entry_url)
        write_json(out / "arkose_context.json", public_arkose_context(current))

        image_catcher = v3.RuyiArkoseImageCatcher(page, label="v4-solver")
        image_catcher.start()
        harness = base.build_solver_harness(
            current["siteKey"], blob, current["surl"]
        )
        origin_info = replace_document_low_traffic(
            page, current["websiteURL"], harness
        )
        write_json(out / "solver" / "origin.json", origin_info)
        if args.debug_screenshots:
            base.screenshot(page, out / "solver_screenshots" / "harness_loaded.png")

        result = run_v4_solver_tab(page, image_catcher, args, out)
        write_json(
            out / "local_v11_solver_result.json",
            {key: value for key, value in result.items() if key != "token"},
        )
        if not result.get("ok") or not result.get("token"):
            raise RuntimeError(str(result.get("error") or "local V11 solver returned no token"))
        with contextlib.suppress(Exception):
            image_catcher.stop()
        image_catcher = None
        result["browserTraffic"] = stop_browser_optimizer(optimizer, out)
        optimizer = None
        return result, current
    finally:
        with contextlib.suppress(Exception):
            if image_catcher:
                image_catcher.stop()
        with contextlib.suppress(Exception):
            if optimizer:
                stop_browser_optimizer(optimizer, out)
        if args.keep_open and page is not None:
            with contextlib.suppress(EOFError):
                input("Solver browser is open. Press Enter to close...")
        with contextlib.suppress(Exception):
            if page:
                page.quit()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Persistent HTTP registration + RuyiPage + local Route V11"
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--resume",
        help="existing run directory or persistent_state.json",
    )
    parser.add_argument("--entry-url", default=REGISTER_URL)
    parser.add_argument(
        "--proxy",
        default=os.environ.get("REGISTRATION_PROXY", ""),
        help="blank/direct, host:port, host:port:user:password, or proxy URL",
    )
    parser.add_argument("--protocol-impersonate", default="chrome")
    parser.add_argument("--protocol-user-agent", default="")
    parser.add_argument("--protocol-timeout", type=float, default=45.0)
    country_probe = parser.add_mutually_exclusive_group()
    country_probe.add_argument(
        "--country-probe",
        dest="country_probe",
        action="store_true",
        default=True,
        help="probe the GBR country form before submitting date of birth (default)",
    )
    country_probe.add_argument(
        "--no-country-probe",
        dest="country_probe",
        action="store_false",
        help="skip the country probe for protocol diagnostics",
    )
    parser.add_argument("--email")
    parser.add_argument("--password")
    parser.add_argument("--battle-tag")

    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--skip-egress-check", action="store_true", default=True)
    parser.add_argument("--blob-timeout", type=float, default=35.0)
    parser.add_argument("--debug-screenshots", action="store_true")
    parser.add_argument("--no-resource-blocking", action="store_true")
    parser.add_argument(
        "--static-cache-dir",
        default=os.environ.get(
            "V4_STATIC_CACHE_DIR",
            str(PROJECT_ROOT / ".cache" / "v4_public_static"),
        ),
        help="shared cache for strictly public Arkose static assets",
    )
    parser.add_argument(
        "--no-direct-public-static",
        action="store_true",
        help="keep public static cache misses on the configured browser route",
    )
    direct_images = parser.add_mutually_exclusive_group()
    direct_images.add_argument(
        "--direct-challenge-images",
        dest="direct_challenge_images",
        action="store_true",
        default=True,
        help="download signed Arkose challenge images runner-direct (default)",
    )
    direct_images.add_argument(
        "--no-direct-challenge-images",
        dest="direct_challenge_images",
        action="store_false",
        help="keep signed Arkose challenge images on the configured browser route",
    )
    parser.add_argument("--static-fetch-timeout", type=float, default=8.0)
    parser.add_argument("--static-cache-max-entry-mib", type=float, default=8.0)
    parser.add_argument(
        "--click-style",
        choices=("balanced", "fast", "human", "js"),
        default="balanced",
    )
    parser.add_argument("--human-move-min-ms", type=int, default=800)
    parser.add_argument("--human-move-max-ms", type=int, default=1400)
    parser.add_argument(
        "--rank-v11-url",
        default=os.environ.get("RANK_V11_URL", v3.DEFAULT_RANK_V11_URL),
    )
    parser.add_argument("--rank-v11-timeout", type=float, default=120.0)
    parser.add_argument("--max-waves", type=int, default=8)
    parser.add_argument("--verify-timeout", type=float, default=25.0)
    parser.add_argument("--first-image-timeout", type=float, default=30.0)
    parser.add_argument("--next-image-timeout", type=float, default=18.0)
    parser.add_argument("--after-submit-token-wait", type=float, default=1.2)
    parser.add_argument("--token-timeout", type=float, default=15.0)
    return parser


def resolve_resume_path(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.is_dir():
        path = path / "persistent_state.json"
    if not path.is_file():
        raise FileNotFoundError(f"resume state not found: {path}")
    return path


def main() -> int:
    force_utf8_stdio()
    args = build_parser().parse_args()
    configure_v3_clicks(args)
    resume_path = resolve_resume_path(args.resume) if args.resume else None
    out = (
        resume_path.parent
        if resume_path is not None
        else Path(args.output_dir).expanduser().resolve() / run_id()
    )
    out.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(out)
    setup_logging(out / "run.log")

    state: Optional[PersistentFlowState] = None
    client: Optional[BattleProtocolClient] = None
    identity: dict[str, str] = {}
    proxy = ProxySettings(None, "direct")
    traffic_meter: Optional[ProxyTrafficMeter] = None
    traffic_snapshots: dict[str, dict[str, Any]] = {}
    runtime_proxy_url: Optional[str] = None
    started = time.perf_counter()
    try:
        proxy = parse_proxy(args.proxy)
        if resume_path is not None:
            state = PersistentFlowState.load(resume_path)
            identity = {
                key: str(value)
                for key, value in dict(state.data.get("identity") or {}).items()
            }
            if not all(
                identity.get(key) for key in ("email", "password", "battle_tag")
            ):
                raise RuntimeError("resume state has no complete account identity")
            LOG.info("Resuming persistent state: %s status=%s", resume_path, state.data.get("status"))
        else:
            identity = configured_identity(args)
            state = PersistentFlowState.create(
                out / "persistent_state.json",
                identity=identity,
                profile={
                    "mode": "persistent-http-ruyipage-local-v11",
                    "registrationCountry": REGISTRATION_COUNTRY,
                    "countryProbe": bool(args.country_probe),
                    "proxy": proxy.summary(),
                    "proxyTrafficMeter": bool(proxy.enabled),
                    "publicStaticDirect": bool(
                        proxy.enabled and not args.no_direct_public_static
                    ),
                    "challengeImageDirect": bool(
                        proxy.enabled and args.direct_challenge_images
                    ),
                    "staticCacheDir": str(Path(args.static_cache_dir).expanduser()),
                    "protocolImpersonate": args.protocol_impersonate,
                },
            )
        write_json(out / "account_generated.json", identity)
        LOG.info("Output directory: %s", out)
        LOG.info("Flow: persistent HTTP -> RuyiPage -> local V11 -> HTTP captcha-gate")
        LOG.info("Registration country: %s (fixed)", REGISTRATION_COUNTRY)
        LOG.info("Proxy route: %s auth=%s", proxy.display, proxy.has_auth)
        LOG.info(
            "Public static route: %s; cache=%s",
            "runner-direct" if proxy.enabled and not args.no_direct_public_static else "browser-route",
            Path(args.static_cache_dir).expanduser().resolve(),
        )
        LOG.info("Account: %s", identity["email"])
        LOG.info("BattleTag: %s", identity["battle_tag"])

        if state.data.get("status") == "complete":
            LOG.info("Persistent state is already complete; no network work is required")
            print(f"Account: {identity['email']}")
            print(f"Password: {identity['password']}")
            print(f"BattleTag: {identity.get('battle_tag', '')}")
            return 0

        runtime_proxy_url = proxy.url
        if proxy.enabled and proxy.url:
            traffic_meter = ProxyTrafficMeter(proxy.url)
            runtime_proxy_url = traffic_meter.start()
            LOG.info(
                "Proxy traffic meter started: local=%s upstream=%s",
                runtime_proxy_url,
                proxy.display,
            )
            traffic_snapshots["start"] = capture_proxy_traffic_snapshot(traffic_meter)

        client = BattleProtocolClient(
            state,
            out,
            entry_url=args.entry_url,
            proxy=runtime_proxy_url,
            impersonate=args.protocol_impersonate,
            user_agent=args.protocol_user_agent or None,
            accept_language="en-GB,en;q=0.9",
            timeout=args.protocol_timeout,
        )
        if state.data.get("status") not in {"captcha-gate", "token-ready"}:
            client.run_to_captcha(
                country=REGISTRATION_COUNTRY,
                opt_in=False,
                country_probe=bool(args.country_probe),
            )
        LOG.info("Persistent HTTP flow reached captcha-gate")
        arkose = dict(state.data.get("arkose") or {})
        if not arkose.get("blob"):
            arkose = client.recover_arkose_from_last_response()
        traffic_snapshots["captchaGate"] = capture_proxy_traffic_snapshot(traffic_meter)
        LOG.info(
            "Arkose context: source=%s siteKey=%s blobLength=%s",
            arkose.get("source"),
            arkose.get("siteKey"),
            len(str(arkose.get("blob") or "")),
        )

        token = (
            str(arkose.get("token") or "")
            if state.data.get("status") == "token-ready"
            else ""
        )
        if token:
            health = {"ok": True, "status": "not-required-resumed-token"}
            solve_result = {
                "ok": True,
                "token": token,
                "actions": [],
                "resumedToken": True,
            }
            LOG.info("Using token from persistent state, length=%s", len(token))
        else:
            health = wait_rank_v11_service(
                args.rank_v11_url, args.rank_v11_timeout
            )
            write_json(out / "rank_v11_health.json", health)
            LOG.info(
                "Local V11 ready: device=%s load=%.3fs warmup=%.3fs",
                health.get("device"),
                float(health.get("model_load_seconds") or 0.0),
                float(health.get("warmup_seconds") or 0.0),
            )
            solve_result, arkose = solve_arkose_with_ruyi(
                client,
                arkose,
                args,
                proxy,
                out,
                runtime_proxy_url=runtime_proxy_url,
            )
            token = str(solve_result["token"])
            arkose["token"] = token
            state.checkpoint(
                "token-ready",
                arkose=arkose,
                event={"completed": "local-v11-solver", "tokenLength": len(token)},
            )
            LOG.info("RuyiPage returned Arkose token, length=%s", len(token))

        traffic_snapshots["tokenReady"] = capture_proxy_traffic_snapshot(traffic_meter)
        outcome = client.submit_captcha(token)
        success = outcome.get("status") == "success" and bool(outcome.get("success"))
        registration = {
            "ok": success,
            "email": identity["email"],
            "battleTag": identity["battle_tag"],
            "successSource": "persistent-http-captcha-gate" if success else None,
            "outcome": outcome,
        }
        write_json(out / "registration_result.json", registration)
        write_json(
            out / "summary.json",
            {
                "ok": success,
                "outputDir": str(out),
                "mode": "persistent-http-ruyipage-local-v11",
                "registrationCountry": REGISTRATION_COUNTRY,
                "countryProbe": bool(args.country_probe),
                "challengeImageDirect": bool(
                    proxy.enabled and args.direct_challenge_images
                ),
                "proxy": proxy.summary(),
                "arkose": public_arkose_context(arkose),
                "rankV11": {
                    "url": args.rank_v11_url,
                    "health": health,
                    "actions": solve_result.get("actions") or [],
                },
                "browserTraffic": solve_result.get("browserTraffic") or {},
                "registration": registration,
                "elapsedSeconds": time.perf_counter() - started,
            },
        )
        if not success:
            LOG.error(
                "captcha-gate did not confirm success: status=%s sample=%r",
                outcome.get("status"),
                outcome.get("sample"),
            )
            return 1

        LOG.info("Registration succeeded through the persisted HTTP session")
        print(f"Account: {identity['email']}")
        print(f"Password: {identity['password']}")
        print(f"BattleTag: {identity['battle_tag']}")
        return 0
    except KeyboardInterrupt:
        LOG.warning("Interrupted")
        write_json(
            out / "summary.json",
            {"ok": False, "error": "KeyboardInterrupt", "outputDir": str(out)},
        )
        return 130
    except Exception as exc:
        error_text = redact_proxy_text(
            f"{type(exc).__name__}: {exc}", proxy, args.proxy
        )
        safe_traceback = redact_proxy_text(traceback.format_exc(), proxy, args.proxy)
        LOG.error("Run failed: %s\n%s", error_text, safe_traceback)
        if state is not None:
            with contextlib.suppress(Exception):
                state.checkpoint(
                    str(state.data.get("status") or "failed"),
                    error=error_text,
                    event={"failed": "runner", "errorType": type(exc).__name__},
                )
        failure = {
            "ok": False,
            "error": error_text,
            "outputDir": str(out),
            "registrationCountry": REGISTRATION_COUNTRY,
            "countryProbe": bool(args.country_probe),
            "challengeImageDirect": bool(
                proxy.enabled and args.direct_challenge_images
            ),
            "proxy": proxy.summary(),
            "elapsedSeconds": time.perf_counter() - started,
        }
        if isinstance(exc, v3.UnsupportedCaptchaQuestion):
            failure["unsupportedCaptcha"] = True
            failure["challenge"] = exc.details
        browser_traffic = read_json_object(out / "browser_traffic.json")
        if browser_traffic:
            failure["browserTraffic"] = browser_traffic
        write_json(out / "summary.json", failure)
        return (
            v3.UNSUPPORTED_CAPTCHA_EXIT_CODE
            if isinstance(exc, v3.UnsupportedCaptchaQuestion)
            else 1
        )
    finally:
        if client is not None:
            with contextlib.suppress(Exception):
                client.session.close()
        if traffic_meter is not None:
            with contextlib.suppress(Exception):
                report = traffic_meter.stop()
                write_proxy_traffic_report(out, report)
                phase_report = build_proxy_traffic_phase_report(
                    traffic_snapshots,
                    report,
                )
                write_proxy_traffic_phase_report(out, phase_report)
                log_proxy_traffic_phases(phase_report)
                log_proxy_traffic_targets(report)
                LOG.info(
                    "Proxy traffic total: upload=%.4f MiB download=%.4f MiB "
                    "total=%.4f MiB bytes=%s connections=%s failures=%s",
                    float(report.get("uploadMiB") or 0.0),
                    float(report.get("downloadMiB") or 0.0),
                    float(report.get("totalMiB") or 0.0),
                    int(report.get("totalBytes") or 0),
                    int(report.get("connections") or 0),
                    int(report.get("failures") or 0),
                )


if __name__ == "__main__":
    raise SystemExit(main())
