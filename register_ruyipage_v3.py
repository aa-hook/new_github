# -*- coding: utf-8 -*-
"""RuyiPage 同浏览器新标签 + 本地 Route V11 自动解题。

核心闭环：
  原注册标签 -> 抓 Arkose publicKey/surl/blob -> 同浏览器新标签加载 Arkose
  -> 点击 Verify -> Firefox BiDi 抓 /rtig/image 验证图
  -> 只接受 standing-on-the-same-icons 题型
  -> 本地常驻 Route V11 返回 answer_index
  -> 把 answer_index 当成“点击下一张图按钮 N 次”
  -> Submit，多轮直到 onCompleted token -> 回原标签注入 token。
"""
from __future__ import annotations

import argparse
import base64
import contextlib
import hashlib
import html as html_lib
import io
import logging
import os
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock
from typing import Any, Dict, Optional
from urllib.parse import parse_qs

import requests

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    import ruyipage  # noqa: F401
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "缺少 ruyipage。请先运行：\n"
        "  python -m pip install ruyiPage --upgrade\n"
        "  python -m ruyipage install\n"
    ) from exc

from isolated_proxy_adapter import IsolatedProxyRoute
from register import generate_identity
from ruyipage_manual_register import manual_same_browser_register_ruyipage as base


LOG = logging.getLogger("ruyipage_local_v11")
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ruyipage_local_v11_register" / "runs"
SUPPORTED_QUESTION = (
    "use the arrows to move the characters until they are standing on the same icons as in the picture on the left"
)
DEFAULT_RANK_V11_URL = "http://127.0.0.1:8765"
SUPPORTED_IMAGE_SIZE = (2000, 400)
UNSUPPORTED_CAPTCHA_EXIT_CODE = 42
CLICK_STYLE = "balanced"
FAST_POINTER_MOVE_MS = 20
FAST_CLICK_HOLD_MS = 35
HUMAN_MOVE_MIN_MS = 1000
HUMAN_MOVE_MAX_MS = 2000


class ArkoseCompletionRejected(RuntimeError):
    """Arkose fired onCompleted, but the payload marks the challenge as failed."""

    def __init__(self, reason: str, payload: dict):
        self.reason = reason
        self.payload = dict(payload)
        super().__init__(f"Arkose completion rejected: {reason}")


class UnsupportedCaptchaQuestion(RuntimeError):
    """The current Arkose game is outside the only model supported by V3."""

    def __init__(self, details: dict):
        self.details = dict(details)
        super().__init__(
            "unsupported Arkose question: "
            f"questionMatched={self.details.get('questionMatched')} "
            f"imageSize={self.details.get('imageSize')}"
        )


def image_size(data: bytes) -> Optional[tuple[int, int]]:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            return tuple(im.size)
    except Exception:
        return None


def image_ext(mime: str, data: bytes) -> str:
    mime = (mime or "").lower()
    if "jpeg" in mime or "jpg" in mime or data.startswith(b"\xff\xd8"):
        return ".jpg"
    if "png" in mime or data.startswith(b"\x89PNG"):
        return ".png"
    if "webp" in mime or data.startswith(b"RIFF"):
        return ".webp"
    return ".bin"


def _decode_b64_or_text(text: str) -> bytes:
    s = (text or "").strip()
    if s.startswith("data:") and "," in s:
        s = s.split(",", 1)[1]
    compact = re.sub(r"\s+", "", s)
    if len(compact) >= 16 and len(compact) % 4 == 0 and re.fullmatch(r"[A-Za-z0-9+/=]+", compact):
        with contextlib.suppress(Exception):
            data = base64.b64decode(compact, validate=True)
            if data:
                return data
    with contextlib.suppress(Exception):
        return text.encode("latin1")
    return text.encode("utf-8", "replace")


def decode_bidi_bytes(value: Any) -> bytes:
    """把 Firefox WebDriver BiDi network.getData() 的返回值转成 bytes。"""
    if value is None:
        return b""
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return _decode_b64_or_text(value)
    if not isinstance(value, dict):
        return str(value).encode("utf-8", "replace")
    if "bytes" in value:
        return decode_bidi_bytes(value.get("bytes"))
    if "base64" in value:
        raw = value.get("base64")
        if isinstance(raw, dict):
            raw = raw.get("value")
        if raw:
            with contextlib.suppress(Exception):
                return base64.b64decode(str(raw))
    typ = value.get("type")
    val = value.get("value")
    if typ == "base64" and val:
        with contextlib.suppress(Exception):
            return base64.b64decode(str(val))
    if val is not None:
        return decode_bidi_bytes(val)
    return b""


@dataclass
class RuyiArkoseImageCatcher:
    """全局监听 Firefox BiDi，捕获 Arkose challenge 图片响应体。"""

    page: Any
    label: str = "solver"
    captured_images: list[dict] = field(default_factory=list)
    _driver: Any = None
    _subscription_id: Optional[str] = None
    _collector_id: Optional[str] = None
    _lock: Lock = field(default_factory=Lock)
    _event: Event = field(default_factory=Event)
    _rid_to_idx: Dict[str, int] = field(default_factory=dict)

    def start(self) -> None:
        from ruyipage._bidi import network as bidi_network
        from ruyipage._bidi import session as bidi_session

        self._driver = self.page._driver._browser_driver
        with contextlib.suppress(Exception):
            ret = bidi_network.add_data_collector(
                self._driver,
                events=["responseCompleted"],
                contexts=None,
                data_types=["response"],
                max_encoded_data_size=8 * 1024 * 1024,
            )
            self._collector_id = ret.get("collector")
        ret = bidi_session.subscribe(self._driver, ["network.responseCompleted", "network.fetchError"], contexts=None)
        self._subscription_id = ret.get("subscription")
        self._driver.set_callback("network.responseCompleted", self._on_response, context=None)
        self._driver.set_callback("network.fetchError", self._on_fetch_error, context=None)
        LOG.info("[%s] RuyiPage image capture started collector=%s", self.label, bool(self._collector_id))

    def stop(self) -> None:
        from ruyipage._bidi import network as bidi_network
        from ruyipage._bidi import session as bidi_session

        with contextlib.suppress(Exception):
            if self._driver:
                self._driver.remove_callback("network.responseCompleted", context=None)
                self._driver.remove_callback("network.fetchError", context=None)
        with contextlib.suppress(Exception):
            if self._driver and self._subscription_id:
                bidi_session.unsubscribe(self._driver, subscription=self._subscription_id)
        with contextlib.suppress(Exception):
            if self._driver and self._collector_id:
                bidi_network.remove_data_collector(self._driver, self._collector_id)

    def _on_fetch_error(self, params: Dict[str, Any]) -> None:
        req = params.get("request", {}) or {}
        url = req.get("url", "") or ""
        if "arkoselabs" in url or "/rtig/image" in url:
            LOG.debug("[%s] image fetchError: %s", self.label, url[:180])

    def _on_response(self, params: Dict[str, Any]) -> None:
        req = params.get("request", {}) or {}
        resp = params.get("response", {}) or {}
        url = resp.get("url") or req.get("url") or ""
        rid = req.get("request") or ""
        mime = (resp.get("mimeType") or resp.get("mime") or "").lower()
        status = resp.get("status") or resp.get("statusCode") or 0
        if not rid or not self._looks_like_image(url, mime):
            return
        with self._lock:
            if rid in self._rid_to_idx:
                return
            idx = len(self.captured_images)
            self._rid_to_idx[rid] = idx
            self.captured_images.append(
                {
                    "url": url,
                    "mime": mime,
                    "status": status,
                    "requestId": rid,
                    "timestamp": time.time(),
                    "body_bytes": None,
                }
            )
        LOG.info("[%s] saw image idx=%s status=%s mime=%s url=%s", self.label, idx, status, mime, url[:160])
        self._collect_body(rid, idx)

    @staticmethod
    def _looks_like_image(url: str, mime: str) -> bool:
        u = (url or "").lower()
        if "/rtig/image" in u:
            return True
        if u.startswith("blob:") and "arkoselabs.com" in u:
            return True
        return "arkoselabs.com" in u and mime.startswith("image/") and "/fc/assets/" not in u

    def _collect_body(self, rid: str, idx: int) -> None:
        if not self._driver or not self._collector_id:
            return
        from ruyipage._bidi import network as bidi_network

        try:
            raw = bidi_network.get_data(self._driver, self._collector_id, rid, data_type="response")
            data = decode_bidi_bytes(raw)
        except Exception as exc:
            with self._lock:
                if 0 <= idx < len(self.captured_images):
                    self.captured_images[idx]["body_error"] = f"{type(exc).__name__}: {exc}"
            LOG.debug("[%s] get image body failed idx=%s: %s", self.label, idx, exc)
            return
        if not data:
            return
        sha = hashlib.sha256(data).hexdigest()
        size = image_size(data)
        with self._lock:
            if 0 <= idx < len(self.captured_images):
                rec = self.captured_images[idx]
                rec["body_bytes"] = data
                rec["bytes"] = len(data)
                rec["sha256"] = sha
                rec["size"] = size
        self._event.set()
        LOG.info("[%s] captured image body idx=%s bytes=%s sha256=%s size=%s", self.label, idx, len(data), sha[:12], size)

    def wait_new_challenge(self, seen: set[str], timeout: float, stop_page: Any = None) -> Optional[dict]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if stop_page is not None:
                with contextlib.suppress(Exception):
                    if base.captcha_state(stop_page) in ("success", "rejected"):
                        return None
            with self._lock:
                ready = [dict(r) for r in self.captured_images if r.get("body_bytes")]
            ready.sort(key=lambda r: (0 if "/rtig/image" in (r.get("url") or "").lower() else 1, r.get("timestamp") or 0))
            for rec in ready:
                data = rec.get("body_bytes") or b""
                sha = rec.get("sha256") or hashlib.sha256(data).hexdigest()
                if sha in seen:
                    continue
                size = rec.get("size") or image_size(data)
                if size:
                    w, h = size
                    # Arkose PC challenge strip 常见 2000x400/2400x400；放宽一点，避免误伤新尺寸。
                    if not (w >= 1200 and 300 <= h <= 650):
                        seen.add(sha)
                        LOG.info("[%s] ignore non-challenge image size=%sx%s url=%s", self.label, w, h, (rec.get("url") or "")[:120])
                        continue
                return rec
            self._event.wait(0.5)
            self._event.clear()
        return None


def save_image_record(rec: dict, out_dir: Path, wave: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    data = rec.get("body_bytes") or b""
    sha = rec.get("sha256") or hashlib.sha256(data).hexdigest()
    path = out_dir / f"local_v11_wave_{wave:02d}_{sha[:12]}{image_ext(rec.get('mime') or '', data)}"
    path.write_bytes(data)
    meta = {k: v for k, v in rec.items() if k != "body_bytes"}
    meta.update({"file": str(path), "sha256": sha, "bytes": len(data)})
    base.write_json(out_dir / f"local_v11_wave_{wave:02d}_{sha[:12]}.json", meta)
    return path


def normalize_question_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def challenge_support_details(question_text: str, size: Any) -> dict:
    normalized = normalize_question_text(question_text)
    expected = normalize_question_text(SUPPORTED_QUESTION)
    try:
        width, height = size
        clean_size = (int(width), int(height))
    except (TypeError, ValueError):
        clean_size = None
    question_matched = expected in normalized
    size_matched = clean_size == SUPPORTED_IMAGE_SIZE
    return {
        "supported": bool(question_matched and size_matched),
        "questionMatched": question_matched,
        "sizeMatched": size_matched,
        "imageSize": list(clean_size) if clean_size else None,
        "expectedQuestion": SUPPORTED_QUESTION,
        "textSample": re.sub(r"\s+", " ", str(question_text or "")).strip()[:1000],
    }


def wait_supported_challenge(
    solver_tab: Any,
    size: Any,
    *,
    timeout: float,
    output_path: Path,
) -> dict:
    deadline = time.time() + max(0.0, float(timeout))
    details = challenge_support_details("", size)
    while True:
        text = base.captcha_text(solver_tab)
        details = challenge_support_details(text, size)
        if details["supported"]:
            base.write_json(output_path, details)
            return details
        if time.time() >= deadline:
            base.write_json(output_path, details)
            raise UnsupportedCaptchaQuestion(details)
        time.sleep(0.25)


def rank_v11_endpoint(base_url: str, path: str) -> str:
    return f"{str(base_url).rstrip('/')}/{path.lstrip('/')}"


def ensure_rank_v11_service(base_url: str, timeout: float) -> dict:
    response = requests.get(rank_v11_endpoint(base_url, "health"), timeout=timeout)
    response.raise_for_status()
    result = response.json()
    if not isinstance(result, dict) or not result.get("ok") or result.get("status") != "ready":
        raise RuntimeError(f"rank_v11 service is not ready: {result}")
    return result


def rank_v11_solve_image(
    image_path: Path,
    *,
    base_url: str,
    timeout: float,
    response_path: Path,
) -> tuple[int, dict]:
    payload = {"image": str(image_path.resolve()), "mode": "accurate"}
    LOG.info("rank_v11 solve: image=%s bytes=%s", image_path.name, image_path.stat().st_size)
    response = requests.post(
        rank_v11_endpoint(base_url, "solve"),
        json=payload,
        timeout=timeout,
    )
    try:
        result = response.json()
    except Exception:
        response_path.write_text(response.text[:4000], encoding="utf-8", errors="replace")
        raise
    base.write_json(response_path, result)
    response.raise_for_status()
    if not isinstance(result, dict) or not result.get("ok"):
        raise RuntimeError(f"rank_v11 solve failed: {result}")
    answer = int(result.get("answer_index", -1))
    if not 0 <= answer <= 9:
        raise RuntimeError(f"rank_v11 returned invalid answer_index: {answer}")
    LOG.info(
        "rank_v11 answer=%s fast=%s v9=%s expert=%s switched=%s model_seconds=%.3f",
        answer,
        result.get("fast_index"),
        result.get("legacy_v9_index"),
        result.get("expert_index"),
        result.get("switched_to_v10"),
        float(result.get("model_seconds") or 0.0),
    )
    return answer, result


def solver_state(page) -> dict:
    with contextlib.suppress(Exception):
        state = page.run_js(
            """return (() => {
              const s = window.__ARKOSE_MANUAL__ || {};
              const cp = s.completedPayload || null;
              return {
                status: s.status || null,
                token: s.token || null,
                tokenLength: s.tokenLength || (s.token ? String(s.token).length : 0),
                error: s.error || null,
                completedPayload: cp ? {
                  completed: !!cp.completed,
                  hasToken: !!cp.token,
                  tokenLength: cp.token ? String(cp.token).length : 0,
                  suppressed: !!cp.suppressed,
                  failed: !!cp.failed,
                  error: cp.error == null ? null : String(cp.error),
                  warning: cp.warning == null ? null : String(cp.warning),
                  requested: cp.requested == null ? null : !!cp.requested,
                  recoverable: !!cp.recoverable,
                  width: cp.width == null ? null : Number(cp.width),
                  height: cp.height == null ? null : Number(cp.height),
                  keys: Object.keys(cp).slice(0, 20)
                } : null,
                // 不返回 events：onShown payload 里也会带 token，避免落盘泄漏。
                eventsCount: (s.events || []).length
              };
            })();""",
            timeout=5,
        )
        if isinstance(state, dict):
            return state
    return {}


def completion_rejection_reason(payload: Optional[dict]) -> Optional[str]:
    if not isinstance(payload, dict):
        return None
    if payload.get("failed") is True:
        return "failed=true"
    error = payload.get("error")
    if error:
        return f"error={error}"
    return None


def arkose_token_metadata(token: str) -> dict:
    """Return Arkose token suffix fields without retaining the opaque first segment."""
    text = str(token or "")
    parts = text.split("|")
    fields: Dict[str, str] = {}
    flags: list[str] = []
    for part in parts[1:]:
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
        elif part:
            flags.append(part)
    return {
        "tokenLength": len(text),
        "opaqueLength": len(parts[0]) if parts else 0,
        "fields": fields,
        "flags": flags,
    }


CAPTCHA_GATE_PATH = "/creation/flow/creation-full/step/captcha-gate"


def is_captcha_gate_url(url: str) -> bool:
    return CAPTCHA_GATE_PATH in str(url or "")


def captcha_gate_request_metadata(body: str) -> dict:
    text = str(body or "")
    parsed = parse_qs(text, keep_blank_values=True)
    field_lengths: Dict[str, int] = {}
    token_metadata = None
    for name, values in parsed.items():
        value = str(values[0] if values else "")
        field_lengths[name] = len(value)
        if "arkose" in name.lower():
            token_metadata = arkose_token_metadata(value)
    return {
        "bodyLength": len(text),
        "fieldNames": list(parsed),
        "fieldLengths": field_lengths,
        "arkoseTokenMetadata": token_metadata,
    }


def selected_bidi_headers(headers: Any) -> Dict[str, str]:
    allowed = {"location", "content-type", "x-request-id", "x-correlation-id"}
    selected: Dict[str, str] = {}
    for item in headers or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "").lower()
        if name not in allowed:
            continue
        value = item.get("value")
        if isinstance(value, dict):
            value = value.get("value")
        if value is not None:
            selected[name] = str(value)
    return selected


def sanitize_captcha_gate_response(body: str) -> str:
    return re.sub(
        r"(?i)(\bST=)[^&\"'<>\s]+",
        r"\1<redacted>",
        str(body or ""),
    )


def parse_captcha_gate_response(body: str) -> dict:
    text = str(body or "")

    def attr(name: str) -> Optional[str]:
        match = re.search(rf'{re.escape(name)}=["\']([^"\']*)["\']', text, re.I)
        return html_lib.unescape(match.group(1)).strip() if match else None

    step_id = attr("data-step-id")
    errors_text = attr("data-step-has-errors")
    has_errors = None if errors_text is None else errors_text.lower() == "true"
    player_account_id = attr("data-player-account-id")
    email_match = re.search(
        r'class=["\'][^"\']*step__banner--account-identifier[^"\']*["\'][^>]*>(.*?)</',
        text,
        re.I | re.S,
    )
    account_email = None
    if email_match:
        account_email = re.sub(r"<[^>]+>", "", email_match.group(1))
        account_email = html_lib.unescape(account_email).strip()
    return {
        "stepId": step_id,
        "hasErrors": has_errors,
        "isCreateSuccess": step_id == "create-success" and has_errors is False,
        "playerAccountId": player_account_id,
        "accountEmail": account_email,
    }


def captcha_gate_success(records: Any, expected_email: Optional[str] = None) -> Optional[dict]:
    expected = str(expected_email or "").strip().casefold()
    for record in records or []:
        response = (record or {}).get("response") or {}
        status = int(response.get("status") or 0)
        if not 200 <= status < 300:
            continue
        outcome = response.get("outcome")
        if not isinstance(outcome, dict):
            outcome = parse_captcha_gate_response(response.get("sample") or "")
        if not outcome.get("isCreateSuccess"):
            continue
        actual = str(outcome.get("accountEmail") or "").strip().casefold()
        if expected and actual and actual != expected:
            continue
        return dict(outcome)
    return None


def build_token_result(page: Any, token: str, actions: list[dict]) -> dict:
    state = solver_state(page)
    completed_payload = state.get("completedPayload")
    metadata = arkose_token_metadata(token)
    LOG.info("Arkose completedPayload: %s", completed_payload)
    LOG.info("Arkose token metadata (opaque segment redacted): %s", metadata)
    return {
        "ok": True,
        "token": token,
        "tokenLength": len(token),
        "actions": actions,
        "completedPayload": completed_payload,
        "tokenMetadata": metadata,
    }


@dataclass
class RuyiCaptchaGateCatcher:
    """Capture the Battle.net captcha-gate request/response without storing form values."""

    page: Any
    records: list[dict] = field(default_factory=list)
    _driver: Any = None
    _subscription_id: Optional[str] = None
    _collector_id: Optional[str] = None
    _lock: Lock = field(default_factory=Lock)
    _event: Event = field(default_factory=Event)
    _by_request_id: Dict[str, dict] = field(default_factory=dict)

    def start(self) -> None:
        from ruyipage._bidi import network as bidi_network
        from ruyipage._bidi import session as bidi_session

        self._driver = self.page._driver._browser_driver
        with contextlib.suppress(Exception):
            ret = bidi_network.add_data_collector(
                self._driver,
                events=["beforeRequestSent", "responseCompleted"],
                contexts=None,
                data_types=["request", "response"],
                max_encoded_data_size=4 * 1024 * 1024,
            )
            self._collector_id = ret.get("collector")
        ret = bidi_session.subscribe(
            self._driver,
            ["network.beforeRequestSent", "network.responseCompleted", "network.fetchError"],
            contexts=None,
        )
        self._subscription_id = ret.get("subscription")
        self._driver.set_callback("network.beforeRequestSent", self._on_request, context=None)
        self._driver.set_callback("network.responseCompleted", self._on_response, context=None)
        self._driver.set_callback("network.fetchError", self._on_fetch_error, context=None)
        LOG.info("Battle.net captcha-gate capture started collector=%s", bool(self._collector_id))

    def stop(self) -> None:
        from ruyipage._bidi import network as bidi_network
        from ruyipage._bidi import session as bidi_session

        with contextlib.suppress(Exception):
            if self._driver:
                self._driver.remove_callback("network.beforeRequestSent", context=None)
                self._driver.remove_callback("network.responseCompleted", context=None)
                self._driver.remove_callback("network.fetchError", context=None)
        with contextlib.suppress(Exception):
            if self._driver and self._subscription_id:
                bidi_session.unsubscribe(self._driver, subscription=self._subscription_id)
        with contextlib.suppress(Exception):
            if self._driver and self._collector_id:
                bidi_network.remove_data_collector(self._driver, self._collector_id)

    def _collector_text(self, request_id: str, data_type: str) -> str:
        if not self._driver or not self._collector_id or not request_id:
            return ""
        from ruyipage._bidi import network as bidi_network

        with contextlib.suppress(Exception):
            raw = bidi_network.get_data(self._driver, self._collector_id, request_id, data_type=data_type)
            return base.decode_bidi_body_value(raw)
        return ""

    def _record_for(self, request_id: str, url: str) -> dict:
        with self._lock:
            rec = self._by_request_id.get(request_id)
            if rec is None:
                rec = {"requestId": request_id, "url": url, "capturedAt": time.time()}
                self._by_request_id[request_id] = rec
                self.records.append(rec)
            return rec

    def _on_request(self, params: Dict[str, Any]) -> None:
        req = params.get("request", {}) or {}
        url = req.get("url", "") or ""
        if not is_captcha_gate_url(url):
            return
        request_id = req.get("request", "") or ""
        rec = self._record_for(request_id, url)
        body = base.decode_bidi_body_value(req.get("body"))
        with self._lock:
            rec["method"] = req.get("method")
            rec["request"] = captcha_gate_request_metadata(body)
        LOG.info("captcha-gate request: method=%s metadata=%s", rec.get("method"), rec.get("request"))

    def _on_response(self, params: Dict[str, Any]) -> None:
        req = params.get("request", {}) or {}
        resp = params.get("response", {}) or {}
        url = resp.get("url") or req.get("url") or ""
        if not is_captcha_gate_url(url):
            return
        request_id = req.get("request", "") or ""
        rec = self._record_for(request_id, url)
        request_body = self._collector_text(request_id, "request")
        response_body = self._collector_text(request_id, "response")
        outcome = parse_captcha_gate_response(response_body)
        sanitized_body = sanitize_captcha_gate_response(response_body)
        sample = re.sub(r"\s+", " ", sanitized_body).strip()[:4000]
        with self._lock:
            if request_body:
                rec["request"] = captcha_gate_request_metadata(request_body)
            rec["response"] = {
                "status": resp.get("status") or resp.get("statusCode") or 0,
                "mime": resp.get("mimeType") or resp.get("mime"),
                "headers": selected_bidi_headers(resp.get("headers")),
                "bodyLength": len(response_body),
                "sample": sample,
                "outcome": outcome,
            }
        self._event.set()
        LOG.info(
            "captcha-gate response: status=%s bodyLength=%s sample=%r",
            rec["response"]["status"],
            rec["response"]["bodyLength"],
            sample[:300],
        )

    def _on_fetch_error(self, params: Dict[str, Any]) -> None:
        req = params.get("request", {}) or {}
        url = req.get("url", "") or ""
        if is_captcha_gate_url(url):
            request_id = req.get("request", "") or ""
            rec = self._record_for(request_id, url)
            with self._lock:
                rec["fetchError"] = str(params.get("errorText") or params.get("error") or "fetchError")
            self._event.set()
            LOG.warning("captcha-gate fetchError: %s", rec["fetchError"])

    def wait(self, timeout: float = 3.0) -> list[dict]:
        self._event.wait(max(0.0, timeout))
        return self.snapshot()

    def snapshot(self) -> list[dict]:
        with self._lock:
            return [dict(record) for record in self.records]


def wait_token_quick(page, timeout: float, prefix: str = "") -> Optional[str]:
    deadline = time.time() + timeout
    last_status = object()
    while time.time() < deadline:
        st = solver_state(page)
        if st.get("status") != last_status:
            LOG.info("%sSolver status: %s tokenLength=%s", prefix, st.get("status"), st.get("tokenLength") or 0)
            last_status = st.get("status")
        completed_payload = st.get("completedPayload")
        rejection_reason = completion_rejection_reason(completed_payload)
        if rejection_reason:
            LOG.warning(
                "%sArkose onCompleted returned a rejected completion: %s payload=%s",
                prefix,
                rejection_reason,
                completed_payload,
            )
            raise ArkoseCompletionRejected(rejection_reason, completed_payload)
        if st.get("token"):
            return str(st["token"])
        time.sleep(0.2)
    return None


def run_js_first(page, js: str, *args, timeout: float = 3.0) -> Optional[dict]:
    for ctx in base.all_contexts(page):
        with contextlib.suppress(Exception):
            ret = ctx.run_js(js, *args, timeout=timeout)
            if isinstance(ret, dict) and ret.get("ok"):
                return ret
    return None


def _element_marker(ele: Any) -> str:
    parts = []
    for getter in (
        lambda: getattr(ele, "text", "") or "",
        lambda: ele.attr("aria-label") or "",
        lambda: ele.attr("class") or "",
        lambda: ele.attr("value") or "",
        lambda: str(ele.property("value") or ""),
    ):
        with contextlib.suppress(Exception):
            parts.append(str(getter()))
    return " ".join(parts).strip()


FIND_TARGET_JS = r"""return ((selectors, acceptPattern, rejectPattern) => {
  let accept = null, reject = null;
  try { if (acceptPattern) accept = new RegExp(acceptPattern, 'i'); } catch(e) {}
  try { if (rejectPattern) reject = new RegExp(rejectPattern, 'i'); } catch(e) {}
  const roots = [document], seen = new Set();
  const visible = el => !!el && !el.disabled && el.getAttribute('aria-disabled') !== 'true'
    && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  for (let i = 0; i < roots.length; i++) {
    const root = roots[i];
    if (!root || seen.has(root)) continue;
    seen.add(root);
    try { root.querySelectorAll('*').forEach(el => { if (el.shadowRoot) roots.push(el.shadowRoot); }); } catch(e) {}
    for (const sel of selectors) {
      let els = [];
      try { els = Array.from(root.querySelectorAll(sel)); } catch(e) {}
      for (const el of els) {
        const marker = ((el.textContent || '') + ' ' + (el.value || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.className || '')).trim();
        if (accept && !accept.test(marker)) continue;
        if (reject && reject.test(marker)) continue;
        if (!visible(el)) continue;
        const r = el.getBoundingClientRect();
        if (!r || r.width <= 0 || r.height <= 0) continue;
        return {
          ok: true,
          x: Math.round(r.left + r.width / 2),
          y: Math.round(r.top + r.height / 2),
          width: Math.round(r.width),
          height: Math.round(r.height),
          selector: sel,
          marker: marker.slice(0, 180)
        };
      }
    }
  }
  return {ok:false};
})(arguments[0], arguments[1], arguments[2]);"""


def native_click_at(ctx: Any, x: int, y: int, desc: str) -> bool:
    """用 BiDi input.performActions 在当前 browsing context 坐标点击，速度快且 isTrusted=true。"""
    if CLICK_STYLE == "balanced":
        move_ms = random.randint(85, 170)
        hold_ms = random.randint(55, 115)
    else:
        move_ms = FAST_POINTER_MOVE_MS
        hold_ms = FAST_CLICK_HOLD_MS
    try:
        ctx._driver._browser_driver.run(
            "input.performActions",
            {
                "context": ctx._context_id,
                "actions": [
                    {
                        "type": "pointer",
                        "id": "mouse0",
                        "parameters": {"pointerType": "mouse"},
                        "actions": [
                            {"type": "pointerMove", "x": int(x), "y": int(y), "duration": move_ms},
                            {"type": "pointerDown", "button": 0},
                            {"type": "pause", "duration": hold_ms},
                            {"type": "pointerUp", "button": 0},
                        ],
                    }
                ],
            },
        )
        LOG.info("Native %s click %s at (%s,%s) move=%sms hold=%sms", CLICK_STYLE, desc, int(x), int(y), move_ms, hold_ms)
        return True
    except Exception as exc:
        LOG.debug("native fast click failed for %s: %s: %s", desc, type(exc).__name__, exc)
        return False


def _viewport_size(ctx: Any) -> tuple[int, int]:
    with contextlib.suppress(Exception):
        size = ctx.rect.viewport_size
        return max(1, int(size[0])), max(1, int(size[1]))
    return 1920, 1080


def _clamp(n: float, lo: int, hi: int) -> int:
    return int(min(max(round(n), lo), hi))


def native_human_click_at(ctx: Any, x: int, y: int, desc: str) -> bool:
    """自定义 1-2 秒随机人类轨迹点击，比 RuyiPage 内置 human_move 快很多。"""
    width, height = _viewport_size(ctx)
    tx = _clamp(x, 1, width - 2)
    ty = _clamp(y, 1, height - 2)

    actions_unit = getattr(ctx, "actions", None)
    known = bool(getattr(actions_unit, "_pointer_position_known", False))
    if known:
        sx = _clamp(getattr(actions_unit, "curr_x", tx), 1, width - 2)
        sy = _clamp(getattr(actions_unit, "curr_y", ty), 1, height - 2)
    else:
        sx = _clamp(tx + random.choice((-1, 1)) * random.randint(100, 260), 1, width - 2)
        sy = _clamp(ty + random.choice((-1, 1)) * random.randint(60, 180), 1, height - 2)

    total_ms = random.randint(HUMAN_MOVE_MIN_MS, HUMAN_MOVE_MAX_MS)
    steps = random.randint(9, 15)
    dx = tx - sx
    dy = ty - sy
    # 控制点制造一点弧线；距离很近时也保留小抖动，避免完全机械。
    cx = (sx + tx) / 2 + random.randint(-80, 80)
    cy = (sy + ty) / 2 + random.randint(-55, 55)
    move_actions = [{"type": "pointerMove", "x": sx, "y": sy, "duration": 0}]
    remaining = total_ms
    for i in range(1, steps + 1):
        t = i / steps
        # 二次贝塞尔 + 末端轻微抖动。
        bx = (1 - t) * (1 - t) * sx + 2 * (1 - t) * t * cx + t * t * tx
        by = (1 - t) * (1 - t) * sy + 2 * (1 - t) * t * cy + t * t * ty
        if i < steps:
            bx += random.uniform(-2.5, 2.5)
            by += random.uniform(-2.0, 2.0)
        dur = max(25, int(total_ms / steps + random.randint(-25, 35)))
        remaining -= dur
        move_actions.append({"type": "pointerMove", "x": _clamp(bx, 1, width - 2), "y": _clamp(by, 1, height - 2), "duration": dur})
    if remaining > 25:
        move_actions.append({"type": "pause", "duration": min(remaining, 180)})
    move_actions.extend(
        [
            {"type": "pointerMove", "x": tx + random.randint(-1, 1), "y": ty + random.randint(-1, 1), "duration": random.randint(25, 60)},
            {"type": "pointerDown", "button": 0},
            {"type": "pause", "duration": random.randint(70, 140)},
            {"type": "pointerUp", "button": 0},
        ]
    )

    try:
        ctx._driver._browser_driver.run(
            "input.performActions",
            {
                "context": ctx._context_id,
                "actions": [
                    {
                        "type": "pointer",
                        "id": "mouse0",
                        "parameters": {"pointerType": "mouse"},
                        "actions": move_actions,
                    }
                ],
            },
        )
        with contextlib.suppress(Exception):
            ctx.actions.curr_x = tx
            ctx.actions.curr_y = ty
            ctx.actions._pointer_position_known = True
        LOG.info("Native human click %s at (%s,%s) total=%sms steps=%s", desc, tx, ty, total_ms, steps)
        return True
    except Exception as exc:
        LOG.debug("custom human click failed for %s: %s: %s", desc, type(exc).__name__, exc)
        return False


def native_human_click_element(ctx: Any, ele: Any, desc: str) -> bool:
    """原生可信点击；fast 用短轨迹，human 用真人化轨迹。"""
    if CLICK_STYLE == "human":
        try:
            pos = ele._get_center()
            if pos and native_human_click_at(ctx, int(pos["x"]), int(pos["y"]), f"{desc} marker={_element_marker(ele)[:80]}"):
                return True
        except Exception as exc:
            LOG.debug("human_click failed for %s: %s: %s", desc, type(exc).__name__, exc)
    try:
        ele.click()
        LOG.info("Native fast element click %s: %s", desc, _element_marker(ele)[:120])
        return True
    except Exception as exc:
        LOG.debug("native element click failed for %s: %s: %s", desc, type(exc).__name__, exc)
    return False


def native_click_selectors(
    page: Any,
    selectors: list[str],
    desc: str,
    accept_re: str | None = None,
    reject_re: str | None = None,
    per_context_timeout: float = 0.25,
) -> bool:
    if CLICK_STYLE == "js":
        return False
    if CLICK_STYLE in ("fast", "balanced", "human"):
        for ctx in base.all_contexts(page):
            with contextlib.suppress(Exception):
                target = ctx.run_js(FIND_TARGET_JS, selectors, accept_re or "", reject_re or "", timeout=0.6)
                if isinstance(target, dict) and target.get("ok"):
                    x = int(target["x"])
                    y = int(target["y"])
                    if CLICK_STYLE in ("balanced", "human"):
                        max_dx = max(1, min(9, int((target.get("width") or 20) * 0.22)))
                        max_dy = max(1, min(7, int((target.get("height") or 20) * 0.22)))
                        x += random.randint(-max_dx, max_dx)
                        y += random.randint(-max_dy, max_dy)
                    click_desc = f"{desc} selector={target.get('selector')} marker={target.get('marker', '')[:80]}"
                    clicked = native_human_click_at(ctx, x, y, click_desc) if CLICK_STYLE == "human" else native_click_at(ctx, x, y, click_desc)
                    if clicked:
                        return True

    accept = re.compile(accept_re, re.I) if accept_re else None
    reject = re.compile(reject_re, re.I) if reject_re else None
    for ctx in base.all_contexts(page):
        for sel in selectors:
            with contextlib.suppress(Exception):
                candidates = ctx.eles(sel, timeout=per_context_timeout) or []
                for ele in candidates[:8]:
                    marker = _element_marker(ele)
                    if accept and not accept.search(marker):
                        continue
                    if reject and reject.search(marker):
                        continue
                    if native_human_click_element(ctx, ele, f"{desc} selector={sel}"):
                        return True
    return False


CLICK_ARROW_JS = r"""return ((direction) => {
  const right = direction === 'right';
  const roots = [document], seen = new Set();
  const visible = el => !!el && !el.disabled && el.getAttribute('aria-disabled') !== 'true'
    && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  for (let i = 0; i < roots.length; i++) {
    const root = roots[i];
    if (!root || seen.has(root)) continue;
    seen.add(root);
    try { root.querySelectorAll('*').forEach(el => { if (el.shadowRoot) roots.push(el.shadowRoot); }); } catch(e) {}
    const selectors = right
      ? ['a.right-arrow','[class*="right-arrow"]','button[aria-label*="Next"]','a[aria-label*="Next"]','[aria-label*="next"]','[aria-label*="下一"]']
      : ['a.left-arrow','[class*="left-arrow"]','button[aria-label*="Previous"]','a[aria-label*="Previous"]','[aria-label*="previous"]','[aria-label*="上一"]'];
    for (const sel of selectors) {
      let els = [];
      try { els = Array.from(root.querySelectorAll(sel)); } catch(e) {}
      for (const el of els) {
        if (!visible(el)) continue;
        el.scrollIntoView({block:'center', inline:'center'});
        el.click();
        return {ok:true, selector:sel, aria:el.getAttribute('aria-label')||'', cls:el.className || ''};
      }
    }
  }
  return {ok:false};
})(arguments[0]);"""


CLICK_SUBMIT_JS = r"""return (() => {
  const roots = [document], seen = new Set();
  const visible = el => !!el && !el.disabled && el.getAttribute('aria-disabled') !== 'true'
    && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
  for (let i = 0; i < roots.length; i++) {
    const root = roots[i];
    if (!root || seen.has(root)) continue;
    seen.add(root);
    try { root.querySelectorAll('*').forEach(el => { if (el.shadowRoot) roots.push(el.shadowRoot); }); } catch(e) {}
    const selectors = ['button.sc-nkuzb1-0.yuVdl.button','button.sc-nkuzb1-0.yuVdl','button[type="submit"]','input[type="submit"]','button','[role="button"]'];
    for (const sel of selectors) {
      let els = [];
      try { els = Array.from(root.querySelectorAll(sel)); } catch(e) {}
      for (const el of els) {
        const marker = ((el.textContent || '') + ' ' + (el.value || '') + ' ' + (el.getAttribute('aria-label') || '') + ' ' + (el.className || '')).toLowerCase();
        const isSubmit = /submit|提交|送出|continue/.test(marker) || /yuVdl/.test(marker);
        const verifyOnly = /verify|human|验证|人类/.test(marker) && !/submit|提交/.test(marker);
        if (!isSubmit || verifyOnly || !visible(el)) continue;
        el.scrollIntoView({block:'center', inline:'center'});
        el.click();
        return {ok:true, selector:sel, text:(el.textContent||el.value||'').trim().slice(0,80), aria:el.getAttribute('aria-label')||'', cls:el.className || ''};
      }
    }
  }
  return {ok:false};
})();"""


GET_INDEX_JS = r"""return (() => {
  const imgs = Array.from(document.querySelectorAll('img[aria-label], img'));
  for (const img of imgs) {
    const s = img.getAttribute('aria-label') || '';
    const cls = img.getAttribute('class') || '';
    const style = img.getAttribute('style') || '';
    const nums = s.match(/\d+/g);
    const carousel = /image|图像|圖像|of|项|項/i.test(s) || cls.includes('sc-7csxyx') || style.includes('blob:') || /arkoselabs|rtig|blob:/.test(img.src || '');
    if (!carousel || !nums || nums.length < 1) continue;
    const first = parseInt(nums[0], 10);
    const total = nums.length >= 2 ? parseInt(nums[1], 10) : 12;
    if (first >= 1 && first <= 12 && (nums.length === 1 || (total >= 6 && total <= 12))) return first - 1;
  }
  return -1;
})();"""


def current_index(page) -> int:
    for ctx in base.all_contexts(page):
        with contextlib.suppress(Exception):
            idx = int(ctx.run_js(GET_INDEX_JS, timeout=3))
            if idx >= 0:
                return max(0, min(11, idx))
    return -1


def wait_index_change(page, before: int, timeout: float = 0.9) -> int:
    deadline = time.time() + timeout
    latest = before
    while time.time() < deadline:
        now = current_index(page)
        if now >= 0:
            latest = now
            if before < 0 or now != before:
                return now
        time.sleep(0.04)
    return latest


def click_gap() -> float:
    if CLICK_STYLE == "fast":
        return 0.03 + random.random() * 0.04
    if CLICK_STYLE == "balanced":
        return 0.18 + random.random() * 0.22
    if CLICK_STYLE == "human":
        return 0.25 + random.random() * 0.45
    return 0.12 + random.random() * 0.18


def click_arrow(page, direction: str, timeout: float = 6.0) -> bool:
    deadline = time.time() + timeout
    if direction == "right":
        selectors = [
            'a.right-arrow',
            '[class*="right-arrow"]',
            'button[aria-label*="Next"]',
            'a[aria-label*="Next"]',
            '[aria-label*="next"]',
            '[aria-label*="下一"]',
        ]
        accept_re = r"right-arrow|next|下一|右"
    else:
        selectors = [
            'a.left-arrow',
            '[class*="left-arrow"]',
            'button[aria-label*="Previous"]',
            'a[aria-label*="Previous"]',
            '[aria-label*="previous"]',
            '[aria-label*="上一"]',
        ]
        accept_re = r"left-arrow|previous|prev|上一|左"
    while time.time() < deadline:
        if native_click_selectors(page, selectors, f"{direction} arrow", accept_re=accept_re):
            return True
        if run_js_first(page, CLICK_ARROW_JS, direction, timeout=2.5):
            LOG.warning("Fallback JS click used for %s arrow; this may lower Arkose trust", direction)
            return True
        time.sleep(0.25)
    return False


def click_next_n(page, count: int) -> bool:
    count = max(0, int(count))
    LOG.info("按本地 V11 answer_index 点击下一张图按钮 %s 次", count)
    for i in range(count):
        ok = False
        for attempt in range(3):
            before = current_index(page)
            if not click_arrow(page, "right", timeout=4):
                return False
            after = wait_index_change(page, before)
            if before < 0 or after != before:
                LOG.info("Next click %s/%s ok: before=%s after=%s", i + 1, count, before, after)
                ok = True
                break
            LOG.warning("Next click %s/%s 可能被忽略，retry=%s before=%s after=%s", i + 1, count, attempt + 1, before, after)
        if not ok:
            return False
        time.sleep(click_gap())
    return True


def click_submit(page, timeout: float = 6.0) -> bool:
    deadline = time.time() + timeout
    selectors = [
        'button.sc-nkuzb1-0.yuVdl.button',
        'button.sc-nkuzb1-0.yuVdl',
        'button[type="submit"]',
        'input[type="submit"]',
        'button',
        '[role="button"]',
    ]
    while time.time() < deadline:
        if native_click_selectors(
            page,
            selectors,
            "submit",
            accept_re=r"submit|提交|送出|continue|yuVdl",
            reject_re=r"verify|human|验证|人类",
        ):
            return True
        ret = run_js_first(page, CLICK_SUBMIT_JS, timeout=2.5)
        if ret:
            LOG.warning("Fallback JS click used for Arkose Submit: %s", ret)
            return True
        time.sleep(0.35)
    return False


def ensure_verify_or_image(page, catcher: RuyiArkoseImageCatcher, timeout: float) -> bool:
    deadline = time.time() + timeout
    last_click = 0.0
    verify_selectors = [
        'button[data-theme="home.verifyButton"]',
        'button[aria-label="Verify"]',
        'button[aria-label="验证"]',
        'button',
        '[role="button"]',
    ]
    while time.time() < deadline:
        with catcher._lock:
            if any(r.get("body_bytes") for r in catcher.captured_images):
                return True
        if time.time() - last_click > 1.8:
            clicked = native_click_selectors(
                page,
                verify_selectors,
                "verify",
                accept_re=r"verify|human|验证|人类|home\.verifybutton",
                per_context_timeout=0.2,
            )
            if not clicked:
                clicked = base.click_arkose_verify(page, timeout=2.0)
                if clicked:
                    LOG.warning("Fallback JS click used for Arkose Verify")
            if clicked:
                LOG.info("Clicked solver Arkose Verify")
            last_click = time.time()
        time.sleep(0.4)
    return False


def wait_image_or_token(catcher: RuyiArkoseImageCatcher, seen: set[str], timeout: float, solver_tab: Any) -> tuple[str, Any]:
    """等待下一张图，同时轮询 token；避免 token 已出还白等图片超时。"""
    deadline = time.time() + timeout
    while time.time() < deadline:
        token = wait_token_quick(solver_tab, 0.1)
        if token:
            return "token", token
        rec = catcher.wait_new_challenge(seen, timeout=0.7, stop_page=solver_tab)
        if rec:
            return "image", rec
    token = wait_token_quick(solver_tab, 0.5)
    if token:
        return "token", token
    return "timeout", None


def auto_solve_solver_tab(solver_tab: Any, catcher: RuyiArkoseImageCatcher, args: argparse.Namespace, out: Path) -> dict:
    images_dir = out / "local_v11_images"
    actions: list[dict] = []
    seen: set[str] = set()

    token = wait_token_quick(solver_tab, 2.0, "initial ")
    if token:
        return build_token_result(solver_tab, token, actions)

    if not ensure_verify_or_image(solver_tab, catcher, args.verify_timeout):
        LOG.warning("未确认 Verify 后出现图片，继续等待图片")

    for wave in range(args.max_waves):
        token = wait_token_quick(solver_tab, 1.0, f"wave{wave} pre ")
        if token:
            return build_token_result(solver_tab, token, actions)

        kind, value = wait_image_or_token(
            catcher,
            seen,
            args.first_image_timeout if wave == 0 else args.next_image_timeout,
            solver_tab,
        )
        if kind == "token":
            token = str(value)
            return build_token_result(solver_tab, token, actions)
        rec = value
        if not rec:
            token = wait_token_quick(solver_tab, args.after_submit_token_wait, f"wave{wave} no-image ")
            if token:
                return build_token_result(solver_tab, token, actions)
            state = base.captcha_state(solver_tab)
            sample = base.captcha_text(solver_tab).replace("\n", " ")[:260]
            return {"ok": False, "error": f"no new challenge image at wave={wave}, state={state}", "actions": actions, "sample": sample}

        data = rec.get("body_bytes") or b""
        sha = rec.get("sha256") or hashlib.sha256(data).hexdigest()
        seen.add(sha)
        img_path = save_image_record(rec, images_dir, wave)
        LOG.info("捕获验证图 wave=%s path=%s size=%s url=%s", wave, img_path, rec.get("size"), (rec.get("url") or "")[:160])
        if args.debug_screenshots:
            base.screenshot(solver_tab, out / "solver_screenshots" / f"wave_{wave:02d}_before_answer.png")

        question_details = wait_supported_challenge(
            solver_tab,
            rec.get("size") or image_size(data),
            timeout=args.question_timeout,
            output_path=images_dir / f"local_v11_wave_{wave:02d}_question.json",
        )
        answer, model_result = rank_v11_solve_image(
            img_path,
            base_url=args.rank_v11_url,
            timeout=args.rank_v11_timeout,
            response_path=images_dir / f"local_v11_wave_{wave:02d}_response.json",
        )
        if not click_next_n(solver_tab, answer):
            return {"ok": False, "error": f"failed to click next {answer} times at wave={wave}", "actions": actions}

        time.sleep(0.08 + random.random() * 0.08)
        submit_ok = click_submit(solver_tab)
        action = {
            "wave": wave,
            "image": str(img_path),
            "sha256": sha,
            "answer": answer,
            "clicks": answer,
            "submit": submit_ok,
            "question": question_details,
            "rankV11": {
                "fastIndex": model_result.get("fast_index"),
                "legacyV9Index": model_result.get("legacy_v9_index"),
                "expertIndex": model_result.get("expert_index"),
                "switchedToV10": model_result.get("switched_to_v10"),
                "confidence": model_result.get("confidence"),
                "margin": model_result.get("margin"),
                "modelSeconds": model_result.get("model_seconds"),
            },
        }
        actions.append(action)
        base.write_json(out / "local_v11_actions_latest.json", actions)
        if args.debug_screenshots:
            base.screenshot(solver_tab, out / "solver_screenshots" / f"wave_{wave:02d}_after_submit.png")
        if not submit_ok:
            return {"ok": False, "error": f"submit button failed at wave={wave}, state={base.captcha_state(solver_tab)}", "actions": actions}

        token = wait_token_quick(solver_tab, args.after_submit_token_wait, f"wave{wave} post ")
        if token:
            return build_token_result(solver_tab, token, actions)

    token = wait_token_quick(solver_tab, args.token_timeout, "final ")
    if token:
        return build_token_result(solver_tab, token, actions)
    return {"ok": False, "error": f"max_waves exceeded ({args.max_waves}) without token", "actions": actions}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="RuyiPage same-browser + local Route V11 auto register experiment.")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--keep-open", action="store_true")
    ap.add_argument(
        "--click-style",
        choices=("balanced", "fast", "human", "js"),
        default="balanced",
        help="balanced=较快但带随机坐标/停顿；fast=最快原生短点击；human=慢速真人轨迹；js=仅兜底测试",
    )
    ap.add_argument("--debug-screenshots", action="store_true", help="保存每轮答题前/后的求解标签截图；默认关闭以提速")
    ap.add_argument("--blob-timeout", type=float, default=45.0)
    ap.add_argument("--success-timeout", type=float, default=45.0)
    ap.add_argument("--click-original-verify", action="store_true")
    ap.add_argument("--skip-egress-check", action="store_true")
    ap.add_argument("--shared-proxy", "--solver-proxy", dest="shared_proxy")
    ap.add_argument("--network-mode", type=int, choices=(1, 2))
    ap.add_argument("--isolated-root", default=str(base.DEFAULT_ISOLATED_ROOT))
    ap.add_argument("--proxy-config")
    ap.add_argument("--proxy-core")
    ap.add_argument("--proxy-node-index", type=int)
    ap.add_argument("--proxy-timeout-ms", type=int, default=6000)
    ap.add_argument("--proxy-workers", type=int, default=8)
    ap.add_argument("--rank-v11-url", default=os.environ.get("RANK_V11_URL", DEFAULT_RANK_V11_URL))
    ap.add_argument("--rank-v11-timeout", type=float, default=120.0)
    ap.add_argument("--question-timeout", type=float, default=8.0)
    ap.add_argument("--human-move-min-ms", type=int, default=1000, help="--click-style human 时每次点击轨迹最短毫秒")
    ap.add_argument("--human-move-max-ms", type=int, default=2000, help="--click-style human 时每次点击轨迹最长毫秒")
    ap.add_argument("--max-waves", type=int, default=8)
    ap.add_argument("--verify-timeout", type=float, default=30.0)
    ap.add_argument("--first-image-timeout", type=float, default=35.0)
    ap.add_argument("--next-image-timeout", type=float, default=22.0)
    ap.add_argument("--after-submit-token-wait", type=float, default=2.0)
    ap.add_argument("--token-timeout", type=float, default=20.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    global CLICK_STYLE, HUMAN_MOVE_MIN_MS, HUMAN_MOVE_MAX_MS
    CLICK_STYLE = args.click_style
    HUMAN_MOVE_MIN_MS = max(80, int(args.human_move_min_ms))
    HUMAN_MOVE_MAX_MS = max(HUMAN_MOVE_MIN_MS, int(args.human_move_max_ms))
    out = Path(args.output_dir) / base.run_id()
    out.mkdir(parents=True, exist_ok=True)
    base.setup_logging(out / "run.log")

    mode = base.choose_network_mode(args)
    proxy_route = None
    proxy_url = args.shared_proxy
    proxy_info: Dict[str, Any] = {"mode": "default", "proxyURL": proxy_url}
    if mode == 2:
        proxy_route = IsolatedProxyRoute(
            project_dir=Path(args.isolated_root),
            config_path=Path(args.proxy_config) if args.proxy_config else None,
            core_path=Path(args.proxy_core) if args.proxy_core else None,
            timeout_ms=args.proxy_timeout_ms,
            workers=args.proxy_workers,
            node_index=args.proxy_node_index,
            evidence_dir=out / "proxy_evidence",
            require_explicit_choice=True,
        )
        proxy_info = proxy_route.start()
        proxy_url = proxy_info.get("proxyURL")
    else:
        LOG.info("Network mode 1: use local default network/system proxy")

    LOG.info("输出目录: %s", out.resolve())
    LOG.info("架构: 原注册标签 -> 同浏览器求解标签 -> 本地 V11 解图 -> token 注入")
    LOG.info("唯一支持题型: %r", SUPPORTED_QUESTION)
    LOG.info(
        "Click style: %s, human_move=%s-%sms, debug_screenshots=%s, after_submit_token_wait=%.1fs",
        args.click_style,
        HUMAN_MOVE_MIN_MS,
        HUMAN_MOVE_MAX_MS,
        args.debug_screenshots,
        args.after_submit_token_wait,
    )

    page = None
    solver_tab = None
    blob_catcher = None
    img_catcher = None
    gate_catcher = None
    ca_records: list[dict] = []
    image_records: list[dict] = []
    try:
        rank_v11_health = ensure_rank_v11_service(args.rank_v11_url, args.rank_v11_timeout)
        base.write_json(out / "rank_v11_health.json", rank_v11_health)
        LOG.info(
            "rank_v11 ready: url=%s device=%s load=%.3fs warmup=%.3fs",
            args.rank_v11_url,
            rank_v11_health.get("device"),
            float(rank_v11_health.get("model_load_seconds") or 0.0),
            float(rank_v11_health.get("warmup_seconds") or 0.0),
        )
        acc = generate_identity()
        base.write_json(out / "account_generated.json", acc)
        LOG.info("账号: %s", acc["email"])
        LOG.info("BattleTag: %s", acc["battle_tag"])

        page = base.launch_ruyi_browser(args, proxy_url)
        LOG.info("RuyiPage version=%s, UA=%s", getattr(ruyipage, "__version__", "?"), page.user_agent)
        if not args.skip_egress_check:
            base.verify_browser_egress(page, out, proxy_url)

        base.drive_original_to_battletag(page, acc, out)
        blob_catcher = base.RuyiArkoseCatcher(page)
        blob_catcher.start()

        LOG.info("Submit BattleTag to trigger FunCaptcha")
        base.click_ele(page, "#flow-form-submit-btn", "BattleTag submit")
        time.sleep(2)
        base.screenshot(page, out / "original_screenshots" / "after_battletag_submit.png")

        blob = blob_catcher.wait_for_blob(timeout=min(15.0, args.blob_timeout))
        if args.click_original_verify or not blob:
            LOG.info("Click original Arkose Verify%s", " (forced)" if args.click_original_verify else " because blob was not captured yet")
            clicked = base.click_arkose_verify(page, timeout=25)
            base.write_json(out / "original_verify_click.json", {"clicked": clicked, "forced": bool(args.click_original_verify)})
            time.sleep(1.5)
            base.screenshot(page, out / "original_screenshots" / "after_original_verify_click.png")
            if not blob:
                blob = blob_catcher.wait_for_blob(timeout=args.blob_timeout)
        blob = blob or blob_catcher.captured_blob
        if not blob:
            raise RuntimeError("no Arkose blob captured from original tab through RuyiPage capture")

        ctx = base.detect_arkose_context(page, blob_catcher)
        if not ctx.get("siteKey"):
            raise RuntimeError("Arkose public key not detected")
        ca_records = list(blob_catcher.ca_requests or [])
        with contextlib.suppress(Exception):
            blob_catcher.stop()
            blob_catcher = None

        base.write_json(out / "original_arkose_context.json", {**ctx, "blobLength": len(blob), "hasBlob": True})
        base.write_json(
            out / "solver_task.json",
            {
                "websiteURL": ctx.get("websiteURL"),
                "websitePublicKey": ctx.get("siteKey"),
                "funcaptchaApiJSSubdomain": ctx.get("surl"),
                "data": {"blob": f"<redacted len={len(blob)}>"},
                "blobLength": len(blob),
                "mode": "ruyipage-same-browser-new-tab-local-v11",
                "supportedQuestion": SUPPORTED_QUESTION,
                "rankV11URL": args.rank_v11_url,
            },
        )
        LOG.info("Captured Arkose context: pk=%s, surl=%s, blob_len=%s", ctx.get("siteKey"), ctx.get("surl"), len(blob))

        solver_tab = page.new_tab(background=False)
        img_catcher = RuyiArkoseImageCatcher(page)
        img_catcher.start()
        html = base.build_solver_harness(str(ctx["siteKey"]), blob, str(ctx.get("surl") or base.DEFAULT_SURL))
        (out / "solver_harness.html").write_text(html, encoding="utf-8")
        origin_info = base.replace_document_under_origin(solver_tab, str(ctx["websiteURL"]), html)
        base.write_json(out / "solver_origin.json", origin_info)
        solver_tab.activate()
        base.screenshot(solver_tab, out / "solver_screenshots" / "harness_loaded.png")

        solve_result = auto_solve_solver_tab(solver_tab, img_catcher, args, out)
        base.write_json(out / "local_v11_solver_result.json", {k: v for k, v in solve_result.items() if k != "token"})
        if not solve_result.get("ok"):
            raise TimeoutError(solve_result.get("error") or "local V11 solver tab did not return token")
        token = str(solve_result["token"])
        LOG.info("Solver tab returned onCompleted token, length=%s", len(token))

        image_records = [{k: v for k, v in r.items() if k != "body_bytes"} for r in img_catcher.captured_images]
        base.write_json(out / "captured_image_records.json", image_records)
        with contextlib.suppress(Exception):
            img_catcher.stop()
        img_catcher = None

        try:
            gate_catcher = RuyiCaptchaGateCatcher(page)
            gate_catcher.start()
        except Exception as exc:
            gate_catcher = None
            LOG.warning("captcha-gate capture start failed: %s: %s", type(exc).__name__, exc)

        inject_result = base.inject_token_to_original(page, token)
        base.write_json(out / "token_injection_result.json", inject_result)
        LOG.info("Original tab token injection result: %s", inject_result)
        base.screenshot(page, out / "original_screenshots" / "after_token_injection.png")

        gate_records = gate_catcher.wait(timeout=min(8.0, args.success_timeout)) if gate_catcher else []
        base.write_json(out / "captcha_gate_records.json", gate_records)
        gate_outcome = captcha_gate_success(gate_records, acc["email"])
        if gate_outcome:
            success = True
            success_source = "captcha-gate-response"
            LOG.info("captcha-gate confirmed create-success: %s", gate_outcome)
        else:
            success = base.wait_registration_success(page, acc["email"], timeout=args.success_timeout)
            success_source = "page-dom" if success else None
        reg_result = {
            "ok": bool(success),
            "email": acc["email"],
            "battleTag": acc["battle_tag"],
            "url": page.url,
            "successSource": success_source,
        }
        if gate_outcome:
            reg_result["captchaGateOutcome"] = gate_outcome
        if success:
            base.screenshot(page, out / "original_screenshots" / "registration_success.png")
            LOG.info("注册成功；本地 V11 求解 token 已被原注册标签接受")
        else:
            reg_result["captchaState"] = base.captcha_state(page)
            reg_result["sample"] = base.captcha_text(page).replace("\n", " ")[:300]
            base.screenshot(page, out / "original_screenshots" / "registration_not_confirmed.png")
        base.write_json(out / "registration_result.json", reg_result)

        base.write_json(
            out / "summary.json",
            {
                "ok": bool(success),
                "outputDir": str(out.resolve()),
                "mode": "ruyipage-same-browser-new-tab-local-v11",
                "networkMode": mode,
                "proxy": proxy_info,
                "siteKey": ctx.get("siteKey"),
                "surl": ctx.get("surl"),
                "blobLength": len(blob),
                "tokenLength": len(token),
                "completedPayload": solve_result.get("completedPayload"),
                "tokenMetadata": solve_result.get("tokenMetadata"),
                "rankV11": {
                    "url": args.rank_v11_url,
                    "supportedQuestion": SUPPORTED_QUESTION,
                    "health": rank_v11_health,
                    "actions": solve_result.get("actions") or [],
                },
                "injectResult": inject_result,
                "registration": reg_result,
                "captchaGateRecords": gate_records,
                "caRecords": ca_records,
                "imageRecords": image_records,
            },
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        LOG.warning("收到 Ctrl+C，准备退出")
        base.write_json(out / "summary.json", {"ok": False, "error": "KeyboardInterrupt", "outputDir": str(out.resolve())})
        return 130
    except Exception as exc:
        LOG.error("Run failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        failure_summary = {"ok": False, "error": f"{type(exc).__name__}: {exc}", "outputDir": str(out.resolve())}
        if isinstance(exc, UnsupportedCaptchaQuestion):
            failure_summary["unsupportedCaptcha"] = True
            failure_summary["unsupportedCaptchaExitCode"] = UNSUPPORTED_CAPTCHA_EXIT_CODE
            failure_summary["challenge"] = exc.details
        if isinstance(exc, ArkoseCompletionRejected):
            failure_summary["completedPayload"] = exc.payload
            base.write_json(
                out / "local_v11_solver_result.json",
                {"ok": False, "error": str(exc), "completedPayload": exc.payload},
            )
        if gate_catcher is not None:
            with contextlib.suppress(Exception):
                failure_summary["captchaGateRecords"] = gate_catcher.snapshot()
                base.write_json(out / "captcha_gate_records.json", failure_summary["captchaGateRecords"])
        base.write_json(out / "summary.json", failure_summary)
        with contextlib.suppress(Exception):
            if page:
                base.screenshot(page, out / "original_screenshots" / "error_original_page.png")
        with contextlib.suppress(Exception):
            if solver_tab:
                base.screenshot(solver_tab, out / "solver_screenshots" / "error_solver_page.png")
        return UNSUPPORTED_CAPTCHA_EXIT_CODE if isinstance(exc, UnsupportedCaptchaQuestion) else 1
    finally:
        with contextlib.suppress(Exception):
            if img_catcher:
                img_catcher.stop()
        with contextlib.suppress(Exception):
            if gate_catcher:
                gate_catcher.stop()
        with contextlib.suppress(Exception):
            if blob_catcher:
                blob_catcher.stop()
        if args.keep_open and page is not None:
            try:
                input("浏览器保持打开。检查完后按 Enter 关闭...")
            except EOFError:
                pass
        with contextlib.suppress(Exception):
            if page:
                page.quit()
        if proxy_route is not None:
            proxy_route.stop()


if __name__ == "__main__":
    raise SystemExit(main())
