# -*- coding: utf-8 -*-
"""Two-browser FunCaptcha snapshot runner.

新增用途：
1. 按 register.py / register_capture_images.py 的注册步骤跑到 BattleTag 后的 Arkose。
2. 原注册浏览器点击 Verify，等题图真正加载后截图。
3. 用原浏览器抓到的 blob/public key/surl 打开第二个独立浏览器的 Arkose harness。
4. 第二浏览器点击 Verify，等题图真正加载后截图。

只抓图，不求解，不提交最终注册；不修改原有 register.py。
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlsplit

from cloakbrowser import launch

from captcha_image_collector import CDPImageCatcher, click_verify_button, save_cdp_images
from register import (
    CDPBlobCatcher,
    CDP_DEBUG_PORT,
    COUNTRY,
    REGISTER_URL,
    create_cloak_browser,
    generate_identity,
)
from register_capture_images import close_cookie_banner, fill_birthday, wait_or_refresh


logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("two_browser_snapshot")

ORIGINAL_PORT = int(os.environ.get("ORIGINAL_CDP_PORT", str(CDP_DEBUG_PORT)))
SOLVER_PORT = int(os.environ.get("SOLVER_CDP_PORT", str(CDP_DEBUG_PORT + 1)))
DEFAULT_SURL = "blizzard-api.arkoselabs.com"


def _truthy(value: Optional[str], default: bool = True) -> bool:
    if value in (None, ""):
        return default
    return str(value).strip().lower() not in ("0", "false", "no", "off")


def _run_id() -> str:
    idx = os.environ.get("MATRIX_INDEX", "local")
    return f"run_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def screenshot(page, path: Path, full_page: bool = True) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        page.screenshot(path=str(path), full_page=full_page, timeout=30_000)
        logger.info("screenshot saved: %s", path)
        return True
    except Exception as exc:
        logger.warning("full screenshot failed %s: %s: %s", path.name, type(exc).__name__, exc)
        try:
            page.screenshot(path=str(path), timeout=30_000)
            logger.info("viewport screenshot saved: %s", path)
            return True
        except Exception as exc2:
            logger.warning("viewport screenshot failed %s: %s: %s", path.name, type(exc2).__name__, exc2)
            return False


def screenshot_arkose_frame(page, path: Path) -> bool:
    """Best-effort: 截 visible Arkose iframe 或本地 harness container。"""
    selectors = [
        'iframe[src*="arkoselabs.com"]',
        'iframe[src*="funcaptcha.com"]',
        "#arkose-container",
        "#fc-iframe-wrap",
        "#root",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    for sel in selectors:
        try:
            loc = page.locator(sel)
            for i in range(min(loc.count(), 8)):
                item = loc.nth(i)
                try:
                    if item.is_visible(timeout=1500):
                        item.screenshot(path=str(path), timeout=30_000)
                        logger.info("frame/container screenshot saved via %s[%s]: %s", sel, i, path)
                        return True
                except Exception:
                    pass
        except Exception:
            pass
    logger.warning("no visible Arkose frame/container for %s", path)
    return False


def normalize_surl(value: Optional[str]) -> str:
    text = (value or "").strip()
    if not text:
        return DEFAULT_SURL
    if "://" not in text:
        text = "https://" + text
    parsed = urlsplit(text)
    return parsed.netloc or parsed.path.split("/", 1)[0] or DEFAULT_SURL


def origin_from_url(value: str) -> str:
    parsed = urlsplit(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid URL: {value!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def safe_json_for_script(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False).replace("</", "<\\/")


def build_solver_harness(public_key: str, blob: str, surl: str, language: str = "en-US") -> str:
    api_url = f"https://{normalize_surl(surl)}/v2/{public_key}/api.js"
    cfg = safe_json_for_script(
        {"publicKey": public_key, "blob": blob, "apiUrl": api_url, "language": language}
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Arkose Snapshot Worker</title>
  <style>
    html,body{{margin:0;min-height:100%;background:#0f172a;color:#e5e7eb;font-family:Arial,sans-serif}}
    header{{padding:12px 16px;background:#111827;border-bottom:1px solid #334155}}
    #status{{margin-top:6px;font-size:13px;color:#bae6fd}}
    #arkose-container{{width:100%;min-height:650px;display:flex;justify-content:center;align-items:flex-start;padding-top:18px;background:#f8fafc;box-sizing:border-box}}
    #events{{margin:0;padding:10px 16px;max-height:160px;overflow:auto;background:#020617;color:#cbd5e1;font-size:12px}}
  </style>
</head>
<body>
  <header>
    <strong>Independent Arkose snapshot worker</strong>
    <div id="status">loading Arkose Client API...</div>
  </header>
  <div id="arkose-container"></div>
  <pre id="events"></pre>
  <script>
  (() => {{
    const cfg = {cfg};
    const state = window.__ARKOSE_SNAPSHOT__ = {{
      apiUrl: cfg.apiUrl, apiReady: false, runCalled: false, token: null,
      status: 'boot', error: null, events: []
    }};
    let enforcement = null;
    function safe(v) {{ try {{ return JSON.parse(JSON.stringify(v)); }} catch (_) {{ return String(v); }} }}
    function emit(name, payload) {{
      const ev = {{name, at: Date.now(), payload: safe(payload)}};
      state.events.push(ev); state.status = name;
      const s = document.getElementById('status');
      if (s) s.textContent = name + (payload ? ': ' + JSON.stringify(safe(payload)).slice(0, 240) : '');
      const out = document.getElementById('events');
      if (out) out.textContent = state.events.slice(-16).map(e => new Date(e.at).toLocaleTimeString() + ' ' + e.name).join('\\n');
      console.log('[ARKOSE-SNAPSHOT]', name, payload || '');
    }}
    function runIt() {{
      if (!enforcement) {{ emit('run-before-ready', null); return; }}
      try {{ state.runCalled = true; enforcement.run(); emit('run', null); }}
      catch (e) {{ state.error = String(e && (e.stack || e.message) || e); emit('run-error', state.error); }}
    }}
    window.setupEnforcement = function(myEnforcement) {{
      enforcement = myEnforcement; state.apiReady = true;
      emit('api-ready', {{publicKey: cfg.publicKey, hasBlob: !!cfg.blob}});
      try {{
        myEnforcement.setConfig({{
          publicKey: cfg.publicKey,
          selector: '#arkose-container',
          mode: 'inline',
          language: cfg.language,
          data: cfg.blob ? {{blob: cfg.blob}} : {{}},
          onReady: r => {{ emit('onReady', r); if (!state.runCalled) setTimeout(runIt, 100); }},
          onShow: r => emit('onShow', r),
          onShown: r => emit('onShown', r),
          onHide: r => emit('onHide', r),
          onReset: r => {{ state.runCalled = false; emit('onReset', r); }},
          onWarning: r => emit('onWarning', r),
          onFailed: r => {{ state.error = safe(r); emit('onFailed', r); }},
          onError: r => {{ state.error = safe(r); emit('onError', r); }},
          onCompleted: r => {{ state.token = r && r.token ? String(r.token) : null; emit('onCompleted', {{tokenLength: state.token ? state.token.length : 0}}); }}
        }});
      }} catch (e) {{ state.error = String(e && (e.stack || e.message) || e); emit('setConfig-error', state.error); }}
    }};
    const script = document.createElement('script');
    script.id = 'arkose-client-api'; script.src = cfg.apiUrl; script.async = true; script.defer = true;
    script.setAttribute('data-callback', 'setupEnforcement');
    script.onload = () => emit('script-loaded', cfg.apiUrl);
    script.onerror = e => {{ state.error = 'api.js load failed'; emit('script-error', String(e)); }};
    document.head.appendChild(script); emit('script-added', cfg.apiUrl);
  }})();
  </script>
</body>
</html>"""


def replace_document_under_origin(page, website_url: str, html: str) -> Dict[str, Any]:
    expected_origin = origin_from_url(website_url)
    nav_error = ""
    try:
        page.goto(expected_origin + "/", wait_until="domcontentloaded", timeout=30_000)
    except Exception as exc:
        nav_error = f"{type(exc).__name__}: {exc}"
    try:
        page.evaluate("window.stop()")
    except Exception:
        pass
    actual = page.evaluate("() => ({url: location.href, origin: location.origin})")
    if actual.get("origin") != expected_origin:
        raise RuntimeError(f"origin mismatch: expected={expected_origin}, actual={actual}, nav_error={nav_error}")
    page.evaluate(
        """html => {
            window.stop();
            document.open();
            document.write(html);
            document.close();
        }""",
        html,
    )
    return {"expectedOrigin": expected_origin, "beforeReplace": actual, "navigationError": nav_error}


def challenge_dom_summary(page) -> Dict[str, Any]:
    js = r"""() => {
      const visible = el => !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
      const isArkoseDoc = /arkose|funcaptcha|client-api|blizzard-api/i.test(location.href)
        || !!document.querySelector('#arkose-container,#fc-iframe-wrap');
      const imgs = [...document.querySelectorAll('img')]
        .filter(img => visible(img) && (img.naturalWidth || 0) > 20 && (img.naturalHeight || 0) > 20)
        .map(img => ({src: String(img.currentSrc || img.src || '').slice(0, 180), w: img.naturalWidth, h: img.naturalHeight, aria: img.getAttribute('aria-label') || ''}));
      const canvases = [...document.querySelectorAll('canvas')]
        .filter(c => visible(c) && (c.width || 0) > 20 && (c.height || 0) > 20)
        .map(c => ({w: c.width, h: c.height}));
      const bg = [...document.querySelectorAll('*')].filter(el => {
        if (!visible(el)) return false;
        return /blob:|rtig|arkose|funcaptcha|image/i.test(getComputedStyle(el).backgroundImage || '');
      }).slice(0, 12).map(el => ({tag: el.tagName, cls: String(el.className || '').slice(0, 80), bg: String(getComputedStyle(el).backgroundImage || '').slice(0, 180)}));
      const buttons = [...document.querySelectorAll('button')].filter(visible).map(b => (b.innerText || b.getAttribute('aria-label') || '').trim()).filter(Boolean).slice(0, 12);
      const text = (document.body && document.body.innerText || '').replace(/\s+/g, ' ').trim().slice(0, 500);
      const arkoseImgs = imgs.filter(img => /blob:|rtig|arkose|funcaptcha|client-api|blizzard-api|image/i.test(img.src + ' ' + img.aria));
      const ok = (isArkoseDoc && (imgs.length > 0 || canvases.length > 0 || bg.length > 0)) || arkoseImgs.length > 0 || bg.length > 0;
      return {url: location.href, title: document.title, isArkoseDoc, images: imgs, arkoseImages: arkoseImgs, canvases, bg, buttons, text, ok};
    }"""
    frames = []
    ok = False
    try:
        frame_list = list(page.frames)
    except Exception:
        frame_list = []
    for idx, frame in enumerate(frame_list):
        try:
            info = frame.evaluate(js)
            info["frameIndex"] = idx
            frames.append(info)
            ok = ok or bool(info.get("ok"))
        except Exception as exc:
            frames.append({"frameIndex": idx, "error": f"{type(exc).__name__}: {exc}"})
    return {"ok": ok, "frames": frames}


def wait_challenge_image(page, catcher: Optional[CDPImageCatcher], timeout: float) -> Dict[str, Any]:
    deadline = time.time() + timeout
    last_dom: Dict[str, Any] = {}
    while time.time() < deadline:
        ready_count = 0
        if catcher is not None:
            try:
                ready_count = len([r for r in catcher.captured_images if r.get("body_bytes")])
                if ready_count >= 1:
                    time.sleep(2.0)  # 给 renderer 画图
                    return {"ok": True, "reason": "cdp-image-body", "readyImages": ready_count, "dom": challenge_dom_summary(page)}
            except Exception:
                pass
        try:
            last_dom = challenge_dom_summary(page)
            if last_dom.get("ok"):
                time.sleep(2.0)
                return {"ok": True, "reason": "dom-visible-image-or-canvas", "readyImages": ready_count, "dom": last_dom}
        except Exception as exc:
            last_dom = {"error": f"{type(exc).__name__}: {exc}"}
        time.sleep(0.75)
    return {"ok": False, "reason": "timeout", "dom": last_dom}


def click_verify_and_snapshot(page, catcher: Optional[CDPImageCatcher], out: Path, prefix: str, wait_timeout: float) -> Dict[str, Any]:
    write_json(out / f"{prefix}_before_verify_dom.json", challenge_dom_summary(page))
    clicked = False
    for attempt in range(1, 16):
        try:
            if click_verify_button(page):
                clicked = True
                logger.info("%s clicked Arkose Verify on attempt %s", prefix, attempt)
                break
        except Exception as exc:
            logger.debug("%s verify click failed: %s: %s", prefix, type(exc).__name__, exc)
        time.sleep(1.0)
    if not clicked:
        logger.warning("%s Verify button not clicked; still waiting for challenge image", prefix)
    wait_result = wait_challenge_image(page, catcher, wait_timeout)
    write_json(out / f"{prefix}_image_wait_result.json", wait_result)
    screenshot(page, out / f"{prefix}_after_verify_fullpage.png", full_page=True)
    screenshot_arkose_frame(page, out / f"{prefix}_arkose_frame.png")
    return {"clickedVerify": clicked, "imageWait": wait_result}


def detect_arkose_context(page, catcher: Optional[CDPBlobCatcher]) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "found": False,
        "siteKey": None,
        "surl": None,
        "websiteURL": page.url,
        "siteOrigin": origin_from_url(page.url),
        "userAgent": None,
        "candidateURLs": [],
        "dataArkoseSrc": None,
    }
    try:
        result["userAgent"] = page.evaluate("() => navigator.userAgent")
        dom = page.evaluate(
            """() => {
              const candidates = [];
              const capture = document.querySelector('#capture-arkose');
              if (capture) {
                const src = capture.getAttribute('data-arkose-src') || '';
                if (src) candidates.push(src);
              }
              document.querySelectorAll('script[src], iframe[src]').forEach(el => {
                const src = el.src || el.getAttribute('src') || '';
                if (/arkoselabs|funcaptcha/i.test(src)) candidates.push(src);
              });
              return {hasCaptureInput: !!capture, dataArkoseSrc: capture ? (capture.getAttribute('data-arkose-src') || '') : '', candidates: Array.from(new Set(candidates))};
            }"""
        )
        result["found"] = bool(dom.get("hasCaptureInput") or dom.get("candidates"))
        result["dataArkoseSrc"] = dom.get("dataArkoseSrc") or None
        result["candidateURLs"].extend(dom.get("candidates") or [])
    except Exception:
        pass
    try:
        for frame in page.frames:
            url = frame.url or ""
            if re.search(r"arkoselabs|funcaptcha", url, re.I):
                result["candidateURLs"].append(url)
                result["found"] = True
    except Exception:
        pass
    if catcher and catcher.captured_pk:
        result["siteKey"] = catcher.captured_pk
        result["found"] = True
    for candidate in result["candidateURLs"]:
        if not result["siteKey"]:
            m = re.search(r"(?:/v\d+/|[?&#]pk=|#)([0-9A-F]{8}-[0-9A-F-]{27,})", candidate, re.I)
            if m:
                result["siteKey"] = m.group(1)
        if not result["surl"]:
            try:
                host = urlsplit(candidate if "://" in candidate else "https:" + candidate).netloc
                if re.search(r"arkoselabs\.com$|funcaptcha\.com$", host, re.I):
                    result["surl"] = host
            except Exception:
                pass
    result["candidateURLs"] = list(dict.fromkeys(result["candidateURLs"]))
    result["surl"] = normalize_surl(result.get("surl"))
    return result


def create_solver_browser(user_agent: Optional[str], headless: bool):
    args = [
        f"--remote-debugging-port={SOLVER_PORT}",
        "--remote-allow-origins=*",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-quic",
    ]
    browser = launch(headless=headless, args=args, locale="en-US")
    ctx_args: Dict[str, Any] = {"viewport": {"width": 1280, "height": 900}, "bypass_csp": True}
    if user_agent:
        ctx_args["user_agent"] = user_agent
    context = browser.new_context(**ctx_args)
    page = context.new_page()
    page.set_default_timeout(20_000)
    return browser, context, page


def drive_original_to_battletag(page, acc: Dict[str, str], out: Path) -> None:
    logger.info("open: %s", REGISTER_URL)
    page.goto(REGISTER_URL, wait_until="domcontentloaded", timeout=60_000)
    time.sleep(2)
    close_cookie_banner(page)

    logger.info("step email: %s", acc["email"])
    page.locator("#accountName").fill(acc["email"])
    page.locator("#submit").click()
    time.sleep(2)

    wait_or_refresh(page, "#capture-country", "country selector")
    page.reload()
    time.sleep(3)
    wait_or_refresh(page, "#capture-country", "country selector after reload")
    page.select_option("#capture-country", COUNTRY)
    time.sleep(1.5)

    close_cookie_banner(page)
    logger.info("step birthday: %s-%s-%s", acc["birth_year"], acc["birth_month"], acc["birth_day"])
    fill_birthday(page, acc)
    time.sleep(0.5)
    page.locator("#flow-form-submit-btn").click()
    time.sleep(2)

    logger.info("step name: %s %s", acc["first_name"], acc["last_name"])
    try:
        page.wait_for_selector("#capture-first-name", timeout=8000)
    except Exception:
        page.reload()
        time.sleep(5)
        page.wait_for_selector("#capture-first-name", timeout=10000)
    page.locator("#capture-first-name").fill(acc["first_name"])
    page.locator("#capture-last-name").fill(acc["last_name"])
    time.sleep(0.5)
    page.locator("#flow-form-submit-btn").click()
    time.sleep(2)

    actual_email = page.locator("#capture-email").input_value()
    if actual_email != acc["email"]:
        screenshot(page, out / "error_email_mismatch.png")
        raise RuntimeError(f"email mismatch actual={actual_email} expected={acc['email']}")
    page.locator("#flow-form-submit-btn").click()
    time.sleep(2)

    logger.info("step legal checkboxes")
    page.evaluate(
        """() => {
            ['#capture-opt-in-blizzard-news-special-offers','#legal-checkboxes > label > input.step__checkbox'].forEach(function(sel){
                var el = document.querySelector(sel);
                if (el && !el.checked) {
                    var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'checked').set;
                    s.call(el, true);
                    el.dispatchEvent(new Event('change', {bubbles:true}));
                    el.dispatchEvent(new Event('input', {bubbles:true}));
                }
            });
        }"""
    )
    time.sleep(0.5)
    page.locator("#flow-form-submit-btn").click()
    time.sleep(2)

    logger.info("step password")
    wait_or_refresh(page, "#capture-password", "password input")
    page.locator("#capture-password").fill(acc["password"])
    time.sleep(0.5)
    page.locator("#flow-form-submit-btn").click()
    time.sleep(2)

    logger.info("step battletag: %s", acc["battle_tag"])
    wait_or_refresh(page, "#capture-battletag", "BattleTag input")
    page.evaluate(
        """
        ([elId, val]) => {
            var el = document.querySelector(elId);
            var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            s.call(el, val);
            el.dispatchEvent(new Event('input', {bubbles:true}));
            el.dispatchEvent(new Event('change', {bubbles:true}));
        }
        """,
        ["#capture-battletag", acc["battle_tag"]],
    )
    time.sleep(0.5)
    try:
        page.wait_for_function(
            '() => { const btn = document.querySelector("#flow-form-submit-btn"); return btn && !btn.disabled; }',
            timeout=5000,
        )
    except Exception:
        logger.warning("submit button enable wait timeout")
    screenshot(page, out / "original_before_battletag_submit.png")


def main() -> int:
    wait_timeout = float(os.environ.get("CAPTCHA_IMAGE_WAIT", "75"))
    blob_timeout = float(os.environ.get("BLOB_WAIT", "90"))
    headless = _truthy(os.environ.get("HEADLESS"), default=True)
    out = Path(os.environ.get("SNAPSHOT_OUTPUT_DIR", "funcaptcha_snapshot_debug")) / _run_id()
    out.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", out.resolve())

    acc = generate_identity()
    write_json(
        out / "account_generated.json",
        {k: acc[k] for k in ("email", "battle_tag", "first_name", "last_name", "birth_year", "birth_month", "birth_day")},
    )

    original_browser = solver_browser = None
    blob_catcher = original_img = solver_blob = solver_img = None
    try:
        # create_cloak_browser 与 register.py 一致，固定 original CDP 端口 9222/headless。
        original_browser, _, original_page = create_cloak_browser()
        drive_original_to_battletag(original_page, acc, out)

        blob_catcher = CDPBlobCatcher(debug_port=ORIGINAL_PORT, label="ORIGINAL-BLOB")
        blob_catcher.start()
        original_img = CDPImageCatcher(debug_port=ORIGINAL_PORT, label="ORIGINAL-IMAGE")
        original_img.start()

        logger.info("submit BattleTag to trigger FunCaptcha")
        original_page.locator("#flow-form-submit-btn").click()
        time.sleep(2)
        screenshot(original_page, out / "original_after_battletag_submit.png")

        blob_before_verify = blob_catcher.wait_for_blob(timeout=min(20.0, blob_timeout))
        original_result = click_verify_and_snapshot(original_page, original_img, out, "original", wait_timeout)
        blob = blob_catcher.captured_blob or blob_before_verify or blob_catcher.wait_for_blob(timeout=blob_timeout)
        if not blob:
            raise RuntimeError("no Arkose blob captured from original browser")
        ctx = detect_arkose_context(original_page, blob_catcher)
        if not ctx.get("siteKey"):
            raise RuntimeError("Arkose public key not detected")
        public_ctx = {**ctx, "blobLength": len(blob), "blobCapturedBeforeVerify": bool(blob_before_verify)}
        write_json(out / "original_arkose_context.json", public_ctx)
        save_cdp_images(original_img, str(out / "original_cdp_images"), prefix="original_captcha")

        logger.info("launch solver browser: port=%s headless=%s", SOLVER_PORT, headless)
        solver_browser, _, solver_page = create_solver_browser(ctx.get("userAgent"), headless=headless)
        solver_page.on("console", lambda msg: logger.info("solver-console[%s] %s", msg.type, msg.text))
        solver_blob = CDPBlobCatcher(debug_port=SOLVER_PORT, label="SOLVER-BLOB")
        solver_blob.start()
        solver_img = CDPImageCatcher(debug_port=SOLVER_PORT, label="SOLVER-IMAGE")
        solver_img.start()

        html = build_solver_harness(str(ctx["siteKey"]), blob, str(ctx.get("surl") or DEFAULT_SURL))
        (out / "solver_harness.html").write_text(html, encoding="utf-8")
        origin_info = replace_document_under_origin(solver_page, str(ctx["websiteURL"]), html)
        write_json(out / "solver_origin.json", origin_info)
        screenshot(solver_page, out / "solver_harness_loaded.png")

        solver_result = click_verify_and_snapshot(solver_page, solver_img, out, "solver", wait_timeout)
        save_cdp_images(solver_img, str(out / "solver_cdp_images"), prefix="solver_captcha")

        write_json(
            out / "summary.json",
            {
                "ok": True,
                "outputDir": str(out.resolve()),
                "siteKey": ctx.get("siteKey"),
                "surl": ctx.get("surl"),
                "blobLength": len(blob),
                "original": original_result,
                "solver": solver_result,
                "originalImageRecords": len(original_img.captured_images) if original_img else 0,
                "solverImageRecords": len(solver_img.captured_images) if solver_img else 0,
            },
        )
        logger.info("done: %s", out.resolve())
        return 0
    except Exception as exc:
        logger.error("snapshot failed: %s: %s", type(exc).__name__, exc, exc_info=True)
        write_json(out / "summary.json", {"ok": False, "error": f"{type(exc).__name__}: {exc}", "outputDir": str(out.resolve())})
        try:
            if "original_page" in locals():
                screenshot(original_page, out / "error_original_page.png")
        except Exception:
            pass
        try:
            if "solver_page" in locals():
                screenshot(solver_page, out / "error_solver_page.png")
        except Exception:
            pass
        return 1
    finally:
        for c in (blob_catcher, original_img, solver_blob, solver_img):
            try:
                if c:
                    c.stop()
            except Exception:
                pass
        keep = float(os.environ.get("SNAPSHOT_KEEP_BROWSER_SECONDS", "0") or "0")
        if keep > 0:
            time.sleep(keep)
        for b in (solver_browser, original_browser):
            try:
                if b:
                    b.close()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
