import string, time, random, logging, os, re, json, threading, subprocess, sys, hashlib
from urllib.parse import unquote
from pathlib import Path
from typing import Optional, Any, Dict
from dataclasses import dataclass
import requests as _req
import websocket
from cloakbrowser import launch

try:
    from captcha_image_collector import CDPImageCatcher, click_submit_button, _click_in_frames
except Exception:
    CDPImageCatcher = None
    click_submit_button = None
    _click_in_frames = None

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger('battle_net')

REGISTER_URL = 'https://account.battle.net/creation/flow/creation-full'
COUNTRY = 'GBR'
BATTLE_TAG_BASE = 'Amireux'
CDP_DEBUG_PORT = 9222
CAPMONSTER_API_KEY = os.environ.get('CAPMONSTER_API_KEY', '')
_CAPMONSTER_FIRST_DEFAULT = '1' if (os.environ.get('GITHUB_ACTIONS') and CAPMONSTER_API_KEY) else '0'
CAPMONSTER_FIRST = os.environ.get('CAPMONSTER_FIRST', _CAPMONSTER_FIRST_DEFAULT).lower() not in ('0', 'false', 'no')
LOCAL_DICE_ENABLED = os.environ.get('LOCAL_DICE_ENABLED', '1').lower() not in ('0', 'false', 'no')
LOCAL_DICE_MAX_WAVES = int(os.environ.get('LOCAL_DICE_MAX_WAVES', '8'))
LOCAL_DICE_CONF = float(os.environ.get('LOCAL_DICE_CONF', '0.25'))
LOCAL_DICE_DIR = Path(os.environ.get('LOCAL_DICE_DIR', str(Path(__file__).resolve().parent / 'yolo')))
LOCAL_DICE_SOLVER = Path(os.environ.get('LOCAL_DICE_SOLVER', str(LOCAL_DICE_DIR / 'solve.py')))
LOCAL_DICE_WEIGHTS = Path(os.environ.get('LOCAL_DICE_WEIGHTS', str(LOCAL_DICE_DIR / 'models' / 'best.onnx')))
CAPTCHA_SOLVE_DIR = Path(os.environ.get('CAPTCHA_SOLVE_DIR', 'captcha_solve_debug'))

FIRST_NAMES = ['Natha','Narin','Nan','Nahon','Nafeh','Naira','Nina','Myie','Myle','Minh','Musa','Mogan','Monia','Demris','Delnn','Deler','Deisi','Dera','Decon','Dayan','Aziah','Ayy','Avia','Anti','Akibi']
LAST_NAMES = ['MEEZ','BUS','VGHN','PKS','DASON','SANO','NORIS','LOVE','SEE','CURY','PWERS','SCTZ','BAKER','GUAN','PAGE','MUZ','BAL','BBS','TER','GSS','FTZGD','STES','DOYLE','SHERN','SAURS','WSE','CON','GIL','ALO','GRER','PALA','SON','WATS','NUNZ','BOOE','COEZ']
MONTHS = ['01','02','03','04','05','06','07','08','09','10','11','12']
LETTERS_A_M = 'abcdefghijklm'
LETTERS_N_Z = 'nopqrstuvwxyz'

def random_pick(chars, count=1):
    return ''.join(random.choice(chars) for _ in range(count))

def random_digits(count=1):
    return random_pick(string.digits, count)

def generate_identity():
    first = random.choice(FIRST_NAMES); last = random.choice(LAST_NAMES)
    email_local = (last.lower() + random_pick(LETTERS_N_Z) + random_digits(2)
                 + first.lower() + random_digits(1) + random_pick(LETTERS_A_M) + random_pick(LETTERS_N_Z))
    return {
        'first_name': first.lower(), 'last_name': last.lower(),
        'email': f'{email_local}@outlook.com', 'password': email_local,
        'birth_year': str(random.randint(1970, 2000)),
        'birth_month': random.choice(MONTHS),
        'birth_day': str(random.randint(10, 28)),
        'battle_tag': f'{BATTLE_TAG_BASE}{random_digits(2)}',
    }


# ═══════════════════════════════ Local Dice ONNX Solver ═══════════════════════════════
def _all_frames(page):
    """Return page + known child frames, compatible with Playwright Page/Frame."""
    out, seen, stack = [], set(), [page]
    while stack:
        p = stack.pop(0)
        if id(p) in seen:
            continue
        seen.add(id(p))
        out.append(p)
        try:
            children = list(getattr(p, 'frames', None) or getattr(p, 'child_frames', None) or [])
        except Exception:
            children = []
        for ch in children:
            if id(ch) not in seen:
                stack.append(ch)
    return out


def _eval_first(page, script: str, default=None):
    for frame in _all_frames(page):
        try:
            value = frame.evaluate(script)
            if value is not None:
                return value
        except Exception:
            pass
    return default


def _captcha_state(page) -> str:
    states = []
    script = r'''() => {
        const text = (document.body && document.body.innerText) || '';
        if (document.querySelector('#success-icon > svg > path')) return 'success';

        // Arkose wrong-answer copy may not have error/alert class; scan full text.
        if (
            /that\s+was\s+not\s+quite\s+right/i.test(text) ||
            /make\s+sure\s+that\s+the\s+dice\s+add\s+up/i.test(text) ||
            /try\s+again/i.test(text) ||
            /not\s+quite\s+right/i.test(text) ||
            /incorrect|invalid|\u5931\u8d25|\u65e0\u6548|\u4e0d\u6b63\u786e|\u91cd\u8bd5/i.test(text)
        ) return 'rejected';

        const err = document.querySelector('[class*="error"], [role="alert"]');
        if (err && /\u65e0\u6548|invalid|\u5931\u8d25|incorrect|try again|not quite right/i.test(err.textContent || '')) return 'rejected';
        const visible = el => !!el && !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
        const submit = [...document.querySelectorAll('button')].some(b => /\u63d0\u4ea4|submit/i.test(b.textContent || '') && visible(b));
        const challengeImg = [...document.querySelectorAll('img[aria-label*="\u56fe\u50cf"], img[aria-label*="Image"]')].some(visible);
        const verify = [...document.querySelectorAll('button')].some(b => /\u9a8c\u8bc1|verify/i.test(b.textContent || '') && visible(b));
        if (submit || challengeImg || verify) return 'active';
        return 'gone';
    }'''
    for frame in _all_frames(page):
        try:
            state = frame.evaluate(script)
            if state in ('success', 'rejected'):
                return state
            if state:
                states.append(state)
        except Exception:
            pass
    if 'active' in states:
        return 'active'
    if 'gone' in states:
        return 'gone'
    return 'active'


def _captcha_text(page) -> str:
    parts = []
    for frame in _all_frames(page):
        try:
            txt = frame.locator('body').inner_text(timeout=800)
            if txt:
                parts.append(txt)
        except Exception:
            pass
    return '\n'.join(parts)


def _is_registration_success(page, expected_email: Optional[str] = None) -> bool:
    """Detect Battle.net final success page.

    The URL can remain /creation/flow/creation-full after success. Scan page
    and child frames; use success copy plus the expected email when available.
    """
    expected_email = (expected_email or '').strip().lower()
    js = r'''([expectedEmail]) => {
        const text = document.body ? (document.body.innerText || '') : '';
        const lower = text.toLowerCase();
        const hasIcon = !!document.querySelector(
            '#success-icon > svg > path, #success-icon, [data-testid*="success"], [class*="success"] svg'
        );
        const hasAllSet = /you['’]?re\s+all\s+set|all\s+set/i.test(text);
        const hasCreated = /account\s+has\s+been\s+created|has\s+been\s+created/i.test(text);
        const hasDownloadApp = /download\s+battle\.net\s+app/i.test(text);
        const hasEmail = expectedEmail ? lower.includes(String(expectedEmail).toLowerCase()) : true;
        const strongSuccess = hasAllSet && (hasCreated || hasDownloadApp);
        const success = !!(hasIcon || (strongSuccess && hasEmail) || (strongSuccess && !expectedEmail));
        return {
            success,
            hasIcon,
            hasAllSet,
            hasCreated,
            hasDownloadApp,
            hasEmail,
            sample: text.slice(0, 220)
        };
    }'''
    for frame in _all_frames(page):
        try:
            result = frame.evaluate(js, [expected_email])
            if isinstance(result, dict) and result.get('success'):
                logger.info(f'✅ 检测到注册成功页: {result}')
                return True
        except Exception:
            pass
        try:
            text = frame.locator('body').inner_text(timeout=1000)
            low = text.lower()
            has_email = expected_email in low if expected_email else True
            strong_success = (
                ("you're all set" in low or "all set" in low)
                and ("account has been created" in low or "download battle.net app" in low)
            )
            if strong_success and has_email:
                logger.info('✅ 检测到注册成功页: text fallback')
                return True
        except Exception:
            pass
    return False

def _wait_registration_success(page, expected_email: Optional[str] = None, timeout: float = 45.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if _is_registration_success(page, expected_email):
            return True
        try:
            state = _captcha_state(page)
            if state == 'rejected':
                logger.warning('Arkose value rejected while waiting registration success; stop waiting early')
                return False
        except Exception:
            pass
        time.sleep(0.5)
    return False


def _wait_post_local_dice_result(page, timeout: float = 10.0) -> Optional[bool]:
    """After browser-side dice clicks, wait for the parent page verdict.

    Local dice clicks can close the Arkose iframe while the parent form still
    rejects the arkose value (for example: "Value is invalid").  Do not treat
    iframe disappearance alone as final success; detect invalid/active gate fast
    so the caller can fallback instead of wasting 45s on success-page wait.
    """
    deadline = time.time() + timeout
    last_state = None
    while time.time() < deadline:
        if _is_registration_success(page):
            return True
        try:
            state = _captcha_state(page)
            last_state = state
            if state == 'rejected':
                sample = _captcha_text(page).replace('\n', ' ')[:220]
                logger.warning(f'Local dice ended but parent page rejected/invalid: {sample}')
                return False
            if state == 'active':
                text = _captcha_text(page).lower()
                if 'i am a human' in text or 'value is invalid' in text or 'detecting humanity' in text:
                    logger.warning('Local dice ended but page is still on humanity gate; fallback needed')
                    return False
        except Exception:
            pass
        time.sleep(0.4)
    logger.info(f'No success/reject verdict {timeout:.0f}s after local dice, last_state={last_state}')
    return None


def _get_current_candidate_index(page) -> int:
    """Return current Arkose carousel candidate index (0-11), or -1 if unknown."""
    idx = _eval_first(page, r'''() => {
        const imgs = [...document.querySelectorAll('img[aria-label]')];
        for (const img of imgs) {
            const s = img.getAttribute('aria-label') || '';
            const cls = img.getAttribute('class') || '';
            const style = img.getAttribute('style') || '';
            const nums = s.match(/\d+/g);
            const looksLikeCarousel = /image|图像|圖像|共|of|项|項/i.test(s)
                || cls.includes('sc-7csxyx')
                || style.includes('blob:');
            if (!looksLikeCarousel || !nums || nums.length < 1) continue;
            const first = parseInt(nums[0], 10);
            const total = nums.length >= 2 ? parseInt(nums[1], 10) : 12;
            if (first >= 1 && first <= 12 && (total === 12 || nums.length === 1)) {
                return first - 1;
            }
        }
        return null;
    }''', default=None)
    try:
        if idx is None:
            return -1
        return max(0, min(11, int(idx)))
    except Exception:
        return -1


def _click_arrow(page, direction: str) -> bool:
    if _click_in_frames is None:
        return False
    if direction == 'right':
        selectors = [
            'a.right-arrow',
            '[class*="right-arrow"]',
            'a[aria-label*="下一"]',
            'a[aria-label*="Next"]',
            'a[aria-label*="next"]',
        ]
    else:
        selectors = [
            'a.left-arrow',
            '[class*="left-arrow"]',
            'a[aria-label*="上一"]',
            'a[aria-label*="Previous"]',
            'a[aria-label*="previous"]',
        ]
    return _click_in_frames(page, selectors)


def _wait_candidate_change(page, previous: int, timeout: float = 1.2) -> int:
    deadline = time.time() + timeout
    latest = previous
    while time.time() < deadline:
        now = _get_current_candidate_index(page)
        if now >= 0:
            latest = now
            if previous < 0 or now != previous:
                return now
        time.sleep(0.06)
    return latest


def _select_candidate(page, answer_index: int) -> bool:
    """Move Arkose carousel to answer_index and verify the final active index.

    Older code only clicked arrows N times. On Arkose the carousel animation can
    occasionally ignore a too-fast click, so we now read the active image aria-label
    after each click and repair if needed.
    """
    answer_index = max(0, min(11, int(answer_index)))
    current = _get_current_candidate_index(page)
    if current == answer_index:
        logger.info(f'🎯 当前已在候选图 index={answer_index}')
        return True
    if current < 0:
        logger.warning('⚠️ 无法读取当前候选 index，按 index=0 估算点击步数')
        current = 0

    for attempt in range(3):
        if current == answer_index:
            logger.info(f'🎯 候选图已切到 index={answer_index}')
            return True
        forward = (answer_index - current) % 12
        backward = (current - answer_index) % 12
        direction = 'right' if forward <= backward else 'left'
        steps = forward if direction == 'right' else backward
        logger.info(f'➡️ 切换候选图: current={current}, answer={answer_index}, {direction} x {steps}, attempt={attempt + 1}')
        for _ in range(steps):
            before = _get_current_candidate_index(page)
            if not _click_arrow(page, direction):
                logger.warning(f'⚠️ 点击 {direction} arrow 失败')
                return False
            after = _wait_candidate_change(page, before, timeout=1.3)
            if after >= 0:
                current = after
            else:
                time.sleep(0.25)
        final = _get_current_candidate_index(page)
        if final >= 0:
            current = final
            logger.info(f'🔎 候选图切换校验: current={current}, target={answer_index}')
        else:
            logger.warning('⚠️ 候选图切换后仍无法读取 index，按点击结果继续')
            return True

    ok = current == answer_index
    if not ok:
        logger.warning(f'⚠️ 候选图最终 index 不一致: current={current}, target={answer_index}')
    return ok


def _record_sha(rec: Dict[str, Any]) -> Optional[str]:
    if rec.get('sha256'):
        return rec.get('sha256')
    data = rec.get('body_bytes')
    return hashlib.sha256(data).hexdigest() if data else None


def _wait_new_captcha_image(image_catcher, seen_shas, timeout: float = 30.0, page=None, stop_on_gone: bool = True):
    """Return the next Arkose challenge strip, not a completion/auxiliary asset.

    The collector also sees Arkose-owned blob: images.  After the final answer,
    these may be 200x200 PNG completion resources rather than another challenge.
    Only 2000x400 and 2400x400 strips represent a selectable challenge here.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if page is not None:
            state = _captcha_state(page)
            if state in ('success', 'rejected') or (stop_on_gone and state == 'gone'):
                return None
        try:
            image_catcher.wait_for_image(min_count=1, timeout=1.0)
            ready = []
            for rec in image_catcher.captured_images:
                if not rec.get('body_bytes'):
                    continue
                sha = _record_sha(rec)
                if not sha or sha in seen_shas:
                    continue
                url = (rec.get('url') or '').lower()
                # Prefer the original Arkose image response. The browser often
                # converts it into a blob: URL afterwards; blob is only fallback.
                priority = 0 if '/rtig/image' in url else 1 if url.startswith('blob:') else 2
                ready.append((priority, rec.get('timestamp') or 0, rec))
            if ready:
                ready.sort(key=lambda x: (x[0], x[1]))
                for _, _, rec in ready:
                    size = _image_record_size(rec)
                    if size is not None:
                        w, h = size
                        is_challenge_strip = w in (2000, 2400) and 350 <= h <= 450
                        if not is_challenge_strip:
                            sha = _record_sha(rec)
                            if sha:
                                seen_shas.add(sha)
                            logger.info(
                                '⏭️ 忽略非题目 Arkose 图片: size=%sx%s, url=%s',
                                w, h, (rec.get('url') or '')[:120]
                            )
                            continue
                    return rec
        except Exception:
            pass
        time.sleep(0.2)
    return None


def _image_record_size(rec: Dict[str, Any]):
    data = rec.get('body_bytes')
    if not data:
        return None
    try:
        from PIL import Image
        import io
        with Image.open(io.BytesIO(data)) as im:
            return tuple(im.size)
    except Exception:
        return None


def _classify_captcha_image(path: Path) -> Dict[str, Any]:
    info = {'kind': 'unknown', 'width': None, 'height': None, 'reason': ''}
    try:
        from PIL import Image
        with Image.open(path) as im:
            w, h = im.size
        info.update({'width': w, 'height': h})
        if w == 2400 and 350 <= h <= 450:
            info.update({'kind': 'dice', 'reason': 'size=2400x400'})
        elif w == 2000 and 350 <= h <= 450:
            info.update({'kind': 'other', 'reason': 'size=2000x400'})
        else:
            info.update({'kind': 'unknown', 'reason': f'unexpected_size={w}x{h}'})
    except Exception as e:
        info.update({'kind': 'unknown', 'reason': f'image_open_error={type(e).__name__}: {e}'})
    return info


def _save_captcha_image_record(rec: Dict[str, Any], wave: int) -> Path:
    CAPTCHA_SOLVE_DIR.mkdir(parents=True, exist_ok=True)
    data = rec.get('body_bytes') or b''
    sha = _record_sha(rec) or hashlib.sha256(data).hexdigest()
    mime = (rec.get('mime') or '').lower()
    ext = '.jpg' if 'jpeg' in mime or 'jpg' in mime else '.png' if 'png' in mime else '.webp' if 'webp' in mime else '.bin'
    path = CAPTCHA_SOLVE_DIR / f'dice_wave_{wave:02d}_{sha[:12]}{ext}'
    path.write_bytes(data)
    return path


def _capture_debug_screenshot(page, filename: str) -> Optional[Path]:
    """Best-effort screenshot for GitHub artifacts; never break registration flow."""
    try:
        CAPTCHA_SOLVE_DIR.mkdir(parents=True, exist_ok=True)
        path = CAPTCHA_SOLVE_DIR / filename
        try:
            page.screenshot(path=str(path), full_page=True)
        except TypeError:
            page.screenshot(path=str(path))
        logger.info(f'📸 已保存调试截图: {path}')
        return path
    except Exception as e:
        logger.debug(f'Debug screenshot failed ({filename}): {type(e).__name__}: {e}')
        return None


def _is_dice_image(path: Path, page=None) -> bool:
    return _classify_captcha_image(path).get('kind') == 'dice'


def _extract_json_from_text(text: str):
    text = text or ''
    start = text.find('{')
    end = text.rfind('}')
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except Exception:
            return None
    return None


_LOCAL_DICE_SOLVE_IMAGE = None
_LOCAL_DICE_IMPORT_ERROR = None


def _load_local_dice_solve_image():
    """Load yolo/solve.py once so each wave does not spawn/re-import a process."""
    global _LOCAL_DICE_SOLVE_IMAGE, _LOCAL_DICE_IMPORT_ERROR
    if _LOCAL_DICE_SOLVE_IMAGE is not None:
        return _LOCAL_DICE_SOLVE_IMAGE
    if _LOCAL_DICE_IMPORT_ERROR is not None:
        raise _LOCAL_DICE_IMPORT_ERROR
    try:
        import importlib.util
        if str(LOCAL_DICE_DIR) not in sys.path:
            sys.path.insert(0, str(LOCAL_DICE_DIR))
        spec = importlib.util.spec_from_file_location('local_dice_solve_module', str(LOCAL_DICE_SOLVER))
        if spec is None or spec.loader is None:
            raise ImportError(f'cannot load solver module: {LOCAL_DICE_SOLVER}')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _LOCAL_DICE_SOLVE_IMAGE = module.solve_image
        return _LOCAL_DICE_SOLVE_IMAGE
    except Exception as e:
        _LOCAL_DICE_IMPORT_ERROR = e
        raise


def _run_local_dice_solver(image_path: Path) -> Dict[str, Any]:
    image_path = Path(image_path).resolve()
    if not LOCAL_DICE_SOLVER.exists():
        return {'status': 'error', 'error': f'missing solver: {LOCAL_DICE_SOLVER}'}
    if not LOCAL_DICE_WEIGHTS.exists():
        return {'status': 'error', 'error': f'missing weights: {LOCAL_DICE_WEIGHTS}'}
    if not image_path.exists():
        return {'status': 'error', 'error': f'image not found before inference: {image_path}'}

    # Fast path: in-process solve with cached ONNX/YOLO sessions. Set
    # LOCAL_DICE_SUBPROCESS=1 to force the old subprocess mode for debugging.
    if os.environ.get('LOCAL_DICE_SUBPROCESS', '0').lower() not in ('1', 'true', 'yes'):
        try:
            solve_image = _load_local_dice_solve_image()
            result = solve_image(
                image_path,
                weights=LOCAL_DICE_WEIGHTS,
                conf=LOCAL_DICE_CONF,
                device='cpu',
                verbose=False,
            )
            result['returncode'] = 0 if result.get('status') == 'unique_match' and result.get('answer_index') is not None else 2
            result['runner'] = 'inprocess'
            return result
        except Exception as e:
            logger.warning(f'Local dice in-process solver failed, fallback to subprocess: {type(e).__name__}: {e}')

    cmd = [
        sys.executable,
        str(LOCAL_DICE_SOLVER),
        str(image_path),
        '--weights', str(LOCAL_DICE_WEIGHTS),
        '--json',
        '--device', 'cpu',
        '--conf', str(LOCAL_DICE_CONF),
    ]
    try:
        p = subprocess.run(cmd, cwd=str(LOCAL_DICE_DIR), capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=80)
    except Exception as e:
        return {'status': 'error', 'error': f'{type(e).__name__}: {e}'}
    data = _extract_json_from_text(p.stdout) or _extract_json_from_text(p.stderr)
    if not data:
        data = {'status': 'error', 'error': (p.stderr or p.stdout or '').strip()[:500], 'returncode': p.returncode}
    data['returncode'] = p.returncode
    data['runner'] = 'subprocess'
    return data


def try_solve_dice_challenge(page, image_catcher) -> Optional[bool]:
    """Solve current Arkose dice challenge in-browser.

    Returns:
        True  - local dice solver completed/submitted challenge.
        False - local dice solver detected dice but failed after interacting.
        None  - not a dice challenge or local solver unavailable; caller may fallback.
    """
    if not LOCAL_DICE_ENABLED:
        return None
    if image_catcher is None:
        logger.info('ℹ️ 未启用 CDP image catcher，跳过本地骰子求解')
        return None
    if click_submit_button is None or _click_in_frames is None:
        logger.warning('⚠️ captcha_image_collector helpers 不可用，跳过本地骰子求解')
        return None
    if not LOCAL_DICE_SOLVER.exists() or not LOCAL_DICE_WEIGHTS.exists():
        logger.warning(f'⚠️ 本地骰子模型不存在: solver={LOCAL_DICE_SOLVER}, weights={LOCAL_DICE_WEIGHTS}')
        return None

    seen_shas = set()
    interacted = False
    for wave in range(LOCAL_DICE_MAX_WAVES):
        rec = _wait_new_captcha_image(
            image_catcher,
            seen_shas,
            timeout=35.0 if wave == 0 else 18.0,
            page=page,
            # wave=0 can briefly report gone before the first image arrives; do not fallback too early.
            stop_on_gone=interacted,
        )
        if not rec:
            state = _captcha_state(page)
            logger.info(f'ℹ️ 未等到新验证码图片 wave={wave}, state={state}')
            if interacted and state in ('success', 'gone'):
                return True
            return None if not interacted else False
        sha = _record_sha(rec)
        if sha:
            seen_shas.add(sha)
        img_path = _save_captcha_image_record(rec, wave)
        rec_size = _image_record_size(rec)
        url = rec.get('url') or ''
        url_kind = 'rtig' if '/rtig/image' in url.lower() else 'blob' if url.lower().startswith('blob:') else 'other'
        logger.info(
            f'🖼️ 捕获验证码图片 wave={wave}: {img_path} '
            f'url_kind={url_kind}, size={rec_size}, mime={rec.get("mime")}, url={url[:160]}'
        )

        image_info = _classify_captcha_image(img_path)
        logger.info(
            f'🔎 验证码图片分类 wave={wave}: kind={image_info.get("kind")}, '
            f'size={image_info.get("width")}x{image_info.get("height")}, '
            f'reason={image_info.get("reason")}'
        )
        if image_info.get('kind') != 'dice':
            logger.info('ℹ️ 当前图片不是 2400x400 骰子长图，不调用 ONNX，交给 CapMonster')
            return None if not interacted else False

        result = _run_local_dice_solver(img_path)
        status = result.get('status')
        answer = result.get('answer_index')
        target = result.get('target_number')
        error = result.get('error')
        logger.info(
            f'🎲 ONNX 求解 wave={wave}: status={status}, target={target}, '
            f'answer={answer}, error={error}, runner={result.get("runner")}, '
            f'sums={result.get("candidate_sums")}'
        )

        if status != 'unique_match' or answer is None:
            detail_path = CAPTCHA_SOLVE_DIR / f'dice_wave_{wave:02d}_failed.json'
            detail_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
            logger.warning(f'⚠️ 本地骰子未唯一命中，交给 CapMonster: {detail_path}')
            return None if not interacted else False

        if not _select_candidate(page, int(answer)):
            return False
        time.sleep(0.25 + random.random() * 0.2)
        if not click_submit_button(page):
            state = _captcha_state(page)
            logger.warning(f'⚠️ 未找到验证码提交按钮, state={state}')
            return state in ('success', 'gone')
        interacted = True
        logger.info(f'✅ 已提交骰子答案 index={answer}')
        time.sleep(1.0 + random.random() * 0.5)

        state = _captcha_state(page)
        if state == 'success':
            logger.info('✅ 本地骰子验证通过')
            return True
        if state == 'rejected':
            logger.warning('⚠️ 本地骰子答案被拒')
            return False

    logger.warning(f'⚠️ 本地骰子达到最大轮数 {LOCAL_DICE_MAX_WAVES}')
    return False


# ═══════════════════════════════ CDPBlobCatcher ═══════════════════════════════
class CDPBlobCatcher:
    def __init__(self, debug_port=9222, ws_url=None, label=''):
        self.debug_port = debug_port
        self._ws_url_override = ws_url
        self._label = label
        self.ws = None; self.msg_id = 0
        self.captured_blob = None; self.captured_pk = None
        self.fc_requests = []
        self._lock = threading.Lock()
        self._running = False; self._thread = None
        self._blob_event = threading.Event()
        self._traffic_bytes = 0
        self._requests_log = {}

    def _send(self, method, params=None, session_id=None):
        with self._lock:
            self.msg_id += 1; mid = self.msg_id
            payload = {'id': mid, 'method': method, 'params': params or {}}
            if session_id: payload['sessionId'] = session_id
            try: self.ws.send(json.dumps(payload))
            except Exception: pass
            return mid

    def _get_browser_ws_url(self):
        if self._ws_url_override: return self._ws_url_override
        return _req.get(f'http://localhost:{self.debug_port}/json/version', timeout=5).json()['webSocketDebuggerUrl']

    def start(self):
        ws_url = self._get_browser_ws_url()
        pfx = f'[{self._label}] ' if self._label else ''
        try:
            vi = _req.get(f'http://localhost:{self.debug_port}/json/version', timeout=5).json()
            logger.info(f'{pfx}🌐 Browser: {vi.get("Browser", "?")}')
        except Exception: pass
        try: self.ws = websocket.create_connection(ws_url, max_size=None, suppress_origin=True)
        except TypeError: self.ws = websocket.create_connection(ws_url, max_size=None, origin='chrome://devtools')
        self._send('Network.enable', {'maxPostDataSize': 131072})
        self._send('Target.setAutoAttach', {'autoAttach': True, 'waitForDebuggerOnStart': True, 'flatten': True})
        self._running = True; self._traffic_bytes = 0
        self._thread = threading.Thread(target=self._loop, daemon=True); self._thread.start()
        logger.info(f'{pfx}🔌 CDP blob 抓取器已启动')

    def _loop(self):
        while self._running:
            try:
                raw = self.ws.recv()
                if not raw: continue
                msg = json.loads(raw)
                method = msg.get('method'); params = msg.get('params', {}); sid = msg.get('sessionId')
                if method == 'Network.requestWillBeSent':
                    req = params.get('request', {}); rid = params.get('requestId'); url = req.get('url', '')
                    if rid: self._requests_log[rid] = {'url': url, 'type': params.get('type', 'Other'), 'method': req.get('method', 'GET'), 'size': 0, 'status': 0}
                    if '/fc/gt2/' in url: self._handle_fc(url, req.get('postData', ''))
                    continue
                if method == 'Network.responseReceived':
                    rid = params.get('requestId'); resp = params.get('response', {})
                    if rid and rid in self._requests_log: self._requests_log[rid]['status'] = resp.get('status', 0)
                    continue
                if method == 'Network.loadingFinished':
                    rid = params.get('requestId'); size = params.get('encodedDataLength', 0)
                    if rid and rid in self._requests_log:
                        self._requests_log[rid]['size'] = size
                        url = self._requests_log[rid].get('url', '')
                        if '127.0.0.1' not in url and 'localhost' not in url: self._traffic_bytes += size
                    continue
                if method == 'Target.attachedToTarget':
                    ns = params.get('sessionId'); waiting = params.get('waitingForDebugger', False)
                    self._send('Target.setAutoAttach', {'autoAttach': True, 'waitForDebuggerOnStart': True, 'flatten': True}, session_id=ns)
                    self._send('Fetch.enable', {'patterns': [{'urlPattern': '*/fc/gt2/*', 'requestStage': 'Request'}]}, session_id=ns)
                    self._send('Network.enable', {'maxPostDataSize': 131072}, session_id=ns)
                    if waiting: self._send('Runtime.runIfWaitingForDebugger', {}, session_id=ns)
                    continue
                if method == 'Fetch.requestPaused':
                    req = params.get('request', {}); url = req.get('url', ''); rid = params.get('requestId')
                    if '/fc/gt2/' in url:
                        body = req.get('postData', '')
                        if not body and req.get('hasPostData'):
                            mid = self._send('Fetch.getRequestPostData', {'requestId': rid}, session_id=sid)
                            body = self._wait_result(mid)
                        self._handle_fc(url, body)
                    self._send('Fetch.continueRequest', {'requestId': rid}, session_id=sid)
                    continue
            except Exception:
                if self._running: time.sleep(0.05)

    def _handle_fc(self, url, body):
        if url not in self.fc_requests: self.fc_requests.append(url)
        m = re.search(r'/fc/gt2/public_key/([0-9A-F-]+)', url, re.I)
        if m: self.captured_pk = m.group(1)
        if body:
            bm = re.search(r'data\[blob\]=([^&]+)', body) or re.search(r'(?:^|&)bda=([^&]+)', body)
            if bm:
                new_blob = unquote(bm.group(1))
                if new_blob != self.captured_blob:
                    self.captured_blob = new_blob
                    logger.info(f'📦 CDP 抓到 blob! 长度 {len(self.captured_blob)}, pk={self.captured_pk}')
                self._blob_event.set()

    def _wait_result(self, target_id, timeout=2.0):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                raw = self.ws.recv(); msg = json.loads(raw)
                if msg.get('id') == target_id: return msg.get('result', {}).get('postData', '')
            except Exception: break
        return ''

    def reset_blob(self):
        self.captured_blob = None; self._blob_event.clear()

    def wait_for_blob(self, timeout=30.0):
        if self._blob_event.wait(timeout): return self.captured_blob
        return None

    def stop(self):
        self._running = False
        try:
            if self.ws: self.ws.close()
        except Exception: pass


# ═══════════════════════════════ CapMonster Solver ═══════════════════════════════
CAPMONSTER_CREATE_TASK = 'https://api.capmonster.cloud/createTask'
CAPMONSTER_GET_RESULT = 'https://api.capmonster.cloud/getTaskResult'

@dataclass
class CapMonsterSolverConfig:
    api_key: str
    poll_interval: float = 2.5
    max_wait: float = 120.0
    user_agent: Optional[str] = None

class CapMonsterFunCaptchaSolver:
    def __init__(self, config: CapMonsterSolverConfig):
        self.config = config
        self._session = _req.Session()
        self._session.headers.update({'Content-Type': 'application/json'})

    def detect(self, page):
        result = {'found': False, 'siteKey': None, 'surl': None, 'callback': None, 'fcTokenField': None}

        try:
            verify_btns = page.locator('//div[@id="root"]//button').all()
            for btn in verify_btns:
                try:
                    txt = (btn.text_content() or '').strip()
                    if '验证' in txt or 'Verify' in txt.lower():
                        result['found'] = True
                        break
                except Exception:
                    pass
        except Exception:
            pass

        caps = page.locator('#capture-arkose').all()
        if caps:
            result['found'] = True
            result['fcTokenField'] = '#capture-arkose'
            dsrc = caps[0].get_attribute('data-arkose-src') or ''
            m = re.search(r'/v\d+/([0-9A-F-]+)/api\.js', dsrc, re.I)
            if m: result['siteKey'] = m.group(1)
            sm = re.search(r'//([a-zA-Z0-9_.-]*arkoselabs\.com)', dsrc, re.I)
            if sm: result['surl'] = sm.group(1)

        if not result['found']:
            try:
                iframes = page.locator('iframe[src*=".arkoselabs.com"], iframe[src*="funcaptcha.com"]').all()
                if iframes:
                    result['found'] = True
                    src = iframes[0].get_attribute('src') or ''
                    if not result['siteKey']:
                        m = re.search(r'#([0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12})', src, re.I)
                        if m: result['siteKey'] = m.group(1)
                    if not result['surl']:
                        sm = re.search(r'//([a-zA-Z0-9_.-]*arkoselabs\.com)', src, re.I)
                        if sm: result['surl'] = sm.group(1)
            except Exception:
                pass

        if not result['found'] or not result['siteKey']:
            try:
                gcf = page.locator('iframe#game-core-frame, iframe[title="视觉挑战"]').all()
                if gcf:
                    result['found'] = True
                    src = gcf[0].get_attribute('src') or ''
                    if not result['siteKey']:
                        m = re.search(r'[?&]pk=([0-9A-F-]+)', src, re.I)
                        if m: result['siteKey'] = m.group(1)
                    if not result['surl']:
                        sm = re.search(r'//([a-zA-Z0-9_.-]*arkoselabs\.com)', src, re.I)
                        if sm: result['surl'] = sm.group(1)
            except Exception:
                pass

        return result

    def _create_task(self, website_url, website_public_key, data_blob=None, surl=None, user_agent=None):
        task = {'type': 'FunCaptchaTask', 'websiteURL': website_url, 'websitePublicKey': website_public_key}
        if data_blob: task['data'] = json.dumps({'blob': data_blob})
        if surl and surl != 'client-api.arkoselabs.com': task['funcaptchaApiJSSubdomain'] = surl
        final_ua = user_agent or self.config.user_agent or 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36'
        task['userAgent'] = final_ua
        try:
            r = self._session.post(CAPMONSTER_CREATE_TASK, json={'clientKey': self.config.api_key, 'task': task}, timeout=15)
            data = r.json()
            task_id = data.get('taskId')
            if not task_id: logger.error(f'createTask 失败: {str(data)[:200]}'); return None
            logger.info(f'📡 CapMonster taskId={task_id}')
            return task_id
        except Exception as e:
            logger.error(f'createTask 异常: {e}')
            return None

    def _poll_task(self, task_id):
        deadline = time.time() + self.config.max_wait
        pc = 0
        while time.time() < deadline:
            pc += 1
            try:
                r = self._session.post(CAPMONSTER_GET_RESULT, json={'clientKey': self.config.api_key, 'taskId': task_id}, timeout=10)
                data = r.json()
            except Exception as e:
                logger.warning(f'轮询异常: {e}')
                time.sleep(self.config.poll_interval)
                continue
            status = data.get('status')
            logger.info(f'🔄 轮询 #{pc}: status={status}')
            if status == 'ready':
                token = data.get('solution', {}).get('token')
                if token: logger.info(f'✅ 求解成功 (token 长度 {len(token)})'); return token
                return None
            eid = data.get('errorId')
            if (eid not in (None, 0)) or status in ('error', 'failed'):
                logger.error(f'❌ 求解失败: {str(data)[:200]}')
                return None
            time.sleep(self.config.poll_interval)
        logger.error(f'⏱️ 轮询超时 ({pc} 次)')
        return None

    def solve(self, page, blob=None):
        info = self.detect(page)
        if not info.get('found'): logger.warning('未检测到 FunCaptcha'); return None
        site_key = info.get('siteKey')
        if not site_key: logger.error('未找到 siteKey'); return None
        website_url = page.url
        user_agent = self.config.user_agent
        if not user_agent:
            try: user_agent = page.evaluate('() => navigator.userAgent')
            except Exception: pass
        logger.info(f'📋 siteKey={site_key}, surl={info.get("surl")}, blob={"有("+str(len(blob))+")" if blob else "无"}')
        task_id = self._create_task(website_url=website_url, website_public_key=site_key, data_blob=blob, surl=info.get('surl'), user_agent=user_agent)
        if not task_id: return None
        return self._poll_task(task_id)

    def _click_arkose_verify_button(self, page, max_depth=3):
        def try_click_in_frame(p):
            for sel in ['button[data-theme="home.verifyButton"]', 'button[aria-label="验证"]', 'button[aria-label="Verify"]']:
                try:
                    btns = p.locator(sel).all()
                    for btn in btns:
                        try:
                            if btn.is_visible():
                                btn.click(force=True, timeout=2000)
                                return True
                        except Exception:
                            pass
                except Exception:
                    pass
            try:
                btns = p.locator('button').all()
                for btn in btns:
                    try:
                        text = (btn.text_content() or '').strip().lower()
                        aria = (btn.get_attribute('aria-label') or '').strip().lower()
                        marker = f'{text} {aria}'
                        is_verify = (
                            'i am a human' in marker
                            or 'verify' in marker
                            or '验证' in marker
                            or '我是人类' in marker
                            or '人类' in marker
                        )
                        if is_verify and btn.is_visible():
                            btn.click(force=True, timeout=2000)
                            return True
                    except Exception:
                        pass
            except Exception:
                pass
            return False

        def recurse(p, depth):
            if try_click_in_frame(p):
                return True
            if depth >= max_depth:
                return False
            for frame in p.frames:
                if frame == p.main_frame and depth > 0:
                    continue
                try:
                    if recurse(frame, depth + 1):
                        return True
                except Exception:
                    pass
            return False

        return recurse(page, 0)

    def _refresh_and_capture_fresh_blob_for_capmonster(self, page, blob_catcher, old_blob=None,
                                                       detect_timeout=30.0, click_timeout=20.0,
                                                       blob_timeout=20.0):
        """Reload the gate while CDP is still listening, then return a fresh Arkose blob.

        Used only after local browser-side dice clicks end in a parent-page
        rejection (for example "Value is invalid"). The old blob belongs to
        the rejected page/session, so it must be cleared before reload and must
        never be passed to CapMonster.
        """
        if blob_catcher is None:
            logger.error('Fresh blob retry requested but blob_catcher is not available')
            return None

        old_len = len(old_blob) if old_blob else 0
        logger.warning(f'Refreshing page for CapMonster fallback with fresh Arkose blob (old_len={old_len})')

        try:
            blob_catcher.reset_blob()
            logger.info('CDP blob catcher reset before reload; listener remains active')
        except Exception as e:
            logger.error(f'Failed to reset blob catcher before reload: {type(e).__name__}: {e}')
            return None

        try:
            try:
                page.reload(wait_until='domcontentloaded', timeout=20000)
            except TypeError:
                page.reload()
            logger.info('Page reloaded; waiting for FunCaptcha to reappear')
        except Exception as e:
            logger.error(f'Page reload failed before fresh blob capture: {type(e).__name__}: {e}')
            return None

        detected = False
        deadline = time.time() + detect_timeout
        while time.time() < deadline:
            try:
                info = self.detect(page)
                if info.get('found'):
                    logger.info(f'FunCaptcha re-detected after refresh (siteKey={info.get("siteKey")})')
                    detected = True
                    break
            except Exception as e:
                logger.debug(f'Detect after refresh failed: {e}')
            time.sleep(0.5)
        if not detected:
            logger.warning('FunCaptcha was not re-detected after refresh; still trying verify click/new blob wait')

        clicked = False
        click_deadline = time.time() + click_timeout
        while time.time() < click_deadline:
            try:
                if self._click_arkose_verify_button(page):
                    clicked = True
                    logger.info('Clicked Arkose verify button after refresh')
                    break
            except Exception as e:
                logger.debug(f'Verify click after refresh failed: {e}')
            time.sleep(1.0)
        if not clicked:
            logger.warning('Verify button was not clicked after refresh; waiting for already-captured blob anyway')

        try:
            new_blob = blob_catcher.wait_for_blob(timeout=blob_timeout)
        except Exception as e:
            logger.error(f'Fresh blob wait failed: {type(e).__name__}: {e}')
            return None

        if not new_blob:
            logger.error('Fresh blob retry failed: no new blob captured after refresh')
            return None
        if old_blob and new_blob == old_blob:
            logger.error('Fresh blob retry captured the same blob as the rejected session; refusing to reuse it')
            return None

        logger.info(f'Fresh Arkose blob captured for CapMonster fallback (new_len={len(new_blob)}, old_len={old_len})')
        return new_blob

    def inject_token(self, page, token):
        try:
            result = page.evaluate('''
                ([token]) => {
                    var form = document.querySelector('form#flow-form') || document.querySelector('form[action*="captcha-gate"]');
                    if (!form) return {ok:false, reason:'no-form'};
                    var arkose = form.querySelector('input[name="arkose"]') || document.querySelector('#capture-arkose');
                    if (!arkose) return {ok:false, reason:'no-arkose-input'};
                    var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    setter.call(arkose, token);
                    arkose.dispatchEvent(new Event('input', {bubbles:true}));
                    arkose.dispatchEvent(new Event('change', {bubbles:true}));
                    try {
                        if (typeof form.requestSubmit === 'function') form.requestSubmit();
                        else form.submit();
                        return {ok:true, method:'requestSubmit', arkoseSet: arkose.value === token};
                    } catch(e) { return {ok:false, reason:'submit-error:'+e.message}; }
                }
            ''', [token])
            if result and result.get('ok'): logger.info(f'💉 token 注入成功: {result.get("method")}'); return True
            logger.warning(f'注入失败: {result}')
        except Exception as e: logger.warning(f'注入异常: {e}')
        return False

    def solve_and_inject(self, page, timeout=90.0, blob_catcher=None, image_catcher=None):
        deadline = time.time() + timeout
        detected = False
        pc = 0
        while time.time() < deadline:
            pc += 1
            try:
                result = self.detect(page)
                if pc % 10 == 0: logger.info(f'🔄 已探测 {pc} 次, found={result.get("found")}')
                if result.get('found'):
                    detected = True
                    logger.info(f'🎯 FunCaptcha 已检测到 (siteKey={result.get("siteKey")})')
                    _capture_debug_screenshot(page, 'funcaptcha_detected.png')
                    break
            except Exception as e: logger.warning(f'探测异常: {e}')
            time.sleep(0.5)
        if not detected: logger.warning('⏱️ 等待 FunCaptcha 超时 (正常)'); return True

        blob_before = blob_catcher.captured_blob if blob_catcher else None
        if blob_before:
            logger.info(f'📦 已有点击前 blob (长度 {len(blob_before)}), 跳过点击后 CDP 等待')
        elif blob_catcher:
            blob_catcher.reset_blob()

        def _try_click(t=30.0):
            dl = time.time() + t
            while time.time() < dl:
                if self._click_arkose_verify_button(page): return True
                time.sleep(2.0)
            return False

        if not _try_click():
            logger.warning('⚠️ 30s 未点到验证按钮, 刷新重试')
            page.reload(); time.sleep(5)
            if blob_catcher and not blob_before: blob_catcher.reset_blob()
            _try_click()
        _capture_debug_screenshot(page, 'funcaptcha_after_verify.png')

        blob_new = None
        if blob_catcher is not None and not blob_before:
            logger.info('⏳ 等待 CDP 抓取 blob (点击后)...')
            blob_new = blob_catcher.wait_for_blob(timeout=15.0)
            if blob_new: logger.info(f'📦 CDP 抓到点击后 blob (长度 {len(blob_new)})')

        blob = blob_before or blob_new
        if blob:
            logger.info(f'📦 使用 blob (长度 {len(blob)}, 来源: {"点击后" if blob_new else "点击前"})')
        else:
            logger.warning('⚠️ 无 blob (点击前后都没抓到)')

        capmonster_blob_refreshed = False
        if CAPMONSTER_FIRST and self.config.api_key:
            logger.info('CAPMONSTER_FIRST enabled; skip local ONNX dice and use captured fresh blob first')
            local_dice = None
        else:
            local_dice = try_solve_dice_challenge(page, image_catcher)
        if local_dice is True:
            logger.info('Local ONNX dice completed; checking parent page verdict')
            post_local = _wait_post_local_dice_result(page, timeout=20.0)
            if post_local is True:
                logger.info('Registration success detected after local dice')
                return True
            if post_local is None:
                if not self.config.api_key or blob_catcher is None:
                    logger.info('No rejection after local dice and no CapMonster/blob catcher fallback available; hand off to registration success wait')
                    return True
                logger.warning('No final success/reject after local dice; refreshing for CapMonster fallback')
            else:
                logger.warning('Local dice did not produce usable parent token; trying CapMonster fallback')
            if not self.config.api_key:
                logger.error('No CAPMONSTER_API_KEY; cannot retry rejected local dice with CapMonster')
                return False
            fresh_blob = self._refresh_and_capture_fresh_blob_for_capmonster(
                page,
                blob_catcher=blob_catcher,
                old_blob=blob,
                detect_timeout=30.0,
                click_timeout=20.0,
                blob_timeout=20.0,
            )
            if not fresh_blob:
                logger.error('Could not obtain a fresh blob after rejected local dice; aborting CapMonster fallback')
                return False
            blob = fresh_blob
            capmonster_blob_refreshed = True
            local_dice = False
        if local_dice is False:
            logger.warning('Local ONNX dice failed/rejected; trying CapMonster fallback')
            if not self.config.api_key:
                logger.error('No CAPMONSTER_API_KEY; cannot retry failed local dice with CapMonster')
                return False
            if blob_catcher is not None and not capmonster_blob_refreshed:
                fresh_blob = self._refresh_and_capture_fresh_blob_for_capmonster(
                    page,
                    blob_catcher=blob_catcher,
                    old_blob=blob,
                    detect_timeout=30.0,
                    click_timeout=20.0,
                    blob_timeout=20.0,
                )
                if not fresh_blob:
                    logger.error('Could not obtain a fresh blob after failed local dice; aborting CapMonster fallback')
                    return False
                blob = fresh_blob
        else:
            logger.info('Non-dice challenge or local model unavailable; using CapMonster')

        if not self.config.api_key:
            logger.error('No CAPMONSTER_API_KEY; cannot use CapMonster fallback')
            return False

        token = self.solve(page, blob=blob)
        if not token:
            if LOCAL_DICE_ENABLED and image_catcher is not None:
                logger.warning('CapMonster failed; trying local ONNX dice fallback if current challenge is dice')
                local_after_capmonster = try_solve_dice_challenge(page, image_catcher)
                if local_after_capmonster is True:
                    post_local = _wait_post_local_dice_result(page, timeout=20.0)
                    if post_local is True:
                        logger.info('Registration success detected after CapMonster-failed local dice fallback')
                        return True
                    if post_local is None:
                        logger.info('No rejection after CapMonster-failed local dice fallback; hand off to registration success wait')
                        return True
                    logger.warning('Local dice fallback after CapMonster failure was rejected')
                elif local_after_capmonster is None:
                    logger.warning('Local dice fallback skipped: current challenge is not dice or local solver unavailable')
                else:
                    logger.warning('Local dice fallback after CapMonster failure failed')
            return False
        ok = self.inject_token(page, token)

        dl2 = time.time() + 15.0
        while time.time() < dl2:
            try:
                state = page.evaluate('''() => {
                    if (document.querySelector('#success-icon > svg > path')) return 'success';
                    var err = document.querySelector('[class*="error"], [role="alert"]');
                    if (err && /无效|invalid|失败/i.test(err.textContent || '')) return 'rejected';
                    return null;
                }''')
                if state == 'success': logger.info('✅ 验证通过!'); return True
                if state == 'rejected': logger.warning('⚠️ token 被拒'); return False
            except Exception: pass
            time.sleep(0.4)
        logger.info('ℹ️ 15s 未见结果页, 交由后续判定')
        return ok


# ═══════════════════════════════ 浏览器 ═══════════════════════════════
def create_cloak_browser():
    launch_args = [
        f'--remote-debugging-port={CDP_DEBUG_PORT}',
        '--remote-allow-origins=*',
        '--no-sandbox',
        '--disable-dev-shm-usage',
        '--disable-gpu',
    ]

    browser = launch(
        headless=True,
        args=launch_args,
    )

    context = browser.new_context(viewport={'width': 1920, 'height': 1080})
    page = context.new_page()

    logger.info('🦊 CloakBrowser 已启动 (Chromium 146, headless=True)')

    return browser, context, page


# ═══════════════════════════════ 注册流程 ═══════════════════════════════
def register_one(acc):
    browser = None
    solver = None
    if CAPMONSTER_API_KEY or LOCAL_DICE_ENABLED:
        solver = CapMonsterFunCaptchaSolver(CapMonsterSolverConfig(api_key=CAPMONSTER_API_KEY, max_wait=300.0))
        if CAPMONSTER_API_KEY:
            logger.info('🛡️ CapMonster 求解器已就绪')
        if LOCAL_DICE_ENABLED:
            logger.info(f'🎲 本地 ONNX 骰子求解已启用: {LOCAL_DICE_WEIGHTS}')

    blob_catcher = None
    image_catcher = None
    try:
        browser, context, page = create_cloak_browser()

        def _wait_or_refresh(selector, desc, timeout=25):
            try:
                is_error = page.evaluate('''() => {
                    return document.querySelector('#main-frame-error') !== null
                        || document.body?.classList?.contains('neterror')
                        || document.title.includes('无法访问')
                }''')
            except Exception:
                is_error = False
            if is_error:
                logger.warning(f'⚠️ 检测到错误页 ({desc}), 立即刷新...')
                page.reload(); time.sleep(3)
            try:
                page.wait_for_selector(selector, timeout=timeout * 1000)
                logger.info(f'✅ {desc} 已就绪')
                return page.locator(selector).first
            except Exception:
                logger.warning(f'⚠️ {desc} 超时 (等{timeout}s), 硬刷新...')
                page.reload(); time.sleep(5)
                page.wait_for_selector(selector, timeout=timeout * 1000)
                logger.info(f'🔄 刷新后 {desc} 已出现')
                return page.locator(selector).first

        # ── 打开页面 ──
        logger.info(f'📄 打开: {REGISTER_URL}')
        page.goto(REGISTER_URL, wait_until='domcontentloaded')
        time.sleep(2)

        # ── Cookie 横幅 ──
        try:
            cookie_btn = page.locator('button#onetrust-reject-all-handler, button.ot-reject-all, button[id*="reject"]')
            if cookie_btn.count() > 0:
                cookie_btn.first.click(timeout=3000)
                logger.info('🍪 已关闭 Cookie 横幅')
        except Exception:
            pass

        # ── 步骤1: 邮箱 ──
        logger.info(f'📧 {acc["email"]}')
        page.locator('#accountName').fill(acc['email'])
        time.sleep(0.5)
        page.locator('#submit').click()
        time.sleep(2)

        # ── 步骤2: 国家 ──
        _wait_or_refresh('#capture-country', '国家选择器')
        logger.info('🔄 刷新页面确保完整加载...')
        page.reload(); time.sleep(3)
        _wait_or_refresh('#capture-country', '国家选择器(刷新后)')
        logger.info(f'🌍 国家: {COUNTRY}')
        page.select_option('#capture-country', COUNTRY)
        time.sleep(1.5)

        # ── 步骤3: 生日 ──
        try:
            cb = page.locator('button#onetrust-reject-all-handler, button[id*="reject"]')
            if cb.count() > 0 and cb.first.is_visible():
                cb.first.click()
                logger.info('🍪 关闭晚出现的 Cookie')
                time.sleep(0.5)
        except Exception:
            pass

        logger.info(f'🎂 {acc["birth_year"]}-{acc["birth_month"]}-{acc["birth_day"]}')
        page.locator('[name="dob-plain"]').click()
        time.sleep(0.8)
        page.evaluate('''
            ([year, month, day]) => {
                var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                var c = document.querySelector('#dob-field-active'); if (!c) return;
                c.querySelectorAll('input').forEach(function(inp){
                    var cls = inp.className || '';
                    if (cls.indexOf('--yyyy')!==-1) setter.call(inp, year);
                    else if (cls.indexOf('--mm')!==-1) setter.call(inp, month);
                    else if (cls.indexOf('--dd')!==-1) setter.call(inp, day);
                    inp.dispatchEvent(new Event('input', {bubbles:true}));
                    inp.dispatchEvent(new Event('change', {bubbles:true}));
                });
            }
        ''', [acc['birth_year'], acc['birth_month'], acc['birth_day']])
        time.sleep(0.5)
        page.locator('#flow-form-submit-btn').click()
        time.sleep(2)

        # ── 步骤4: 姓名 ──
        logger.info(f'👤 {acc["first_name"]} {acc["last_name"]}')
        try:
            page.wait_for_selector('#capture-first-name', timeout=8000)
            page.locator('#capture-first-name').click(); time.sleep(0.2)
            page.locator('#capture-first-name').fill(acc['first_name']); time.sleep(0.3)
            page.locator('#capture-last-name').fill(acc['last_name']); time.sleep(0.3)
        except Exception:
            logger.warning('⚠️ 姓名框未出现, 刷新重试')
            page.reload(); time.sleep(5)
            page.wait_for_selector('#capture-first-name', timeout=10000)
            page.locator('#capture-first-name').click(); time.sleep(0.2)
            page.locator('#capture-first-name').fill(acc['first_name']); time.sleep(0.3)
            page.locator('#capture-last-name').fill(acc['last_name']); time.sleep(0.3)
            logger.info('🔄 刷新后姓名填写成功')
        page.locator('#flow-form-submit-btn').click()
        time.sleep(2)

        # ── 步骤5: 邮箱确认 ──
        actual_email = page.locator('#capture-email').input_value()
        if actual_email == acc['email']:
            logger.info(f'✅ 邮箱一致: {actual_email}')
            page.locator('#flow-form-submit-btn').click()
            time.sleep(2)
        else:
            logger.error(f'❌ 邮箱不一致! {actual_email}')
            page.screenshot(path='error_email.png')
            return False

        # ── 步骤6: 协议 ──
        logger.info('📋 勾选协议')
        time.sleep(1)
        page.evaluate('''() => {
            ['#capture-opt-in-blizzard-news-special-offers','#legal-checkboxes > label > input.step__checkbox'].forEach(function(sel){
                var el = document.querySelector(sel);
                if (el && !el.checked) {
                    var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'checked').set;
                    s.call(el, true);
                    el.dispatchEvent(new Event('change', {bubbles:true}));
                    el.dispatchEvent(new Event('input', {bubbles:true}));
                }
            });
        }''')
        time.sleep(0.5)
        page.locator('#flow-form-submit-btn').click()
        time.sleep(2)

        # ── 步骤7: 密码 ──
        logger.info(f'🔑 密码: {acc["password"]}')
        _wait_or_refresh('#capture-password', '密码输入框')
        page.locator('#capture-password').fill(acc['password'])
        time.sleep(0.5)
        page.locator('#flow-form-submit-btn').click()
        time.sleep(2)

        # ── 步骤8: BattleTag + CDP + FunCaptcha ──
        logger.info(f'🏷️ {acc["battle_tag"]}')
        _wait_or_refresh('#capture-battletag', 'BattleTag输入框')

        page.evaluate('''
            ([elId, val]) => {
                var el = document.querySelector(elId);
                var s = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, "value").set;
                s.call(el, val);
                el.dispatchEvent(new Event("input", {bubbles:true}));
                el.dispatchEvent(new Event("change", {bubbles:true}));
            }
        ''', ['#capture-battletag', acc['battle_tag']])
        time.sleep(0.5)

        try:
            page.wait_for_function(
                '() => { const btn = document.querySelector("#flow-form-submit-btn"); return btn && !btn.disabled; }',
                timeout=5000
            )
        except Exception:
            logger.warning('⚠️ submit-btn 启用等待超时')

        if solver and CAPMONSTER_API_KEY:
            try:
                blob_catcher = CDPBlobCatcher(debug_port=CDP_DEBUG_PORT, label='1')
                blob_catcher.start()
            except Exception as e:
                logger.warning(f'⚠️ CDP 启动失败: {e}')
                blob_catcher = None
        if solver and LOCAL_DICE_ENABLED and CDPImageCatcher is not None:
            try:
                image_catcher = CDPImageCatcher(debug_port=CDP_DEBUG_PORT, label='dice')
                image_catcher.start()
            except Exception as e:
                logger.warning(f'⚠️ CDP 图片抓取器启动失败: {e}')
                image_catcher = None

        logger.info('➡️ 提交 BattleTag')
        page.locator('#flow-form-submit-btn').click()
        time.sleep(2)

        if solver:
            logger.info('⏳ 等待 FunCaptcha 弹窗出现...')
            try:
                ok = solver.solve_and_inject(page, timeout=90.0, blob_catcher=blob_catcher, image_catcher=image_catcher)
            finally:
                if blob_catcher: blob_catcher.stop()
                if image_catcher: image_catcher.stop()
            if not ok:
                logger.error('❌ FunCaptcha 求解失败')
                page.screenshot(path='error_captcha.png')
                return False

        # ── 步骤9: 成功页 ──
        logger.info('⏳ 等待注册成功...')
        try:
            if not _wait_registration_success(page, acc['email'], timeout=45.0):
                raise TimeoutError('registration success page not detected')
            logger.info('✅ 注册成功!')
            with open('registered_account.txt', 'a', encoding='utf-8') as f:
                f.write(f'账号：{acc["email"]}\n密码：{acc["password"]}\n\n')
            logger.info(f'💾 已保存: {acc["email"]}')
            page.screenshot(path='success.png')
            return True
        except Exception as e:
            try:
                final_state = _captcha_state(page)
                sample = _captcha_text(page).replace('\n', ' ')[:220]
            except Exception:
                final_state = 'unknown'
                sample = ''
            logger.error(f'Registration success wait failed: {type(e).__name__}: {e}, captcha_state={final_state}, URL={page.url[:100]}, sample={sample}')
            page.screenshot(path='error_timeout.png')
            return False

    except Exception as e:
        logger.error(f'❌ 异常: {type(e).__name__}: {e}', exc_info=True)
        try:
            if browser and browser.contexts:
                for p in browser.contexts[0].pages:
                    p.screenshot(path='error_exception.png')
        except Exception:
            pass
        return False
    finally:
        if blob_catcher:
            try: blob_catcher.stop()
            except Exception: pass
        if image_catcher:
            try: image_catcher.stop()
            except Exception: pass
        if browser:
            try: browser.close()
            except Exception: pass


def main():
    acc = generate_identity()
    logger.info('=' * 50)
    logger.info('🚀 战网自动注册 — CloakBrowser 免费版 (直连)')
    logger.info(f'   邮箱: {acc["email"]}')
    logger.info(f'   BattleTag: {acc["battle_tag"]}')
    logger.info('=' * 50)

    ok = register_one(acc)
    logger.info(f'\n🏁 注册结束: {"✅ 成功" if ok else "❌ 失败"}')
    if not ok:
        sys.exit(1)


if __name__ == '__main__':
    main()
