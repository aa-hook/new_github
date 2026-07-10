import importlib.util
import sys
import types
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REGISTER_PATH = ROOT / "register.py"


def load_register_module():
    sys.modules.setdefault("cloakbrowser", types.SimpleNamespace(launch=lambda *a, **k: None))
    spec = importlib.util.spec_from_file_location("register_under_test", REGISTER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class FakePage:
    def __init__(self, events=None):
        self.url = "https://account.battle.net/creation/flow/creation-full"
        self.reload_calls = 0
        self.events = events if events is not None else []
        self.frames = []
        self.main_frame = self

    def reload(self, *args, **kwargs):
        self.events.append("reload")
        self.reload_calls += 1

    def evaluate(self, *args, **kwargs):
        return "success"


class FakeBlobCatcher:
    def __init__(self, events=None, old_blob="old_blob", new_blob="new_blob"):
        self.captured_blob = old_blob
        self.new_blob = new_blob
        self.events = events if events is not None else []

    def reset_blob(self):
        self.events.append("reset_blob")
        self.captured_blob = None

    def wait_for_blob(self, timeout=30.0):
        self.events.append(("wait_for_blob", timeout))
        self.captured_blob = self.new_blob
        return self.new_blob


class FakeLocator:
    def __init__(self, items):
        self._items = items

    def all(self):
        return list(self._items)


class FakeButtonPage(FakePage):
    def __init__(self):
        super().__init__([])
        self.clicked = False
        self.button = FakeButton("I am a human", self)

    def locator(self, selector):
        if selector == "button":
            return FakeLocator([self.button])
        return FakeLocator([])


class FakeButton:
    def __init__(self, text, page):
        self._text = text
        self._page = page

    def text_content(self):
        return self._text

    def get_attribute(self, name):
        return ""

    def is_visible(self):
        return True

    def click(self, *args, **kwargs):
        self._page.clicked = True


class FreshBlobRetryTests(unittest.TestCase):
    def test_rejected_local_dice_refreshes_and_uses_new_blob_for_capmonster(self):
        reg = load_register_module()
        original_try_solve = reg.try_solve_dice_challenge
        original_wait_post = reg._wait_post_local_dice_result
        try:
            reg.try_solve_dice_challenge = lambda page, image_catcher: True
            reg._wait_post_local_dice_result = lambda page, timeout=10.0: False

            config = reg.CapMonsterSolverConfig(api_key="capmonster-key")
            solver = reg.CapMonsterFunCaptchaSolver(config)
            events = []
            page = FakePage(events)
            blob_catcher = FakeBlobCatcher(events)
            solved_blobs = []
            click_events = []

            solver.detect = lambda p: {"found": True, "siteKey": "SITEKEY", "surl": "blizzard-api.arkoselabs.com"}

            def fake_click(p):
                click_events.append("click")
                return True

            solver._click_arkose_verify_button = fake_click
            solver.solve = lambda p, blob=None: solved_blobs.append(blob) or "capmonster-token"
            solver.inject_token = lambda p, token: True

            ok = solver.solve_and_inject(page, timeout=5.0, blob_catcher=blob_catcher, image_catcher=None)

            self.assertTrue(ok)
            self.assertEqual(page.reload_calls, 1)
            self.assertEqual(blob_catcher.events[0], "reset_blob")
            self.assertIn(("wait_for_blob", 20.0), blob_catcher.events)
            self.assertGreaterEqual(len(click_events), 2)
            self.assertEqual(solved_blobs, ["new_blob"])
            self.assertLess(events.index("reset_blob"), events.index("reload"))
        finally:
            reg.try_solve_dice_challenge = original_try_solve
            reg._wait_post_local_dice_result = original_wait_post

    def test_local_dice_false_refreshes_before_capmonster(self):
        reg = load_register_module()
        original_try_solve = reg.try_solve_dice_challenge
        try:
            reg.try_solve_dice_challenge = lambda page, image_catcher: False

            solver = reg.CapMonsterFunCaptchaSolver(reg.CapMonsterSolverConfig(api_key="capmonster-key"))
            events = []
            page = FakePage(events)
            blob_catcher = FakeBlobCatcher(events, old_blob="old_blob", new_blob="fresh_blob")
            click_events = []
            solved_blobs = []

            solver.detect = lambda p: {"found": True, "siteKey": "SITEKEY", "surl": "blizzard-api.arkoselabs.com"}
            solver._click_arkose_verify_button = lambda p: click_events.append("click") or True
            solver.solve = lambda p, blob=None: solved_blobs.append(blob) or "capmonster-token"
            solver.inject_token = lambda p, token: True

            ok = solver.solve_and_inject(page, timeout=5.0, blob_catcher=blob_catcher, image_catcher=None)

            self.assertTrue(ok)
            self.assertEqual(page.reload_calls, 1)
            self.assertEqual(solved_blobs, ["fresh_blob"])
            self.assertIn("reset_blob", events)
            self.assertIn(("wait_for_blob", 20.0), events)
            self.assertGreaterEqual(len(click_events), 2)
        finally:
            reg.try_solve_dice_challenge = original_try_solve

    def test_no_verdict_after_local_dice_refreshes_before_capmonster(self):
        reg = load_register_module()
        original_try_solve = reg.try_solve_dice_challenge
        original_wait_post = reg._wait_post_local_dice_result
        try:
            reg.try_solve_dice_challenge = lambda page, image_catcher: True
            reg._wait_post_local_dice_result = lambda page, timeout=10.0: None

            solver = reg.CapMonsterFunCaptchaSolver(reg.CapMonsterSolverConfig(api_key="capmonster-key"))
            events = []
            page = FakePage(events)
            blob_catcher = FakeBlobCatcher(events, old_blob="old_blob", new_blob="fresh_after_no_verdict")
            click_events = []
            solved_blobs = []

            solver.detect = lambda p: {"found": True, "siteKey": "SITEKEY", "surl": "blizzard-api.arkoselabs.com"}
            solver._click_arkose_verify_button = lambda p: click_events.append("click") or True
            solver.solve = lambda p, blob=None: solved_blobs.append(blob) or "capmonster-token"
            solver.inject_token = lambda p, token: True

            ok = solver.solve_and_inject(page, timeout=5.0, blob_catcher=blob_catcher, image_catcher=None)

            self.assertTrue(ok)
            self.assertEqual(page.reload_calls, 1)
            self.assertEqual(solved_blobs, ["fresh_after_no_verdict"])
            self.assertIn("reset_blob", events)
            self.assertIn(("wait_for_blob", 20.0), events)
            self.assertGreaterEqual(len(click_events), 2)
        finally:
            reg.try_solve_dice_challenge = original_try_solve
            reg._wait_post_local_dice_result = original_wait_post

    def test_capmonster_first_skips_local_dice_and_uses_initial_blob(self):
        reg = load_register_module()
        original_try_solve = reg.try_solve_dice_challenge
        original_capmonster_first = getattr(reg, "CAPMONSTER_FIRST", False)
        try:
            reg.CAPMONSTER_FIRST = True

            def fail_if_called(page, image_catcher):
                raise AssertionError("local dice should be skipped in CAPMONSTER_FIRST mode")

            reg.try_solve_dice_challenge = fail_if_called
            solver = reg.CapMonsterFunCaptchaSolver(reg.CapMonsterSolverConfig(api_key="capmonster-key"))
            events = []
            page = FakePage(events)
            blob_catcher = FakeBlobCatcher(events, old_blob="initial_blob", new_blob="unused")
            solved_blobs = []

            solver.detect = lambda p: {"found": True, "siteKey": "SITEKEY", "surl": "blizzard-api.arkoselabs.com"}
            solver._click_arkose_verify_button = lambda p: True
            solver.solve = lambda p, blob=None: solved_blobs.append(blob) or "capmonster-token"
            solver.inject_token = lambda p, token: True

            ok = solver.solve_and_inject(page, timeout=5.0, blob_catcher=blob_catcher, image_catcher=None)

            self.assertTrue(ok)
            self.assertEqual(page.reload_calls, 0)
            self.assertEqual(solved_blobs, ["initial_blob"])
            self.assertNotIn("reset_blob", events)
        finally:
            reg.CAPMONSTER_FIRST = original_capmonster_first
            reg.try_solve_dice_challenge = original_try_solve

    def test_verify_click_falls_back_to_i_am_human_button_text(self):
        reg = load_register_module()
        solver = reg.CapMonsterFunCaptchaSolver(reg.CapMonsterSolverConfig(api_key="capmonster-key"))
        page = FakeButtonPage()

        clicked = solver._click_arkose_verify_button(page)

        self.assertTrue(clicked)
        self.assertTrue(page.clicked)

    def test_main_exits_nonzero_when_registration_fails(self):
        reg = load_register_module()
        original_generate = reg.generate_identity
        original_register_one = reg.register_one
        try:
            reg.generate_identity = lambda: {
                "email": "failure@example.com",
                "password": "pw",
                "battle_tag": "Tag1",
                "first_name": "a",
                "last_name": "b",
                "birth_year": "1990",
                "birth_month": "01",
                "birth_day": "10",
            }
            reg.register_one = lambda acc: False

            with self.assertRaises(SystemExit) as cm:
                reg.main()

            self.assertEqual(cm.exception.code, 1)
        finally:
            reg.generate_identity = original_generate
            reg.register_one = original_register_one


if __name__ == "__main__":
    unittest.main()
