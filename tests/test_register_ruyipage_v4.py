import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import register_ruyipage_v4 as app


def test_proxy_blank_means_direct():
    proxy = app.parse_proxy("")

    assert proxy.enabled is False
    assert proxy.url is None
    assert proxy.display == "direct"
    assert proxy.summary()["hasAuth"] is False


def test_proxy_shorthand_is_normalized_and_redacted():
    proxy = app.parse_proxy("127.0.0.1:8080:user name:p@ss:word")

    assert proxy.url == "http://user%20name:p%40ss%3Aword@127.0.0.1:8080"
    assert proxy.display == "http://127.0.0.1:8080"
    assert proxy.has_auth is True
    assert "user" not in repr(proxy.summary())
    assert "p@ss" not in repr(proxy.summary())


def test_encoded_proxy_url_is_canonicalized_once():
    proxy = app.parse_proxy("http://user%20name:p%40ss@proxy.example:3128")

    assert proxy.url == "http://user%20name:p%40ss@proxy.example:3128"
    assert proxy.display == "http://proxy.example:3128"


def test_ruyi_launch_uses_proxy_but_never_logs_credentials(tmp_path, monkeypatch, caplog):
    calls = {}
    caplog.set_level("INFO")

    class FakePage:
        def set_bypass_csp(self, enabled):
            calls["bypass"] = enabled

    class FakeOptions:
        def __init__(self):
            self.preferences = {}

        def quick_start(self, **kwargs):
            calls["kwargs"] = kwargs

        def set_pref(self, key, value):
            self.preferences[key] = value

        def set_browser_path(self, value):
            calls["browser_path"] = value

    def fake_page(options):
        calls["preferences"] = dict(options.preferences)
        return FakePage()

    monkeypatch.setattr(app.base.ruyipage, "FirefoxOptions", FakeOptions)
    monkeypatch.setattr(app.base.ruyipage, "FirefoxPage", fake_page)
    monkeypatch.setattr(
        app.base.ruyipage,
        "resolve_firefox_path",
        lambda _path: "/runtime/firefox",
    )
    proxy = app.parse_proxy("proxy.test:8080:user:secret")

    app.launch_ruyi_browser(
        SimpleNamespace(headless=False, output_dir=str(tmp_path)), proxy
    )

    assert calls["kwargs"]["proxy"] == "http://user:secret@proxy.test:8080"
    assert calls["preferences"] == app._FIREFOX_LOW_TRAFFIC_PREFS
    assert calls["preferences"]["services.settings.server"] == "data:,"
    assert calls["preferences"]["network.connectivity-service.enabled"] is False
    assert calls["preferences"]["media.gmp-manager.updateEnabled"] is False
    assert calls["preferences"]["media.gmp-gmpopenh264.enabled"] is False
    assert calls["preferences"]["network.proxy.no_proxies_on"] == ",".join(
        app._FIREFOX_BACKGROUND_DIRECT_HOSTS
    )
    assert calls["browser_path"] == "/runtime/firefox"
    assert calls["bypass"] is True
    assert "secret" not in caplog.text
    assert "user" not in caplog.text
    assert "http://proxy.test:8080" in caplog.text


def test_proxy_values_are_redacted_from_errors():
    raw = "proxy.test:8080:user:secret"
    proxy = app.parse_proxy(raw)
    message = f"failed via {raw} normalized={proxy.url}"

    redacted = app.redact_proxy_text(message, proxy, raw)

    assert redacted == (
        "failed via http://proxy.test:8080 normalized=http://proxy.test:8080"
    )
    assert "user" not in redacted
    assert "secret" not in redacted


def test_v11_wait_retries_until_background_service_is_ready(monkeypatch):
    attempts = []

    def fake_health(_url, _timeout):
        attempts.append(_timeout)
        if len(attempts) < 3:
            raise ConnectionError("starting")
        return {"ok": True, "status": "ready"}

    monkeypatch.setattr(app.v3, "ensure_rank_v11_service", fake_health)
    monkeypatch.setattr(app.time, "sleep", lambda _seconds: None)

    assert app.wait_rank_v11_service("http://127.0.0.1:8765", 5)["status"] == "ready"
    assert len(attempts) == 3


@pytest.mark.parametrize(
    "value",
    [
        "host",
        "host:not-a-port",
        "host:70000",
        "ftp://host:21",
        "http://user@host:8080",
        "http://host:8080/path",
    ],
)
def test_invalid_proxy_is_rejected(value):
    with pytest.raises(ValueError):
        app.parse_proxy(value)


def test_country_probe_is_default_and_can_only_be_disabled_explicitly():
    parser = app.build_parser()

    assert parser.parse_args([]).country_probe is True
    assert parser.parse_args(["--country-probe"]).country_probe is True
    assert parser.parse_args(["--no-country-probe"]).country_probe is False


def test_public_static_optimizer_is_enabled_by_default_and_can_be_disabled(tmp_path):
    parser = app.build_parser()

    default = parser.parse_args(["--static-cache-dir", str(tmp_path)])
    disabled = parser.parse_args(
        ["--static-cache-dir", str(tmp_path), "--no-direct-public-static"]
    )
    proxy_images = parser.parse_args(
        ["--static-cache-dir", str(tmp_path), "--no-direct-challenge-images"]
    )

    assert default.no_direct_public_static is False
    assert default.direct_challenge_images is True
    assert default.static_cache_max_entry_mib == 8.0
    assert disabled.no_direct_public_static is True
    assert proxy_images.direct_challenge_images is False


def test_low_traffic_filter_keeps_arkose_images_and_blocks_only_nonessential_assets():
    assert not app.should_block_resource(
        "https://blizzard-api.arkoselabs.com/rtig/image?challenge=0"
    )
    assert not app.should_block_resource(
        "https://blizzard-api.arkoselabs.com/fc/assets/font.woff2"
    )
    assert app.should_block_resource("https://cdn.example.net/font.woff2")
    assert app.should_block_resource("https://www.google-analytics.com/collect")
    assert not app.should_block_resource("https://account.battle.net/main.js")


def test_undecodable_matched_challenge_is_reclassified_as_retryable(
    monkeypatch,
    tmp_path,
):
    original = app.v3.UnsupportedCaptchaQuestion(
        {
            "questionMatched": True,
            "sizeMatched": False,
            "imageSize": None,
        }
    )

    def fail(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(app.v3, "auto_solve_solver_tab", fail)

    with pytest.raises(RuntimeError, match="retry required") as caught:
        app.run_v4_solver_tab(None, None, SimpleNamespace(), tmp_path)

    assert type(caught.value) is RuntimeError
    assert caught.value.__cause__ is original


def test_real_unsupported_challenge_keeps_exit_classification(monkeypatch, tmp_path):
    original = app.v3.UnsupportedCaptchaQuestion(
        {
            "questionMatched": True,
            "sizeMatched": False,
            "imageSize": [1000, 400],
        }
    )

    def fail(*_args, **_kwargs):
        raise original

    monkeypatch.setattr(app.v3, "auto_solve_solver_tab", fail)

    with pytest.raises(app.v3.UnsupportedCaptchaQuestion) as caught:
        app.run_v4_solver_tab(None, None, SimpleNamespace(), tmp_path)

    assert caught.value is original


def test_protocol_success_metadata_is_authoritative():
    outcome = app.BattleProtocolClient.__module__
    assert outcome == "battle_protocol_flow_v4"

    from battle_protocol_flow_v4 import classify_registration_response

    result = classify_registration_response(
        """
        <i id="step-meta-data" data-step-id="create-success"
           data-step-has-errors="false"></i>
        <i id="player-id" data-player-account-id="123456"></i>
        <p class="step__banner--account-identifier">user@example.com</p>
        """,
        "user@example.com",
    )

    assert result["status"] == "success"
    assert result["playerAccountId"] == "123456"
    assert result["accountEmail"] == "user@example.com"


def test_main_returns_v11_token_to_original_http_session(tmp_path, monkeypatch):
    calls = {}

    class FakeMeter:
        def __init__(self, upstream_url):
            calls["meter_upstream"] = upstream_url
            self.snapshot_index = 0

        def start(self):
            return "http://127.0.0.1:43210"

        def snapshot(self):
            snapshots = (
                (0, 0, 0),
                (30, 270, 1),
                (70, 730, 2),
            )
            upload, download, connections = snapshots[self.snapshot_index]
            self.snapshot_index += 1
            return {
                "enabled": True,
                "uploadBytes": upload,
                "downloadBytes": download,
                "totalBytes": upload + download,
                "uploadMiB": round(upload / (1024 * 1024), 4),
                "downloadMiB": round(download / (1024 * 1024), 4),
                "totalMiB": round((upload + download) / (1024 * 1024), 4),
                "connections": connections,
                "failures": 0,
                "durationSeconds": float(self.snapshot_index),
            }

        def stop(self):
            return {
                "enabled": True,
                "uploadBytes": 100,
                "downloadBytes": 900,
                "totalBytes": 1000,
                "uploadMiB": 0.0001,
                "downloadMiB": 0.0009,
                "totalMiB": 0.001,
                "connections": 2,
                "failures": 0,
            }

    class FakeClient:
        def __init__(self, state, output_dir, **kwargs):
            calls["client"] = kwargs
            self.state = state

        def run_to_captcha(self, **kwargs):
            calls["run_to_captcha"] = kwargs

        def recover_arkose_from_last_response(self):
            return {
                "blob": "B" * 120,
                "siteKey": "SITE_KEY",
                "surl": "blizzard-api.arkoselabs.com",
                "websiteURL": "https://account.battle.net/creation/flow/creation-full",
                "source": "test",
            }

        def submit_captcha(self, token):
            calls["submitted_token"] = token
            return {
                "status": "success",
                "success": True,
                "playerAccountId": "123",
                "sample": "created",
            }

    monkeypatch.setattr(app, "BattleProtocolClient", FakeClient)
    monkeypatch.setattr(
        app.v3,
        "ensure_rank_v11_service",
        lambda *_args, **_kwargs: {
            "ok": True,
            "status": "ready",
            "device": "cpu",
            "model_load_seconds": 0.1,
            "warmup_seconds": 0.2,
        },
    )
    monkeypatch.setattr(
        app,
        "generate_identity",
        lambda: {
            "first_name": "first",
            "last_name": "last",
            "email": "user@example.com",
            "password": "secret",
            "birth_year": "1990",
            "birth_month": "01",
            "birth_day": "15",
            "battle_tag": "Player12",
        },
    )

    def fake_solve(
        client, context, args, proxy, out, runtime_proxy_url=None
    ):
        calls["solver_proxy"] = runtime_proxy_url
        return (
            {"ok": True, "token": "ARKOSE_TOKEN", "actions": [{"wave": 0}]},
            dict(context),
        )

    monkeypatch.setattr(app, "solve_arkose_with_ruyi", fake_solve)
    monkeypatch.setattr(app, "ProxyTrafficMeter", FakeMeter)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "register_ruyipage_v4.py",
            "--output-dir",
            str(tmp_path),
            "--proxy",
            "10.0.0.1:8080:u:p",
        ],
    )

    assert app.main() == 0
    assert calls["run_to_captcha"] == {
        "country": "GBR",
        "opt_in": False,
        "country_probe": True,
    }
    assert calls["meter_upstream"] == "http://u:p@10.0.0.1:8080"
    assert calls["client"]["proxy"] == "http://127.0.0.1:43210"
    assert calls["solver_proxy"] == "http://127.0.0.1:43210"
    assert calls["submitted_token"] == "ARKOSE_TOKEN"

    run_dir = next(tmp_path.glob("run_*"))
    result = json.loads((run_dir / "registration_result.json").read_text("utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert result["ok"] is True
    assert summary["registrationCountry"] == "GBR"
    assert summary["countryProbe"] is True
    assert summary["proxy"]["hasAuth"] is True
    assert summary["proxyTraffic"]["totalBytes"] == 1000
    phases = summary["proxyTrafficPhases"]
    assert phases["complete"] is True
    assert phases["phases"]["protocolToCaptcha"]["totalBytes"] == 300
    assert phases["phases"]["arkoseSolver"]["totalBytes"] == 500
    assert phases["phases"]["captchaSubmit"]["totalBytes"] == 200
    assert phases["accountedBytes"] == 1000
    assert phases["unaccountedBytes"] == 0
    traffic = json.loads((run_dir / "proxy_traffic.json").read_text("utf-8"))
    assert traffic["uploadBytes"] == 100
    assert traffic["downloadBytes"] == 900
    phase_file = json.loads(
        (run_dir / "proxy_traffic_phases.json").read_text("utf-8")
    )
    assert phase_file["phases"] == phases["phases"]
    assert "secret" not in json.dumps(summary)


def test_resume_with_saved_token_skips_browser_and_model(tmp_path, monkeypatch):
    run_dir = tmp_path / "existing-run"
    state = app.PersistentFlowState.create(
        run_dir / "persistent_state.json",
        identity={
            "email": "resume@example.com",
            "password": "resume-password",
            "battle_tag": "Resume77",
        },
    )
    state.checkpoint(
        "token-ready",
        arkose={
            "blob": "B" * 120,
            "token": "SAVED_TOKEN",
            "siteKey": "SITE_KEY",
            "surl": "blizzard-api.arkoselabs.com",
            "websiteURL": "https://account.battle.net/creation/flow/creation-full",
        },
    )
    calls = {}

    class FakeResumeClient:
        def __init__(self, state, output_dir, **kwargs):
            calls["created"] = True

        def submit_captcha(self, token):
            calls["token"] = token
            return {"status": "success", "success": True}

    monkeypatch.setattr(app, "BattleProtocolClient", FakeResumeClient)
    monkeypatch.setattr(
        app.v3,
        "ensure_rank_v11_service",
        lambda *_args, **_kwargs: pytest.fail("V11 health must be skipped"),
    )
    monkeypatch.setattr(
        app,
        "solve_arkose_with_ruyi",
        lambda *_args, **_kwargs: pytest.fail("browser solver must be skipped"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["register_ruyipage_v4.py", "--resume", str(run_dir)],
    )

    assert app.main() == 0
    assert calls["created"] is True
    assert calls["token"] == "SAVED_TOKEN"
    result = json.loads((run_dir / "registration_result.json").read_text("utf-8"))
    assert result["ok"] is True
