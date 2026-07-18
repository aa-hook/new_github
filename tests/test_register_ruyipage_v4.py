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

    def fake_launch(**kwargs):
        calls["kwargs"] = kwargs
        return FakePage()

    monkeypatch.setattr(app.base.ruyipage, "launch", fake_launch)
    proxy = app.parse_proxy("proxy.test:8080:user:secret")

    app.launch_ruyi_browser(
        SimpleNamespace(headless=False, output_dir=str(tmp_path)), proxy
    )

    assert calls["kwargs"]["proxy"] == "http://user:secret@proxy.test:8080"
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

    def fake_solve(client, context, args, proxy, out):
        calls["solver_proxy"] = proxy.url
        return (
            {"ok": True, "token": "ARKOSE_TOKEN", "actions": [{"wave": 0}]},
            dict(context),
        )

    monkeypatch.setattr(app, "solve_arkose_with_ruyi", fake_solve)
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
    assert calls["client"]["proxy"] == "http://u:p@10.0.0.1:8080"
    assert calls["solver_proxy"] == "http://u:p@10.0.0.1:8080"
    assert calls["submitted_token"] == "ARKOSE_TOKEN"

    run_dir = next(tmp_path.glob("run_*"))
    result = json.loads((run_dir / "registration_result.json").read_text("utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text("utf-8"))
    assert result["ok"] is True
    assert summary["registrationCountry"] == "GBR"
    assert summary["countryProbe"] is True
    assert summary["proxy"]["hasAuth"] is True
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
