import pytest

import register_ruyipage_v2 as app


def test_failed_completion_payload_is_rejected_even_when_it_contains_a_token(monkeypatch):
    state = {
        "status": "onCompleted",
        "token": "FAILED_TOKEN",
        "tokenLength": 12,
        "completedPayload": {
            "completed": True,
            "hasToken": True,
            "failed": True,
            "error": None,
            "recoverable": False,
        },
    }
    monkeypatch.setattr(app, "solver_state", lambda _page: state)

    with pytest.raises(app.ArkoseCompletionRejected, match="failed=true"):
        app.wait_token_quick(object(), timeout=0.1)


def test_successful_completion_payload_returns_token(monkeypatch):
    state = {
        "status": "onCompleted",
        "token": "GOOD_TOKEN",
        "tokenLength": 10,
        "completedPayload": {
            "completed": True,
            "hasToken": True,
            "failed": False,
            "error": None,
            "recoverable": False,
        },
    }
    monkeypatch.setattr(app, "solver_state", lambda _page: state)

    assert app.wait_token_quick(object(), timeout=0.1) == "GOOD_TOKEN"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"failed": True, "error": None}, "failed=true"),
        ({"failed": False, "error": "DENIED"}, "error=DENIED"),
        ({"failed": False, "error": None}, None),
        (None, None),
    ],
)
def test_completion_rejection_reason(payload, expected):
    assert app.completion_rejection_reason(payload) == expected


def test_token_metadata_redacts_opaque_token_and_keeps_suffix_fields():
    token = "OPAQUE_SECRET|r=us-east-1|pk=SITE_KEY|at=40|ag=101|sup=1|plain-flag"

    metadata = app.arkose_token_metadata(token)

    assert metadata == {
        "tokenLength": len(token),
        "opaqueLength": len("OPAQUE_SECRET"),
        "fields": {
            "r": "us-east-1",
            "pk": "SITE_KEY",
            "at": "40",
            "ag": "101",
            "sup": "1",
        },
        "flags": ["plain-flag"],
    }
    assert "OPAQUE_SECRET" not in repr(metadata)


def test_build_token_result_persists_completion_and_redacted_metadata(monkeypatch):
    completion = {
        "completed": True,
        "hasToken": True,
        "failed": False,
        "suppressed": False,
        "error": None,
        "recoverable": False,
    }
    monkeypatch.setattr(
        app,
        "solver_state",
        lambda _page: {"status": "onCompleted", "completedPayload": completion},
    )

    result = app.build_token_result(object(), "SECRET|r=eu-west-1|ag=101", [{"wave": 0}])

    assert result["ok"] is True
    assert result["token"] == "SECRET|r=eu-west-1|ag=101"
    assert result["completedPayload"] == completion
    assert result["tokenMetadata"]["opaqueLength"] == len("SECRET")
    assert result["tokenMetadata"]["fields"] == {"r": "eu-west-1", "ag": "101"}


def test_captcha_gate_request_metadata_keeps_lengths_without_raw_values():
    body = "arkose=OPAQUE_SECRET%7Cr%3Dus-east-1%7Cag%3D101&email=user%40example.com"

    metadata = app.captcha_gate_request_metadata(body)

    assert metadata["bodyLength"] == len(body)
    assert metadata["fieldNames"] == ["arkose", "email"]
    assert metadata["fieldLengths"] == {
        "arkose": len("OPAQUE_SECRET|r=us-east-1|ag=101"),
        "email": len("user@example.com"),
    }
    assert metadata["arkoseTokenMetadata"]["fields"] == {"r": "us-east-1", "ag": "101"}
    assert "OPAQUE_SECRET" not in repr(metadata)
    assert "user@example.com" not in repr(metadata)


def test_captcha_gate_url_filter():
    assert app.is_captcha_gate_url(
        "https://account.battle.net/creation/flow/creation-full/step/captcha-gate"
    )
    assert not app.is_captcha_gate_url("https://account.battle.net/creation/flow/creation-full")


def test_selected_bidi_headers_decodes_location_and_content_type():
    headers = [
        {"name": "Location", "value": {"type": "string", "value": "/creation/success"}},
        {"name": "Content-Type", "value": {"type": "string", "value": "text/html"}},
        {"name": "Set-Cookie", "value": {"type": "string", "value": "secret=1"}},
    ]

    assert app.selected_bidi_headers(headers) == {
        "location": "/creation/success",
        "content-type": "text/html",
    }
