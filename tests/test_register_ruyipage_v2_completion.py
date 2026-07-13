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
