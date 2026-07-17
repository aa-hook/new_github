from pathlib import Path

import pytest

import register_ruyipage_v3 as app


class FakeResponse:
    def __init__(self, payload, *, status_code=200, text=""):
        self.payload = payload
        self.status_code = status_code
        self.text = text

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise app.requests.HTTPError(f"HTTP {self.status_code}")


def test_supported_question_is_case_and_whitespace_insensitive():
    question = (
        "  USE THE ARROWS to move the characters until they are standing\n"
        "on the same icons as in the picture on the left  "
    )

    details = app.challenge_support_details(question, (2000, 400))

    assert details["supported"] is True
    assert details["questionMatched"] is True
    assert details["sizeMatched"] is True
    assert details["imageSize"] == [2000, 400]


@pytest.mark.parametrize(
    ("question", "size", "question_matched", "size_matched"),
    [
        ("pick the image that is the correct way up", (2000, 400), False, True),
        (app.SUPPORTED_QUESTION, (1200, 400), True, False),
        (app.SUPPORTED_QUESTION, None, True, False),
        (app.SUPPORTED_QUESTION, "invalid", True, False),
    ],
)
def test_unsupported_question_or_layout_is_rejected(
    question, size, question_matched, size_matched
):
    details = app.challenge_support_details(question, size)

    assert details["supported"] is False
    assert details["questionMatched"] is question_matched
    assert details["sizeMatched"] is size_matched


def test_wait_supported_challenge_writes_evidence(tmp_path, monkeypatch):
    output = tmp_path / "question.json"
    monkeypatch.setattr(
        app.base,
        "captcha_text",
        lambda _tab: f"Arkose prompt: {app.SUPPORTED_QUESTION}",
    )

    details = app.wait_supported_challenge(
        object(), (2000, 400), timeout=0, output_path=output
    )

    assert details["supported"] is True
    assert output.is_file()
    assert app.SUPPORTED_QUESTION in output.read_text(encoding="utf-8")


def test_wait_unsupported_challenge_raises_and_writes_evidence(tmp_path, monkeypatch):
    output = tmp_path / "question.json"
    monkeypatch.setattr(app.base, "captcha_text", lambda _tab: "different challenge")

    with pytest.raises(app.UnsupportedCaptchaQuestion) as raised:
        app.wait_supported_challenge(
            object(), (2000, 400), timeout=0, output_path=output
        )

    assert app.UNSUPPORTED_CAPTCHA_EXIT_CODE == 42
    assert raised.value.details["questionMatched"] is False
    assert output.is_file()


def test_health_check_requires_ready_service(monkeypatch):
    calls = []

    def fake_get(url, timeout):
        calls.append((url, timeout))
        return FakeResponse({"ok": True, "status": "ready", "device": "cpu"})

    monkeypatch.setattr(app.requests, "get", fake_get)

    result = app.ensure_rank_v11_service("http://127.0.0.1:8765/", 12.5)

    assert result["device"] == "cpu"
    assert calls == [("http://127.0.0.1:8765/health", 12.5)]


def test_local_model_request_uses_accurate_mode_and_persists_result(
    tmp_path, monkeypatch
):
    image = tmp_path / "challenge.jpg"
    image.write_bytes(b"image")
    output = tmp_path / "response.json"
    calls = []
    payload = {
        "ok": True,
        "answer_index": 6,
        "fast_index": 6,
        "legacy_v9_index": 5,
        "expert_index": 6,
        "switched_to_v10": False,
        "model_seconds": 0.5,
    }

    def fake_post(url, *, json, timeout):
        calls.append((url, json, timeout))
        return FakeResponse(payload)

    monkeypatch.setattr(app.requests, "post", fake_post)

    answer, result = app.rank_v11_solve_image(
        image,
        base_url="http://127.0.0.1:8765/",
        timeout=30,
        response_path=output,
    )

    assert answer == 6
    assert result == payload
    assert calls == [
        (
            "http://127.0.0.1:8765/solve",
            {"image": str(image.resolve()), "mode": "accurate"},
            30,
        )
    ]
    assert '"answer_index": 6' in output.read_text(encoding="utf-8")


@pytest.mark.parametrize("answer", [-1, 10, "bad"])
def test_local_model_rejects_invalid_answer_index(tmp_path, monkeypatch, answer):
    image = tmp_path / "challenge.jpg"
    image.write_bytes(b"image")
    monkeypatch.setattr(
        app.requests,
        "post",
        lambda *args, **kwargs: FakeResponse({"ok": True, "answer_index": answer}),
    )

    with pytest.raises((RuntimeError, ValueError)):
        app.rank_v11_solve_image(
            image,
            base_url="http://127.0.0.1:8765",
            timeout=30,
            response_path=tmp_path / "response.json",
        )
