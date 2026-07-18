from pathlib import Path

from battle_protocol_flow_v4 import (
    BattleProtocolClient,
    PersistentFlowState,
    build_step_fields,
    detect_arkose_context,
    parse_flow_form,
    validate_transition,
)


class FakeResponse:
    def __init__(self, body: str, url: str, status: int = 200) -> None:
        self.text = body
        self.content = body.encode("utf-8")
        self.url = url
        self.status_code = status
        self.headers = {"content-type": "text/html;charset=UTF-8"}


class FakeSession:
    def __init__(self, responses: list[FakeResponse]) -> None:
        from curl_cffi.requests import Cookies

        self.responses = list(responses)
        self.headers = {}
        self.cookies = Cookies()
        self.calls = []

    def request(self, method: str, url: str, **kwargs):
        self.calls.append({"method": method, "url": url, **kwargs})
        return self.responses.pop(0)


def flow_form(step: str, csrf: str, controls: str = "") -> str:
    return f"""
    <form id="flow-form" method="post"
          action="https://HOST/creation/flow/creation-full/step/{step}">
      <input type="hidden" name="_csrf" value="{csrf}">
      {controls}
    </form>
    """


def test_get_started_fields_force_gbr_and_preserve_server_dob_format():
    form = parse_flow_form(
        flow_form(
            "get-started",
            "csrf-1",
            '<input type="hidden" name="dob-format" value="DMY">',
        ),
        "https://HOST/creation/flow/creation-full",
    )

    fields = build_step_fields(
        form,
        "get-started",
        {"birth_year": "1994", "birth_month": "02", "birth_day": "10"},
        country="GBR",
    )

    assert ("country", "GBR") in fields
    assert ("dob-format", "DMY") in fields
    assert ("dob-day", "10") in fields
    assert ("dob-month", "02") in fields


def test_tou_form_selection_and_redirect_transitions_are_supported():
    html = """
    <form id="flow-form" action="creation-full/step/initial-tou-agreement">
      <input type="hidden" name="_csrf" value="country-csrf">
      <select name="country"><option selected value="GBR">UK</option></select>
    </form>
    <form action="creation-full/step/initial-tou-agreement">
      <input type="hidden" name="_csrf" value="tou-csrf">
      <input type="checkbox" name="tou-agreements-explicit" value="doc;1">
    </form>
    """

    form = parse_flow_form(
        html,
        "https://HOST/creation/flow/creation-full",
        preferred_control="tou-agreements-explicit",
    )

    assert form.csrf == "tou-csrf"
    validate_transition("initial-tou-agreement", "row-redirect-to-tassadar")
    validate_transition("row-redirect-to-tassadar", "initial-tou-agreement")


def test_server_rendered_arkose_context_is_detected():
    blob = "B" * 717
    context = detect_arkose_context(
        '<input data-arkose-src="//fixture-api.arkoselabs.com/v2/'
        'E8A75615-1CBA-5DFF-8032-D16BCF234E10/api.js" '
        f'data-arkose-exchange-data="{blob}">',
        "https://HOST/creation/flow/creation-full",
    )

    assert context["blob"] == blob
    assert context["siteKey"] == "E8A75615-1CBA-5DFF-8032-D16BCF234E10"
    assert context["surl"] == "fixture-api.arkoselabs.com"


def test_default_country_probe_flow_reaches_captcha_and_submits_token(tmp_path: Path):
    entry = "https://HOST/creation/flow/creation-full"
    login = """
    <form method="post" action="https://LOGIN_HOST/login/en/?flowTrackingId=test">
      <input type="hidden" name="csrftoken" value="login-csrf">
      <input name="accountName" type="email">
    </form>
    """
    legal = """
      <input type="hidden" name="opt-in-blizzard-news-special-offers" value="false">
      <input type="checkbox" name="tou-agreements-implicit" value="agreement-a">
    """
    blob = "C" * 180
    responses = [
        FakeResponse(login, "https://LOGIN_HOST/login/en/"),
        FakeResponse(
            flow_form(
                "get-started",
                "csrf-1",
                '<input type="hidden" name="dob-format" value="DMY">',
            ),
            entry,
        ),
        FakeResponse(
            flow_form(
                "get-started",
                "csrf-probed",
                '<input type="hidden" name="dob-format" value="DMY">',
            ),
            entry,
        ),
        FakeResponse(flow_form("provide-name", "csrf-2"), entry),
        FakeResponse(flow_form("provide-credentials", "csrf-3"), entry),
        FakeResponse(flow_form("legal-and-opt-ins", "csrf-4", legal), entry),
        FakeResponse(flow_form("set-password", "csrf-5"), entry),
        FakeResponse(flow_form("set-battletag", "csrf-6"), entry),
        FakeResponse(
            flow_form("captcha-gate", "csrf-7")
            + f'<input data-arkose-exchange-data="{blob}">',
            entry,
        ),
        FakeResponse(
            """
            <i id="step-meta-data" data-step-id="create-success"
               data-step-has-errors="false"></i>
            <i id="player-id" data-player-account-id="987"></i>
            <p class="step__banner--account-identifier">user@example.test</p>
            """,
            entry,
        ),
    ]
    session = FakeSession(responses)
    state = PersistentFlowState.create(
        tmp_path / "state.json",
        identity={
            "email": "user@example.test",
            "password": "Password123A",
            "battle_tag": "Fixture77",
            "first_name": "Test",
            "last_name": "User",
            "birth_year": "1994",
            "birth_month": "02",
            "birth_day": "10",
        },
    )
    client = BattleProtocolClient(
        state,
        tmp_path,
        entry_url=entry,
        proxy="http://u:p@proxy.test:8080",
        session=session,
    )

    form = client.run_to_captcha(country="GBR")
    outcome = client.submit_captcha("TOKEN_FROM_RUYI")

    assert form.step == "captcha-gate"
    assert state.data["arkose"]["blob"] == blob
    assert state.data["countryProbed"] is True
    assert outcome["status"] == "success"
    assert outcome["playerAccountId"] == "987"
    assert state.data["status"] == "complete"
    assert b"TOKEN_FROM_RUYI" in session.calls[-1]["data"]
    assert session.calls[2]["method"] == "PUT"
    assert b"csrf-1" in session.calls[2]["data"]
    assert session.calls[3]["method"] == "POST"
    assert b"csrf-probed" in session.calls[3]["data"]
    assert all(
        call["proxies"]["https"] == "http://u:p@proxy.test:8080"
        for call in session.calls
    )
    assert not session.responses
