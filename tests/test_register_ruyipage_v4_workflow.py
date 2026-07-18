from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "register-ruyipage-v4.yml"


def load_workflow():
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def register_steps():
    return load_workflow()["jobs"]["register"]["steps"]


def step_command(name):
    return next(step["run"] for step in register_steps() if step.get("name") == name)


def test_dispatch_has_count_parallel_and_optional_proxy_inputs():
    inputs = load_workflow()["on"]["workflow_dispatch"]["inputs"]

    assert list(inputs) == ["count", "max_parallel", "proxy"]
    assert inputs["proxy"]["default"] == ""
    assert "ip:port:username:password" in inputs["proxy"]["description"]


def test_matrix_supports_256_jobs_but_caps_parallelism_at_20():
    command = load_workflow()["jobs"]["prepare"]["steps"][0]["run"]
    strategy = load_workflow()["jobs"]["register"]["strategy"]

    assert 'bounded_int("count", os.environ["REQUESTED_COUNT"], 256)' in command
    assert '"max_parallel", os.environ["REQUESTED_MAX_PARALLEL"], 20' in command
    assert "min(requested_parallel, count, 20)" in command
    assert "needs.prepare.outputs.max_parallel" in strategy["max-parallel"]
    assert strategy["fail-fast"] == "false"


def test_dependency_cache_is_minimal_and_has_no_wheelhouse():
    workflow = load_workflow()
    steps = workflow["jobs"]["deps-cache"]["steps"]
    text = WORKFLOW.read_text(encoding="utf-8").lower()

    cache_steps = [step for step in steps if str(step.get("uses", "")).startswith("actions/cache@")]
    assert len(cache_steps) == 2
    assert "wheelhouse" not in text
    assert "requirements-ruyipage-v4.txt" in text
    assert "cloakbrowser" not in text
    assert "playwright" not in text
    assert "selenium" not in text
    assert "mihomo" not in text


def test_registration_uses_http_v4_ruyi_local_v11_and_optional_proxy():
    command = step_command("Run HTTP + RuyiPage V4 registration")
    step = next(
        item
        for item in register_steps()
        if item.get("name") == "Run HTTP + RuyiPage V4 registration"
    )

    assert "register_ruyipage_v4.py" in command
    assert '--proxy "$REGISTRATION_PROXY"' in command
    assert "--country-probe" in command
    assert 'echo "::add-mask::$REGISTRATION_PROXY"' in command
    assert "--click-style balanced" in command
    assert "for attempt in 1 2 3" in command
    assert 'if [ "$last_rc" -eq 42 ]' in command
    assert step["env"]["REGISTRATION_PROXY"] == "${{ inputs.proxy }}"


def test_v11_starts_in_background_while_http_flow_begins():
    command = step_command("Start persistent local V11 service")

    assert "nohup .venv/bin/python rank_v11/server.py" in command
    assert "rank_v11_server.pid" in command
    assert "wait_ready.py" not in command


def test_v4_outputs_and_merged_artifact_are_version_isolated():
    text = WORKFLOW.read_text(encoding="utf-8")

    assert "ruyipage_http_v11_register/runs/**" in text
    assert "ruyi-v4-http-local-reg-data-${{ matrix.index }}" in text
    assert "ruyi-v4-http-local-reg-accounts-${{ matrix.index }}" in text
    assert "all-ruyipage-v4-http-local-registered-accounts" in text


def test_v4_requirements_keep_only_the_ruyi_http_and_v11_runtime():
    requirements = (ROOT / "requirements-ruyipage-v4.txt").read_text(
        encoding="utf-8"
    ).lower()

    assert "ruyipage" in requirements
    assert "curl_cffi" in requirements
    assert "beautifulsoup4" in requirements
    assert "cloakbrowser" not in requirements
    assert "playwright" not in requirements
    assert "selenium" not in requirements
