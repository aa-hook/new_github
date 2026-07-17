from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "register-ruyipage-v3.yml"


def load_workflow():
    return yaml.load(WORKFLOW.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def register_steps():
    return load_workflow()["jobs"]["register"]["steps"]


def step_command(name):
    return next(step["run"] for step in register_steps() if step.get("name") == name)


def test_dispatch_exposes_only_count_and_parallelism():
    workflow = load_workflow()

    assert list(workflow["on"]["workflow_dispatch"]["inputs"]) == [
        "count",
        "max_parallel",
    ]


def test_local_v11_service_is_persistent_cpu_accurate_and_waited_until_ready():
    command = step_command("Start persistent local V11 service")

    assert "rank_v11/server.py" in command
    assert "--device cpu" in command
    assert "--mode accurate" in command
    assert "--cpu-threads 2" in command
    assert "rank_v11/wait_ready.py" in command
    assert "rank_v11_server.pid" in command
    assert "cat rank_v11_server.log" in command


def test_registration_uses_v3_local_service_and_stops_retrying_on_exit_42():
    command = step_command("Run RuyiPage v3 registration")

    assert "for attempt in 1 2 3" in command
    assert "register_ruyipage_v3.py" in command
    assert '--rank-v11-url "$RANK_V11_URL"' in command
    assert '--click-style human' in command
    assert 'if [ "$last_rc" -eq 42 ]' in command
    assert "exit 42" in command
    assert "yescaptcha" not in command.lower()


def test_workflow_has_no_yescaptcha_secret_or_remote_classifier():
    text = WORKFLOW.read_text(encoding="utf-8").lower()

    assert "yescaptcha" not in text
    assert "clientkey" not in text
    assert "funcaptchaclassification" not in text


def test_v11_package_and_all_checkpoints_are_present():
    assert (ROOT / "rank_v11" / "server.py").is_file()
    assert (ROOT / "rank_v11" / "solve.py").is_file()
    manifest = ROOT / "rank_v11" / "models" / "model_manifest.json"
    assert manifest.is_file()
    for checkpoint in (
        "route_v11_fast_v10.pt",
        "route_v11_legacy_v9.pt",
        "route_v11_expert336.pt",
    ):
        assert (manifest.parent / checkpoint).stat().st_size > 1_000_000


def test_v3_artifacts_use_local_output_tree():
    workflow = load_workflow()
    upload = next(
        step
        for step in workflow["jobs"]["register"]["steps"]
        if step.get("name") == "Upload screenshots and debug data"
    )

    assert "rank_v11_server.log" in upload["with"]["path"]
    assert "ruyipage_local_v11_register/runs/**" in upload["with"]["path"]
